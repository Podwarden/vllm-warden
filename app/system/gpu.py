import asyncio
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

NVIDIA_SMI_CMD = [
    "nvidia-smi",
    # power.draw is the 6th field — added in S7 (#124) so the stats sampler
    # gets util/mem/power in a SINGLE nvidia-smi acquisition per tick
    # (CTO decision #6: no second pass). Driver versions that don't support
    # power.draw on the queried card report "[Not Supported]"/"[N/A]" — the
    # parser turns those into ``power_w=None`` and the sampler logs+skips
    # the power write rather than corrupting the bucket.
    "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,power.draw",
    "--format=csv,noheader,nounits",
]

# Pull memory.free too for the live-probe endpoint (`/api/system/gpus`).
# Older `query_gpus()` shape stays the same so the existing DB-backed
# sampler does not change behaviour.
NVIDIA_SMI_LIVE_CMD = [
    "nvidia-smi",
    # compute_cap is the 8th field — added in #176 so fit-preview can cross
    # the detected GPU's CUDA compute capability (e.g. 8.6 for an A4000) with
    # the candidate's quant/dtype and warn on emulated/unsupported combos.
    # Drivers/cards that don't report it print "[Not Supported]"/"[N/A]"; the
    # parser turns those into ``compute_cap=None`` and keeps the row (same
    # degradation pattern as power.draw above).
    "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,compute_cap",
    "--format=csv,noheader,nounits",
]

NVIDIA_SMI_APPS_CMD = [
    "nvidia-smi",
    "--query-compute-apps=pid,gpu_uuid,process_name,used_memory",
    "--format=csv,noheader,nounits",
]


@dataclass
class GpuInfo:
    index: int
    name: str
    memory_total_mib: int
    memory_used_mib: int
    utilization_pct: int
    # S7 (#124) — current power draw in watts. ``None`` when the driver
    # reports "[Not Supported]" / "[N/A]" or the field can't be parsed.
    # New rows on cards that DO report power get a real float; the sampler
    # only writes to ``power_samples`` when this field is non-None.
    power_w: float | None = None


@dataclass
class GpuLive:
    index: int
    uuid: str
    name: str
    memory_total_mib: int
    memory_used_mib: int
    memory_free_mib: int
    utilization_pct: int
    # #176 — CUDA compute capability (e.g. 8.6 for sm_86). ``None`` when the
    # driver reports "[Not Supported]" / "[N/A]" or the field can't be parsed;
    # the capability-warning layer treats None as "can't classify" and emits
    # no warnings rather than fabricating one.
    compute_cap: float | None = None


@dataclass
class GpuComputeApp:
    pid: int
    gpu_uuid: str
    process_name: str
    memory_mib: int


@dataclass
class GpuSnapshot:
    """Result of a live probe of nvidia-smi: per-GPU stats + compute holders."""

    gpus: list[GpuLive] = field(default_factory=list)
    apps: list[GpuComputeApp] = field(default_factory=list)
    probe_error: str | None = None


def _parse_optional_float(raw: str) -> float | None:
    """Parse a numeric nvidia-smi field that may report "[Not Supported]" /
    "[N/A]" on cards or drivers without the capability. Returns the float
    on success, None on the documented sentinels, and None (with a debug
    log) on any other non-numeric value so a malformed row never crashes
    the sampler.
    """
    s = raw.strip()
    if not s or s.startswith("[") or s.upper() in {"N/A", "NOT SUPPORTED"}:
        return None
    try:
        return float(s)
    except ValueError:
        logger.debug("nvidia-smi non-numeric optional float skipped: %r", raw)
        return None


def parse_nvidia_smi_csv(stdout: str) -> list[GpuInfo]:
    out = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        # S7 (#124) — accept either 5-col (pre-power) or 6-col (with power.draw)
        # rows. Older nvidia-smi builds in dev fixtures sometimes emit the
        # 5-col form; the runtime image's nvidia-smi reliably emits 6.
        if len(parts) not in (5, 6):
            continue
        try:
            power_w = _parse_optional_float(parts[5]) if len(parts) == 6 else None
            out.append(GpuInfo(
                index=int(parts[0]),
                name=parts[1],
                memory_total_mib=int(parts[2]),
                memory_used_mib=int(parts[3]),
                utilization_pct=int(parts[4]),
                power_w=power_w,
            ))
        except ValueError:
            logger.warning("nvidia-smi malformed row skipped: %r", line)
            continue
    return out


def parse_nvidia_smi_live_csv(stdout: str) -> list[GpuLive]:
    """Parser for NVIDIA_SMI_LIVE_CMD (8 cols: uuid + memory.free + compute_cap).

    #176 — accept either 7-col (pre-compute_cap) or 8-col rows so older
    nvidia-smi builds in dev fixtures don't break; the runtime image emits 8.
    The compute_cap field degrades to None on "[Not Supported]"/"[N/A]" /
    unparseable values (same pattern as power.draw) without dropping the row.
    """
    out: list[GpuLive] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) not in (7, 8):
            continue
        try:
            compute_cap = _parse_optional_float(parts[7]) if len(parts) == 8 else None
            out.append(GpuLive(
                index=int(parts[0]),
                uuid=parts[1],
                name=parts[2],
                memory_total_mib=int(parts[3]),
                memory_used_mib=int(parts[4]),
                memory_free_mib=int(parts[5]),
                utilization_pct=int(parts[6]),
                compute_cap=compute_cap,
            ))
        except ValueError:
            logger.warning("nvidia-smi (live) malformed row skipped: %r", line)
            continue
    return out


def parse_compute_apps_csv(stdout: str) -> list[GpuComputeApp]:
    """Parser for nvidia-smi --query-compute-apps. Memory may be reported as
    '[Not Supported]' or '[N/A]' on some MIG / consumer cards — those rows are
    skipped, not crashed on."""
    out: list[GpuComputeApp] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        pid_s, gpu_uuid, process_name, used_mem = parts
        try:
            pid = int(pid_s)
            memory_mib = int(used_mem)
        except ValueError:
            logger.debug("nvidia-smi compute-apps non-numeric row skipped: %r", line)
            continue
        out.append(GpuComputeApp(
            pid=pid,
            gpu_uuid=gpu_uuid,
            process_name=process_name,
            memory_mib=memory_mib,
        ))
    return out


async def _run_nvidia_smi(args: list[str], *, timeout: float = 5.0) -> str | None:  # noqa: ASYNC109
    """Run nvidia-smi with `args` and return stdout, or None if it failed."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except (TimeoutError, FileNotFoundError) as e:
        logger.warning("nvidia-smi unavailable: %s", e)
        return None
    if proc.returncode != 0:
        logger.warning("nvidia-smi exit %d: %s", proc.returncode, stderr.decode())
        return None
    return stdout.decode()


async def query_gpus() -> list[GpuInfo]:
    """Run nvidia-smi and parse output. Returns [] if not available."""
    stdout = await _run_nvidia_smi(NVIDIA_SMI_CMD)
    if stdout is None:
        return []
    return parse_nvidia_smi_csv(stdout)


async def query_gpu_snapshot() -> GpuSnapshot:
    """Live probe: per-GPU stats + compute holders.

    If nvidia-smi is missing or fails, returns a snapshot with empty lists and
    a populated `probe_error` so the API can surface the cause without 500'ing.
    """
    live_out = await _run_nvidia_smi(NVIDIA_SMI_LIVE_CMD)
    if live_out is None:
        return GpuSnapshot(gpus=[], apps=[], probe_error="nvidia-smi unavailable")
    gpus = parse_nvidia_smi_live_csv(live_out)
    apps_out = await _run_nvidia_smi(NVIDIA_SMI_APPS_CMD)
    apps = parse_compute_apps_csv(apps_out) if apps_out is not None else []
    return GpuSnapshot(gpus=gpus, apps=apps, probe_error=None)
