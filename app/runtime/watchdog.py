"""Engine watchdog: detect a dead engine under a live wrapper, preserve the
evidence, and restore the model to its pre-crash condition.

Why this exists (2026-08-17). ``RuntimeSupervisor._watch_exit`` awaits
``handle.wait()`` on the process the driver spawned — for the local driver that
is the ``vllm serve`` **wrapper**. vLLM v1 runs ``EngineCore`` and the TP workers
underneath it. Four times in one day EngineCore died while the wrapper stayed
alive, so the wait never returned, ``on_exit`` never fired, the row kept
``status='loaded'`` and the proxy forwarded to a dead engine indefinitely:

    794285  uvicorn (warden api)
    800137  └─ vllm serve          <-- ALIVE, RSS 2.1 GB
    801311     └─ resource_tracker   <-- only child left

The wrapper is also an unreliable signal in the other direction: at 07:39 it did
exit, at 09:03 it was still alive seven minutes later. So this watchdog does not
watch processes at all — it asks the engine's own /health endpoint, which is the
same signal the load path already trusts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from app.db.database import open_db
from app.db.repos.models import ModelRepo
from app.db.repos.settings import SettingsRepo

logger = logging.getLogger(__name__)

PROBE_TIMEOUT_S = 5.0


# Runtime-tunable knobs, read from the settings KV on every tick so an operator's
# change in the Settings UI takes effect on the next pass rather than at the next
# warden restart. The env-derived Settings values are the fallback/default.
WATCHDOG_KEYS = (
    "watchdog_enabled",
    "watchdog_restore_on_boot",
    "watchdog_interval_s",
    "watchdog_failure_threshold",
    "watchdog_max_restarts",
)


async def live_config(settings) -> dict:
    """Effective watchdog config: KV overrides on top of the env defaults."""
    cfg = {
        "enabled": settings.watchdog_enabled,
        "restore_on_boot": True,
        "interval_s": settings.watchdog_interval_s,
        "failure_threshold": settings.watchdog_failure_threshold,
        "max_restarts": settings.watchdog_max_restarts,
    }
    try:
        async with open_db(settings.db_path) as db:
            kv = await SettingsRepo(db).get_many(list(WATCHDOG_KEYS))
    except Exception:  # noqa: BLE001 — a KV read must never stop the watchdog
        logger.warning("watchdog: could not read settings KV; using defaults")
        return cfg
    if (v := kv.get("watchdog_enabled")) is not None:
        cfg["enabled"] = v in ("1", "true", "True")
    if (v := kv.get("watchdog_restore_on_boot")) is not None:
        cfg["restore_on_boot"] = v in ("1", "true", "True")
    for key, name in (
        ("watchdog_interval_s", "interval_s"),
        ("watchdog_failure_threshold", "failure_threshold"),
        ("watchdog_max_restarts", "max_restarts"),
    ):
        if (v := kv.get(key)) is not None:
            try:
                cfg[name] = int(v)
            except ValueError:
                logger.warning("watchdog: ignoring non-numeric %s=%r", key, v)
    return cfg


async def probe_health(host: str, port: int) -> tuple[bool, str]:
    """Probe the engine's own /health. Returns (healthy, detail-for-the-record).

    Both a non-200 and a connection error count as unhealthy. The detail string
    is stored in the crash report: if a dead engine ever answers 200 here, that
    record is what proves this probe is too weak and must escalate to a real
    generation.
    """
    url = f"http://{host}:{port}/health"
    try:
        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_S) as c:
            r = await c.get(url)
        return r.status_code == 200, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as exc:  # noqa: BLE001 — any failure to reach it is unhealthy
        return False, f"{type(exc).__name__}: {exc}"


def _write(path: Path, text: str) -> None:
    path.write_text(text, errors="replace")


def _capture_command(path: Path, argv: list[str]) -> None:
    out = subprocess.run(  # noqa: S603
        argv, capture_output=True, text=True, timeout=15, check=False
    )
    _write(path, out.stdout or out.stderr)


def _copy_tail(src: Path, dst: Path, max_bytes: int) -> None:
    """Copy at most the last ``max_bytes`` of ``src`` into ``dst``.

    The crash is at the end of the file, so the tail is the part anyone reads.
    Copying the whole log made /data occupancy scale with uptime rather than
    with a fixed retention count (see Settings.watchdog_crash_log_bytes).
    """
    size = src.stat().st_size
    with src.open("rb") as fh:
        if size > max_bytes:
            fh.seek(size - max_bytes)
        with dst.open("wb") as out:
            if size > max_bytes:
                out.write(
                    b"[truncated: copied the last %d of %d bytes]\n"
                    % (max_bytes, size)
                )
            shutil.copyfileobj(fh, out)


def capture_evidence(
    settings, model_id: str, *, report: dict
) -> Path | None:
    """Write the crash scene to disk BEFORE recovery destroys it.

    Every step is individually guarded: evidence collection must never prevent a
    restart. A missing artefact is logged, not raised.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    crash_dir = settings.crashes_dir / model_id / stamp
    try:
        crash_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.exception("watchdog: cannot create crash dir for %s", model_id)
        return None

    for name, fn in (
        # The live log is reused by the next load, so the crash tail would
        # otherwise be buried under a fresh startup.
        ("engine.log", lambda: _copy_tail(
            settings.logs_dir / f"{model_id}.log", crash_dir / "engine.log",
            settings.watchdog_crash_log_bytes)),
        # Shows whether the wrapper outlived the engine — the signature failure.
        ("processes.txt", lambda: _capture_command(
            crash_dir / "processes.txt",
            ["ps", "-eo", "pid,ppid,stat,etime,rss,args"])),
        ("meminfo.txt", lambda: _write(
            crash_dir / "meminfo.txt",
            Path("/proc/meminfo").read_text()
            # Node-wide and NOT namespaced: settles OOM questions with no node
            # access. It read 0 across all four deaths, refuting the OOM theory.
            + "\n--- /proc/vmstat oom ---\n"
            + "\n".join(ln for ln in Path("/proc/vmstat").read_text().splitlines()
                        if "oom" in ln))),
        ("gpu.txt", lambda: _capture_command(
            crash_dir / "gpu.txt", ["nvidia-smi"])),
    ):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 — best effort, never fatal
            logger.warning("watchdog: could not capture %s: %s", name, exc)

    try:
        _write(crash_dir / "report.json", json.dumps(report, indent=2, default=str))
    except Exception:  # noqa: BLE001
        logger.exception("watchdog: could not write report.json")
    return crash_dir


def prune_crash_dirs(settings, model_id: str) -> None:
    """Keep only the newest N crash dirs so a crash loop cannot fill the volume."""
    root = settings.crashes_dir / model_id
    try:
        dirs = sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: d.name)
    except OSError:
        return
    for stale in dirs[: max(0, len(dirs) - settings.watchdog_crash_keep)]:
        try:
            shutil.rmtree(stale)
        except OSError:
            logger.warning("watchdog: could not prune %s", stale)


class RestartBudget:
    """Crash-loop guard. An engine that dies on startup must not be restarted
    forever: that burns GPU cycles and buries the first crash under identical
    reports."""

    def __init__(self, max_restarts: int, window_s: float) -> None:
        self._max = max_restarts
        self._window = window_s
        self._events: dict[str, list[float]] = {}

    def allow(self, model_id: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        events = [t for t in self._events.get(model_id, []) if now - t < self._window]
        self._events[model_id] = events
        return len(events) < self._max

    def record(self, model_id: str, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self._events.setdefault(model_id, []).append(now)

    def reset(self, model_id: str) -> None:
        self._events.pop(model_id, None)

    def reconfigure(self, max_restarts: int) -> None:
        """Apply a live settings change without losing the crash history."""
        self._max = max_restarts


def _gpu_compute_apps() -> list[tuple[int, int]]:
    """[(pid, used_MiB)] from nvidia-smi. Empty on any failure."""
    try:
        out = subprocess.run(  # noqa: S603
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except Exception:  # noqa: BLE001
        return []
    apps = []
    for line in out.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            apps.append((int(parts[0]), int(parts[1])))
    return apps


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            errors="replace"
        )
    except OSError:
        return ""


def _ppid(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("PPid:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        return None
    return None


def _has_live_ancestor(pid: int, live_pids: set[int]) -> bool:
    """Walk up to see whether this process still belongs to a running engine.

    The container shares the host PID namespace, so orphans are reparented to the
    container shim rather than PID 1 — "ppid == 1" is NOT a valid orphan test here.
    Ancestry against the supervisor's live handles is.
    """
    seen = 0
    cur = _ppid(pid)
    while cur and cur > 1 and seen < 20:
        if cur in live_pids:
            return True
        cur = _ppid(cur)
        seen += 1
    return False


def reap_orphan_gpu_holders(live_pids: set[int]) -> list[int]:
    """SIGKILL vLLM worker processes still holding VRAM with no live engine.

    ``unload(force=True)`` only reaps the process the driver spawned — the
    ``vllm serve`` wrapper. The TP workers are its grandchildren and they SURVIVE
    it: observed 2026-08-17 with four ``VLLM::Worker_TP*`` processes holding
    10550 MiB each long after EngineCore and the wrapper were gone. Any reload
    then dies with "Free memory on device cuda:0 (4.98/15.6 GiB) is less than
    desired (14.82 GiB)", which is a wasted restart and a wasted crash budget.

    Only processes that both (a) hold GPU memory, (b) look like vLLM workers, and
    (c) have no living engine ancestor are killed — so a healthy model's workers,
    which still have their wrapper, are never touched.
    """
    killed = []
    for pid, _used in _gpu_compute_apps():
        cmd = _cmdline(pid)
        if "VLLM::" not in cmd and "vllm" not in cmd.lower():
            continue  # not ours; never touch a foreign GPU process
        if _has_live_ancestor(pid, live_pids):
            continue  # belongs to a running engine
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
            logger.warning("watchdog: killed orphaned GPU holder %s (%s)", pid, cmd[:60])
        except (ProcessLookupError, PermissionError) as exc:
            logger.warning("watchdog: could not kill orphan %s: %s", pid, exc)
    return killed


async def wait_for_gpu_release(
    live_pids: set[int], *, timeout_s: float = 30.0, interval_s: float = 2.0
) -> bool:
    """Wait until no orphaned vLLM process holds VRAM. Freeing is not instant
    after SIGKILL, and reloading too early reproduces the very failure we are
    recovering from."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        remaining = [
            pid for pid, _ in _gpu_compute_apps()
            if ("VLLM::" in _cmdline(pid) or "vllm" in _cmdline(pid).lower())
            and not _has_live_ancestor(pid, live_pids)
        ]
        if not remaining:
            return True
        await asyncio.sleep(interval_s)
    return False


async def _restart(settings, app_state, model_id: str, overrides) -> str:
    """Force-unload the corpse, then reload through the shared load path."""
    sup = app_state.supervisor
    # force=True is the step a plain reload misses: the stale wrapper still holds
    # its port and GPU claims, so a fresh spawn would fail on both.
    await sup.unload(model_id, force=True)

    # force-unload kills only the wrapper the driver spawned; the TP workers are
    # its grandchildren and survive, still holding all the VRAM.
    live_pids = {p for p in (sup.get_pid(m) for m in sup._handles) if p}  # noqa: SLF001
    killed = reap_orphan_gpu_holders(live_pids)
    if killed and not await wait_for_gpu_release(live_pids):
        logger.error(
            "watchdog: VRAM still held after killing %s; reloading anyway", killed
        )

    async with open_db(settings.db_path) as db:
        model = await ModelRepo(db).get(model_id)
    if model is None:
        return "model row vanished"

    # Imported here: routes_api imports heavy siblings, and a module-level import
    # would make the runtime package depend on the API package.
    from app.models.routes_api import start_engine  # noqa: PLC0415

    # The load ROUTE marks the row 'loading' before calling start_engine; this
    # path must do the same or the row reads 'loaded' for the whole reload and
    # lies to the UI. Deliberately here and not inside start_engine: an extra
    # write at the top of the shared path serializes against SQLite's 30s
    # busy_timeout and can delay every engine spawn.
    # Captured BEFORE the 'loading' write below, which clears prior_status by
    # design (see ModelRepo.update_status). Without this the failure path
    # cannot put the flag back and the row loses its only claim to a retry.
    was = model.prior_status or "loaded"

    async with open_db(settings.db_path) as db:
        await ModelRepo(db).update_status(model_id, "loading")

    port = app_state.port_allocator.allocate()
    try:
        await start_engine(
            settings, sup, app_state.port_allocator, model, port, overrides
        )
    except Exception as exc:  # noqa: BLE001 — re-raised after repairing state
        # A spawn that fails here would otherwise strand the row in 'loading'
        # FOREVER: nothing resets it, on_exit never fires (no process was ever
        # supervised long enough to exit), and no recovery path inspects
        # 'loading'. Prod sat dead 11.5h in exactly this state on 2026-08-18 --
        # the sweep restarted a crashed engine, the respawn produced no engine
        # log at all, and the row was still 'loading' when the pod restarted
        # hours later.
        #
        # Put it back to 'failed' and restore the restart flag, so the next
        # sweep retries it under the RestartBudget rather than the row silently
        # leaving the recoverable set.
        async with open_db(settings.db_path) as db:
            await ModelRepo(db).update_status(
                model_id, "failed", last_error=f"restart failed to spawn: {exc}"
            )
            await ModelRepo(db).set_prior_status(model_id, was)
        app_state.port_allocator.release(port)
        raise

    async with open_db(settings.db_path) as db:
        row = await ModelRepo(db).get(model_id)
    return row.status if row else "unknown"


def wants_restart(row) -> bool:
    """True when a row is a crashed engine that WAS serving, so restart it.

    One predicate, deliberately, for both ways a serving engine can end up
    dead: the warden restarted underneath it (boot reconciliation sets
    prior_status), or the engine died while the warden kept running
    (``on_exit`` sets it). Those used to be handled by different mechanisms
    and only the first one worked.

    This reads ``prior_status``, NOT ``last_error``. The old code compared
    last_error against an exact sentence, so any diagnosis written on the
    crash path displaced the sentinel and silently disabled recovery -- the
    2026-08-18 outage, where the model sat dead for 1h43m advertising a
    completely unrelated "requires trust_remote_code" message.

    ``loading`` counts too, and that is safe for a non-obvious reason: the
    ONLY writer of ``prior_status='loading'`` is boot reconciliation, i.e.
    "the warden restarted while this model was mid-load". ``on_exit`` sets
    ``prior_status`` exclusively for rows that were ``loaded``, so a model
    that failed on its own merits during a user-initiated load never gets the
    flag and is never retried here. Bad configs still stay down for a human.

    Without ``loading``, a model interrupted while it was being auto-restarted
    became permanently unrecoverable -- which is exactly how prod stayed dead
    for 11.5 hours on 2026-08-18/19: the engine crashed, the sweep restarted
    it, the respawn stalled in ``loading``, the pod later restarted, and the
    row was then ineligible forever.

    ``pulling`` and ``unloading`` are deliberately excluded: a half-finished
    pull is not a serving model, and an unload is an operator's stated intent.
    """
    return row.status == "failed" and (row.prior_status or "") in (
        "loaded",
        "loading",
    )


async def restore_after_warden_restart(settings, app_state) -> list[str]:
    """Reload models that were serving when the warden itself went down.

    The engine is a child of the warden process, so a warden restart takes the
    engine with it and boot reconciliation marks the row `failed`. Nothing then
    brings it back: observed 2026-08-17, when a routine redeploy left the model
    unloaded and every request 404'd until a human loaded it by hand.

    Only rows marked as having been *loaded* are restored. A model that was
    mid-pull, or that a human deliberately unloaded, is left alone.
    """
    async with open_db(settings.db_path) as db:
        rows = await ModelRepo(db).list_all()
    candidates = [r for r in rows if wants_restart(r)]
    restored = []
    for row in candidates:
        logger.warning(
            "watchdog: %s was loaded before the warden restarted — restoring", row.id
        )
        try:
            outcome = await _restart(settings, app_state, row.id, None)
            restored.append(f"{row.id}:{outcome}")
        except Exception:  # noqa: BLE001 — one bad model must not block the others
            logger.exception("watchdog: could not restore %s", row.id)
    return restored


async def restart_crashed_models(settings, app_state, budget) -> list[str]:
    """Restart engines that died while the warden stayed up.

    ``check_once`` only probes rows the DB believes are ``loaded``. When the
    engine's own wrapper exits, the supervisor's ``_watch_exit`` gets there
    first and flips the row to ``failed`` -- at which point the health-probe
    loop stops looking at it forever and NOTHING restarts it. That is the gap
    that kept prod down twice on 2026-08-18 (09:15 and 13:11), both times
    until a human noticed.

    Shares the SAME ``RestartBudget`` as the probe path on purpose: a model
    flapping between the two paths must still hit one combined ceiling rather
    than getting a fresh allowance from each.

    Evidence is captured BEFORE the restart, because the restart overwrites
    the live engine log with a fresh startup. The probe path already did this;
    the exit path did not, which is why /data/crashes was empty after a real
    crash.
    """
    async with open_db(settings.db_path) as db:
        rows = await ModelRepo(db).list_all()
    out = []
    for row in (r for r in rows if wants_restart(r)):
        if not budget.allow(row.id):
            logger.error(
                "watchdog: %s exceeded the restart budget — leaving it failed", row.id
            )
            # Drop the flag so the sweep stops retrying it every tick and the
            # row settles into a state an operator has to look at.
            async with open_db(settings.db_path) as db:
                await ModelRepo(db).set_prior_status(row.id, None)
            out.append(f"{row.id}:budget-exhausted")
            continue
        logger.warning("watchdog: %s died while serving — restarting", row.id)
        capture_evidence(
            settings,
            row.id,
            report={
                "model_id": row.id,
                "detected_by": "exit-path sweep",
                "last_error": row.last_error,
                "prior_status": row.prior_status,
            },
        )
        prune_crash_dirs(settings, row.id)
        budget.record(row.id)
        try:
            out.append(f"{row.id}:{await _restart(settings, app_state, row.id, None)}")
        except Exception:  # noqa: BLE001 — one bad model must not block the others
            logger.exception("watchdog: could not restart %s", row.id)
    return out


async def check_once(
    settings, app_state, state: dict, budget: RestartBudget, cfg: dict | None = None
) -> None:
    """One watchdog pass over every model the DB believes is loaded."""
    cfg = cfg or {
        "failure_threshold": settings.watchdog_failure_threshold,
        "max_restarts": settings.watchdog_max_restarts,
    }
    threshold = cfg["failure_threshold"]
    sup = app_state.supervisor
    async with open_db(settings.db_path) as db:
        rows = await ModelRepo(db).list_all()

    # 'loading' is deliberately skipped: the load path owns that window and has
    # its own timeout. Probing it would restart a model that is merely slow.
    loaded = [r for r in rows if r.status == "loaded"]
    live = {r.id for r in loaded}
    for gone in [k for k in state if k not in live]:
        state.pop(gone, None)

    for row in loaded:
        port = sup.get_port(row.id)
        if port is None:
            continue  # no runtime record yet; nothing to probe
        host = sup.get_host(row.id) or "127.0.0.1"
        healthy, detail = await probe_health(host, port)
        if healthy:
            if state.get(row.id):
                logger.info("watchdog: %s recovered on its own", row.id)
            state[row.id] = 0
            budget.reset(row.id)
            continue

        state[row.id] = state.get(row.id, 0) + 1
        fails = state[row.id]
        logger.warning(
            "watchdog: %s health probe failed (%d/%d): %s",
            row.id, fails, threshold, detail,
        )
        if fails < threshold:
            continue

        state[row.id] = 0
        handle = sup._handles.get(row.id)  # noqa: SLF001 — exit code is the evidence
        overrides = sup.get_overrides(row.id)   # preserves an auto-capped context
        report = {
            "model_id": row.id,
            "served_model_name": row.served_model_name,
            "detected_at": datetime.now(UTC).isoformat(),
            "consecutive_failures": fails,
            # If a dead engine ever answers 200 here, this field is the proof that
            # /health is too weak a probe and it must escalate to a generation.
            "health_probe": detail,
            # Negative => killed by that signal. This is the discriminator between
            # an external kill and a native crash, and it is exactly what was
            # missing when the root cause could not be determined.
            "wrapper_returncode": getattr(handle, "returncode", None),
            "wrapper_pid": sup.get_pid(row.id),
            "config": {
                "hf_repo": row.hf_repo,
                "gpu_indices": row.gpu_indices,
                "tensor_parallel_size": row.tensor_parallel_size,
                "max_model_len": row.max_model_len,
                "gpu_memory_utilization": row.gpu_memory_utilization,
                "extra_args": row.extra_args,
            },
            "effective_overrides": overrides,
        }

        if not budget.allow(row.id):
            report["restart"] = "refused: crash-loop guard"
            capture_evidence(settings, row.id, report=report)
            prune_crash_dirs(settings, row.id)
            msg = (
                f"engine died {cfg['max_restarts']}+ times within "
                f"{settings.watchdog_restart_window_s:.0f}s — automatic restart "
                f"disabled; see {settings.crashes_dir / row.id}"
            )
            logger.error("watchdog: %s", msg)
            async with open_db(settings.db_path) as db:
                await ModelRepo(db).update_status(row.id, "failed", last_error=msg)
            continue

        # Evidence BEFORE recovery — the restart destroys the scene.
        crash_dir = capture_evidence(settings, row.id, report=report)
        budget.record(row.id)
        logger.error(
            "watchdog: %s engine is dead (%s); evidence in %s — restarting",
            row.id, detail, crash_dir,
        )
        try:
            outcome = await _restart(settings, app_state, row.id, overrides)
        except Exception as exc:  # noqa: BLE001 — a failed restart must not kill the loop
            logger.exception("watchdog: restart of %s failed", row.id)
            outcome = f"restart raised: {type(exc).__name__}: {exc}"
        report["restart"] = outcome
        if crash_dir is not None:
            try:
                (crash_dir / "report.json").write_text(
                    json.dumps(report, indent=2, default=str)
                )
            except OSError:
                logger.warning("watchdog: could not update report.json")
        prune_crash_dirs(settings, row.id)
        logger.warning("watchdog: %s restart finished with status=%s", row.id, outcome)


async def run_watchdog_forever(settings, app_state) -> None:
    state: dict[str, int] = {}
    cfg = await live_config(settings)
    budget = RestartBudget(cfg["max_restarts"], settings.watchdog_restart_window_s)
    logger.info(
        "engine watchdog started (every %ss, %s consecutive failures to act, "
        "restore-on-boot=%s)",
        cfg["interval_s"], cfg["failure_threshold"], cfg["restore_on_boot"],
    )
    if cfg["restore_on_boot"]:
        try:
            restored = await restore_after_warden_restart(settings, app_state)
            if restored:
                logger.warning("watchdog: restored after warden restart: %s", restored)
        except Exception:
            logger.exception("watchdog: post-restart restore failed; continuing")
    while True:
        try:
            cfg = await live_config(settings)
            budget.reconfigure(cfg["max_restarts"])
            if cfg["enabled"]:
                await check_once(settings, app_state, state, budget, cfg)
                # Runs AFTER check_once so a model the probe path just gave up
                # on is picked up on the next tick rather than restarted twice
                # in the same pass. Covers the engines that died with their
                # wrapper, which check_once can never see (it only looks at
                # rows the DB believes are 'loaded').
                swept = await restart_crashed_models(settings, app_state, budget)
                if swept:
                    logger.warning("watchdog: restarted crashed engines: %s", swept)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("watchdog iteration failed; continuing")
        await asyncio.sleep(cfg["interval_s"])
