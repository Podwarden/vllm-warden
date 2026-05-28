"""Build subprocess env for vLLM. THIS FIXES THE 2026-05-08 BUG.

CUDA_VISIBLE_DEVICES is derived from model.gpu_indices (a DB column populated by the
user's wizard/CRUD selection). It is NEVER inherited from the parent process. The
parent vllm-warden container is launched with all GPUs visible (the launcher gives
the container `--gpus all` so the supervisor can dispatch any GPU); each per-model
subprocess MUST have CUDA_VISIBLE_DEVICES restricted to exactly that model's
gpu_indices, in the order specified, so vLLM's logical device 0 == gpu_indices[0].

extra_env keys are filtered through an allowlist and cannot override CUDA_VISIBLE_DEVICES,
HF_HOME, the HF token, or PATH.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# Keys that extra_env is never permitted to override — security/incident lockdown.
#
# PYTHONUNBUFFERED is locked even though no ALLOWED_ENV_PREFIXES currently
# accepts the bare "PYTHON" prefix (so today an operator-supplied value would
# be silently dropped by the allowlist). The lock is defence-in-depth against
# a future hand adding a "PYTHON_" prefix for legitimate escape hatches like
# PYTHONFAULTHANDLER / PYTHONHASHSEED — that change would otherwise let
# PYTHONUNBUFFERED=0 flow through and silently undo the v2026.05.15.5 fix
# (block-buffered subprocess stdout swallowing fast-crash tracebacks).
HARD_LOCKED_ENV_KEYS: frozenset[str] = frozenset({
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "HF_HOME",
    "HUGGING_FACE_HUB_TOKEN",
    "HF_TOKEN",
    "PATH",
    "PYTHONUNBUFFERED",
})

# extra_env keys are accepted iff they start with one of these prefixes OR are
# exactly CUDA_MODULE_LOADING.  Everything else is silently dropped.
ALLOWED_ENV_PREFIXES: tuple[str, ...] = (
    "VLLM_",
    "TRITON_",
    "NCCL_",
    "PYTORCH_",
    "TORCH_",
    "OMP_",
)
ALLOWED_ENV_EXACT: frozenset[str] = frozenset({"CUDA_MODULE_LOADING"})


def _filter_extra_env(extra_env: dict[str, str]) -> dict[str, str]:
    """Validate and filter extra_env before merging into the subprocess env.

    Raises ValueError if any hard-locked key is present in extra_env.
    Silently drops keys that don't match the allowlist.
    """
    for key in extra_env:
        if key in HARD_LOCKED_ENV_KEYS:
            raise ValueError(
                f"extra_env key '{key}' is hard-locked and cannot be overridden"
            )

    filtered = {
        key: value
        for key, value in extra_env.items()
        if any(key.startswith(prefix) for prefix in ALLOWED_ENV_PREFIXES)
        or key in ALLOWED_ENV_EXACT
    }
    dropped = sorted(set(extra_env) - set(filtered))
    if dropped:
        log.info(
            "env_builder: dropped %d extra_env key(s) not on allowlist: %s",
            len(dropped),
            dropped,
        )
    return filtered


def build_subprocess_env(model, *, hf_token: str, data_dir: str) -> dict[str, str]:
    """Construct the env dict for a vLLM subprocess.

    Returns a closed dict. Caller passes this dict as the env= kwarg to
    asyncio.create_subprocess_exec, which uses ONLY this env (no inheritance).

    extra_env from the model row is merged in after validation: allowed keys
    override defaults (e.g. VLLM_LOGGING_LEVEL=DEBUG overrides INFO), but
    hard-locked keys (CUDA_VISIBLE_DEVICES, HF_HOME, HUGGING_FACE_HUB_TOKEN,
    PATH) cannot be overridden.
    """
    if not model.gpu_indices:
        raise ValueError("gpu_indices must be non-empty")
    if model.tensor_parallel_size != len(model.gpu_indices):
        raise ValueError(
            f"tensor_parallel_size ({model.tensor_parallel_size}) must equal "
            f"len(gpu_indices) ({len(model.gpu_indices)})"
        )

    extra_env = getattr(model, "extra_env", {}) or {}
    filtered = _filter_extra_env(extra_env)

    env = {
        "VLLM_LOGGING_LEVEL": "INFO",
        # Pin device ordering to PCI bus so vLLM's logical 0..N matches
        # gpu_indices on heterogeneous-GPU hosts (e.g. pw_prod `bonus` mixes
        # Quadro RTX 4000 + A4000). Without this, NVML may reorder by SM count
        # and break gpu_indices semantics.
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        # v2026.05.15.5 — Python defaults to block-buffered stdout when
        # stdout is a file fd (the supervisor's behaviour — see
        # supervisor.py opens the log with O_WRONLY|O_CREAT|O_APPEND
        # and passes it as stdout=). On a fast crash (rc=1 within <1s)
        # the subprocess can exit before the buffered output is
        # flushed, so the operator sees nothing in the log even though
        # Python printed the traceback. PYTHONUNBUFFERED=1 forces
        # line-buffered stdio and guarantees flush on exit — defence in
        # depth alongside the routes_logs open-or-create fix.
        "PYTHONUNBUFFERED": "1",
    }
    # Apply allowed extra_env first so hard-locked keys below always win.
    env.update(filtered)

    # Hard-locked keys are set last and cannot be overridden.
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in model.gpu_indices)
    env["HF_HOME"] = f"{data_dir}/hf-cache"
    env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    env["HF_TOKEN"] = hf_token
    env["PATH"] = "/usr/local/bin:/usr/bin:/bin"

    return env
