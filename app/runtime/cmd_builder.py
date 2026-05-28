"""Build the `vllm serve` argv from a ModelRow.

The `overrides` parameter lets callers apply a load-config (quantization,
tensor-parallel size, gpu_memory_utilization, max_model_len, max_num_seqs)
in memory without mutating the models row — useful for ad-hoc reload
flows. When `overrides=None`, behaviour matches the row defaults.
"""
from __future__ import annotations

import os
import re

# vLLM 0.20.0 requires GGUF models to be addressed as ``repo_id:quant_type``
# (e.g. ``unsloth/Qwen3.6-27B-GGUF:Q5_K_M``). #85 (v17.13) started shipping
# per-file GGUF downloads via the ``filename`` column, but the launcher kept
# emitting ``--model <hf_repo>`` only — every GGUF deployment died at vllm
# subprocess startup with rc=1. We recover the quant tag from the filename
# (``...-Q5_K_M.gguf``). The pattern allows extended variants like
# ``UD-Q4_K_XL`` by anchoring on ``-Q`` + a digit + arbitrary
# letters/digits/underscores up to ``.gguf``. When the filename is missing or
# the regex doesn't match we fall back to the bare hf_repo — same behaviour as
# pre-#85, so safetensors / non-quantized rows are unaffected. See issue #100.
_GGUF_QUANT_RE = re.compile(r"-(Q\d[_A-Za-z0-9]*)\.gguf$", re.IGNORECASE)


def _engine_bind_host() -> str:
    """Interface vLLM binds its HTTP server to. Loopback is correct for the
    in-container subprocess driver (engine shares the control-plane netns)
    and keeps the engine off the container's external interfaces. The docker
    socket driver runs the engine as a SEPARATE container, so the published
    port only works if vLLM binds 0.0.0.0 — that deployment sets
    ``VW_ENGINE_BIND_HOST=0.0.0.0``. Read per-call so the env is honoured
    without re-importing the module."""
    return os.environ.get("VW_ENGINE_BIND_HOST", "127.0.0.1")


# Override keys recognised by build_vllm_args. Any other key is a bug
# upstream (the supervisor builds the dict) and is ignored silently to
# stay forward-compatible with future load-config dimensions.
_OVERRIDE_KEYS = frozenset({
    "quantization",
    "tensor_parallel_size",
    "gpu_memory_utilization",
    "max_model_len",
    "max_num_seqs",
})


def build_vllm_args(
    model,
    *,
    port: int,
    overrides: dict | None = None,
) -> list[str]:
    """Construct the argv vector for `vllm serve <args>`.

    `overrides` may contain any subset of:
      - quantization           → emit --quantization <value>
      - tensor_parallel_size   → replace --tensor-parallel-size
      - gpu_memory_utilization → replace --gpu-memory-utilization
      - max_model_len          → replace --max-model-len (omit if None)
      - max_num_seqs           → emit --max-num-seqs <value>

    Overrides are read-only against the model row; the caller decides what
    to pass and is responsible for never persisting this to the DB.
    """
    ov = overrides or {}

    tp = ov.get("tensor_parallel_size", model.tensor_parallel_size)
    gpu_mem = ov.get("gpu_memory_utilization", model.gpu_memory_utilization)
    # `max_model_len` may be explicitly None either on the row or as an
    # override — treat both as "let vLLM pick the default" by omitting the flag.
    max_len = ov["max_model_len"] if "max_model_len" in ov else model.max_model_len
    # `quantization` lives only as an override + DB column; fall back to the
    # row's `quantization` attr if present (added by migration 0011) so the
    # legacy code path stays None-clean when the column hasn't been set.
    quantization = ov.get("quantization", getattr(model, "quantization", None))
    # `max_num_seqs` likewise — DB column from 0011, override slot in v2.
    max_num_seqs = ov.get("max_num_seqs", getattr(model, "max_num_seqs", None))

    # `parallelism_strategy` (#88) — wizard's tp/pp/auto choice from migration
    # 0014. ``auto`` and ``tp`` both emit ``--tensor-parallel-size`` (legacy
    # behaviour); ``pp`` swaps to ``--pipeline-parallel-size``. N is identical
    # in either case because ``ModelCreate._tp_consistent`` validates
    # ``tensor_parallel_size == len(gpu_indices)`` — there is one parallelism
    # dimension wide, only the flag name differs. Single-host PP is fine on
    # vLLM (CTO-decided in #82 plan); we do NOT block at builder level.
    # Legacy ``ModelRow`` instances (no column from migration 0014) decode with
    # ``parallelism_strategy='auto'`` default so this branch is safe pre-#85.
    strategy = getattr(model, "parallelism_strategy", "auto")
    parallelism_flag = (
        "--pipeline-parallel-size" if strategy == "pp" else "--tensor-parallel-size"
    )

    # GGUF: append ``:quant_type`` extracted from ``filename`` (#100). The
    # ``.lower().endswith(".gguf")`` guard skips safetensors rows even if some
    # future caller populates ``filename`` on them (e.g. partial downloads).
    model_arg = model.hf_repo
    filename = getattr(model, "filename", None)
    if filename and filename.lower().endswith(".gguf"):
        m = _GGUF_QUANT_RE.search(filename)
        if m:
            model_arg = f"{model.hf_repo}:{m.group(1)}"

    args: list[str] = [
        "--model", model_arg,
        "--host", _engine_bind_host(),
        "--port", str(port),
        "--served-model-name", model.served_model_name,
        parallelism_flag, str(tp),
        "--gpu-memory-utilization", str(gpu_mem),
    ]
    # dtype / max_model_len are optional on the model row — vLLM picks safe
    # defaults (auto / model-config max) when omitted. Emitting "--dtype None"
    # passes None into asyncio.create_subprocess_exec, which raises
    # "expected str, bytes or os.PathLike object, not NoneType" — the subprocess
    # never actually starts and the failure surfaces as last_error on the model
    # row rather than as a vLLM log line. Only forward these flags when set.
    if model.dtype:
        args += ["--dtype", model.dtype]
    if max_len is not None:
        args += ["--max-model-len", str(max_len)]
    if quantization:
        args += ["--quantization", str(quantization)]
    if max_num_seqs is not None:
        args += ["--max-num-seqs", str(max_num_seqs)]
    if model.hf_revision:
        args += ["--revision", model.hf_revision]
    # #106: GGUF repos that omit ``config.json`` (common for unsloth republishes)
    # require ``--hf-config-path <original_repo>`` so vLLM can find a config to
    # load. ``--tokenizer`` covers the same upstream-vs-quant split for the
    # tokenizer. Both columns are NULL on legacy / non-GGUF rows; the
    # ``getattr`` keeps stand-in test rows (no migration 0015 column) working.
    hf_config_repo = getattr(model, "hf_config_repo", None)
    if hf_config_repo:
        args += ["--hf-config-path", hf_config_repo]
    tokenizer_repo = getattr(model, "tokenizer_repo", None)
    if tokenizer_repo:
        args += ["--tokenizer", tokenizer_repo]
    # #173 part B — run vLLM's V1 scheduler in priority mode so the engine's
    # own waiting queue is ordered by the per-request ``priority`` field the
    # proxy injects (see app/proxy/routes.py). With all-equal priorities this
    # policy is identical to FCFS, so it's a safe unconditional default; a user
    # who wants plain FCFS can override by putting ``--scheduling-policy fcfs``
    # in ``extra_args`` (appended last → wins in argparse).
    args += ["--scheduling-policy", "priority"]
    extra_args = list(getattr(model, "extra_args", []) or [])
    args += extra_args
    return args
