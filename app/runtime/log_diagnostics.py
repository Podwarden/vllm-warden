"""Turn a vLLM engine-log tail into an actionable, operator-facing error.

When a load fails, the control plane historically reported only a generic
``last_error`` ("vllm subprocess exited unexpectedly (rc=1)"). The real cause
is in the engine log (``{data_dir}/logs/{model_id}.log``). This pure parser
scans the tail and, for the failure modes we can recognise, produces a message
that tells the operator exactly what to change.

Matching is on STABLE tokens, not exact phrasing — vLLM's wording drifts across
versions and even contains typos (the real d5 string says ``models's``). We key
off invariants and capture numbers opportunistically. No match → ``None`` so
the caller keeps its existing generic message.

The two failure modes we distinguish carefully (both observed live on d5):

  1. **KV cache too small for context.** vLLM prints its OWN profiled fit
     estimate ("the estimated maximum model length is N"). That estimate beats
     any static heuristic, so we surface N as ``recommended_max_model_len`` and
     tell the operator to cap to it (or raise gpu_memory_utilization). This is
     the ``tencent/Hy-MT2-1.8B`` footgun.

  2. **No room for the cache blocks at all** ("No available memory for the
     cache blocks", typically with a NEGATIVE "Available KV cache memory"). The
     weights + overhead already exceed the budget; lowering ``max_model_len``
     does NOT help. We must NOT suggest it — only more/larger GPUs or a higher
     gpu_memory_utilization. ``recommended_max_model_len`` stays ``None``.

There is deliberately NO auto-retry here: the post-crash path only REPORTS.
Auto-relaunching on a parsed estimate risks oscillation; the static pre-spawn
preflight (``app/models/load_preflight.py``) is the only place we adjust a
launch, and only for the NULL-max_model_len case.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EngineDiagnosis:
    """An actionable diagnosis extracted from an engine log.

    ``message`` is operator-facing (goes straight into ``last_error``).
    ``recommended_max_model_len`` is vLLM's own profiled fit estimate when the
    failure was a context/KV overflow and we could capture it, else ``None``.
    """

    message: str
    recommended_max_model_len: int | None = None


# --- Variant 2: no room for the cache blocks (weights/util, NOT context). ----
# Checked FIRST because this line co-occurs with "KV cache" text and would
# otherwise be misclassified as a context overflow. Lowering max_model_len does
# not help here.
_NO_CACHE_BLOCKS_RE = re.compile(
    r"no\s+available\s+memory\s+for\s+the\s+cache\s+blocks", re.IGNORECASE
)

# --- Variant 1: KV cache too small for the requested context. ----------------
# Stable tokens: "max seq len" + "KV cache", or an explicit decrease-max_model_len
# instruction. vLLM's authoritative estimate, when present:
#   "the estimated maximum model length is 157216"
_MAX_SEQ_LEN_RE = re.compile(r"max\s+seq\s+len", re.IGNORECASE)
_KV_CACHE_RE = re.compile(r"kv\s+cache", re.IGNORECASE)
_DECREASE_MAXLEN_RE = re.compile(
    r"(decreas|lower|reduc)\w*\s+`?max_model_len", re.IGNORECASE
)
_ESTIMATED_MAX_LEN_RE = re.compile(
    r"estimated\s+maximum\s+model\s+length\s+is\s+(\d+)", re.IGNORECASE
)
# Model's declared max seq len, e.g. "max seq len (262144)".
_MODEL_SEQ_LEN_RE = re.compile(r"max\s+seq\s+len\s*\((\d+)\)", re.IGNORECASE)
# KV-cache GiB figures: "16.0 GiB KV cache is needed" / "available KV cache
# memory (9.6 GiB)". Captured loosely for the message only.
_KV_NEEDED_GIB_RE = re.compile(
    r"([\d.]+)\s*GiB\s+KV\s+cache\s+is\s+needed", re.IGNORECASE
)
_KV_AVAIL_GIB_RE = re.compile(
    r"available\s+KV\s+cache\s+memory\s*\(([\d.]+)\s*GiB\)", re.IGNORECASE
)

# --- CUDA OOM — torch exception class or the human string. -------------------
_CUDA_OOM_RE = re.compile(r"(cuda out of memory|outofmemoryerror)", re.IGNORECASE)

# --- trust_remote_code required prompt. --------------------------------------
_TRUST_REMOTE_CODE_RE = re.compile(r"trust_remote_code", re.IGNORECASE)


def diagnose_engine_log(text: str) -> EngineDiagnosis | None:
    """Scan an engine-log tail and return an actionable diagnosis, or None."""
    if not text or not text.strip():
        return None

    # Variant 2 first — it shares vocabulary with Variant 1 but the fix differs.
    if _NO_CACHE_BLOCKS_RE.search(text):
        return EngineDiagnosis(
            message=(
                "GPU has no room for the KV cache after loading the weights — "
                "the model is too large for the selected GPU(s) at this "
                "gpu_memory_utilization. Use more/larger GPUs or raise "
                "gpu_memory_utilization."
            ),
            recommended_max_model_len=None,
        )

    # Variant 1 — KV / context overflow.
    has_max_seq = bool(_MAX_SEQ_LEN_RE.search(text))
    has_kv_cache = bool(_KV_CACHE_RE.search(text))
    has_decrease = bool(_DECREASE_MAXLEN_RE.search(text))
    if (has_max_seq and has_kv_cache) or has_decrease:
        est_m = _ESTIMATED_MAX_LEN_RE.search(text)
        recommended = int(est_m.group(1)) if est_m else None

        model_len_m = _MODEL_SEQ_LEN_RE.search(text)
        model_len = model_len_m.group(1) if model_len_m else None
        kv_needed_m = _KV_NEEDED_GIB_RE.search(text)
        kv_avail_m = _KV_AVAIL_GIB_RE.search(text)

        if recommended is not None:
            parts = ["Context too long for GPU memory:"]
            if model_len and kv_needed_m:
                parts.append(
                    f" model wants {model_len} tokens (needs "
                    f"{kv_needed_m.group(1)} GiB KV cache)"
                )
            elif model_len:
                parts.append(f" model wants {model_len} tokens")
            if kv_avail_m:
                parts.append(f" but only {kv_avail_m.group(1)} GiB is available.")
            else:
                parts.append(".")
            parts.append(
                f" vLLM estimates the max workable context here is {recommended} "
                f"— set max_model_len <= {recommended}, or raise "
                f"gpu_memory_utilization."
            )
            return EngineDiagnosis(
                message="".join(parts), recommended_max_model_len=recommended
            )

        # No authoritative estimate to capture — still actionable.
        return EngineDiagnosis(
            message=(
                "KV cache too small for the requested context. Lower "
                "max_model_len or raise gpu_memory_utilization for the selected "
                "GPU(s)."
            ),
            recommended_max_model_len=None,
        )

    # CUDA out of memory.
    if _CUDA_OOM_RE.search(text):
        return EngineDiagnosis(
            message=(
                "GPU ran out of memory loading the model. Use fewer/larger GPUs, "
                "lower gpu_memory_utilization, or reduce max_model_len."
            ),
            recommended_max_model_len=None,
        )

    # trust_remote_code required.
    if _TRUST_REMOTE_CODE_RE.search(text):
        return EngineDiagnosis(
            message=(
                "This model requires trust_remote_code to load. Enable "
                "trust_remote_code for this model (it executes code from the "
                "model repo — only do this for repos you trust), then retry."
            ),
            recommended_max_model_len=None,
        )

    return None
