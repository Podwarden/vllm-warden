"""Unit tests for the engine-log diagnostic parser.

``diagnose_engine_log`` scans an engine-log tail and turns a raw vLLM crash
traceback into an actionable, operator-facing message. It matches on STABLE
tokens (not exact phrasing) because vLLM's wording drifts across versions.
Returns ``None`` when nothing matches so the caller keeps its generic message.

The two KV-related fixtures below are the EXACT strings observed on d5 (real
vLLM output), including their quirks (``models's`` typo, a stray ``(`` before
``16.0 GiB``). The parser must match these tolerantly — never anchor on exact
punctuation.
"""
from __future__ import annotations

from app.runtime.log_diagnostics import diagnose_engine_log

# --- Variant 1: KV cache too small for the requested context. ---------------
# vLLM prints its OWN authoritative fit estimate ("the estimated maximum model
# length is 157216"). That estimate is far better than our static math, so the
# parser MUST capture it and use it as the recommendation.
_KV_OVERFLOW_LOG = (
    "ValueError: To serve at least one request with the models's max seq len "
    "(262144), (16.0 GiB KV cache is needed, which is larger than the available "
    "KV cache memory (9.6 GiB). Based on the available memory, the estimated "
    "maximum model length is 157216. Try increasing `gpu_memory_utilization` or "
    "decreasing `max_model_len` when initializing the engine. See "
    "https://docs.vllm.ai/en/latest/configuration/conserving_memory/ for more "
    "details."
)

# --- Variant 2: no room for cache blocks at all (weights/util, NOT context). -
# Seen alongside "Available KV cache memory: -1.06 GiB" — negative means weights
# + overhead already exceed the budget; lowering max_model_len will NOT help.
_NO_CACHE_BLOCKS_LOG = (
    "INFO 05-25 worker.py:1 Available KV cache memory: -1.06 GiB\n"
    "ValueError: No available memory for the cache blocks. Try increasing "
    "`gpu_memory_utilization` when initializing the engine. See "
    "https://docs.vllm.ai/en/latest/configuration/conserving_memory/ for more "
    "details."
)

_CUDA_OOM_LOG = (
    "INFO 05-25 12:00:01 model_runner.py:1 Loading weights\n"
    "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB.\n"
    "GPU 0 has a total capacity of 15.99 GiB of which 1.20 GiB is free.\n"
)

_TRUST_REMOTE_CODE_LOG = (
    "ValueError: Loading this model requires you to execute the configuration "
    "file in that repo on your local machine. Make sure you have read the code "
    "there to avoid malicious use, then set the option `trust_remote_code=True` "
    "to remove this error.\n"
)

_UNRELATED_LOG = (
    "INFO 05-25 12:00:01 api_server.py:1 Started server process\n"
    "INFO 05-25 12:00:02 api_server.py:1 Application startup complete.\n"
    "INFO 05-25 12:00:03 api_server.py:1 Uvicorn running on http://0.0.0.0:8000\n"
)


def test_kv_overflow_uses_vllm_estimated_max_len():
    """Variant 1: vLLM's own profiled estimate (157216) wins over static math."""
    diag = diagnose_engine_log(_KV_OVERFLOW_LOG)
    assert diag is not None
    # The recommendation is vLLM's authoritative estimate, NOT a parenthesized
    # token count.
    assert diag.recommended_max_model_len == 157216
    # Message names the numbers the operator needs and what to change.
    assert "262144" in diag.message
    assert "157216" in diag.message
    assert "max_model_len" in diag.message
    assert "gpu_memory_utilization" in diag.message


def test_no_cache_blocks_does_not_recommend_lowering_max_len():
    """Variant 2: weights already overflow the budget; lowering max_model_len
    will NOT help, so no recommendation and the message must NOT suggest it."""
    diag = diagnose_engine_log(_NO_CACHE_BLOCKS_LOG)
    assert diag is not None
    assert diag.recommended_max_model_len is None
    msg = diag.message.lower()
    # Points at the real fix (bigger/more GPUs or higher util)...
    assert "gpu_memory_utilization" in msg or "gpu" in msg
    # ...and crucially does NOT tell the operator to lower max_model_len.
    assert "max_model_len" not in msg


def test_cuda_oom():
    diag = diagnose_engine_log(_CUDA_OOM_LOG)
    assert diag is not None
    assert "out of memory" in diag.message.lower()
    assert "gpu_memory_utilization" in diag.message
    assert diag.recommended_max_model_len is None


def test_trust_remote_code():
    diag = diagnose_engine_log(_TRUST_REMOTE_CODE_LOG)
    assert diag is not None
    assert "trust_remote_code" in diag.message
    assert diag.recommended_max_model_len is None


def test_no_match_returns_none():
    assert diagnose_engine_log(_UNRELATED_LOG) is None


def test_empty_text_returns_none():
    assert diagnose_engine_log("") is None
    assert diagnose_engine_log("   \n  ") is None


def test_kv_overflow_without_estimate_still_actionable():
    """A KV-overflow line with no 'estimated maximum model length' phrase still
    produces an actionable message, just without a numeric recommendation."""
    text = (
        "ValueError: The model's max seq len is larger than the maximum number "
        "of tokens that can be stored in KV cache. Try decreasing "
        "`max_model_len`."
    )
    diag = diagnose_engine_log(text)
    assert diag is not None
    assert "max_model_len" in diag.message
    assert diag.recommended_max_model_len is None


def test_kv_overflow_case_insensitive_tokens():
    """The KV/context branch keys off STABLE tokens case-insensitively so it
    survives vLLM wording drift."""
    text = (
        "Error: Model's MAX SEQ LEN exceeds the KV CACHE capacity. The "
        "estimated maximum model length is 9000. Decrease max_model_len."
    )
    diag = diagnose_engine_log(text)
    assert diag is not None
    assert diag.recommended_max_model_len == 9000


def test_oom_via_outofmemoryerror_token():
    text = "RuntimeError: OutOfMemoryError raised during weight allocation"
    diag = diagnose_engine_log(text)
    assert diag is not None
    assert "out of memory" in diag.message.lower()
