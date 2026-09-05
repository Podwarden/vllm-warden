"""vLLM's Prometheus dialect: the names, and the derivations they need.

Every ``vllm:`` string in this repository lives in this file. They came out of
app/stats/live_engine.py's build_frame(), which named 31 of them inline while
also doing rate math and SSE framing; sub-project C split the three jobs once a
second dialect existed to justify the seam.

Two things here are NOT renamings, and they are why a name-lookup table would
not have been enough:

* ``kv_tokens_total`` is composed from the LABELS of vllm:cache_config_info
  (block_size * num_gpu_blocks), not read from a series.
* several counters have two spellings across vLLM versions -- 0.25.1 renamed
  gpu_cache_usage_perc to kv_cache_usage_perc and appended _total to others --
  so the lookups go through Metrics.value_any and take whichever exists. That
  version-drift tolerance is why an absent metric reads None rather than
  crashing the stream, and it must survive any tidying.
"""

from __future__ import annotations

from app.runtime.backends.metrics import EngineReading
from app.stats.prometheus import Metrics, parse_prometheus


def _kv_tokens_total(m: Metrics) -> float | None:
    info = m.info("vllm:cache_config_info")
    if info is None:
        return None
    try:
        return float(int(info["block_size"]) * int(info["num_gpu_blocks"]))
    except (KeyError, ValueError):
        return None


def read(body: str) -> EngineReading | None:
    """Parse vLLM exposition text into an EngineReading, or None if unusable."""
    if not body or not body.strip():
        return None
    m = Metrics(parse_prometheus(body))
    return EngineReading(
        requests_running=m.value("vllm:num_requests_running"),
        requests_waiting=m.value("vllm:num_requests_waiting"),
        waiting_capacity=m.value(
            "vllm:num_requests_waiting_by_reason", reason="capacity"
        ),
        waiting_deferred=m.value(
            "vllm:num_requests_waiting_by_reason", reason="deferred"
        ),
        kv_cache_usage_perc=m.value_any(
            "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"
        ),
        kv_tokens_total=_kv_tokens_total(m),
        engine_sleep_state=m.value("vllm:engine_sleep_state"),
        prompt_tokens_total=m.value("vllm:prompt_tokens_total"),
        generation_tokens_total=m.value("vllm:generation_tokens_total"),
        preemptions_total=m.value("vllm:num_preemptions_total"),
        prefix_cache_hits=m.value_any(
            "vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"
        ),
        prefix_cache_queries=m.value_any(
            "vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries"
        ),
        mm_cache_hits=m.value_any("vllm:mm_cache_hits_total", "vllm:mm_cache_hits"),
        mm_cache_queries=m.value_any(
            "vllm:mm_cache_queries_total", "vllm:mm_cache_queries"
        ),
        external_prefix_cache_hits=m.value_any(
            "vllm:external_prefix_cache_hits_total", "vllm:external_prefix_cache_hits"
        ),
        external_prefix_cache_queries=m.value_any(
            "vllm:external_prefix_cache_queries_total",
            "vllm:external_prefix_cache_queries",
        ),
        flops_per_gpu_total=m.value("vllm:estimated_flops_per_gpu_total"),
        finished_stop=m.value("vllm:request_success_total", finished_reason="stop"),
        finished_length=m.value("vllm:request_success_total", finished_reason="length"),
        finished_abort=m.value("vllm:request_success_total", finished_reason="abort"),
        ttft_hist=m.histogram("vllm:time_to_first_token_seconds"),
        itl_hist=m.histogram("vllm:inter_token_latency_seconds"),
        tpot_hist=m.histogram("vllm:time_per_output_token_seconds"),
        e2e_hist=m.histogram("vllm:e2e_request_latency_seconds"),
    )
