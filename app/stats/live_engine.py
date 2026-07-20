"""Live engine-metrics SSE — ``GET /api/stats/live`` (Plane A).

Scrapes the loaded model's vLLM ``/metrics`` endpoint (aggregate Prometheus
text) and emits a parsed JSON frame over SSE, mirroring the header-metrics SSE
in ``app/header/routes_api.py`` (ticket auth, ``sse_headers``, ~2s cadence, 15s
keepalive, shared TTL cache so multi-tab collapses to one scrape).

Two data planes back the live-stats dashboard (see docs/live-stats-spec.md).
This module is Plane A: the *aggregate* engine truth vLLM exposes on
``127.0.0.1:{engine_port}/metrics``. Per-request KV/context truth is Plane B
(``app/stats/live_requests.py``), a separate module — they never share a file.

Design notes:

* **Parser** — a tiny inline Prometheus text parser (no new dependency).
  ``vllm:`` metric names are matched; a renamed/absent name maps to ``null``
  rather than raising (vLLM 0.25.1 renamed several — e.g. ``gpu_cache_usage_perc``
  → ``kv_cache_usage_perc``). All parse/extract helpers are pure functions so
  the unit tests exercise them against a captured metrics blob.
* **Cache** — ``_MetricsCache`` keyed on ``model_id`` with a ~1.5s TTL and a
  single ``asyncio.Lock`` collapses N concurrent tabs to one scrape, mirroring
  ``_ProbeCache`` in ``app/system/routes_gpus.py``. The cached entry carries the
  scrape's monotonic timestamp so every tab derives per-second rates against a
  consistent clock.
* **Rates** — counters are cumulative; per-second rates (generation/prompt
  tokens, preemptions) and the interval prefix-cache hit rate are deltas vs the
  *connection's* previous frame. The first frame emits ``null`` rates. A counter
  going backwards (engine restart) also yields ``null`` for that tick.
* **Percentiles** — histogram p50/p90/p99 via cumulative ``_bucket`` count
  interpolation; means via ``_sum`` / ``_count``.
* **Fail-open** — a scrape failure is caught and surfaced as ``scrape_error`` on
  an otherwise-null frame; it never kills the stream (mirrors the header SSE).
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.db.database import open_db
from app.models.routes_logs import require_sse_ticket
from app.utils.sse import sse_headers

router = APIRouter(prefix="/api/stats", tags=["stats-live"])


# --------------------------------------------------------------------------- #
# Cadence / cache tuning
# --------------------------------------------------------------------------- #

# SSE emit cadence. Override via ``VW_STATS_LIVE_INTERVAL_S`` (float seconds).
# Floor at 0.5s so a misconfigured env can't pin a CPU (same clamp as the
# header metrics stream).
def _interval_seconds() -> float:
    raw = os.environ.get("VW_STATS_LIVE_INTERVAL_S")
    if not raw:
        return 2.0
    try:
        v = float(raw)
    except ValueError:
        return 2.0
    return max(0.5, v)


# SSE keepalive cadence — same rationale as the header stream. With the default
# 2s emit interval this never fires; it covers an operator who bumps the emit
# interval up for a quiet dashboard.
KEEPALIVE_INTERVAL_S: float = 15.0

# Shared-cache TTL. ~1.5s absorbs multi-tab burst polling into one scrape while
# staying fresh enough to feel live at the 2s emit cadence.
STATS_LIVE_CACHE_TTL_S: float = 1.5

# httpx timeout for the /metrics scrape. Short — the endpoint is loopback and a
# slow scrape should surface as scrape_error, not stall the whole stream.
SCRAPE_TIMEOUT_S: float = 5.0


# --------------------------------------------------------------------------- #
# Prometheus text parser (inline, no new dependency)
# --------------------------------------------------------------------------- #

# A metric line is ``name{labels} value [timestamp]`` or ``name value``.
_LINE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+(.+)$")
# One label pair: ``key="value"`` where value may contain escaped quotes.
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')


def _unescape(v: str) -> str:
    return v.replace("\\\\", "\\").replace('\\"', '"').replace("\\n", "\n")


def _parse_value(raw: str) -> float | None:
    tok = raw.split()[0] if raw else ""
    try:
        return float(tok)
    except ValueError:
        return None


@dataclass(frozen=True)
class PromSample:
    name: str
    labels: dict[str, str]
    value: float


def parse_prometheus(text: str) -> list[PromSample]:
    """Parse Prometheus exposition text into a flat list of samples.

    ``# HELP`` / ``# TYPE`` comment lines and blanks are skipped. A malformed
    sample line is dropped rather than raising — a partial scrape must never
    crash the stream.
    """
    out: list[PromSample] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        name, label_blob, raw_val = m.group(1), m.group(2), m.group(3)
        val = _parse_value(raw_val)
        if val is None:
            continue
        labels: dict[str, str] = {}
        if label_blob:
            for k, v in _LABEL_RE.findall(label_blob):
                labels[k] = _unescape(v)
        out.append(PromSample(name=name, labels=labels, value=val))
    return out


class Metrics:
    """Accessor over parsed Prometheus samples.

    All lookups aggregate (sum) across label series with the same metric name
    so a metric that vLLM happens to split by an extra label (e.g. an engine
    index) collapses to a single engine-wide figure. A name with no matching
    series returns ``None`` — never a crash — which is how a renamed/absent
    metric turns into ``null`` in the frame.
    """

    def __init__(self, samples: list[PromSample]) -> None:
        self._samples = samples
        self._by_name: dict[str, list[PromSample]] = defaultdict(list)
        for s in samples:
            self._by_name[s.name].append(s)

    def _matching(self, name: str, label_filter: dict[str, str]):
        for s in self._by_name.get(name, ()):
            if all(s.labels.get(k) == v for k, v in label_filter.items()):
                yield s

    def value(self, name: str, **label_filter: str) -> float | None:
        """Sum of matching series' values, or ``None`` if none match."""
        vals = [s.value for s in self._matching(name, label_filter)]
        return sum(vals) if vals else None

    def value_any(self, *names: str, **label_filter: str) -> float | None:
        """First non-None ``value`` across alternative metric names.

        vLLM 0.25.1 renamed several counters and sometimes appends ``_total``;
        callers pass the plausible spellings and take whichever exists.
        """
        for name in names:
            v = self.value(name, **label_filter)
            if v is not None:
                return v
        return None

    def info(self, name: str) -> dict[str, str] | None:
        """Labels of the first series with ``name`` (Prometheus ``*_info``)."""
        for s in self._by_name.get(name, ()):
            return s.labels
        return None

    def histogram(self, base: str):
        """Return ``(buckets, sum, count)`` for a histogram, or ``None``.

        ``buckets`` is a list of ``(le, cumulative_count)`` sorted ascending,
        counts summed across any non-``le`` label series. Returns ``None`` when
        the histogram is entirely absent.
        """
        bmap: dict[float, float] = defaultdict(float)
        saw_bucket = False
        for s in self._by_name.get(base + "_bucket", ()):
            le = s.labels.get("le")
            if le is None:
                continue
            try:
                le_f = float(le)
            except ValueError:
                continue
            bmap[le_f] += s.value
            saw_bucket = True
        total_sum = self.value(base + "_sum")
        total_count = self.value(base + "_count")
        if not saw_bucket and total_count is None:
            return None
        buckets = sorted(bmap.items())
        return buckets, total_sum, total_count


def hist_quantile(buckets: list[tuple[float, float]], q: float) -> float | None:
    """Interpolate a quantile from cumulative histogram buckets.

    ``buckets`` is ascending ``(le, cumulative_count)`` including the ``+Inf``
    bucket. Linear interpolation within the bucket that straddles the rank —
    the standard Prometheus ``histogram_quantile`` shape. Returns ``None`` for
    an empty or all-zero histogram.
    """
    if not buckets:
        return None
    total = buckets[-1][1]
    if not total or total <= 0:
        return None
    rank = q * total
    prev_le = 0.0
    prev_c = 0.0
    for le, c in buckets:
        if rank <= c:
            if math.isinf(le):
                # Rank falls in the open-ended top bucket; the best finite
                # answer is the last finite boundary we passed.
                return prev_le if prev_c > 0 else None
            if c <= prev_c:
                return le
            frac = (rank - prev_c) / (c - prev_c)
            return prev_le + frac * (le - prev_le)
        if not math.isinf(le):
            prev_le = le
        prev_c = c
    return prev_le


def hist_mean(sum_v: float | None, count_v: float | None) -> float | None:
    if sum_v is None or count_v is None or count_v <= 0:
        return None
    return sum_v / count_v


def _pct(hist, q: float) -> float | None:
    if hist is None:
        return None
    buckets, _sum, _count = hist
    return hist_quantile(buckets, q)


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den <= 0:
        return None
    return num / den


def _int_or_none(v: float | None) -> int | None:
    return int(round(v)) if v is not None else None


# --------------------------------------------------------------------------- #
# Frame construction
# --------------------------------------------------------------------------- #


@dataclass
class RateState:
    """Per-connection snapshot of cumulative counters + scrape clock, used to
    derive per-second rates on the *next* frame."""

    monotonic: float
    gen: float | None
    prompt: float | None
    preempt: float | None
    prefix_hits: float | None
    prefix_queries: float | None


def _rate(cur: float | None, prev: float | None, dt: float) -> float | None:
    """Per-second rate of a cumulative counter, or ``None``.

    ``None`` when either endpoint is missing, ``dt`` is non-positive, or the
    counter went backwards (an engine restart reset it — a spurious spike we
    refuse to emit).
    """
    if cur is None or prev is None or dt <= 0 or cur < prev:
        return None
    return (cur - prev) / dt


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _null_frame(model, model_id, max_model_len, scrape_error: str) -> dict:
    return {
        "ts": _now_iso(),
        "model": model,
        "model_id": model_id,
        "max_model_len": max_model_len,
        "engine": None,
        "throughput": None,
        "cache": None,
        "latency": None,
        "mfu": None,
        "finished": None,
        "scrape_error": scrape_error,
    }


def build_frame(
    m: Metrics,
    *,
    model: str | None,
    model_id: str | None,
    max_model_len: int | None,
    prev: RateState | None,
    scrape_monotonic: float,
) -> tuple[dict, RateState]:
    """Build the ``data:`` frame from parsed metrics; return ``(frame, state)``.

    ``state`` is the counter snapshot the caller stores to compute rates on the
    next tick. When ``prev`` is ``None`` (first frame) all rates are ``null``.
    """
    dt = (scrape_monotonic - prev.monotonic) if prev else 0.0

    # --- raw cumulative counters (also fed into next-frame rate state) ---
    gen_total = m.value("vllm:generation_tokens_total")
    prompt_total = m.value("vllm:prompt_tokens_total")
    preempt_total = m.value("vllm:num_preemptions_total")
    prefix_hits = m.value_any("vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits")
    prefix_queries = m.value_any(
        "vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries"
    )

    # --- KV cache: usage% -> absolute tokens via cache_config_info ---
    kv_perc = m.value("vllm:kv_cache_usage_perc")
    if kv_perc is None:  # 0.25.1 rename fallback
        kv_perc = m.value("vllm:gpu_cache_usage_perc")
    kv_total: int | None = None
    info = m.info("vllm:cache_config_info")
    if info is not None:
        try:
            block_size = int(info["block_size"])
            num_gpu_blocks = int(info["num_gpu_blocks"])
            kv_total = block_size * num_gpu_blocks
        except (KeyError, ValueError):
            kv_total = None
    kv_used = (
        int(round(kv_perc * kv_total))
        if (kv_perc is not None and kv_total is not None)
        else None
    )

    engine = {
        "num_requests_running": _int_or_none(m.value("vllm:num_requests_running")),
        "num_requests_waiting": _int_or_none(m.value("vllm:num_requests_waiting")),
        "waiting_by_reason": {
            "capacity": _int_or_none(
                m.value("vllm:num_requests_waiting_by_reason", reason="capacity")
            ),
            "deferred": _int_or_none(
                m.value("vllm:num_requests_waiting_by_reason", reason="deferred")
            ),
        },
        "kv_cache_usage_perc": kv_perc,
        "kv_tokens_used": kv_used,
        "kv_tokens_total": kv_total,
        "engine_sleep_state": _int_or_none(m.value("vllm:engine_sleep_state")),
        "preemptions_total": _int_or_none(preempt_total),
        "preemptions_per_s": _rate(preempt_total, prev.preempt if prev else None, dt),
    }

    throughput = {
        "prompt_tokens_per_s": _rate(prompt_total, prev.prompt if prev else None, dt),
        "generation_tokens_per_s": _rate(gen_total, prev.gen if prev else None, dt),
        "prompt_tokens_total": _int_or_none(prompt_total),
        "generation_tokens_total": _int_or_none(gen_total),
    }

    # Interval prefix-cache hit rate = delta hits / delta queries over the tick.
    interval_hit_rate: float | None = None
    if prev is not None and dt > 0:
        dh = _rate(prefix_hits, prev.prefix_hits, dt)  # per-s, reuse reset guard
        dq = _rate(prefix_queries, prev.prefix_queries, dt)
        if dh is not None and dq is not None and dq > 0:
            interval_hit_rate = dh / dq
    cache = {
        "prefix_hit_rate": interval_hit_rate,
        "prefix_hit_rate_cumulative": _ratio(prefix_hits, prefix_queries),
        "mm_hit_rate_cumulative": _ratio(
            m.value_any("vllm:mm_cache_hits_total", "vllm:mm_cache_hits"),
            m.value_any("vllm:mm_cache_queries_total", "vllm:mm_cache_queries"),
        ),
        "external_prefix_hit_rate_cumulative": _ratio(
            m.value_any(
                "vllm:external_prefix_cache_hits_total",
                "vllm:external_prefix_cache_hits",
            ),
            m.value_any(
                "vllm:external_prefix_cache_queries_total",
                "vllm:external_prefix_cache_queries",
            ),
        ),
    }

    ttft = m.histogram("vllm:time_to_first_token_seconds")
    itl = m.histogram("vllm:inter_token_latency_seconds")
    tpot = m.histogram("vllm:time_per_output_token_seconds")
    e2e = m.histogram("vllm:e2e_request_latency_seconds")
    latency = {
        "ttft_p50": _pct(ttft, 0.5),
        "ttft_p90": _pct(ttft, 0.9),
        "ttft_p99": _pct(ttft, 0.99),
        "ttft_mean": hist_mean(ttft[1], ttft[2]) if ttft else None,
        "itl_p50": _pct(itl, 0.5),
        "itl_p99": _pct(itl, 0.99),
        "tpot_p50": _pct(tpot, 0.5),
        "e2e_p50": _pct(e2e, 0.5),
        "e2e_p90": _pct(e2e, 0.9),
        "e2e_p99": _pct(e2e, 0.99),
    }

    # MFU (Model FLOPs Utilization). We expose the cumulative estimated FLOPs
    # counter vLLM reports; a true MFU% needs peak device FLOPs (dtype-specific)
    # which we don't resolve in v1, so ``mfu_estimate`` stays null. Formula for
    # a later iteration:
    #   mfu = (d(flops_per_gpu_total)/dt) / peak_flops_per_gpu(dtype)
    # i.e. the per-second FLOPs rate divided by the GPU's advertised peak FLOPs
    # for the engine's compute dtype.
    mfu = {
        "flops_per_gpu_total": m.value("vllm:estimated_flops_per_gpu_total"),
        "mfu_estimate": None,
    }

    finished = {
        "stop": _int_or_none(
            m.value("vllm:request_success_total", finished_reason="stop")
        ),
        "length": _int_or_none(
            m.value("vllm:request_success_total", finished_reason="length")
        ),
        "abort": _int_or_none(
            m.value("vllm:request_success_total", finished_reason="abort")
        ),
    }

    frame = {
        "ts": _now_iso(),
        "model": model,
        "model_id": model_id,
        "max_model_len": max_model_len,
        "engine": engine,
        "throughput": throughput,
        "cache": cache,
        "latency": latency,
        "mfu": mfu,
        "finished": finished,
        "scrape_error": None,
    }
    state = RateState(
        monotonic=scrape_monotonic,
        gen=gen_total,
        prompt=prompt_total,
        preempt=preempt_total,
        prefix_hits=prefix_hits,
        prefix_queries=prefix_queries,
    )
    return frame, state


# --------------------------------------------------------------------------- #
# Shared TTL scrape cache (mirror of ``_ProbeCache``, keyed by model_id)
# --------------------------------------------------------------------------- #


@dataclass
class ScrapeResult:
    metrics: Metrics | None
    error: str | None
    monotonic: float


async def _default_fetch(host: str, port: int, at: float) -> ScrapeResult:
    """Scrape ``/metrics`` once and parse. Never raises — a failed scrape is a
    ``ScrapeResult`` with ``error`` set and ``metrics=None`` (fail-open)."""
    url = f"http://{host}:{port}/metrics"
    try:
        async with httpx.AsyncClient(timeout=SCRAPE_TIMEOUT_S) as client:
            r = await client.get(url)
        r.raise_for_status()
        return ScrapeResult(metrics=Metrics(parse_prometheus(r.text)), error=None, monotonic=at)
    except Exception as exc:  # noqa: BLE001 — any scrape failure is surfaced, not raised
        return ScrapeResult(
            metrics=None, error=str(exc) or exc.__class__.__name__, monotonic=at
        )


class _MetricsCache:
    """One in-process cache for the vLLM ``/metrics`` scrape, keyed by model_id.

    A single ``asyncio.Lock`` guards the cache read and the underlying scrape so
    concurrent requests inside the TTL window collapse to one HTTP GET — the
    multi-tab de-dup goal. The cached ``ScrapeResult`` carries the scrape's
    monotonic timestamp so every consumer derives rates against a shared clock.
    Lives on ``app.state.stats_live_cache`` so tests can inject a clock + fetch.
    """

    def __init__(
        self,
        *,
        ttl: float = STATS_LIVE_CACHE_TTL_S,
        clock=time.monotonic,
        fetch=_default_fetch,
    ) -> None:
        self._ttl = ttl
        self._clock = clock
        self._fetch = fetch
        self._lock = asyncio.Lock()
        self._cache: dict[str, ScrapeResult] = {}
        self.invocations = 0  # exposed for tests

    async def get(self, model_id: str, host: str, port: int) -> ScrapeResult:
        async with self._lock:
            now = self._clock()
            hit = self._cache.get(model_id)
            if hit is not None and (now - hit.monotonic) < self._ttl:
                return hit
            self.invocations += 1
            res = await self._fetch(host, port, now)
            self._cache[model_id] = res
            return res


def _get_cache(request: Request) -> _MetricsCache:
    cache = getattr(request.app.state, "stats_live_cache", None)
    if cache is None:
        cache = _MetricsCache()
        request.app.state.stats_live_cache = cache
    return cache


async def _loaded_model(db_path) -> tuple[str | None, str | None, int | None]:
    """Return ``(model_id, served_model_name, max_model_len)`` of the loaded
    model, or ``(None, None, None)``. Same 'loaded' semantics as the header
    stream: ``models.status='loaded'`` with a ``model_runtime`` row."""
    async with open_db(db_path) as db:
        cur = await db.execute(
            "SELECT m.id, m.served_model_name, m.max_model_len "
            "FROM models m JOIN model_runtime r ON r.model_id = m.id "
            "WHERE m.status = 'loaded' "
            "LIMIT 1"
        )
        row = await cur.fetchone()
    if row is None:
        return (None, None, None)
    return (row[0], row[1], row[2])


# --------------------------------------------------------------------------- #
# SSE endpoint
# --------------------------------------------------------------------------- #


@router.get("/live")
async def stream_live(request: Request, _user: str = Depends(require_sse_ticket)):
    """Stream live vLLM engine metrics as SSE events (one JSON frame per tick).

    Structure mirrors the header-metrics SSE: an immediate first frame, then an
    emit every ``_interval_seconds()`` with ``is_disconnected`` checks and a
    belt-and-suspenders 15s keepalive. Each tick resolves the loaded model,
    pulls a (possibly cached) scrape, and builds a frame — computing rates
    against this connection's previous frame.
    """
    cache = _get_cache(request)
    settings = request.app.state.settings
    supervisor = request.app.state.supervisor
    interval = _interval_seconds()

    async def _one_frame(prev: RateState | None) -> tuple[dict, RateState | None]:
        """Resolve target, scrape (cached), and build one frame.

        Returns ``(frame, new_state_or_None)``. On any failure (no model
        loaded, no port, scrape error) returns a null frame with
        ``scrape_error`` set and ``None`` state so rate accounting resumes
        cleanly once the engine is back.
        """
        model_id, model_name, max_model_len = await _loaded_model(settings.db_path)
        if model_id is None:
            return _null_frame(None, None, None, "no model loaded"), None
        port = supervisor.get_port(model_id)
        if port is None:
            return (
                _null_frame(model_name, model_id, max_model_len, "engine not running"),
                None,
            )
        host = supervisor.get_host(model_id) or "127.0.0.1"
        res = await cache.get(model_id, host, port)
        if res.metrics is None:
            return (
                _null_frame(model_name, model_id, max_model_len, res.error or "scrape failed"),
                None,
            )
        return build_frame(
            res.metrics,
            model=model_name,
            model_id=model_id,
            max_model_len=max_model_len,
            prev=prev,
            scrape_monotonic=res.monotonic,
        )

    async def gen():
        prev: RateState | None = None
        last_yield_at = time.monotonic()
        # Immediate first frame so the consumer isn't blank for ``interval``s.
        try:
            frame, state = await _one_frame(prev)
            if state is not None:
                prev = state
            yield f"data: {json.dumps(frame)}\n\n"
            last_yield_at = time.monotonic()
        except Exception:  # noqa: BLE001 — first-tick failure falls through to the loop
            pass

        while True:
            if await request.is_disconnected():
                return
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                return
            if await request.is_disconnected():
                return
            try:
                frame, state = await _one_frame(prev)
                if state is not None:
                    prev = state
                yield f"data: {json.dumps(frame)}\n\n"
                last_yield_at = time.monotonic()
            except Exception as exc:  # noqa: BLE001 — never let a frame error kill the stream
                yield f"data: {json.dumps(_null_frame(None, None, None, str(exc) or exc.__class__.__name__))}\n\n"
                last_yield_at = time.monotonic()
            if time.monotonic() - last_yield_at >= KEEPALIVE_INTERVAL_S:
                yield ": keepalive\n\n"
                last_yield_at = time.monotonic()

    return StreamingResponse(gen(), media_type="text/event-stream", headers=sse_headers())
