# Live Stats Dashboard — API contract (LOCKED)

Branch: `feature/live-stats-dashboard`. New page only — `/stats` stays untouched.

## Why two data planes

vLLM `/metrics` (Prometheus, on `127.0.0.1:{engine_port}/metrics`) is **aggregate**.
It exposes engine-wide gauges/counters/histograms but **NOT** per-request KV blocks or
per-session context. So:

- **Engine plane** (aggregate truth from vLLM `/metrics`) → `GET /api/stats/live` (SSE).
- **Proxy plane** (per-request truth the warden already computes on the forward path) →
  `GET /api/stats/requests` (JWT GET snapshot, frontend polls ~1.5s).

The two are independent modules with a fixed seam. `main.py` router registration and
stub modules are scaffolded up front so the two backend slices never touch the same file.

---

## Plane A — Engine metrics  (owner: dev-1)

### Module: `app/stats/live_engine.py`
Router already registered in `main.py`. Endpoint: `GET /api/stats/live`.

- **Auth**: SSE ticket, exactly like `app/header/routes_api.py` — `Depends(require_sse_ticket)`,
  `sse_headers()`, `StreamingResponse(media_type="text/event-stream")`.
- **Source**: scrape `http://{host}:{port}/metrics` where host/port come from the loaded
  model's runtime — resolve like the proxy does:
  ```
  sup = request.app.state.supervisor
  # find the single loaded model (see app/header/routes_api.py::_active_model)
  port = sup.get_port(model_id); host = sup.get_host(model_id) or "127.0.0.1"
  ```
- **Cache**: shared TTL cache (~1.5s) keyed on model_id so N browser tabs collapse to one
  scrape — mirror `_ProbeCache` in `app/system/routes_gpus.py`. Store `(monotonic_ts, parsed)`.
- **Cadence**: emit every `VW_STATS_LIVE_INTERVAL_S` (default 2.0, floor 0.5), immediate first
  frame, 15s keepalive comment — copy the header SSE loop structure verbatim.
- **Rates**: counters are cumulative. Compute per-second rates (`tokens_per_s`,
  `preemptions_per_s`, prefix-cache hit rate over the interval) as deltas vs the connection's
  previous frame. First frame: rates = null.
- **Parsing**: tiny inline Prometheus text parser (no new dependency). Match `vllm:` metric
  names. Histograms: read `_sum` and `_count` for a running mean; read `_bucket` cumulative
  counts to derive p50/p90/p99 by bucket interpolation. Absolute KV tokens =
  `kv_cache_usage_perc * (cache_config_info block_size * num_gpu_blocks)`.

### Frame shape (`data:` JSON)
```jsonc
{
  "ts": "2026-07-19T20:01:02.345Z",
  "model": "qwopus3.6-27b-coder-fp8-model",   // served name, or null if none loaded
  "model_id": "90b6c566e02afa8f",             // or null
  "max_model_len": 224800,                    // from model row / engine, for context bars
  "engine": {
    "num_requests_running": 3,
    "num_requests_waiting": 0,
    "waiting_by_reason": {"capacity": 0, "deferred": 0},
    "kv_cache_usage_perc": 0.871,             // 0..1
    "kv_tokens_used": 195900,                 // derived absolute
    "kv_tokens_total": 224800,                // block_size * num_gpu_blocks
    "engine_sleep_state": 0,                  // 0 awake
    "preemptions_total": 12,
    "preemptions_per_s": 0.0
  },
  "throughput": {
    "prompt_tokens_per_s": 0.0,
    "generation_tokens_per_s": 41.7,
    "prompt_tokens_total": 12345678,
    "generation_tokens_total": 726779
  },
  "cache": {
    "prefix_hit_rate": 0.62,                  // interval delta hits/queries, null on 1st frame
    "prefix_hit_rate_cumulative": 0.55,
    "mm_hit_rate_cumulative": null,
    "external_prefix_hit_rate_cumulative": null
  },
  "latency": {                                // seconds; mean + percentiles from histograms
    "ttft_p50": 0.21, "ttft_p90": 0.8, "ttft_p99": 1.9, "ttft_mean": 0.34,
    "itl_p50": 0.021, "itl_p99": 0.09,
    "tpot_p50": 0.024,
    "e2e_p50": 8.1, "e2e_p90": 30.2, "e2e_p99": 95.0
  },
  "mfu": {                                    // Model FLOPs Utilization, derived, null if counters absent
    "flops_per_gpu_total": 1.2e15,
    "mfu_estimate": null                      // dev-1 documents formula; may be null in v1
  },
  "finished": {                               // request_success_total{finished_reason}
    "stop": 40210, "length": 118, "abort": 33
  },
  "scrape_error": null                        // string if the /metrics scrape failed this tick
}
```
Unknown/absent metrics → `null` (0.25.1 renamed some; never crash on a missing name).

---

## Plane B — Live request registry  (owner: dev-2)

### Module: `app/proxy/request_registry.py` (new) + `app/stats/live_requests.py` (endpoint)
Endpoint (already registered): `GET /api/stats/requests` — **JWT** (`Depends(require_jwt)`),
plain JSON snapshot, frontend polls ~1.5s. (No SSE — keep the hot path free of stream fan-out.)

### Registry (`app/proxy/request_registry.py`)
In-process, single uvicorn worker. **Lock-light**: `dict[str, LiveRequest]` guarded by a
short `asyncio.Lock` only for insert/delete; field updates during streaming are plain
attribute writes (last-write-wins is fine — the reader tolerates a one-tick-stale field, same
rationale as `ActiveRequestCounter.count()`). Replaces/absorbs `ActiveRequestCounter` (keep
`count()` working so `GET /api/admin/active-requests` and its Playwright test stay green).

`LiveRequest` fields:
```
id: str                # uuid4 hex, minted at register (Math.random-free: use uuid)
token_id: str | None
token_name: str | None
client_ip: str | None  # first X-Forwarded-For hop, else X-Real-Ip, else request.client.host
model: str             # served name
path: str              # /v1/chat/completions | /v1/completions
prompt_tokens: int     # already computed at routes.py:112
completion_tokens: int # updated as SSE deltas accumulate (streaming) / at end (non-stream)
max_model_len: int     # model row, for the context-window bar
started_monotonic: float
started_iso: str
phase: str             # "prefill" until first delta, then "decode"; "done" is deregistered
orphan: bool           # True once client disconnect detected but forward still draining
```

### Hook points in `app/proxy/routes.py::_forward`
1. After `prompt_tokens` computed (line ~112) and slot acquired: `reg.register(LiveRequest(...))`.
   Client IP + token already in scope (`token`, `request.headers`).
2. Streaming `gen()`: on first non-empty `delta` set `phase="decode"`; periodically (or each
   delta batch) update `completion_tokens` from the running tokenizer count OR a cheap
   incremental count — **do not** add a heavy per-delta tokenize; approximate with
   `len(accumulated)`-based estimate is acceptable if documented, but prefer reusing the
   final `tok_cache.count` and updating `completion_tokens` every ~0.5s from a coarse counter.
3. In the `finally` blocks (both stream + non-stream): `reg.deregister(id)`.
4. Orphan: when `await request.is_disconnected()` is true (streaming loop can check) but the
   upstream is still producing, set `orphan=True` before deregister. This is the same
   orphan-on-disconnect signature diagnosed on prod — surfacing it live is a goal.

Registration must be **fail-open**: any registry exception is swallowed so it can never break
a proxied request. Wrap hook calls in try/except.

### Snapshot shape (`GET /api/stats/requests`)
```jsonc
{
  "ts": "2026-07-19T20:01:02.345Z",
  "count": 3,
  "requests": [
    {
      "id": "a1b2...", "token_name": "hermes-bot", "client_ip": "10.42.5.185",
      "model": "qwopus3.6-27b-coder-fp8-model", "path": "/v1/chat/completions",
      "prompt_tokens": 65231, "completion_tokens": 1804, "context_tokens": 67035,
      "max_model_len": 224800, "context_pct": 0.298,
      "elapsed_s": 42.1, "phase": "decode", "orphan": false
    }
  ],
  "by_token": [
    {"token_name": "hermes-bot", "requests": 2, "context_tokens": 130000, "prompt_tokens": 128000, "completion_tokens": 2000}
  ],
  "by_ip": [
    {"client_ip": "10.42.5.185", "requests": 3, "context_tokens": 190000}
  ]
}
```
`context_tokens = prompt_tokens + completion_tokens`; `context_pct = context_tokens / max_model_len`.
Token name/IP are metadata only — **never** emit token plaintext/hash/secret columns.

---

## Plane C — Frontend  (owner: frontend agent, uses frontend-design skill)

New page: `frontend/src/app/stats/live/page.tsx` (route `/ui/stats/live`). `/stats` untouched.

- Consume `GET /api/stats/live` via an SSE client modeled on
  `frontend/src/lib/header-metrics-stream.ts` (ticket mint at `/api/auth/sse-ticket`,
  ref-counted singleton, exp-backoff reconnect). New lib: `frontend/src/lib/live-stats-stream.ts`.
- Consume `GET /api/stats/requests` via `useSWR` + `authFetchJSON` with `refreshInterval: 1500`,
  pause when tab hidden (see `/stats` `REFRESH_MS` pattern).
- Panels (design freely per frontend-design skill, but cover):
  - **Engine core load**: running / waiting (+by reason) big numbers + sparkline.
  - **KV cache**: gauge `kv_cache_usage_perc` + "X / Y tokens" absolute, preemptions/s.
  - **Live requests table**: token · client IP · model · phase · elapsed · orphan badge, with a
    **per-session context-window bar** (`context_tokens / max_model_len`) per row.
  - **Aggregations**: by token and by client IP (from `by_token` / `by_ip`).
  - **Throughput**: gen tokens/s + prompt tokens/s.
  - **Latency**: TTFT / ITL / TPOT / e2e percentiles.
  - **Cache hit rate**: prefix-cache hit rate. **MFU** if present. **finished-reason** breakdown.
- Types live in a new `frontend/src/lib/live-stats.ts` matching the two shapes above.
- Add a nav link/tab from `/stats` → `/stats/live` (a single `<Link>`; do not restructure `/stats`).

## Constraints (all slices)
- Everything runs in Docker. Tests: backend `pytest` and frontend `lint`/`typecheck` run via
  the project Docker images / make targets — never host node/python.
- Fail-open on the proxy hot path. Lock-light. Single worker assumption is fine.
- No secrets: token name + client IP only; never token plaintext/hash.
- Staging deploy only. No prod deploy/restart without explicit user confirmation.
