// Frontend types + tiny formatters for the two live-stats data planes
// (docs/live-stats-spec.md). Two independent backend shapes:
//
//   - Plane A "engine" — GET /api/stats/live (SSE). Aggregate truth scraped
//     from vLLM /metrics. See LiveEngineFrame.
//   - Plane B "requests" — GET /api/stats/requests (JWT JSON, polled ~1.5s).
//     Per-request truth the warden computes on the forward path. See
//     LiveRequestsSnapshot.
//
// These types pin the contract by hand: the /api/stats/* routes return bare
// dicts (no FastAPI response_model), so openapi-typescript can't reach the
// inner shape — same rationale as stats-v2.ts. A drift here fails typecheck.
//
// Unknown / absent engine metrics arrive as `null` (vLLM 0.25.1 renamed some
// names) — every optional numeric is therefore `number | null`.

// ---- Plane A: GET /api/stats/live (SSE frame) -----------------------------

export interface LiveEngine {
  num_requests_running: number;
  num_requests_waiting: number;
  /** Free-form reason → waiting count, e.g. {capacity, deferred}. */
  waiting_by_reason: Record<string, number>;
  kv_cache_usage_perc: number; // 0..1
  kv_tokens_used: number; // derived absolute
  kv_tokens_total: number; // block_size * num_gpu_blocks
  engine_sleep_state: number; // 0 = awake
  preemptions_total: number;
  preemptions_per_s: number | null; // null on the connection's first frame
}

export interface LiveThroughput {
  prompt_tokens_per_s: number | null;
  generation_tokens_per_s: number | null;
  prompt_tokens_total: number;
  generation_tokens_total: number;
}

export interface LiveCache {
  /** Interval delta hits/queries — null on the first frame. */
  prefix_hit_rate: number | null;
  prefix_hit_rate_cumulative: number | null;
  mm_hit_rate_cumulative: number | null;
  external_prefix_hit_rate_cumulative: number | null;
}

export interface LiveLatency {
  ttft_p50: number | null;
  ttft_p90: number | null;
  ttft_p99: number | null;
  ttft_mean: number | null;
  itl_p50: number | null;
  itl_p99: number | null;
  tpot_p50: number | null;
  e2e_p50: number | null;
  e2e_p90: number | null;
  e2e_p99: number | null;
}

export interface LiveMfu {
  flops_per_gpu_total: number | null;
  /** Model FLOPs Utilization 0..1, or null if the formula isn't wired yet. */
  mfu_estimate: number | null;
}

export interface LiveEngineFrame {
  ts: string;
  /** Served model name, or null when no model is loaded (engine idle). */
  model: string | null;
  model_id: string | null;
  /** From the model row / engine — the denominator for context bars. */
  max_model_len: number | null;
  engine: LiveEngine;
  throughput: LiveThroughput;
  cache: LiveCache;
  latency: LiveLatency;
  mfu: LiveMfu | null;
  /** request_success_total{finished_reason} → count. */
  finished: Record<string, number>;
  /** Non-null string when the /metrics scrape failed this tick. */
  scrape_error: string | null;
}

// ---- Plane B: GET /api/stats/requests (JSON snapshot) ---------------------

export interface LiveRequestRow {
  id: string;
  token_name: string | null;
  client_ip: string | null;
  model: string;
  path: string; // /v1/chat/completions | /v1/completions
  prompt_tokens: number;
  completion_tokens: number;
  context_tokens: number; // prompt + completion
  max_model_len: number;
  context_pct: number; // context_tokens / max_model_len, 0..1
  elapsed_s: number;
  phase: string; // "prefill" | "decode"
  orphan: boolean;
}

export interface LiveByToken {
  token_name: string | null;
  requests: number;
  context_tokens: number;
  prompt_tokens: number;
  completion_tokens: number;
}

export interface LiveByIp {
  client_ip: string | null;
  requests: number;
  context_tokens: number;
}

export interface LiveRequestsSnapshot {
  ts: string;
  count: number;
  requests: LiveRequestRow[];
  by_token: LiveByToken[];
  by_ip: LiveByIp[];
}

// ---- Pressure scale -------------------------------------------------------
//
// The single visual metaphor that ties the engine-wide KV cache gauge to the
// per-session context bars: both are "how full is this?" meters on the same
// green→amber→red scale. Thresholds are shared so a row that's hot means the
// same thing as a KV gauge that's hot.

export type Pressure = "healthy" | "warm" | "hot";

export const PRESSURE_WARM = 0.6;
export const PRESSURE_HOT = 0.85;

export function pressureOf(fraction: number): Pressure {
  if (fraction >= PRESSURE_HOT) return "hot";
  if (fraction >= PRESSURE_WARM) return "warm";
  return "healthy";
}

// ---- Formatters -----------------------------------------------------------
// Pure. Kept beside the types so the page and any future export path agree.

const DASH = "—";

/** Abbreviate a token/FLOP-style count: 195900 → "195.9k", 1.2e6 → "1.2M". */
export function formatCompact(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return DASH;
  const abs = Math.abs(n);
  if (abs >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return Math.round(n).toString();
}

/** Full integer with locale separators, or "—" for missing. */
export function formatInt(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return DASH;
  return Math.round(n).toLocaleString();
}

/** A 0..1 fraction as a whole-percent string: 0.871 → "87%". */
export function formatPct(fraction: number | null | undefined, digits = 0): string {
  if (fraction === null || fraction === undefined || Number.isNaN(fraction)) return DASH;
  return `${(fraction * 100).toFixed(digits)}%`;
}

/** Seconds → operator-friendly latency: <1s in ms, else "8.1 s". */
export function formatLatency(s: number | null | undefined): string {
  if (s === null || s === undefined || Number.isNaN(s)) return DASH;
  if (s < 1) return `${Math.round(s * 1000)} ms`;
  return `${s.toFixed(1)} s`;
}

/** Elapsed seconds → "m:ss" (or "s.s" under 10s for a live feel). */
export function formatElapsed(s: number | null | undefined): string {
  if (s === null || s === undefined || Number.isNaN(s)) return DASH;
  if (s < 10) return `${s.toFixed(1)}s`;
  const total = Math.floor(s);
  const m = Math.floor(total / 60);
  const sec = total % 60;
  if (m === 0) return `${sec}s`;
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

/** FLOP/s → "1.2 PFLOP/s". */
export function formatFlops(f: number | null | undefined): string {
  if (f === null || f === undefined || Number.isNaN(f)) return DASH;
  if (f >= 1e15) return `${(f / 1e15).toFixed(1)} PFLOP/s`;
  if (f >= 1e12) return `${(f / 1e12).toFixed(1)} TFLOP/s`;
  if (f >= 1e9) return `${(f / 1e9).toFixed(1)} GFLOP/s`;
  return `${Math.round(f)} FLOP/s`;
}
