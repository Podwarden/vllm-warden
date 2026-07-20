"use client";

// /ui/stats/live — realtime engine + per-session cockpit.
//
// The static /stats tiles answer "what happened over the last hour". This
// page answers "what is the engine doing THIS SECOND": who is connected, how
// much KV cache and context window each live session is holding, and whether
// the box is saturating or dropping requests on the floor.
//
// Two independent data planes (docs/live-stats-spec.md):
//   - Engine (aggregate) — GET /api/stats/live over SSE, ~2s cadence, via the
//     ref-counted singleton in live-stats-stream.ts.
//   - Requests (per-session) — GET /api/stats/requests polled every 1.5s with
//     useSWR, paused when the tab is hidden.
//
// Design: an instrument cluster in the app's existing slate/emerald language.
// One visual metaphor carries the page — a green→amber→red "pressure" meter,
// used identically for the engine-wide KV cache gauge and each row's
// context-window bar. Context windows are logical claims on the physical KV
// cache, so sharing the scale is literally true, not decorative. The pressure
// FILL colors are deliberately theme-independent (data encoding must not
// shift under the retro theme remap); all chrome uses the remapped
// slate/emerald utilities so light + dark themes stay honest.

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import useSWR from "swr";
import { authFetchJSON } from "@/lib/auth-fetch";
import { formatTps } from "@/lib/stats-v2";
import { cn } from "@/lib/utils";
import { useLiveStats, type LiveStatsState } from "@/lib/live-stats-stream";
import {
  formatCompact,
  formatElapsed,
  formatFlops,
  formatInt,
  formatLatency,
  formatPct,
  pressureOf,
  type LiveEngineFrame,
  type LiveRequestsSnapshot,
  type Pressure,
} from "@/lib/live-stats";

// Per-request registry poll. 1.5s matches the spec cadence; pause when the
// tab is hidden so a background tab doesn't keep the registry read warm.
const REQ_REFRESH_MS = 1_500;
const reqRefreshInterval = () =>
  typeof document !== "undefined" && document.hidden ? 0 : REQ_REFRESH_MS;

// Rolling sparkline history depth (SSE frames, ~2s each → ~80s of trail).
const HIST_LEN = 40;

// Theme-independent pressure fills — see the module header. These specific
// classes are NOT in the retro-theme override list, so they render the same
// green / amber / red in every theme, keeping the 3-level scale legible.
const PRESSURE_FILL: Record<Pressure, string> = {
  healthy: "bg-emerald-500",
  warm: "bg-amber-500",
  hot: "bg-red-500",
};

// ===========================================================================
// Page
// ===========================================================================

export default function LiveStatsPage() {
  const engine = useLiveStats();
  const frame = engine.frame;

  const requests = useSWR<LiveRequestsSnapshot>(
    "/api/stats/requests",
    authFetchJSON,
    { refreshInterval: reqRefreshInterval, keepPreviousData: true },
  );

  // Rolling trails for the sparklines, appended once per distinct engine
  // frame. Kept in state (not a ref) so the sparklines re-render on push.
  const [hist, setHist] = useState<{
    running: number[];
    kv: number[];
    gen: number[];
  }>({ running: [], kv: [], gen: [] });
  const lastTs = useRef<string | null>(null);

  useEffect(() => {
    if (!frame || frame.model === null) return;
    if (frame.ts === lastTs.current) return;
    lastTs.current = frame.ts;
    const cap = (arr: number[], v: number) =>
      [...arr, v].slice(-HIST_LEN);
    setHist((h) => ({
      running: cap(h.running, frame.engine.num_requests_running),
      kv: cap(h.kv, frame.engine.kv_cache_usage_perc * 100),
      gen: cap(h.gen, frame.throughput.generation_tokens_per_s ?? 0),
    }));
  }, [frame]);

  const noModel = frame !== null && frame.model === null;
  const showSkeleton = frame === null && engine.status !== "terminal-error";

  return (
    <div className="space-y-5" data-testid="live-stats-page">
      {/* Header ------------------------------------------------------- */}
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="text-2xl font-semibold">Live</h1>
          {frame && frame.model ? (
            <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-700/60 bg-emerald-900/30 px-2.5 py-0.5">
              <span
                aria-hidden="true"
                className="h-1.5 w-1.5 rounded-full bg-emerald-400"
              />
              <span className="font-mono text-[11px] text-emerald-200">
                {frame.model}
              </span>
            </span>
          ) : noModel ? (
            <span className="rounded-full border border-slate-700 bg-slate-800/50 px-2.5 py-0.5 text-[11px] uppercase tracking-wide text-slate-400">
              idle
            </span>
          ) : null}
        </div>
        <div className="flex items-center gap-3 text-xs">
          <ConnectionStatus state={engine} />
          {frame && <LastUpdated ts={frame.ts} />}
          <Link
            href="/stats"
            className="rounded-md border border-slate-700 px-2.5 py-1 text-slate-300 transition-colors hover:bg-slate-800 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
          >
            ← Stats
          </Link>
        </div>
      </header>

      {/* Terminal / scrape banners ------------------------------------ */}
      {engine.status === "terminal-error" && (
        <Banner tone="error">
          Live engine stream disconnected
          {engine.errorCode ? ` (HTTP ${engine.errorCode})` : ""}. Reload the
          page to reconnect.
        </Banner>
      )}
      {frame?.scrape_error && (
        <Banner tone="warn">
          Metrics scrape failed: {frame.scrape_error}. Showing the last good
          values.
        </Banner>
      )}

      {/* No model loaded --------------------------------------------- */}
      {noModel ? (
        <EmptyState
          title="No model loaded"
          body="The engine is idle. Load a model to see live KV-cache, throughput, and per-session telemetry."
        />
      ) : (
        <>
          {/* Engine hero — the headline: load + KV pressure ---------- */}
          <EngineHero frame={frame} hist={hist} skeleton={showSkeleton} />

          {/* Live requests — the centerpiece ------------------------- */}
          <LiveRequestsPanel
            data={requests.data}
            error={requests.error}
            isLoading={requests.isLoading && !requests.data}
          />

          {/* Throughput + latency ------------------------------------ */}
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <ThroughputPanel frame={frame} genHist={hist.gen} skeleton={showSkeleton} />
            <LatencyPanel frame={frame} skeleton={showSkeleton} />
          </div>

          {/* Aggregations -------------------------------------------- */}
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <ByTokenPanel data={requests.data} />
            <ByIpPanel data={requests.data} />
          </div>

          {/* Cache / MFU / finished ---------------------------------- */}
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
            <CachePanel frame={frame} skeleton={showSkeleton} />
            <MfuPanel frame={frame} skeleton={showSkeleton} />
            <FinishedPanel frame={frame} skeleton={showSkeleton} />
          </div>
        </>
      )}
    </div>
  );
}

// ===========================================================================
// Header bits
// ===========================================================================

const STATUS_META: Record<
  LiveStatsState["status"],
  { dot: string; label: string; pulse: boolean }
> = {
  connecting: { dot: "bg-slate-400", label: "Connecting", pulse: true },
  connected: { dot: "bg-emerald-400", label: "Live", pulse: true },
  reconnecting: { dot: "bg-amber-400", label: "Reconnecting", pulse: true },
  "terminal-error": { dot: "bg-red-500", label: "Disconnected", pulse: false },
};

function ConnectionStatus({ state }: { state: LiveStatsState }) {
  const meta = STATUS_META[state.status];
  return (
    <span
      className="inline-flex items-center gap-1.5 text-slate-400"
      role="status"
      aria-label={`Engine stream ${meta.label}`}
    >
      <span
        aria-hidden="true"
        className={cn(
          "h-2 w-2 rounded-full",
          meta.dot,
          meta.pulse && "motion-safe:animate-pulse",
        )}
      />
      {meta.label}
    </span>
  );
}

// Self-contained 1s ticker so "updated Ns ago" stays live without re-rendering
// the whole page each second.
function LastUpdated({ ts }: { ts: string }) {
  const [, setNow] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setNow((n) => n + 1), 1_000);
    return () => clearInterval(id);
  }, []);
  const ms = Date.parse(ts);
  if (Number.isNaN(ms)) return null;
  const ago = Math.max(0, Math.round((Date.now() - ms) / 1000));
  return (
    <span className="tabular-nums text-slate-500" title={ts}>
      updated {ago}s ago
    </span>
  );
}

// ===========================================================================
// Engine hero
// ===========================================================================

function EngineHero({
  frame,
  hist,
  skeleton,
}: {
  frame: LiveEngineFrame | null;
  hist: { running: number[]; kv: number[]; gen: number[] };
  skeleton: boolean;
}) {
  if (skeleton || !frame) {
    return (
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-4">
        <SkeletonCard className="h-32" />
        <SkeletonCard className="h-32" />
        <SkeletonCard className="h-32 lg:col-span-2" />
      </div>
    );
  }

  const e = frame.engine;
  const kv = Math.max(0, Math.min(1, e.kv_cache_usage_perc));
  const reasons = Object.entries(e.waiting_by_reason ?? {}).filter(
    ([, v]) => v > 0,
  );

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-4">
      {/* Running ---------------------------------------------------- */}
      <Panel title="Running">
        <BigNumber value={e.num_requests_running} accent />
        <p className="mt-0.5 text-xs text-slate-500">requests decoding now</p>
        <Sparkline data={hist.running} className="mt-3 text-emerald-400" />
      </Panel>

      {/* Waiting ---------------------------------------------------- */}
      <Panel title="Waiting">
        <BigNumber value={e.num_requests_waiting} />
        <p className="mt-0.5 text-xs text-slate-500">queued for a slot</p>
        <div className="mt-3 flex flex-wrap gap-1.5">
          {reasons.length === 0 ? (
            <span className="text-xs text-slate-600">no backlog</span>
          ) : (
            reasons.map(([reason, count]) => (
              <span
                key={reason}
                className="rounded border border-amber-800/40 bg-amber-950/40 px-1.5 py-0.5 text-[11px] text-amber-300"
              >
                {reason} {count}
              </span>
            ))
          )}
        </div>
      </Panel>

      {/* KV cache pressure — the emotional center ------------------- */}
      <Panel
        title="KV cache pressure"
        className="lg:col-span-2"
        right={
          <span className="text-xs text-slate-500">
            {e.engine_sleep_state === 0 ? "awake" : `sleep ${e.engine_sleep_state}`}
          </span>
        }
      >
        <div className="flex items-end justify-between gap-4">
          <div>
            <BigNumber value={formatPct(kv)} />
            <p className="mt-0.5 font-mono text-xs text-slate-400">
              {formatCompact(e.kv_tokens_used)} / {formatCompact(e.kv_tokens_total)}{" "}
              tokens
            </p>
          </div>
          <div className="text-right">
            <p className="font-mono text-lg font-semibold tabular-nums text-slate-100">
              {e.preemptions_per_s === null
                ? "—"
                : e.preemptions_per_s.toFixed(2)}
            </p>
            <p className="text-xs text-slate-500">preemptions / s</p>
          </div>
        </div>
        <Meter fraction={kv} className="mt-3 h-3.5" />
        <div className="mt-1.5 flex justify-between text-[10px] uppercase tracking-wide text-slate-600">
          <span>0</span>
          <span>{formatInt(e.preemptions_total)} preempted total</span>
          <span>{formatCompact(e.kv_tokens_total)}</span>
        </div>
      </Panel>
    </div>
  );
}

// ===========================================================================
// Live requests table — the centerpiece
// ===========================================================================

function LiveRequestsPanel({
  data,
  error,
  isLoading,
}: {
  data: LiveRequestsSnapshot | undefined;
  error: unknown;
  isLoading: boolean;
}) {
  const rows = data?.requests ?? [];
  return (
    <Panel
      title="Live requests"
      right={
        <span className="tabular-nums text-xs text-slate-500">
          {data ? `${data.count} active` : ""}
        </span>
      }
      bodyClassName="p-0"
    >
      {isLoading ? (
        <div className="p-4">
          <SkeletonBar />
        </div>
      ) : error && !data ? (
        <p className="p-4 text-sm text-red-400">
          Failed to load live requests
          {error instanceof Error ? `: ${error.message}` : "."}
        </p>
      ) : rows.length === 0 ? (
        <p className="p-6 text-center text-sm text-slate-500">
          No active requests. Live sessions appear here the moment they hit the
          engine.
        </p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[56rem] text-sm">
            <thead className="border-b border-slate-800 text-left text-[11px] uppercase tracking-wider text-slate-500">
              <tr>
                <th className="px-4 py-2 font-medium">Token</th>
                <th className="px-3 py-2 font-medium">Client IP</th>
                <th className="px-3 py-2 font-medium">Model</th>
                <th className="px-3 py-2 font-medium">Phase</th>
                <th className="px-3 py-2 font-medium">Context window</th>
                <th className="px-3 py-2 text-right font-medium">Elapsed</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/70">
              {rows.map((r) => (
                <tr key={r.id} className="text-slate-200">
                  <td className="px-4 py-2.5">
                    <span className="font-mono text-xs">
                      {r.token_name ?? (
                        <span className="text-slate-500">anonymous</span>
                      )}
                    </span>
                    {r.orphan && (
                      <span
                        title="Client disconnected but the upstream is still draining."
                        className="ml-2 rounded bg-red-950/50 px-1.5 py-0.5 text-[10px] font-medium uppercase text-red-300"
                      >
                        orphan
                      </span>
                    )}
                  </td>
                  <td className="px-3 py-2.5 font-mono text-xs text-slate-400">
                    {r.client_ip ?? "—"}
                  </td>
                  <td className="max-w-[12rem] truncate px-3 py-2.5 font-mono text-xs text-slate-400">
                    {r.model}
                  </td>
                  <td className="px-3 py-2.5">
                    <PhasePill phase={r.phase} />
                  </td>
                  <td className="px-3 py-2.5">
                    <ContextBar
                      used={r.context_tokens}
                      total={r.max_model_len}
                      pct={r.context_pct}
                    />
                  </td>
                  <td className="px-3 py-2.5 text-right font-mono tabular-nums text-slate-300">
                    {formatElapsed(r.elapsed_s)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}

function PhasePill({ phase }: { phase: string }) {
  const decode = phase === "decode";
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] font-medium",
        decode
          ? "bg-emerald-900/40 text-emerald-300"
          : "bg-amber-900/40 text-amber-300",
      )}
    >
      <span
        aria-hidden="true"
        className={cn(
          "h-1.5 w-1.5 rounded-full",
          decode ? "bg-emerald-400" : "bg-amber-400 motion-safe:animate-pulse",
        )}
      />
      {phase}
    </span>
  );
}

// Per-session context bar — the same pressure meter as the KV gauge, sized for
// a table row. Fill color is theme-independent; the numeric label carries the
// exact figure so the meaning never depends on color alone.
function ContextBar({
  used,
  total,
  pct,
}: {
  used: number;
  total: number;
  pct: number;
}) {
  const clamped = Math.max(0, Math.min(1, pct));
  return (
    <div className="flex min-w-[14rem] items-center gap-2.5">
      <Meter fraction={clamped} className="h-2 flex-1" showTicks={false} />
      <span className="shrink-0 font-mono text-xs tabular-nums text-slate-400">
        {formatCompact(used)}/{formatCompact(total)}
      </span>
      <span className="w-9 shrink-0 text-right font-mono text-xs font-medium tabular-nums text-slate-200">
        {Math.round(clamped * 100)}%
      </span>
    </div>
  );
}

// ===========================================================================
// Throughput / latency
// ===========================================================================

function ThroughputPanel({
  frame,
  genHist,
  skeleton,
}: {
  frame: LiveEngineFrame | null;
  genHist: number[];
  skeleton: boolean;
}) {
  if (skeleton || !frame) return <PanelSkeleton title="Throughput" />;
  const t = frame.throughput;
  return (
    <Panel title="Throughput">
      <div className="flex items-end justify-between gap-4">
        <div>
          <p className="flex items-baseline gap-1.5">
            <span className="font-mono text-3xl font-semibold tabular-nums text-slate-100">
              {formatTps(t.generation_tokens_per_s ?? 0)}
            </span>
            <span className="text-sm text-slate-400">gen tok/s</span>
          </p>
          <p className="mt-1 font-mono text-xs text-slate-500">
            prompt {formatTps(t.prompt_tokens_per_s ?? 0)} tok/s
          </p>
        </div>
        <Sparkline data={genHist} className="h-10 w-28 text-emerald-400" />
      </div>
      <div className="mt-3 grid grid-cols-2 gap-3 border-t border-slate-800 pt-3">
        <MiniStat label="gen total" value={formatCompact(t.generation_tokens_total)} />
        <MiniStat label="prompt total" value={formatCompact(t.prompt_tokens_total)} />
      </div>
    </Panel>
  );
}

function LatencyPanel({
  frame,
  skeleton,
}: {
  frame: LiveEngineFrame | null;
  skeleton: boolean;
}) {
  if (skeleton || !frame) return <PanelSkeleton title="Latency" />;
  const l = frame.latency;
  return (
    <Panel
      title="Latency"
      right={<span className="text-[11px] text-slate-600">percentiles</span>}
    >
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-[11px] uppercase tracking-wider text-slate-600">
              <th className="pb-1.5 font-medium" />
              <th className="pb-1.5 pr-3 text-right font-medium">p50</th>
              <th className="pb-1.5 pr-3 text-right font-medium">p90</th>
              <th className="pb-1.5 pr-3 text-right font-medium">p99</th>
              <th className="pb-1.5 text-right font-medium">mean</th>
            </tr>
          </thead>
          <tbody className="font-mono tabular-nums text-slate-300">
            <LatencyRow label="TTFT" p50={l.ttft_p50} p90={l.ttft_p90} p99={l.ttft_p99} mean={l.ttft_mean} />
            <LatencyRow label="ITL" p50={l.itl_p50} p99={l.itl_p99} />
            <LatencyRow label="TPOT" p50={l.tpot_p50} />
            <LatencyRow label="E2E" p50={l.e2e_p50} p90={l.e2e_p90} p99={l.e2e_p99} />
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

function LatencyRow({
  label,
  p50,
  p90,
  p99,
  mean,
}: {
  label: string;
  p50?: number | null;
  p90?: number | null;
  p99?: number | null;
  mean?: number | null;
}) {
  const cell = (v: number | null | undefined) =>
    v === undefined ? "" : formatLatency(v);
  return (
    <tr className="border-t border-slate-800/70">
      <td className="py-1.5 pr-3 font-sans text-xs uppercase tracking-wide text-slate-500">
        {label}
      </td>
      <td className="py-1.5 pr-3 text-right text-slate-100">{cell(p50)}</td>
      <td className="py-1.5 pr-3 text-right">{cell(p90)}</td>
      <td className="py-1.5 pr-3 text-right">{cell(p99)}</td>
      <td className="py-1.5 text-right text-slate-400">{cell(mean)}</td>
    </tr>
  );
}

// ===========================================================================
// Aggregations
// ===========================================================================

function ByTokenPanel({ data }: { data: LiveRequestsSnapshot | undefined }) {
  const rows = [...(data?.by_token ?? [])].sort(
    (a, b) => b.context_tokens - a.context_tokens,
  );
  return (
    <Panel title="By token" bodyClassName="p-0">
      {rows.length === 0 ? (
        <p className="p-4 text-sm text-slate-500">No active sessions.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="border-b border-slate-800 text-left text-[11px] uppercase tracking-wider text-slate-500">
              <tr>
                <th className="px-4 py-2 font-medium">Token</th>
                <th className="px-3 py-2 text-right font-medium">Reqs</th>
                <th className="px-3 py-2 text-right font-medium">Context</th>
                <th className="px-3 py-2 text-right font-medium">Prompt</th>
                <th className="px-3 py-2 text-right font-medium">Compl.</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/70 font-mono tabular-nums text-slate-300">
              {rows.map((r, i) => (
                <tr key={`${r.token_name ?? "anon"}-${i}`}>
                  <td className="px-4 py-2 text-xs">
                    {r.token_name ?? <span className="text-slate-500">anonymous</span>}
                  </td>
                  <td className="px-3 py-2 text-right">{r.requests}</td>
                  <td className="px-3 py-2 text-right text-slate-100">
                    {formatCompact(r.context_tokens)}
                  </td>
                  <td className="px-3 py-2 text-right">{formatCompact(r.prompt_tokens)}</td>
                  <td className="px-3 py-2 text-right">{formatCompact(r.completion_tokens)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}

function ByIpPanel({ data }: { data: LiveRequestsSnapshot | undefined }) {
  const rows = [...(data?.by_ip ?? [])].sort(
    (a, b) => b.context_tokens - a.context_tokens,
  );
  return (
    <Panel title="By client IP" bodyClassName="p-0">
      {rows.length === 0 ? (
        <p className="p-4 text-sm text-slate-500">No active sessions.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="border-b border-slate-800 text-left text-[11px] uppercase tracking-wider text-slate-500">
              <tr>
                <th className="px-4 py-2 font-medium">Client IP</th>
                <th className="px-3 py-2 text-right font-medium">Reqs</th>
                <th className="px-3 py-2 text-right font-medium">Context</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/70 font-mono tabular-nums text-slate-300">
              {rows.map((r, i) => (
                <tr key={`${r.client_ip ?? "unknown"}-${i}`}>
                  <td className="px-4 py-2 text-xs">{r.client_ip ?? "—"}</td>
                  <td className="px-3 py-2 text-right">{r.requests}</td>
                  <td className="px-3 py-2 text-right text-slate-100">
                    {formatCompact(r.context_tokens)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}

// ===========================================================================
// Cache / MFU / finished
// ===========================================================================

function CachePanel({
  frame,
  skeleton,
}: {
  frame: LiveEngineFrame | null;
  skeleton: boolean;
}) {
  if (skeleton || !frame) return <PanelSkeleton title="Prefix cache" />;
  const c = frame.cache;
  const interval = c.prefix_hit_rate;
  return (
    <Panel title="Prefix cache">
      <BigNumber value={interval === null ? "—" : formatPct(interval)} />
      <p className="mt-0.5 text-xs text-slate-500">hit rate, this interval</p>
      {interval !== null && (
        <Meter fraction={interval} className="mt-3 h-2" showTicks={false} tone="neutral" />
      )}
      <div className="mt-3 space-y-1 border-t border-slate-800 pt-3 text-xs">
        <KeyVal k="cumulative" v={formatPct(c.prefix_hit_rate_cumulative)} />
        {c.mm_hit_rate_cumulative !== null && (
          <KeyVal k="multimodal" v={formatPct(c.mm_hit_rate_cumulative)} />
        )}
        {c.external_prefix_hit_rate_cumulative !== null && (
          <KeyVal k="external" v={formatPct(c.external_prefix_hit_rate_cumulative)} />
        )}
      </div>
    </Panel>
  );
}

function MfuPanel({
  frame,
  skeleton,
}: {
  frame: LiveEngineFrame | null;
  skeleton: boolean;
}) {
  if (skeleton || !frame) return <PanelSkeleton title="MFU" />;
  const mfu = frame.mfu;
  const estimate = mfu?.mfu_estimate ?? null;
  return (
    <Panel title="MFU">
      {estimate === null && (mfu?.flops_per_gpu_total ?? null) === null ? (
        <div className="flex h-full flex-col justify-center py-2">
          <p className="text-sm text-slate-500">Not reported</p>
          <p className="mt-1 text-xs text-slate-600">
            Model FLOPs Utilization isn&apos;t exposed by this engine build.
          </p>
        </div>
      ) : (
        <>
          <BigNumber value={estimate === null ? "—" : formatPct(estimate)} />
          <p className="mt-0.5 text-xs text-slate-500">model FLOPs utilization</p>
          {estimate !== null && (
            <Meter fraction={estimate} className="mt-3 h-2" showTicks={false} tone="neutral" />
          )}
          <div className="mt-3 border-t border-slate-800 pt-3 text-xs">
            <KeyVal k="compute" v={formatFlops(mfu?.flops_per_gpu_total)} />
          </div>
        </>
      )}
    </Panel>
  );
}

// Ordered so the "good" outcome (stop) reads first; unusual outcomes (length,
// abort) get the warm/hot colors that match their operational meaning.
const FINISHED_COLOR: Record<string, string> = {
  stop: "bg-emerald-500",
  length: "bg-amber-500",
  abort: "bg-red-500",
};
const FINISHED_TEXT: Record<string, string> = {
  stop: "text-emerald-300",
  length: "text-amber-300",
  abort: "text-red-300",
};

function FinishedPanel({
  frame,
  skeleton,
}: {
  frame: LiveEngineFrame | null;
  skeleton: boolean;
}) {
  const entries = useMemo(
    () => Object.entries(frame?.finished ?? {}).filter(([, v]) => v > 0),
    [frame],
  );
  if (skeleton || !frame) return <PanelSkeleton title="Finished reasons" />;
  const total = entries.reduce((acc, [, v]) => acc + v, 0);
  return (
    <Panel title="Finished reasons">
      {total === 0 ? (
        <p className="py-2 text-sm text-slate-500">No completions yet.</p>
      ) : (
        <>
          <div className="flex h-2.5 w-full overflow-hidden rounded-full bg-slate-800">
            {entries.map(([reason, count]) => (
              <div
                key={reason}
                className={cn("h-full", FINISHED_COLOR[reason] ?? "bg-slate-500")}
                style={{ width: `${(count / total) * 100}%` }}
                title={`${reason}: ${formatInt(count)}`}
              />
            ))}
          </div>
          <ul className="mt-3 space-y-1.5 text-xs">
            {entries.map(([reason, count]) => (
              <li key={reason} className="flex items-center justify-between gap-2">
                <span className="flex items-center gap-1.5">
                  <span
                    aria-hidden="true"
                    className={cn(
                      "h-2 w-2 rounded-sm",
                      FINISHED_COLOR[reason] ?? "bg-slate-500",
                    )}
                  />
                  <span className={cn(FINISHED_TEXT[reason] ?? "text-slate-300")}>
                    {reason}
                  </span>
                </span>
                <span className="font-mono tabular-nums text-slate-400">
                  {formatInt(count)}
                  <span className="ml-1.5 text-slate-600">
                    {formatPct(count / total)}
                  </span>
                </span>
              </li>
            ))}
          </ul>
        </>
      )}
    </Panel>
  );
}

// ===========================================================================
// Shared primitives
// ===========================================================================

function Panel({
  title,
  right,
  children,
  className,
  bodyClassName,
}: {
  title: React.ReactNode;
  right?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section
      className={cn(
        "rounded-lg border border-slate-800 bg-slate-900/50",
        className,
      )}
    >
      <div className="flex items-center justify-between gap-2 border-b border-slate-800 px-4 py-2.5">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          {title}
        </h2>
        {right}
      </div>
      <div className={cn("p-4", bodyClassName)}>{children}</div>
    </section>
  );
}

function BigNumber({
  value,
  accent,
}: {
  value: React.ReactNode;
  accent?: boolean;
}) {
  return (
    <span
      className={cn(
        "font-mono text-4xl font-semibold tabular-nums",
        accent ? "text-emerald-300" : "text-slate-100",
      )}
    >
      {value}
    </span>
  );
}

function MiniStat({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <p className="text-[11px] uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-0.5 font-mono text-sm tabular-nums text-slate-200">{value}</p>
    </div>
  );
}

function KeyVal({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-2">
      <span className="text-slate-500">{k}</span>
      <span className="font-mono tabular-nums text-slate-300">{v}</span>
    </div>
  );
}

// The pressure meter — the page's signature element. `tone="neutral"` draws a
// flat emerald bar (for hit-rate / MFU, where higher is simply better rather
// than a saturation warning); the default derives green/amber/red from the
// shared pressure thresholds. Fill colors are theme-independent by design.
function Meter({
  fraction,
  className,
  showTicks = true,
  tone = "pressure",
}: {
  fraction: number;
  className?: string;
  showTicks?: boolean;
  tone?: "pressure" | "neutral";
}) {
  const clamped = Math.max(0, Math.min(1, fraction));
  const fill =
    tone === "neutral" ? "bg-emerald-500" : PRESSURE_FILL[pressureOf(clamped)];
  return (
    <div
      className={cn(
        "relative w-full overflow-hidden rounded-full bg-slate-800",
        className,
      )}
      role="meter"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(clamped * 100)}
    >
      <div
        className={cn(
          "h-full rounded-full transition-[width] duration-500 ease-out motion-reduce:transition-none",
          fill,
        )}
        style={{ width: `${clamped * 100}%` }}
      />
      {showTicks && tone === "pressure" && (
        <>
          <span
            aria-hidden="true"
            className="absolute inset-y-0 left-[60%] w-px bg-slate-950/40"
          />
          <span
            aria-hidden="true"
            className="absolute inset-y-0 left-[85%] w-px bg-slate-950/40"
          />
        </>
      )}
    </div>
  );
}

// Rolling SVG sparkline. Deliberately dependency-free (not recharts): it
// re-renders every ~2s with the whole page, and a 40-point polyline is far
// cheaper than a Recharts tree. Stroke follows `currentColor`, so the caller
// sets the color via a text-* class on the wrapper.
function Sparkline({ data, className }: { data: number[]; className?: string }) {
  if (data.length < 2) {
    return <div className={cn("h-7 w-full", className)} aria-hidden="true" />;
  }
  const w = 100;
  const h = 28;
  const max = Math.max(...data);
  const min = Math.min(...data);
  const range = max - min || 1;
  const points = data
    .map((v, i) => {
      const x = (i / (data.length - 1)) * w;
      const y = h - ((v - min) / range) * (h - 2) - 1;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      preserveAspectRatio="none"
      className={cn("h-7 w-full", className)}
      aria-hidden="true"
    >
      <polyline
        points={points}
        fill="none"
        stroke="currentColor"
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}

// ===========================================================================
// States: banner / empty / skeletons
// ===========================================================================

function Banner({
  tone,
  children,
}: {
  tone: "warn" | "error";
  children: React.ReactNode;
}) {
  return (
    <div
      role={tone === "error" ? "alert" : "status"}
      className={cn(
        "rounded-md border px-4 py-2.5 text-sm",
        tone === "error"
          ? "border-red-800/40 bg-red-950/30 text-red-300"
          : "border-amber-800/40 bg-amber-950/40 text-amber-300",
      )}
    >
      {children}
    </div>
  );
}

function EmptyState({ title, body }: { title: string; body: string }) {
  return (
    <div className="rounded-lg border border-dashed border-slate-700 bg-slate-900/30 p-10 text-center">
      <p className="text-base font-medium text-slate-300">{title}</p>
      <p className="mx-auto mt-1.5 max-w-md text-sm text-slate-500">{body}</p>
    </div>
  );
}

function SkeletonCard({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        "animate-pulse rounded-lg border border-slate-800 bg-slate-900/50",
        className,
      )}
    />
  );
}

function SkeletonBar() {
  return <div className="h-10 w-full animate-pulse rounded bg-slate-800/60" />;
}

function PanelSkeleton({ title }: { title: string }) {
  return (
    <Panel title={title}>
      <div className="h-20 animate-pulse rounded bg-slate-800/60" />
    </Panel>
  );
}
