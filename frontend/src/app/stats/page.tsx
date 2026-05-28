"use client";

// /stats — operator dashboard rebuilt on /api/stats/v2 (S7, #124).
//
// Replaces the v1 stats page (which the user called "garbage" in the
// overhaul brief). One round-trip per refresh: `GET /api/stats/v2/overview`
// returns the current snapshot, the four historical series, and the
// active-model list in a single payload. A second endpoint feeds the
// per-key token table.
//
// User brief (locked):
//   "proper stats page (vram, power consumption, tokens (per api key and
//    aggregate), gpu load - both historical and current. current vram
//    and gpu load should be on every page in the header. selected range
//    for stats charts should be stored in browser storage so no need to
//    select again after page reload."
//
// Header-metrics widget (current VRAM% + GPU%) already mounts inside
// NavBar (S2, header-metrics.tsx) — that covers "on every page". This
// page owns the dedicated dashboard view.
//
// Range persistence: usePersistedRange('vw.stats.range') — reload-safe.

import { useMemo, useState } from "react";
import useSWR from "swr";
import { authFetchJSON } from "@/lib/auth-fetch";
import {
  STATS_RANGES,
  formatTps,
  formatWatts,
  mibToGib,
  type StatsRange,
  type StatsV2Overview,
  type StatsV2TokensPerKey,
  type StatsV2TokensPerKeyRow,
} from "@/lib/stats-v2";
import { usePersistedRange } from "@/lib/use-persisted-range";
import { StatCard } from "@/components/stat-card";
import {
  PowerChart,
  TokensChart,
  UtilChart,
  VramChart,
} from "@/components/stats/v2-charts";
import { SystemConfigSection } from "@/components/stats/system-config-section";
import { Skeleton } from "@/components/ui/skeleton";

// localStorage key for the range selector. Documented in the dispatch
// brief as `vw.stats.range`.
const RANGE_KEY = "vw.stats.range";

// 30s poll on /stats matches the cadence of the minute-bucketed data —
// faster would just re-render with identical numbers. Pause when the
// tab is hidden so a background tab doesn't keep the box warm.
const REFRESH_MS = 30_000;
const refreshInterval = () =>
  typeof document !== "undefined" && document.hidden ? 0 : REFRESH_MS;

type Sort = { col: keyof StatsV2TokensPerKeyRow | "total_tokens"; dir: "asc" | "desc" };

export default function StatsPage() {
  const [range, setRange] = usePersistedRange(RANGE_KEY, "1h");
  const [sort, setSort] = useState<Sort>({ col: "total_tokens", dir: "desc" });

  const overview = useSWR<StatsV2Overview>(
    `/api/stats/v2/overview?range=${range}`,
    authFetchJSON,
    { refreshInterval },
  );
  const tpk = useSWR<StatsV2TokensPerKey>(
    `/api/stats/v2/tokens-per-key?range=${range}`,
    authFetchJSON,
    { refreshInterval },
  );

  const data = overview.data;

  // Detect whether the host's GPU reports power: if `current.power_w` is
  // null AND the series has zero entries, we tell the operator the
  // telemetry isn't supported rather than leaving them staring at an
  // empty chart waiting for data that will never arrive.
  const powerSupported = useMemo(() => {
    if (!data) return true;
    if (data.current.power_w !== null) return true;
    return data.series.power.length > 0;
  }, [data]);

  const sortedRows = useMemo(() => {
    const rows = tpk.data?.rows ?? [];
    if (rows.length === 0) return rows;
    const dir = sort.dir === "asc" ? 1 : -1;
    return [...rows].sort((a, b) => {
      const av = a[sort.col];
      const bv = b[sort.col];
      // String columns (name, token_id, prefix) — locale compare.
      if (typeof av === "string" && typeof bv === "string") {
        return av.localeCompare(bv) * dir;
      }
      // Numeric columns — straight subtraction.
      const an = typeof av === "number" ? av : 0;
      const bn = typeof bv === "number" ? bv : 0;
      return (an - bn) * dir;
    });
  }, [tpk.data, sort]);

  function toggleSort(col: Sort["col"]) {
    setSort((s) =>
      s.col === col
        ? { col, dir: s.dir === "asc" ? "desc" : "asc" }
        : { col, dir: col === "name" ? "asc" : "desc" },
    );
  }

  return (
    <div className="space-y-6" data-testid="stats-page">
      {/* Header row — title + range selector --------------------------- */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold">Stats</h1>
          {data && data.active_models.length > 0 && (
            <p
              data-testid="active-models"
              className="mt-1 flex flex-wrap items-center gap-1.5 text-xs text-slate-400"
            >
              <span className="text-slate-500">Loaded:</span>
              {data.active_models.map((m) => (
                <span
                  key={m.id}
                  className="inline-flex items-center gap-1 rounded-full border border-emerald-700/60 bg-emerald-900/30 px-2 py-0.5 text-emerald-200"
                >
                  <span
                    aria-hidden="true"
                    className="h-1.5 w-1.5 rounded-full bg-emerald-400"
                  />
                  <span className="font-mono text-[11px]">
                    {m.served_model_name}
                  </span>
                </span>
              ))}
            </p>
          )}
        </div>
        <div
          role="group"
          aria-label="Time range"
          data-testid="range-selector"
          className="inline-flex h-9 rounded-md border border-slate-700 bg-slate-900/50 p-0.5 text-xs"
        >
          {STATS_RANGES.map((r) => {
            const active = r === range;
            return (
              <button
                key={r}
                type="button"
                onClick={() => setRange(r)}
                data-active={active}
                aria-pressed={active}
                className={
                  "rounded px-3 text-slate-300 transition-colors " +
                  (active
                    ? "bg-emerald-700/60 text-emerald-50 shadow-inner"
                    : "hover:bg-slate-800")
                }
              >
                {r}
              </button>
            );
          })}
        </div>
      </div>

      {overview.error && !overview.data && (
        <p className="text-sm text-red-500">
          Failed to load stats
          {overview.error instanceof Error ? `: ${overview.error.message}` : "."}
        </p>
      )}

      {/* Current snapshot — four tiles --------------------------------- */}
      <section
        aria-label="Current"
        data-testid="current-row"
        className="grid grid-cols-2 gap-3 lg:grid-cols-4"
      >
        {!data ? (
          <>
            <Skeleton className="h-24 w-full" />
            <Skeleton className="h-24 w-full" />
            <Skeleton className="h-24 w-full" />
            <Skeleton className="h-24 w-full" />
          </>
        ) : (
          <>
            <StatCard
              label="VRAM"
              value={
                <span data-testid="tile-vram-value">
                  {mibToGib(data.current.vram_used_mib)} /{" "}
                  {mibToGib(data.current.vram_total_mib)}
                </span>
              }
              unit="GiB"
              hint={`${data.current.vram_pct}% used`}
              title={`${data.current.vram_used_mib} / ${data.current.vram_total_mib} MiB`}
            />
            <StatCard
              label="GPU util"
              value={
                <span data-testid="tile-util-value">
                  {data.current.gpu_util_pct}
                </span>
              }
              unit="%"
              hint="max across GPUs"
            />
            <StatCard
              label="Power"
              value={
                <span data-testid="tile-power-value">
                  {formatWatts(data.current.power_w)}
                </span>
              }
              unit={data.current.power_w === null ? undefined : "W"}
              hint={
                data.current.power_w === null
                  ? "telemetry unavailable"
                  : "sum across GPUs"
              }
              title={
                data.current.power_w === null
                  ? "No GPU on this host reports power.draw."
                  : undefined
              }
            />
            <StatCard
              label="Tokens / sec"
              value={
                <span data-testid="tile-tps-value">
                  {formatTps(data.current.tps)}
                </span>
              }
              hint="last full minute"
            />
          </>
        )}
      </section>

      {/* Historical charts — 2×2 grid on wide, stacked on narrow ------ */}
      <section
        aria-label="Historical"
        data-testid="charts-grid"
        className="grid grid-cols-1 gap-4 lg:grid-cols-2"
      >
        <ChartPanel title="VRAM over time" testid="chart-vram">
          {!data ? (
            <Skeleton className="h-64 w-full" />
          ) : (
            <VramChart points={data.series.vram} range={range} />
          )}
        </ChartPanel>
        <ChartPanel title="GPU utilisation" testid="chart-util">
          {!data ? (
            <Skeleton className="h-64 w-full" />
          ) : (
            <UtilChart points={data.series.util} range={range} />
          )}
        </ChartPanel>
        <ChartPanel title="Power draw" testid="chart-power">
          {!data ? (
            <Skeleton className="h-64 w-full" />
          ) : (
            <PowerChart
              points={data.series.power}
              range={range}
              supported={powerSupported}
            />
          )}
        </ChartPanel>
        <ChartPanel title="Tokens / minute" testid="chart-tokens">
          {!data ? (
            <Skeleton className="h-64 w-full" />
          ) : (
            <TokensChart points={data.series.tokens} range={range} />
          )}
        </ChartPanel>
      </section>

      {/* System Configuration panel (#148) — static-ish HW/OS inventory
          that helps interpret the live numbers above. Owns its own SWR
          fetch + 60s backend cache; safe to render alongside the v2
          overview without compounding load. */}
      <SystemConfigSection />

      {/* Per-key tokens table ----------------------------------------- */}
      <section aria-label="Tokens per API key" className="space-y-3">
        <div className="flex items-baseline justify-between">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
            Tokens per API key
          </h2>
          <span className="text-xs text-slate-500">last {range}</span>
        </div>
        {tpk.isLoading && !tpk.data ? (
          <Skeleton className="h-24 w-full" />
        ) : tpk.error && !tpk.data ? (
          <p className="text-sm text-red-500">
            Failed to load per-key tokens
            {tpk.error instanceof Error ? `: ${tpk.error.message}` : "."}
          </p>
        ) : sortedRows.length === 0 ? (
          <div className="rounded-md border border-dashed border-slate-700 bg-slate-900/30 p-6 text-center text-sm text-slate-400">
            No token usage in this window.
          </div>
        ) : (
          <div
            data-testid="tokens-per-key-table"
            className="overflow-x-auto rounded-md border border-slate-800"
          >
            <table className="w-full text-sm">
              <thead className="border-b border-slate-800 bg-slate-900/50 text-left text-xs uppercase text-slate-400">
                <tr>
                  <SortableHeader
                    label="Name"
                    col="name"
                    sort={sort}
                    onSort={toggleSort}
                  />
                  <th className="px-3 py-2">Prefix</th>
                  <SortableHeader
                    label="Requests"
                    col="requests"
                    sort={sort}
                    onSort={toggleSort}
                    align="right"
                  />
                  <SortableHeader
                    label="Prompt"
                    col="prompt_tokens"
                    sort={sort}
                    onSort={toggleSort}
                    align="right"
                  />
                  <SortableHeader
                    label="Completion"
                    col="completion_tokens"
                    sort={sort}
                    onSort={toggleSort}
                    align="right"
                  />
                  <SortableHeader
                    label="Total"
                    col="total_tokens"
                    sort={sort}
                    onSort={toggleSort}
                    align="right"
                  />
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {sortedRows.map((row) => (
                  <tr
                    key={row.token_id}
                    data-testid="tokens-per-key-row"
                    className="text-slate-200"
                  >
                    <td className="px-3 py-2">
                      {row.name}
                      {row.name === "(unknown)" && (
                        <span
                          className="ml-2 rounded bg-amber-900/40 px-1.5 py-0.5 text-[10px] uppercase text-amber-300"
                          title="The api_tokens row was deleted; usage remains in the rollup."
                        >
                          orphan
                        </span>
                      )}
                    </td>
                    <td className="px-3 py-2 font-mono text-xs text-slate-400">
                      {row.prefix ?? "—"}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums">
                      {row.requests.toLocaleString()}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums">
                      {row.prompt_tokens.toLocaleString()}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums">
                      {row.completion_tokens.toLocaleString()}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums font-semibold text-slate-100">
                      {row.total_tokens.toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

// Internal wrapper for the four chart panels — keeps the section title +
// data-testid pattern in one place instead of repeating the h2 + div boilerplate
// inline four times. Departure from dispatch (which called for charts rendered
// directly inside <section>); see MR description.
interface ChartPanelProps {
  title: string;
  testid: string;
  children: React.ReactNode;
}

function ChartPanel({ title, testid, children }: ChartPanelProps) {
  return (
    <div data-testid={testid} className="space-y-2">
      <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
        {title}
      </h2>
      {children}
    </div>
  );
}

interface SortableHeaderProps {
  label: string;
  col: Sort["col"];
  sort: Sort;
  onSort: (col: Sort["col"]) => void;
  align?: "left" | "right";
}

function SortableHeader({ label, col, sort, onSort, align = "left" }: SortableHeaderProps) {
  const active = sort.col === col;
  const arrow = !active ? "" : sort.dir === "asc" ? " ↑" : " ↓";
  return (
    <th
      className={
        "px-3 py-2 " + (align === "right" ? "text-right" : "text-left")
      }
    >
      <button
        type="button"
        onClick={() => onSort(col)}
        data-testid={`sort-${col}`}
        aria-pressed={active}
        className={
          "uppercase tracking-wider transition-colors " +
          (active ? "text-emerald-300" : "text-slate-400 hover:text-slate-200")
        }
      >
        {label}
        {arrow}
      </button>
    </th>
  );
}
