"use client";

// Historical charts for the redesigned /stats page (S7, #124).
//
// Four charts, all minute-bucketed, all consuming `/api/stats/v2/overview`
// `series` arrays directly — no client-side pivoting (v1's whole pain
// point: see the comment block at the top of @/lib/stats). Each chart
// receives its own slice + the [start, end] bounds derived from the
// range selector.
//
// Visual language matches the v1 charts (gpu-util-chart, throughput-chart)
// so the page reads as one piece: slate panel, emerald primary, sibling
// hues for the secondary series. Animations are off everywhere — a 30s
// poll on a flickering chart is exhausting.

import {
  AreaChart,
  Area,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import {
  withTs,
  type StatsV2VramPoint,
  type StatsV2UtilPoint,
  type StatsV2PowerPoint,
  type StatsV2TokensPoint,
  type StatsRange,
} from "@/lib/stats-v2";
import { rangeBounds } from "@/lib/stats";

// ---- shared tick formatters ---------------------------------------------

function fmtTime(ts: number): string {
  return new Date(ts).toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function fmtDate(ts: number): string {
  return new Date(ts).toLocaleDateString(undefined, {
    month: "short",
    day: "2-digit",
  });
}

function pickTickFormatter(range: StatsRange): (ts: number) => string {
  // 7d resolution would render every-minute HH:MM ticks as a smear.
  return range === "7d" ? fmtDate : fmtTime;
}

// Empty-state / children switcher shared by all four chart components — keeps
// the dashed-border placeholder rendering in one place.
interface ChartShellProps {
  emptyLabel: string;
  hasData: boolean;
  children: React.ReactNode;
}

function ChartShell({ emptyLabel, hasData, children }: ChartShellProps) {
  if (!hasData) {
    return (
      <div className="rounded-lg border border-dashed border-slate-700 bg-slate-900/30 p-8 text-center text-sm text-slate-400">
        {emptyLabel}
      </div>
    );
  }
  return (
    <div className="h-64 w-full rounded-lg border border-slate-700 bg-slate-900/30 p-2">
      {children}
    </div>
  );
}

const TOOLTIP_STYLE = {
  backgroundColor: "#0f172a",
  border: "1px solid #334155",
  fontSize: "12px",
};

// ---- VRAM over time ------------------------------------------------------

interface VramChartProps {
  points: readonly StatsV2VramPoint[];
  range: StatsRange;
}

export function VramChart({ points, range }: VramChartProps) {
  const data = withTs(points);
  const bounds = rangeBounds(range);
  const tickFmt = pickTickFormatter(range);
  // Y-axis ceiling: pin to the largest total_mib in window so the chart
  // doesn't auto-fit to the used floor (which would mask headroom). Fall
  // back to dataMax when no total is reported (e.g. mock fixtures).
  const yMax =
    data.reduce((m, p) => Math.max(m, p.total_mib), 0) || undefined;
  return (
    <ChartShell hasData={data.length > 0} emptyLabel="No VRAM samples in this window.">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <defs>
            <linearGradient id="vram-fill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#34d399" stopOpacity={0.6} />
              <stop offset="95%" stopColor="#34d399" stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#334155" strokeDasharray="3 3" />
          <XAxis
            dataKey="ts"
            type="number"
            domain={bounds}
            allowDataOverflow
            scale="time"
            tickFormatter={tickFmt}
            stroke="#94a3b8"
            fontSize={11}
          />
          <YAxis
            domain={[0, yMax ?? "auto"]}
            tickFormatter={(v) => `${(Number(v) / 1024).toFixed(0)}`}
            stroke="#94a3b8"
            fontSize={11}
            width={56}
            label={{
              value: "GiB",
              angle: -90,
              position: "insideLeft",
              fill: "#64748b",
              fontSize: 11,
            }}
          />
          <Tooltip
            contentStyle={TOOLTIP_STYLE}
            labelFormatter={(v) => tickFmt(Number(v))}
            formatter={(value, name) => [
              typeof value === "number" ? `${(value / 1024).toFixed(1)} GiB` : String(value),
              String(name),
            ]}
          />
          <Area
            type="monotone"
            dataKey="used_mib"
            name="VRAM used"
            stroke="#34d399"
            fill="url(#vram-fill)"
            strokeWidth={2}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

// ---- GPU util over time --------------------------------------------------

interface UtilChartProps {
  points: readonly StatsV2UtilPoint[];
  range: StatsRange;
}

export function UtilChart({ points, range }: UtilChartProps) {
  const data = withTs(points);
  const bounds = rangeBounds(range);
  const tickFmt = pickTickFormatter(range);
  return (
    <ChartShell hasData={data.length > 0} emptyLabel="No GPU util samples in this window.">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <defs>
            <linearGradient id="util-fill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#60a5fa" stopOpacity={0.6} />
              <stop offset="95%" stopColor="#60a5fa" stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#334155" strokeDasharray="3 3" />
          <XAxis
            dataKey="ts"
            type="number"
            domain={bounds}
            allowDataOverflow
            scale="time"
            tickFormatter={tickFmt}
            stroke="#94a3b8"
            fontSize={11}
          />
          <YAxis
            domain={[0, 100]}
            tickFormatter={(v) => `${v}%`}
            stroke="#94a3b8"
            fontSize={11}
            width={48}
          />
          <Tooltip
            contentStyle={TOOLTIP_STYLE}
            labelFormatter={(v) => tickFmt(Number(v))}
            formatter={(value, name) => [
              typeof value === "number" ? `${value.toFixed(0)}%` : String(value),
              String(name),
            ]}
          />
          <Area
            type="monotone"
            dataKey="max_pct"
            name="GPU util (max)"
            stroke="#60a5fa"
            fill="url(#util-fill)"
            strokeWidth={2}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

// ---- Power over time -----------------------------------------------------

interface PowerChartProps {
  points: readonly StatsV2PowerPoint[];
  range: StatsRange;
  /** True iff the host has at least one card that reports power.draw.
   *  Lets the empty state distinguish "no samples yet" from "card can't
   *  do it" — a virtualised host will never produce power data and the
   *  operator should know to stop waiting. */
  supported?: boolean;
}

export function PowerChart({ points, range, supported = true }: PowerChartProps) {
  const data = withTs(points);
  const bounds = rangeBounds(range);
  const tickFmt = pickTickFormatter(range);
  const emptyLabel = supported
    ? "No power samples in this window."
    : "Power telemetry not reported by this host's GPUs.";
  return (
    <ChartShell hasData={data.length > 0} emptyLabel={emptyLabel}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <defs>
            <linearGradient id="power-fill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#fbbf24" stopOpacity={0.6} />
              <stop offset="95%" stopColor="#fbbf24" stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#334155" strokeDasharray="3 3" />
          <XAxis
            dataKey="ts"
            type="number"
            domain={bounds}
            allowDataOverflow
            scale="time"
            tickFormatter={tickFmt}
            stroke="#94a3b8"
            fontSize={11}
          />
          <YAxis
            stroke="#94a3b8"
            fontSize={11}
            width={56}
            tickFormatter={(v) => `${v}`}
            label={{
              value: "W",
              angle: -90,
              position: "insideLeft",
              fill: "#64748b",
              fontSize: 11,
            }}
          />
          <Tooltip
            contentStyle={TOOLTIP_STYLE}
            labelFormatter={(v) => tickFmt(Number(v))}
            formatter={(value, name) => [
              typeof value === "number" ? `${value.toFixed(1)} W` : String(value),
              String(name),
            ]}
          />
          <Area
            type="monotone"
            dataKey="watts"
            name="Power draw"
            stroke="#fbbf24"
            fill="url(#power-fill)"
            strokeWidth={2}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

// ---- Tokens over time ----------------------------------------------------

interface TokensChartProps {
  points: readonly StatsV2TokensPoint[];
  range: StatsRange;
}

export function TokensChart({ points, range }: TokensChartProps) {
  // Two lines on one axis — prompt and completion. Stacking would let
  // the operator read "total throughput" at a glance but obscure the
  // mix; v2's value is exposing the mix, so we render them separately.
  const data = withTs(points);
  const bounds = rangeBounds(range);
  const tickFmt = pickTickFormatter(range);
  return (
    <ChartShell hasData={data.length > 0} emptyLabel="No token usage in this window.">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid stroke="#334155" strokeDasharray="3 3" />
          <XAxis
            dataKey="ts"
            type="number"
            domain={bounds}
            allowDataOverflow
            scale="time"
            tickFormatter={tickFmt}
            stroke="#94a3b8"
            fontSize={11}
          />
          <YAxis
            stroke="#94a3b8"
            fontSize={11}
            tickFormatter={(v) => Number(v).toLocaleString()}
            width={72}
          />
          <Tooltip
            contentStyle={TOOLTIP_STYLE}
            labelFormatter={(v) => tickFmt(Number(v))}
            formatter={(value, name) => [
              typeof value === "number" ? value.toLocaleString() : String(value),
              String(name),
            ]}
          />
          <Legend wrapperStyle={{ fontSize: "12px" }} />
          <Line
            type="monotone"
            dataKey="prompt"
            name="Prompt"
            stroke="#a78bfa"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="completion"
            name="Completion"
            stroke="#34d399"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}
