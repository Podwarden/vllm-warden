"use client";

// Historical charts for the redesigned /stats page (S7, #124).
//
// Three charts, all minute-bucketed, all consuming `/api/stats/v2/overview`
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
  ReferenceLine,
  ResponsiveContainer,
  Legend,
} from "recharts";
import type { Reference } from "@/lib/live-history";
import {
  withTs,
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
    // text-chat-muted feeds `currentColor` inside the SVG: the 24h reference
    // lines below draw with it, so they resolve through the theme tokens
    // rather than a literal colour.
    <div className="h-64 w-full rounded-lg border border-slate-700 bg-slate-900/30 p-2 text-chat-muted">
      {children}
    </div>
  );
}

// ---- fixed 24h reference lines -------------------------------------------
//
// Dashed = the 24h median of BUSY minutes, dotted = the 24h peak. Fixed at
// 24h regardless of the selected window — a reference that moved with the
// view could never answer "is this normal for this box?". In retro-dark
// several tokens share a hue, so the two references differ by dash PATTERN
// and label, never by colour alone. `null` median (no busy minutes in 24h)
// draws no line rather than a baseline invented from idle samples.

// A reference label is placed from where its line actually sits in the
// domain, not from a fixed corner. Two failures drove this:
//
//   * a line at the TOP of the scale (GPU utilisation pinned at 100%, VRAM
//     at the card total) had its label drawn ABOVE the line, outside the
//     plot, where it was clipped by the panel edge;
//   * median and peak both hugged the RIGHT edge, so when the two values are
//     close (511 W against a 555 W peak) the labels sat on top of each other.
//
// So: a line in the upper part of the domain gets its label below, otherwise
// above; and peak keeps the right edge while median takes the left. They can
// never collide horizontally, whatever the two values are.
const LABEL_FLIP_FRACTION = 0.85;

export function refLabelPosition(value: number, yMax: number, side: "Left" | "Right") {
  const high = yMax > 0 && value / yMax > LABEL_FLIP_FRACTION;
  return `inside${high ? "Bottom" : "Top"}${side}` as const;
}

function refLines(
  reference: Reference | undefined,
  fmt: (v: number) => string,
  yMax: number,
): React.ReactNode[] {
  if (!reference) return [];
  const out: React.ReactNode[] = [];
  if (reference.peak !== null) {
    out.push(
      <ReferenceLine
        key="ref-peak"
        y={reference.peak}
        stroke="currentColor"
        strokeOpacity={0.6}
        strokeDasharray="1 3"
        label={{
          value: `24h peak ${fmt(reference.peak)}`,
          position: refLabelPosition(reference.peak, yMax, "Right"),
          fill: "currentColor",
          fontSize: 10,
        }}
      />,
    );
  }
  if (reference.median !== null) {
    out.push(
      <ReferenceLine
        key="ref-median"
        y={reference.median}
        stroke="currentColor"
        strokeOpacity={0.6}
        strokeDasharray="6 4"
        label={{
          value: `24h busy median ${fmt(reference.median)}`,
          position: refLabelPosition(reference.median, yMax, "Left"),
          fill: "currentColor",
          fontSize: 10,
        }}
      />,
    );
  }
  return out;
}

// The peak is included in the y-scale ON PURPOSE: a chart auto-scaled to a
// quiet hour makes that hour look busy, which is the failure the reference
// exists to prevent.
function ceilingWith(dataMax: number, reference: Reference | undefined): number {
  const peak = reference?.peak ?? 0;
  return Math.max(dataMax, peak) * 1.08 || 1;
}

const TOOLTIP_STYLE = {
  backgroundColor: "#0f172a",
  border: "1px solid #334155",
  fontSize: "12px",
};

// There is deliberately no VRAM-over-time chart. Both engines pre-allocate
// weights and KV cache at load (vLLM from gpu_memory_utilization, llama.cpp
// from n_ctx), so VRAM is a step function that moves only on load/unload
// and its chart is a flat block. Current VRAM is on the per-GPU cards in
// SystemConfigSection; the `vram` series stays in the API for other readers.

// ---- GPU util over time --------------------------------------------------

interface UtilChartProps {
  points: readonly StatsV2UtilPoint[];
  range: StatsRange;
  reference?: Reference;
}

export function UtilChart({ points, range, reference }: UtilChartProps) {
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
          {refLines(reference, (v) => `${Math.round(v)}%`, 100)}
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
  reference?: Reference;
}

export function PowerChart({
  points,
  range,
  supported = true,
  reference,
}: PowerChartProps) {
  const data = withTs(points);
  const bounds = rangeBounds(range);
  const tickFmt = pickTickFormatter(range);
  const yMax = ceilingWith(
    data.reduce((m, p) => Math.max(m, p.watts), 0),
    reference,
  );
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
            domain={[0, yMax]}
            stroke="#94a3b8"
            fontSize={11}
            width={56}
            tickFormatter={(v) => `${Math.round(Number(v))}`}
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
          {refLines(reference, (v) => `${Math.round(v)} W`, yMax)}
        </AreaChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

// ---- Tokens over time ----------------------------------------------------

interface TokensChartProps {
  points: readonly StatsV2TokensPoint[];
  range: StatsRange;
  /** Computed over the PROMPT series (the dominant one), like the rest a
   *  fixed 24h reference. */
  reference?: Reference;
}

export function TokensChart({ points, range, reference }: TokensChartProps) {
  // Two lines on one axis — prompt and completion. Stacking would let
  // the operator read "total throughput" at a glance but obscure the
  // mix; v2's value is exposing the mix, so we render them separately.
  const data = withTs(points);
  const bounds = rangeBounds(range);
  const tickFmt = pickTickFormatter(range);
  const yMax = ceilingWith(
    data.reduce((m, p) => Math.max(m, p.prompt, p.completion), 0),
    reference,
  );
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
            domain={[0, yMax]}
            stroke="#94a3b8"
            fontSize={11}
            tickFormatter={(v) => Math.round(Number(v)).toLocaleString()}
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
          {refLines(
            reference,
            (v) => (v >= 1000 ? `${Math.round(v / 1000)}k` : `${Math.round(v)}`),
            yMax,
          )}
        </LineChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}
