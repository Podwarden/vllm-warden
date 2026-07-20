'use client';
// Header metrics widget — compact live VRAM% + GPU% + active-model badge
// mounted inside <NavBar /> (right cluster, between ThemeSwitcher and
// the menu button). One <EventSource> per browser tab, ref-counted by
// frontend/src/lib/header-metrics-stream.ts.
//
// Visual language:
//   - server-rack instrument cluster — three readouts as a single
//     horizontal pill, divided by hairline slate dividers, mono digits.
//   - chrome stays subdued (slate-900/40 + slate-700/60 border) until
//     a probe error or terminal-error promotes the cluster to warning
//     (amber-400) / fault (red-400) state.
//   - the active-model chip is the cluster's identity slot — it gets a
//     persistent emerald dot when loaded so the operator can tell at a
//     glance whether the rack is "running anything" without reading text.
//   - hidden on /login and /setup matches NavBar's own gate.
//
// Restraint notes: this widget is glanceable and SHOULD NOT compete
// with the brand block or the Stats page charts. We deliberately don't
// animate the digits — flicker on every 2s tick is fatiguing.
import { Cpu, MemoryStick } from 'lucide-react';
import { useHeaderMetrics } from '@/lib/header-metrics-stream';

// Format a percentage 0–100 (or null) into a fixed-width readout. We use
// figure-tab-numerals via Tailwind's `tabular-nums` so the digit grid
// doesn't jitter as values move between 1 and 100.
function pct(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '--';
  const clamped = Math.max(0, Math.min(100, Math.round(value)));
  return String(clamped).padStart(2, ' ');
}

// Compact MiB → GiB rendering for the VRAM tooltip — keeps the badge
// itself percentage-only.
function gib(mib: number): string {
  if (!mib) return '0';
  return (mib / 1024).toFixed(1);
}

export function HeaderMetrics() {
  const { status, frame, errorCode } = useHeaderMetrics();

  // Terminal states render a quiet fault chip rather than the cluster.
  // We don't want the widget to nag — if /api/header/metrics/stream is
  // 401 (session expired) or 404 (endpoint missing on an old build),
  // collapse to a hyphenated placeholder so the chrome stays calm.
  const terminal = status === 'terminal-error';
  const reconnecting = status === 'reconnecting';
  const probeError = frame?.probe_error ?? null;

  // Choose an accent color: emerald (healthy + loaded), slate (healthy
  // + idle), amber (reconnecting / probe-error), red (terminal). We
  // deliberately desaturate the idle state so the widget doesn't
  // demand attention when nothing is happening.
  const accent =
    terminal ? 'text-red-400' :
    reconnecting || probeError ? 'text-amber-400' :
    frame?.active_model ? 'text-emerald-400' :
    'text-slate-400';

  const dot =
    terminal ? 'bg-red-400/80' :
    reconnecting || probeError ? 'bg-amber-400/80' :
    frame?.active_model ? 'bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.6)]' :
    'bg-slate-500/70';

  const vramPct = terminal ? null : frame?.vram_pct ?? null;
  const gpuPct = terminal ? null : frame?.gpu_util_pct ?? null;
  const modelLabel = terminal
    ? 'offline'
    : frame?.active_model ?? 'idle';

  // Build a single-line tooltip that surfaces the data the badge omits:
  // per-GPU breakdown, probe error, status hint.
  const tooltipLines: string[] = [];
  if (frame) {
    tooltipLines.push(
      `VRAM ${gib(frame.vram_used_mib)} / ${gib(frame.vram_total_mib)} GiB`,
    );
    for (const g of frame.gpus) {
      const name = g.name ?? `GPU ${g.index}`;
      tooltipLines.push(
        `  ${name}: ${gib(g.memory_used_mib)}/${gib(g.memory_total_mib)} GiB · util ${g.utilization_pct}%`,
      );
    }
    if (frame.active_model) {
      tooltipLines.push(`Loaded: ${frame.active_model}`);
    }
  }
  if (probeError) tooltipLines.push(`Probe error: ${probeError}`);
  if (reconnecting) tooltipLines.push('Reconnecting…');
  if (terminal) {
    tooltipLines.push(
      errorCode === 401
        ? 'Session expired — refresh the page'
        : `Stream unavailable (HTTP ${errorCode ?? '?'})`,
    );
  }
  const tooltip = tooltipLines.join('\n') || 'header metrics';

  return (
    <div
      // role=status so screen readers announce the cluster but don't
      // promote it to a live region (we don't want every 2s tick
      // narrated). aria-live=off forces that.
      role="status"
      aria-live="off"
      aria-label={`Header metrics — VRAM ${pct(vramPct).trim()} percent, GPU ${pct(gpuPct).trim()} percent, ${modelLabel}`}
      title={tooltip}
      data-testid="header-metrics"
      data-status={status}
      className={[
        'hidden md:inline-flex items-center gap-2',
        'h-8 px-2.5 rounded-md',
        'border border-slate-700/60 bg-slate-900/40',
        'font-mono text-xs leading-none',
        'transition-colors duration-300',
        accent,
      ].join(' ')}
    >
      {/* VRAM readout */}
      <span className="inline-flex items-center gap-1.5 tabular-nums">
        <MemoryStick className="h-3.5 w-3.5 opacity-70" aria-hidden="true" />
        <span className="text-slate-400">VRAM</span>
        <span data-testid="header-metrics-vram-pct" className="text-slate-100">
          {pct(vramPct)}
        </span>
        <span className="text-slate-500">%</span>
      </span>

      <span className="h-3 w-px bg-slate-700/80" aria-hidden="true" />

      {/* GPU util readout */}
      <span className="inline-flex items-center gap-1.5 tabular-nums">
        <Cpu className="h-3.5 w-3.5 opacity-70" aria-hidden="true" />
        <span className="text-slate-400">GPU</span>
        <span data-testid="header-metrics-gpu-pct" className="text-slate-100">
          {pct(gpuPct)}
        </span>
        <span className="text-slate-500">%</span>
      </span>

      <span className="h-3 w-px bg-slate-700/80" aria-hidden="true" />

      {/* Active model chip — identity slot. The dot encodes status
          (emerald=loaded, slate=idle, amber=warn, red=fault). */}
      <span className="inline-flex items-center gap-1.5 max-w-[10rem]">
        <span
          aria-hidden="true"
          className={['h-1.5 w-1.5 rounded-full transition-colors', dot].join(' ')}
        />
        <span
          data-testid="header-metrics-model"
          className="truncate text-slate-200"
        >
          {modelLabel}
        </span>
      </span>
    </div>
  );
}
