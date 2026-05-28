"use client";

// ---------------------------------------------------------------------------
// GpuChecklist — single source of truth for GPU-selection UX.
//
// Prop-driven and presentational: it never fetches. Callers pass the live
// inventory (`gpus`, from GET /api/system/gpus) and the configured selection
// (`selected`). The component renders one checkbox per present GPU and, for
// any configured index NOT in the inventory, a distinct "ghost row" so a
// missing GPU is surfaced and repairable rather than silently dropped.
//
// `onChange` always emits a sorted-ascending number[] so callers never have
// to re-sort, and dirty-tracking against a stored (sorted) value is stable.
// ---------------------------------------------------------------------------

export interface GpuInfo {
  index: number;
  name: string;
  memory_total_mib: number;
  memory_used_mib: number;
  utilization_pct: number;
}

interface GpuChecklistProps {
  gpus: GpuInfo[];
  selected: number[];
  onChange: (next: number[]) => void;
  disabled?: boolean;
  /** Forwarded to the component's root element as `aria-describedby`. Lets a
   *  caller (e.g. SettingField's chrome) wire the field hint to this control,
   *  since GpuChecklist owns its own per-GPU checkbox ids and can't consume an
   *  external input id. */
  describedById?: string;
}

function emit(onChange: (n: number[]) => void, set: Set<number>) {
  onChange(Array.from(set).sort((a, b) => a - b));
}

export function GpuChecklist({ gpus, selected, onChange, disabled = false, describedById }: GpuChecklistProps) {
  const selectedSet = new Set(selected);
  const presentIndices = new Set(gpus.map((g) => g.index));
  // Configured indices with no matching present GPU — render as ghost rows.
  const missing = selected.filter((i) => !presentIndices.has(i)).sort((a, b) => a - b);

  function toggle(index: number) {
    const next = new Set(selectedSet);
    if (next.has(index)) next.delete(index);
    else next.add(index);
    emit(onChange, next);
  }

  if (gpus.length === 0 && missing.length === 0) {
    return (
      <p className="text-sm text-slate-500" data-testid="gpu-empty" aria-describedby={describedById}>
        No GPUs detected — saving will still validate against allowed_gpu_indices server-side.
      </p>
    );
  }

  return (
    <div className="space-y-1" aria-describedby={describedById}>
      {missing.length > 0 && (
        <div
          role="alert"
          className="rounded-md border border-amber-600/50 bg-amber-950/40 px-2 py-1.5 text-xs text-amber-300"
        >
          {missing.length === 1 ? "GPU" : "GPUs"} {missing.join(", ")} configured but not present
          in the system. Uncheck to remove, or restore the card before loading.
        </div>
      )}
      <ul className="grid grid-cols-1 gap-1 sm:grid-cols-2" data-testid="gpu-list">
        {gpus.map((g) => {
          const id = `gpu-checklist-${g.index}`;
          const freeGiB = (g.memory_total_mib - g.memory_used_mib) / 1024;
          return (
            <li
              key={g.index}
              className="flex items-center gap-2 rounded-md border border-slate-700 bg-slate-900 px-2 py-1.5 text-xs"
            >
              <input
                id={id}
                type="checkbox"
                className="h-3.5 w-3.5"
                checked={selectedSet.has(g.index)}
                disabled={disabled}
                onChange={() => toggle(g.index)}
              />
              <label htmlFor={id} className="flex-1 cursor-pointer">
                <span className="font-mono text-slate-300">#{g.index}</span>{" "}
                <span>{g.name}</span>{" "}
                <span className="text-slate-500">{freeGiB.toFixed(1)} GiB free</span>
              </label>
            </li>
          );
        })}
        {missing.map((index) => {
          const id = `gpu-checklist-missing-${index}`;
          return (
            <li
              key={`missing-${index}`}
              className="flex items-center gap-2 rounded-md border border-amber-700/60 bg-amber-950/30 px-2 py-1.5 text-xs"
            >
              <input
                id={id}
                type="checkbox"
                className="h-3.5 w-3.5"
                checked={selectedSet.has(index)}
                disabled={disabled}
                onChange={() => toggle(index)}
              />
              <label htmlFor={id} className="flex-1 cursor-pointer text-amber-300">
                GPU {index} — not present
              </label>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
