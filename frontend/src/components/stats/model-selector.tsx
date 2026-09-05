"use client";
// The model selector, shared by /stats, /stats/live and /godmode.
//
// One control, one selection, three surfaces — the operator picks once and
// every page answers about the same models. A per-page selector would let two
// open tabs disagree about what "the numbers" mean.
//
// THE LAST CHECKBOX IS DISABLED, not silently re-ticked. At least one model
// must stay selected (see @/lib/model-selection for why an empty selection is
// unanswerable), and there are two ways to enforce that: refuse the click
// invisibly, or say so. Refusing invisibly leaves the operator unable to tell
// whether the click registered at all — which is this codebase's defining
// defect class, an affordance whose effect is gated on a different condition
// than the affordance itself. So the input carries `disabled` and a title
// explaining it, and the gate and the control are the same fact.
import type { ModelSelection } from "@/lib/model-selection";

export interface SelectableModel {
  id: string;
  served_model_name: string;
  /** Optional: which cards this model occupies, for the tooltip. */
  gpu_indices?: number[];
}

export function ModelSelector({
  models,
  selection,
  label = "Models",
}: {
  models: SelectableModel[];
  selection: ModelSelection;
  label?: string;
}) {
  // Nothing loaded: render nothing rather than an empty box. There is no
  // selection to make, and the pages already have their own "nothing loaded"
  // states that say more than a disabled fieldset would.
  if (models.length === 0) return null;

  const allSelected = selection.selected.length === models.length;

  return (
    <div
      role="group"
      aria-label={label}
      data-testid="model-selector"
      className="flex flex-wrap items-center gap-2 rounded-md border border-slate-800 bg-slate-900/40 px-3 py-2"
    >
      <span className="text-xs uppercase tracking-wider text-slate-500">
        {label}
      </span>
      {models.map((m) => {
        const checked = selection.selected.includes(m.id);
        const locked = checked && selection.isLocked(m.id);
        const cards =
          m.gpu_indices && m.gpu_indices.length > 0
            ? ` · GPU ${m.gpu_indices.join(", ")}`
            : "";
        return (
          <label
            key={m.id}
            data-testid="model-selector-option"
            data-model-id={m.id}
            data-checked={checked}
            data-locked={locked}
            title={
              locked
                ? "At least one model must stay selected."
                : `${m.served_model_name}${cards}`
            }
            className={[
              "inline-flex cursor-pointer items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs transition-colors",
              checked
                ? "border-emerald-700/60 bg-emerald-900/30 text-emerald-100"
                : "border-slate-700 bg-slate-900/60 text-slate-400 hover:text-slate-200",
              locked ? "cursor-not-allowed opacity-80" : "",
            ].join(" ")}
          >
            <input
              type="checkbox"
              className="h-3 w-3 accent-emerald-500 disabled:cursor-not-allowed"
              checked={checked}
              // The gate and the control, one condition. See the header note.
              disabled={locked}
              onChange={() => selection.toggle(m.id)}
              aria-label={m.served_model_name}
            />
            <span className="font-mono">{m.served_model_name}</span>
          </label>
        );
      })}
      <button
        type="button"
        data-testid="model-selector-all"
        onClick={selection.selectAll}
        // Disabled when it would do nothing — the same rule as the checkboxes,
        // applied to the shortcut.
        disabled={allSelected}
        className="rounded border border-slate-700 px-2 py-0.5 text-xs text-slate-400 transition-colors hover:text-slate-100 disabled:cursor-not-allowed disabled:opacity-40"
      >
        All
      </button>
    </div>
  );
}
