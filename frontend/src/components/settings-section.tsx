"use client";

import { useId, useState, type ReactNode } from "react";

/**
 * A collapsible section with a thin top border + uppercase title (S4 design
 * principle: instrument-panel aesthetic — discrete sections of related knobs
 * rather than one flat list).
 *
 * Each section ships with:
 *   - A bold uppercase title (left).
 *   - An optional subtitle (right) — used for the per-section restart-warning
 *     summary on the model settings page.
 *   - A children block that can be hidden via the disclosure triangle.
 *
 * Native `<details>` would handle keyboard a11y for free but doesn't give us
 * a slot for the right-aligned subtitle, so we drive expansion ourselves with
 * `aria-expanded` + a button summary.
 */
export interface SettingsSectionProps {
  title: string;
  subtitle?: ReactNode;
  /** Initial expansion state. Default true — collapsing is opt-in. */
  defaultOpen?: boolean;
  /** Stable test id; defaults to the title slugified. */
  testId?: string;
  children: ReactNode;
}

export function SettingsSection({
  title,
  subtitle,
  defaultOpen = true,
  testId,
  children,
}: SettingsSectionProps) {
  const [open, setOpen] = useState(defaultOpen);
  const bodyId = useId();
  const id = testId ?? `settings-section-${slugify(title)}`;
  return (
    <section
      data-testid={id}
      className="border-t border-slate-700/60 pt-4 first:border-t-0 first:pt-0"
    >
      <div className="flex w-full items-baseline justify-between gap-3">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          aria-controls={bodyId}
          aria-label={title}
          data-testid={`${id}-toggle`}
          className="flex items-center gap-2 text-left focus:outline-none focus:ring-1 focus:ring-amber-600 rounded-sm"
        >
          <span
            aria-hidden
            className={
              "inline-block w-3 text-[10px] text-slate-500 transition-transform " +
              (open ? "rotate-90" : "")
            }
          >
            ▶
          </span>
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            {title}
          </span>
        </button>
        {subtitle && (
          // Subtitle lives OUTSIDE the toggle button so its content does NOT
          // become part of the button's accessible name. Crucial because the
          // model-settings page renders a "1 unsaved" subtitle that would
          // otherwise match `getByRole('button', { name: /save/i })` and
          // collide with the page-level Save action button.
          <span
            className="text-[11px] text-slate-500"
            data-testid={`${id}-subtitle`}
          >
            {subtitle}
          </span>
        )}
      </div>
      <div
        id={bodyId}
        data-testid={`${id}-body`}
        hidden={!open}
        className="mt-3 space-y-4"
      >
        {children}
      </div>
    </section>
  );
}

function slugify(s: string): string {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "");
}
