# GPU Checkbox Selector + Missing-GPU Handling + Freeform Input Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace comma-text GPU entry with checkboxes driven by the live GPU inventory, visibly flag GPUs missing from a saved config, block model-load fast when a configured GPU is absent, and fix the controlled-input bug that also breaks `extra_args`/env.

**Architecture:** A new presentational `GpuChecklist` component is the single source of truth for GPU-selection UX; the add-model modal is refactored onto it and a new `SettingField` `kind: "gpu-set"` wraps it for both settings pages. The three parse-on-change `SettingField` controls (`int-list`, `string-list`, `kv-map`) become small stateful sub-components that own raw text locally and only resync from props on canonical-form divergence. The backend `load_model` route gains a live-probe pre-flight returning a 422 `gpu_index_missing` envelope mirroring the existing fit-preview check.

**Tech Stack:** Next.js 15 / React 19 + TypeScript (frontend, Vitest + Testing Library); FastAPI + SQLite (backend, pytest). Everything runs in Docker — never invoke `npm`/`node`/`python`/`pytest` on the host.

**Spec:** `docs/superpowers/specs/2026-05-26-gpu-checkbox-selector-design.md` (issue podwarden/apps/vllm-warden#175)

---

## Conventions (read once before starting)

- **Run frontend tests** (Vitest) in Docker:
  ```bash
  docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v "$(pwd)":/work -w /work/frontend \
    node:20-alpine sh -c "npm ci --no-audit --no-fund && npx vitest run <TESTFILE>"
  ```
  (After the first `npm ci`, subsequent runs reuse `frontend/node_modules`; you may drop `npm ci` and run `npx vitest run <TESTFILE>` if deps are already installed in the worktree.)
- **Frontend typecheck:** `docker run --rm -v "$(pwd)":/work -w /work/frontend node:20-alpine sh -c "npm ci && npx tsc --noEmit"`
- **Run backend tests:** `make test-unit ARGS=…` is for the whole suite; for one file use:
  ```bash
  docker run --rm -v "$(pwd)":/app -w /app python:3.11-slim \
    sh -c "pip install -q -r requirements-dev.txt && pytest -v tests/unit/models/test_load_endpoint.py"
  ```
- **Backend lint:** `make lint` (pinned ruff).
- All file paths below are relative to the worktree root `/home/ip/projects/vllm-warden-worktrees/gpu-checkbox-selector`.

---

## File Structure

**Create:**
- `frontend/src/components/gpu/gpu-checklist.tsx` — shared `GpuChecklist` presentational component; exports the `GpuInfo` type (lifted here as the single definition).
- `frontend/tests/component/gpu-checklist.test.tsx` — Vitest tests for `GpuChecklist` (present rows, ghost rows, warning banner, sorted emit).

**Modify:**
- `frontend/src/components/settings/setting-field.tsx` — add `kind: "gpu-set"` union member + render branch; convert `int-list`/`string-list`/`kv-map` branches to stateful sub-components (`IntListInput`, `StringListTextarea`, `KvMapTextarea`).
- `frontend/src/components/models/add-model-modal.tsx` — replace the inline GPU `<ul>`/checkbox block with `<GpuChecklist>`; import `GpuInfo` from the new module instead of defining it locally.
- `frontend/src/components/settings/runtime-field.tsx` — `default_gpu_indices` switches from `kind="int-list"` to `kind="gpu-set"`; accept + forward an optional `gpus` prop.
- `frontend/src/components/settings/general-tab.tsx` — fetch `GET /api/system/gpus`, pass `gpus` into the `RuntimeField` for the Defaults section.
- `frontend/src/app/models/[id]/settings/page.tsx` — fetch `GET /api/system/gpus`, pass `gpus` into `SettingFieldFor`; `gpu_indices` switches to `kind="gpu-set"`.
- `app/models/routes_api.py` — add live-probe pre-flight to `load_model` after the allow-list subset check.

**Test (modify/extend):**
- `frontend/tests/component/setting-field.test.tsx` — NEW or extend: regression tests for the freeform typing fix + `gpu-set` behaviour. (No existing `setting-field.test.tsx`; create it.)
- `tests/unit/models/test_load_endpoint.py` — add `gpu_index_missing` pre-flight test.

---

## Task 1: `GpuChecklist` shared component + lifted `GpuInfo` type

**Files:**
- Create: `frontend/src/components/gpu/gpu-checklist.tsx`
- Test: `frontend/tests/component/gpu-checklist.test.tsx`

The component is presentational and prop-driven (no fetching). It renders one checkbox per present GPU, plus a "ghost row" for any `selected` index not present in `gpus`, plus a warning banner when ≥1 ghost row exists. `onChange` always emits a **sorted ascending** `number[]`.

- [ ] **Step 1: Write the failing test**

Create `frontend/tests/component/gpu-checklist.test.tsx`:

```tsx
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import { GpuChecklist, type GpuInfo } from "@/components/gpu/gpu-checklist";

afterEach(cleanup);

const GPUS: GpuInfo[] = [
  { index: 0, name: "NVIDIA RTX A4000", memory_total_mib: 16376, memory_used_mib: 1024, utilization_pct: 5 },
  { index: 1, name: "NVIDIA RTX A4000", memory_total_mib: 16376, memory_used_mib: 0, utilization_pct: 0 },
];

describe("GpuChecklist", () => {
  it("renders one checkbox per present GPU", () => {
    render(<GpuChecklist gpus={GPUS} selected={[0]} onChange={() => {}} />);
    expect(screen.getByLabelText(/#0/)).toBeChecked();
    expect(screen.getByLabelText(/#1/)).not.toBeChecked();
  });

  it("toggling a GPU emits a sorted number[]", () => {
    const onChange = vi.fn();
    render(<GpuChecklist gpus={GPUS} selected={[1]} onChange={onChange} />);
    fireEvent.click(screen.getByLabelText(/#0/));
    expect(onChange).toHaveBeenCalledWith([0, 1]);
  });

  it("renders a removable ghost row + warning banner for a missing configured index", () => {
    const onChange = vi.fn();
    render(<GpuChecklist gpus={GPUS} selected={[0, 5]} onChange={onChange} />);
    const ghost = screen.getByLabelText(/GPU 5 — not present/);
    expect(ghost).toBeChecked();
    expect(screen.getByRole("alert")).toHaveTextContent(/not present/i);
    fireEvent.click(ghost); // unchecking removes it
    expect(onChange).toHaveBeenCalledWith([0]);
  });

  it("shows an empty-state message when no GPUs are present and none selected", () => {
    render(<GpuChecklist gpus={[]} selected={[]} onChange={() => {}} />);
    expect(screen.getByText(/no gpus detected/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run the test, verify it fails**

Run:
```bash
docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v "$(pwd)":/work -w /work/frontend \
  node:20-alpine sh -c "npm ci --no-audit --no-fund && npx vitest run tests/component/gpu-checklist.test.tsx"
```
Expected: FAIL — `Cannot find module '@/components/gpu/gpu-checklist'`.

- [ ] **Step 3: Write the component**

Create `frontend/src/components/gpu/gpu-checklist.tsx`:

```tsx
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
}

function emit(onChange: (n: number[]) => void, set: Set<number>) {
  onChange(Array.from(set).sort((a, b) => a - b));
}

export function GpuChecklist({ gpus, selected, onChange, disabled = false }: GpuChecklistProps) {
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
      <p className="text-sm text-slate-500" data-testid="gpu-empty">
        No GPUs detected — saving will still validate against allowed_gpu_indices server-side.
      </p>
    );
  }

  return (
    <div className="space-y-1">
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
                checked
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
```

- [ ] **Step 4: Run the test, verify it passes**

Run:
```bash
docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v "$(pwd)":/work -w /work/frontend \
  node:20-alpine sh -c "npx vitest run tests/component/gpu-checklist.test.tsx"
```
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/gpu/gpu-checklist.tsx frontend/tests/component/gpu-checklist.test.tsx
git commit -m "feat(gpu): add shared GpuChecklist component (#175)"
```

---

## Task 2: Refactor add-model modal onto `GpuChecklist`

The modal already owns `selectedGpus: Set<number>` and a `toggleGpu` callback; we keep that internal state but render `GpuChecklist` instead of the inline `<ul>`. The modal's `GpuInfo` interface is deleted and imported from the new module (single definition).

**Files:**
- Modify: `frontend/src/components/models/add-model-modal.tsx`
- Test: `frontend/tests/component/add-model-modal.test.tsx` (existing — must stay green; it uses `data-testid="gpu-list"`, which `GpuChecklist` preserves).

- [ ] **Step 1: Confirm the existing modal test references the contract we preserve**

Run:
```bash
grep -n "gpu-list\|add-model-gpu-\|#0\|toggleGpu\|GpuInfo" frontend/tests/component/add-model-modal.test.tsx frontend/tests/component/add-model-modal-shards-warnings.test.tsx
```
Note which selectors the tests assert on. `GpuChecklist` keeps `data-testid="gpu-list"` and a `#<index>`-prefixed label. If a test queries the old per-checkbox id `add-model-gpu-<n>`, update that test in Step 4 to the new label query `getByLabelText(/#<n>/)` — a contract rename, not a behaviour change.

- [ ] **Step 2: Remove the local `GpuInfo` interface; import from the shared module**

In `frontend/src/components/models/add-model-modal.tsx`, delete the local interface (lines ~108-114):

```tsx
interface GpuInfo {
  index: number;
  name: string;
  memory_total_mib: number;
  memory_used_mib: number;
  utilization_pct: number;
}
```

Add to the import block near the top of the file:

```tsx
import { GpuChecklist, type GpuInfo } from "@/components/gpu/gpu-checklist";
```

- [ ] **Step 3: Replace the inline GPU `<ul>` block with `GpuChecklist`**

In `SelectFileStage` replace the entire GPUs `<div>` block (the one at ~lines 1123-1157, beginning `<p ...>GPUs</p>` through the closing `</div>` of that group) with:

```tsx
      <div>
        <p className="text-xs uppercase tracking-wide text-slate-500">GPUs</p>
        <div className="mt-1">
          <GpuChecklist
            gpus={gpus}
            selected={Array.from(selectedGpus).sort((a, b) => a - b)}
            onChange={(next) => onGpusChange(new Set(next))}
          />
        </div>
      </div>
```

Add `onGpusChange` to `SelectFileStageProps` (next to `onToggleGpu`) and to the destructured params:

```tsx
  onGpusChange: (s: Set<number>) => void;
```

Keep the existing `onToggleGpu` prop only if other code paths use it; otherwise remove both `onToggleGpu` and the `toggleGpu` plumbing in favour of `onGpusChange`. Wire the parent (the `<SelectFileStage ... />` usage at ~line 860):

```tsx
          selectedGpus={selectedGpus}
          onGpusChange={setSelectedGpus}
```

Remove the now-unused `toggleGpu` function (~line 602) and its `onToggleGpu={toggleGpu}` prop pass if nothing else references them. (Search first: `grep -n "toggleGpu\|onToggleGpu" frontend/src/components/models/add-model-modal.tsx`.)

- [ ] **Step 4: Update any modal test selectors broken by the contract rename**

If Step 1 found tests querying `add-model-gpu-<n>`, change them to `screen.getByLabelText(/#<n>/)`. Leave all behavioural assertions intact.

- [ ] **Step 5: Run modal + checklist tests, verify pass**

Run:
```bash
docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v "$(pwd)":/work -w /work/frontend \
  node:20-alpine sh -c "npx vitest run tests/component/add-model-modal.test.tsx tests/component/add-model-modal-shards-warnings.test.tsx tests/component/gpu-checklist.test.tsx"
```
Expected: PASS (all suites).

- [ ] **Step 6: Typecheck**

Run:
```bash
docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v "$(pwd)":/work -w /work/frontend \
  node:20-alpine sh -c "npx tsc --noEmit"
```
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/models/add-model-modal.tsx frontend/tests/component/
git commit -m "refactor(gpu): render add-model modal GPU picker via GpuChecklist (#175)"
```

---

## Task 3: `gpu-set` SettingField kind

Add a new discriminated-union member and render branch that wraps `GpuChecklist` in the standard label/hint/restart-badge chrome.

**Files:**
- Modify: `frontend/src/components/settings/setting-field.tsx`
- Test: `frontend/tests/component/setting-field.test.tsx` (create)

- [ ] **Step 1: Write the failing test**

Create `frontend/tests/component/setting-field.test.tsx`:

```tsx
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import { SettingField } from "@/components/settings/setting-field";
import type { GpuInfo } from "@/components/gpu/gpu-checklist";

afterEach(cleanup);

const hint = { label: "GPU indices", hint: "Which GPUs", restart: "model-reload" } as const;
const GPUS: GpuInfo[] = [
  { index: 0, name: "A4000", memory_total_mib: 16376, memory_used_mib: 0, utilization_pct: 0 },
  { index: 1, name: "A4000", memory_total_mib: 16376, memory_used_mib: 0, utilization_pct: 0 },
];

describe("SettingField gpu-set", () => {
  it("renders checkboxes and emits a sorted number[]", () => {
    const onChange = vi.fn();
    render(<SettingField kind="gpu-set" field={hint} value={[1]} gpus={GPUS} onChange={onChange} />);
    fireEvent.click(screen.getByLabelText(/#0/));
    expect(onChange).toHaveBeenCalledWith([0, 1]);
  });
});
```

- [ ] **Step 2: Run, verify fail**

Run:
```bash
docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v "$(pwd)":/work -w /work/frontend \
  node:20-alpine sh -c "npx vitest run tests/component/setting-field.test.tsx"
```
Expected: FAIL — TypeScript/runtime error, `kind="gpu-set"` not assignable.

- [ ] **Step 3: Add the union member, import, and render branch**

In `frontend/src/components/settings/setting-field.tsx`:

Add the import near the top (after the existing imports):

```tsx
import { GpuChecklist, type GpuInfo } from "@/components/gpu/gpu-checklist";
```

Add the props type (after `KvMapProps`, before `SelectProps`):

```tsx
// Checkbox-driven GPU index selector backed by the live inventory. Replaces
// the comma-text `int-list` for gpu_indices / default_gpu_indices: the
// caller passes the present `gpus` and a configured `value`; GpuChecklist
// surfaces any configured-but-absent index as a removable ghost row.
type GpuSetProps = CommonProps & {
  kind: "gpu-set";
  value: number[];
  onChange: (v: number[]) => void;
  gpus: GpuInfo[];
};
```

Add `GpuSetProps` to the `Props` union:

```tsx
type Props =
  | TextProps
  | NumberProps
  | BooleanProps
  | IntListProps
  | StringListProps
  | KvMapProps
  | GpuSetProps
  | SelectProps;
```

Add the render branch inside `renderControl`'s switch (e.g. after `case "kv-map":`):

```tsx
    case "gpu-set":
      return (
        <GpuChecklist
          gpus={props.gpus}
          selected={props.value}
          onChange={props.onChange}
          disabled={disabled}
        />
      );
```

(The `aria-describedby={hintId}` wiring used by inputs isn't applicable to the checkbox group; the hint paragraph above already provides context. No `inputId` is consumed by this branch — that's fine, `useId` tolerates unused ids.)

- [ ] **Step 4: Run, verify pass**

Run:
```bash
docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v "$(pwd)":/work -w /work/frontend \
  node:20-alpine sh -c "npx vitest run tests/component/setting-field.test.tsx"
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/settings/setting-field.tsx frontend/tests/component/setting-field.test.tsx
git commit -m "feat(settings): add gpu-set SettingField kind backed by GpuChecklist (#175)"
```

---

## Task 4: Fix the freeform controlled-input bug (`int-list`, `string-list`, `kv-map`)

Each parse-on-change branch becomes a small internal stateful component that owns the raw text and resyncs from props only on canonical-form divergence. This keeps `int-list` (still used by `runtime-field.tsx` for non-GPU keys) and the `extra_args`/env controls usable.

**Files:**
- Modify: `frontend/src/components/settings/setting-field.tsx`
- Test: `frontend/tests/component/setting-field.test.tsx` (extend)

- [ ] **Step 1: Write the failing regression tests**

Append to `frontend/tests/component/setting-field.test.tsx`:

```tsx
describe("SettingField freeform typing", () => {
  it("int-list keeps a trailing comma while typing a second value", () => {
    function Harness() {
      const [v, setV] = (require("react") as typeof import("react")).useState<number[]>([]);
      return <SettingField kind="int-list" field={{ label: "G", hint: "", restart: "none" }} value={v} onChange={setV} />;
    }
    render(<Harness />);
    const input = screen.getByRole("textbox") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "0," } });
    expect(input.value).toBe("0,"); // comma survives, not wiped to "0"
    fireEvent.change(input, { target: { value: "0,1" } });
    expect(input.value).toBe("0,1");
  });

  it("string-list keeps a trailing newline while starting a second line", () => {
    function Harness() {
      const [v, setV] = (require("react") as typeof import("react")).useState<string[]>([]);
      return <SettingField kind="string-list" field={{ label: "A", hint: "", restart: "none" }} value={v} onChange={setV} />;
    }
    render(<Harness />);
    const ta = screen.getByRole("textbox") as HTMLTextAreaElement;
    fireEvent.change(ta, { target: { value: "--a\n" } });
    expect(ta.value).toBe("--a\n");
    fireEvent.change(ta, { target: { value: "--a\n--b" } });
    expect(ta.value).toBe("--a\n--b");
  });

  it("kv-map keeps a freshly-pressed Enter before the second KEY=value", () => {
    function Harness() {
      const [v, setV] = (require("react") as typeof import("react")).useState<Record<string, string>>({});
      return <SettingField kind="kv-map" field={{ label: "E", hint: "", restart: "none" }} value={v} onChange={setV} />;
    }
    render(<Harness />);
    const ta = screen.getByRole("textbox") as HTMLTextAreaElement;
    fireEvent.change(ta, { target: { value: "A=1\n" } });
    expect(ta.value).toBe("A=1\n");
    fireEvent.change(ta, { target: { value: "A=1\nB=2" } });
    expect(ta.value).toBe("A=1\nB=2");
  });

  it("int-list resyncs when the value prop changes externally", () => {
    const { rerender } = render(
      <SettingField kind="int-list" field={{ label: "G", hint: "", restart: "none" }} value={[0]} onChange={() => {}} />,
    );
    expect((screen.getByRole("textbox") as HTMLInputElement).value).toBe("0");
    rerender(
      <SettingField kind="int-list" field={{ label: "G", hint: "", restart: "none" }} value={[2, 3]} onChange={() => {}} />,
    );
    expect((screen.getByRole("textbox") as HTMLInputElement).value).toBe("2,3");
  });
});
```

- [ ] **Step 2: Run, verify fail**

Run:
```bash
docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v "$(pwd)":/work -w /work/frontend \
  node:20-alpine sh -c "npx vitest run tests/component/setting-field.test.tsx"
```
Expected: FAIL — `input.value` is `"0"` not `"0,"` (the comma-wipe bug), etc.

- [ ] **Step 3: Add `useState`/`useEffect` to imports and write the three stateful sub-components**

In `setting-field.tsx`, change the React import:

```tsx
import { useEffect, useId, useState } from "react";
```

Add these three components above `export function SettingField` (after the `kvMapToText` helper):

```tsx
// ---------------------------------------------------------------------------
// Stateful freeform inputs.
//
// The bug these fix: a control whose displayed value is derived from the
// PARSED result (`value.join(",")`) wipes in-progress text on every
// keystroke — a trailing comma or blank newline never survives the next
// render, so you can never type a second entry. Each control below instead
// owns the raw text locally and emits the parsed value upward, resyncing
// from the prop only when the prop's CANONICAL form diverges from what the
// local text parses to (an external reset: model switch, Reset button).
// Typing "0," parses to [0] whose canonical "0" matches the prop, so no
// resync fires and the comma stays.
// ---------------------------------------------------------------------------

function IntListInput(props: {
  id: string;
  hintId: string;
  value: number[];
  disabled: boolean;
  onChange: (v: number[]) => void;
}) {
  const [text, setText] = useState(() => props.value.join(","));
  useEffect(() => {
    const canonicalProp = props.value.join(",");
    const canonicalLocal = parseIntList(text).join(",");
    if (canonicalProp !== canonicalLocal) setText(canonicalProp);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.value]);
  return (
    <Input
      id={props.id}
      aria-describedby={props.hintId}
      value={text}
      disabled={props.disabled}
      inputMode="numeric"
      placeholder="0,1"
      onChange={(e) => {
        setText(e.target.value);
        props.onChange(parseIntList(e.target.value));
      }}
    />
  );
}

function StringListTextarea(props: {
  id: string;
  hintId: string;
  value: string[];
  disabled: boolean;
  onChange: (v: string[]) => void;
}) {
  const [text, setText] = useState(() => props.value.join("\n"));
  useEffect(() => {
    const canonicalProp = props.value.join("\n");
    const canonicalLocal = parseStringList(text).join("\n");
    if (canonicalProp !== canonicalLocal) setText(canonicalProp);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.value]);
  return (
    <textarea
      id={props.id}
      aria-describedby={props.hintId}
      className="flex min-h-[5rem] w-full rounded-md border border-slate-600 bg-slate-900 px-3 py-2 font-mono text-sm text-slate-100 placeholder:text-slate-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-900 disabled:cursor-not-allowed disabled:opacity-50"
      value={text}
      disabled={props.disabled}
      rows={Math.max(3, text.split("\n").length + 1)}
      placeholder="--worker-use-ray"
      onChange={(e) => {
        setText(e.target.value);
        props.onChange(parseStringList(e.target.value));
      }}
    />
  );
}

function KvMapTextarea(props: {
  id: string;
  hintId: string;
  value: Record<string, string>;
  disabled: boolean;
  onChange: (v: Record<string, string>) => void;
}) {
  const [text, setText] = useState(() => kvMapToText(props.value));
  useEffect(() => {
    const canonicalProp = kvMapToText(props.value);
    const canonicalLocal = kvMapToText(parseKvMap(text));
    if (canonicalProp !== canonicalLocal) setText(canonicalProp);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.value]);
  return (
    <textarea
      id={props.id}
      aria-describedby={props.hintId}
      className="flex min-h-[5rem] w-full rounded-md border border-slate-600 bg-slate-900 px-3 py-2 font-mono text-sm text-slate-100 placeholder:text-slate-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-900 disabled:cursor-not-allowed disabled:opacity-50"
      value={text}
      disabled={props.disabled}
      rows={Math.max(3, text.split("\n").length + 1)}
      placeholder="HF_HUB_OFFLINE=1"
      onChange={(e) => {
        setText(e.target.value);
        props.onChange(parseKvMap(e.target.value));
      }}
    />
  );
}
```

- [ ] **Step 4: Replace the three render branches to use the sub-components**

In `renderControl`, replace the `case "int-list":` branch body:

```tsx
    case "int-list":
      return (
        <IntListInput
          id={inputId}
          hintId={hintId}
          value={props.value}
          disabled={disabled}
          onChange={props.onChange}
        />
      );
```

Replace `case "string-list":`:

```tsx
    case "string-list":
      return (
        <StringListTextarea
          id={inputId}
          hintId={hintId}
          value={props.value}
          disabled={disabled}
          onChange={props.onChange}
        />
      );
```

Replace `case "kv-map":`:

```tsx
    case "kv-map":
      return (
        <KvMapTextarea
          id={inputId}
          hintId={hintId}
          value={props.value}
          disabled={disabled}
          onChange={props.onChange}
        />
      );
```

- [ ] **Step 5: Run, verify pass**

Run:
```bash
docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v "$(pwd)":/work -w /work/frontend \
  node:20-alpine sh -c "npx vitest run tests/component/setting-field.test.tsx"
```
Expected: PASS (all `gpu-set` + freeform + resync tests).

- [ ] **Step 6: Run the broader settings/model-settings suites for regressions**

Run:
```bash
docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v "$(pwd)":/work -w /work/frontend \
  node:20-alpine sh -c "npx vitest run tests/component/model-settings.test.tsx tests/component/settings.test.tsx tests/component/settings-presets-suggest-argv.test.tsx tests/component/system-config-section.test.tsx"
```
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/settings/setting-field.tsx frontend/tests/component/setting-field.test.tsx
git commit -m "fix(settings): keep in-progress text in int-list/string-list/kv-map inputs (#175)"
```

---

## Task 5: Wire `default_gpu_indices` (Settings → General) to `gpu-set`

The general tab fetches the GPU inventory once and threads it through `RuntimeField` into the `gpu-set` field.

**Files:**
- Modify: `frontend/src/components/settings/runtime-field.tsx`
- Modify: `frontend/src/components/settings/general-tab.tsx`
- Test: `frontend/tests/component/settings.test.tsx` (existing — keep green; add a focused assertion if it covers `default_gpu_indices`).

- [ ] **Step 1: Add an optional `gpus` prop to `RuntimeField` and switch `default_gpu_indices` to `gpu-set`**

In `frontend/src/components/settings/runtime-field.tsx`:

Add the import:

```tsx
import type { GpuInfo } from "@/components/gpu/gpu-checklist";
```

Add `gpus` to `RuntimeFieldProps` (the interface at ~line 27) and to the destructured params:

```tsx
interface RuntimeFieldProps {
  fieldKey: RuntimeKey;
  hint: FieldHint;
  draft: /* existing type */;
  setField: /* existing type */;
  disabled: boolean;
  gpus?: GpuInfo[];
}
```

```tsx
export function RuntimeField({
  fieldKey,
  hint,
  draft,
  setField,
  disabled,
  gpus = [],
}: RuntimeFieldProps) {
```

Replace the `case "default_gpu_indices":` branch (~lines 72-81):

```tsx
    case "default_gpu_indices":
      return (
        <SettingField
          kind="gpu-set"
          field={hint}
          value={draft.default_gpu_indices}
          gpus={gpus}
          onChange={(v) => setField("default_gpu_indices", v)}
          disabled={disabled}
        />
      );
```

(Confirm the existing destructure block's exact `draft`/`setField` types by reading lines 27-41 first; only add the `gpus?` line, don't restate the others incorrectly.)

- [ ] **Step 2: Fetch GPUs in `general-tab.tsx` and pass to the Defaults `RuntimeField`**

In `frontend/src/components/settings/general-tab.tsx`:

Add imports:

```tsx
import useSWR from "swr";
import { authFetchJSON } from "@/lib/auth-fetch";
import type { GpuInfo } from "@/components/gpu/gpu-checklist";
```

Inside `GeneralTab()`, after `const disabled = ...`:

```tsx
  // GPU inventory for the default_gpu_indices checkbox picker. /api/system/gpus
  // has a 2 s server-side cache so this is cheap; an empty list (no NVIDIA /
  // probe error) just yields ghost rows for any configured index.
  const { data: gpuData } = useSWR<{ gpus: GpuInfo[] }>("/api/system/gpus", authFetchJSON);
  const gpus = gpuData?.gpus ?? [];
```

Pass `gpus` into the Defaults-section `RuntimeField` only (the `DEFAULTS_KEYS.map` block at ~lines 79-92):

```tsx
              <RuntimeField
                key={k}
                fieldKey={k}
                hint={hint}
                draft={draft}
                setField={setField}
                disabled={disabled}
                gpus={gpus}
              />
```

(The Identity and Hugging Face sections don't need `gpus`; leaving it off there is fine since the prop is optional.)

- [ ] **Step 3: Typecheck + run settings suite**

Run:
```bash
docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v "$(pwd)":/work -w /work/frontend \
  node:20-alpine sh -c "npx tsc --noEmit && npx vitest run tests/component/settings.test.tsx"
```
Expected: no TS errors; settings suite PASS. If `settings.test.tsx` renders `GeneralTab` and doesn't already mock `/api/system/gpus`, add a fetch stub returning `{ gpus: [] }` for that path (mirror how the suite stubs other `/api/...` calls — read the top of the file first). An empty list renders the empty-state message, so existing assertions about `default_gpu_indices` may need updating from a text input to the checkbox group.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/settings/runtime-field.tsx frontend/src/components/settings/general-tab.tsx frontend/tests/component/settings.test.tsx
git commit -m "feat(settings): default_gpu_indices uses GPU checkboxes (#175)"
```

---

## Task 6: Wire per-model `gpu_indices` (model settings page) to `gpu-set` + missing-GPU UI

**Files:**
- Modify: `frontend/src/app/models/[id]/settings/page.tsx`
- Test: `frontend/tests/component/model-settings.test.tsx` (existing — keep green; extend for the ghost row).

- [ ] **Step 1: Fetch GPUs in the page and thread into `SettingFieldFor`**

In `frontend/src/app/models/[id]/settings/page.tsx`:

Add the import:

```tsx
import { type GpuInfo } from "@/components/gpu/gpu-checklist";
```

In the page component (near the existing `useSWR<ModelSettings>` at ~line 212), add:

```tsx
  const { data: gpuData } = useSWR<{ gpus: GpuInfo[] }>("/api/system/gpus", authFetchJSON);
  const gpus = gpuData?.gpus ?? [];
```

Pass `gpus` into the `SettingFieldFor` usage (~line 550):

```tsx
                  <SettingFieldFor
                    key={k}
                    fieldKey={k}
                    hint={hint}
                    draft={draft}
                    gpus={gpus}
                    setDraft={(updater) =>
                      setDraft((d) => (d === null ? d : updater(d)))
                    }
                    disabled={allDisabled}
                  />
```

- [ ] **Step 2: Add `gpus` to `SettingFieldFor` and switch `gpu_indices` to `gpu-set`**

Update the `SettingFieldFor` signature (~line 1026) to accept `gpus`:

```tsx
function SettingFieldFor({
  fieldKey,
  hint,
  draft,
  setDraft,
  disabled,
  gpus,
}: {
  fieldKey: PatchableKey;
  hint: import("@/lib/settings-hints").FieldHint;
  draft: Draft;
  setDraft: (updater: (d: Draft) => Draft) => void;
  disabled: boolean;
  gpus: GpuInfo[];
}) {
```

Replace the `case "gpu_indices":` branch (~lines 1126-1135):

```tsx
    case "gpu_indices":
      return (
        <SettingField
          kind="gpu-set"
          field={hint}
          value={draft.gpu_indices}
          gpus={gpus}
          onChange={(v) => set("gpu_indices", v)}
          disabled={disabled}
        />
      );
```

- [ ] **Step 3: Extend the model-settings test for the ghost row**

Read `frontend/tests/component/model-settings.test.tsx` to see how it stubs `authFetchJSON` / `/api/models/<id>`. Add a `/api/system/gpus` stub and a test: seed the model snapshot with `gpu_indices: [0, 7]` and a probe returning only GPU 0, then assert the page shows `GPU 7 — not present` and the warning banner. Mirror the file's existing `syncResolved` / SWR harness. Example shape (adapt to the file's actual fetch mock):

```tsx
it("shows a ghost row for a configured GPU absent from the probe", async () => {
  // ...stub /api/models/<id> with gpu_indices: [0, 7]
  // ...stub /api/system/gpus -> { gpus: [{ index: 0, name: "A4000", memory_total_mib: 16376, memory_used_mib: 0, utilization_pct: 0 }] }
  renderPage("m1");
  expect(await screen.findByText(/GPU 7 — not present/)).toBeInTheDocument();
  expect(screen.getByRole("alert")).toHaveTextContent(/not present/i);
});
```

- [ ] **Step 4: Typecheck + run model-settings suite**

Run:
```bash
docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v "$(pwd)":/work -w /work/frontend \
  node:20-alpine sh -c "npx tsc --noEmit && npx vitest run tests/component/model-settings.test.tsx"
```
Expected: no TS errors; suite PASS (including the new ghost-row test). If the existing suite asserted a comma text input for `gpu_indices`, update those assertions to the checkbox group.

- [ ] **Step 5: Commit**

```bash
git add "frontend/src/app/models/[id]/settings/page.tsx" frontend/tests/component/model-settings.test.tsx
git commit -m "feat(settings): per-model gpu_indices uses GPU checkboxes + missing-GPU ghost rows (#175)"
```

---

## Task 7: Backend load pre-flight — 422 `gpu_index_missing`

After the allow-list subset check and before `update_status(model_id, "loading")`, probe the live GPU cache and 422 if any configured index is absent. Mirrors the fit-preview envelope.

**Files:**
- Modify: `app/models/routes_api.py` (`load_model`, ~lines 982-983)
- Test: `tests/unit/models/test_load_endpoint.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/models/test_load_endpoint.py`. Reuse the fit-preview fake-probe shape (a duck-typed cache on `app.state.gpu_probe_cache` with an async `get()` returning an object exposing `.gpus` (each with `.index`, `.memory_total_mib`) and `.probe_error`):

```python
from dataclasses import dataclass, field


@dataclass
class _FakeGpuLive:
    index: int
    memory_total_mib: int = 16376


@dataclass
class _FakeSnap:
    gpus: list = field(default_factory=list)
    apps: list = field(default_factory=list)
    probe_error: str | None = None


class _FakeProbeCache:
    def __init__(self, snap):
        self._snap = snap

    async def get(self):
        return self._snap


def test_load_422_when_configured_gpu_absent_from_probe(tmp_data_dir, client):
    """A model whose gpu_indices is allow-listed but physically absent must
    422 gpu_index_missing before the row flips to 'loading'."""
    client.get("/healthz")
    _seed_done_with_pulled_model(
        tmp_data_dir / "vllm-warden.db", allowed=[0, 1, 2, 3], gpus=[0, 2],
        model_id="ghost-gpu",
    )
    # Probe sees only GPU 0 — index 2 is gone (card pulled / re-indexed).
    client.app.state.gpu_probe_cache = _FakeProbeCache(_FakeSnap(gpus=[_FakeGpuLive(index=0)]))
    auth = _jwt_login(client)

    r = client.post("/api/models/ghost-gpu/load", headers={**auth, **csrf_header(client)})
    assert r.status_code == 422
    body = r.json()
    assert body["detail"]["error_code"] == "gpu_index_missing"
    assert body["detail"]["available"] == [0]
    assert 2 in body["detail"]["message"] or "[2]" in body["detail"]["message"]
    # Row must NOT have advanced to loading.
    import sqlite3
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        status = db.execute("SELECT status FROM models WHERE id='ghost-gpu'").fetchone()[0]
    assert status == "pulled"


def test_load_passes_preflight_when_all_gpus_present(tmp_data_dir, client):
    """Allow-listed AND present gpu_indices passes the probe pre-flight (202)."""
    client.get("/healthz")
    _seed_done_with_pulled_model(
        tmp_data_dir / "vllm-warden.db", allowed=[0, 1, 2, 3], gpus=[0, 1],
        model_id="present-gpu",
    )
    client.app.state.gpu_probe_cache = _FakeProbeCache(
        _FakeSnap(gpus=[_FakeGpuLive(index=0), _FakeGpuLive(index=1)])
    )
    auth = _jwt_login(client)

    sup_load = AsyncMock()
    health = AsyncMock(return_value=True)
    with patch("app.runtime.supervisor.Supervisor.load", new=sup_load), \
         patch("app.models.routes_api.wait_for_health", new=health):
        r = client.post("/api/models/present-gpu/load", headers={**auth, **csrf_header(client)})
    assert r.status_code == 202
```

- [ ] **Step 2: Run, verify fail**

Run:
```bash
docker run --rm -v "$(pwd)":/app -w /app python:3.11-slim \
  sh -c "pip install -q -r requirements-dev.txt && pytest -v tests/unit/models/test_load_endpoint.py -k 'preflight or absent or present'"
```
Expected: `test_load_422_when_configured_gpu_absent_from_probe` FAILS (returns 202, not 422). `test_load_passes_preflight_when_all_gpus_present` may already pass (no probe gate yet) — that's fine.

- [ ] **Step 3: Add the pre-flight to `load_model`**

In `app/models/routes_api.py`, inside `load_model`, replace the allow-list block (currently ending at `await ModelRepo(db).update_status(model_id, "loading")`) so the probe check runs **after** the subset check and **before** `update_status`:

```python
        allowed = (await SetupRepo(db).get()).draft.get("allowed_gpu_indices", [])
        if not set(model.gpu_indices).issubset(set(allowed)):
            raise HTTPException(
                422, f"gpu_indices {model.gpu_indices} not subset of allowed {allowed}"
            )

        # Physical-presence pre-flight: a model can be allow-listed yet point
        # at a GPU index that has since vanished (card pulled, driver
        # re-index). Hand vLLM a bad CUDA_VISIBLE_DEVICES and it crashes
        # opaquely; fail fast with the same envelope the fit-preview check
        # uses so the frontend can reuse its handling. Reuse the shared probe
        # cache (same one /api/system/gpus and fit-preview read).
        cache_gpu = getattr(request.app.state, "gpu_probe_cache", None)
        if cache_gpu is None:
            from app.system.routes_gpus import _ProbeCache  # noqa: PLC0415
            cache_gpu = _ProbeCache()
            request.app.state.gpu_probe_cache = cache_gpu
        snap = await cache_gpu.get()
        present = {g.index for g in snap.gpus}
        missing = [i for i in model.gpu_indices if i not in present]
        if missing:
            raise HTTPException(
                422,
                detail={
                    "error_code": "gpu_index_missing",
                    "message": f"gpu_indices {missing} not present in nvidia-smi probe",
                    "available": sorted(present),
                    "probe_error": snap.probe_error,
                },
            )

        await ModelRepo(db).update_status(model_id, "loading")
```

Note: the existing tests `test_load_calls_supervisor_then_health_check`, `test_load_runs_warmup_probe_before_flipping_to_loaded`, etc. do **not** install a fake probe cache, so `cache_gpu` will be a real `_ProbeCache` shelling `nvidia-smi`, which on the CI box returns `gpus: []` with a `probe_error`. That would make `present` empty and 422 those tests. **To keep them green, install a permissive fake probe in those tests** OR have the pre-flight treat a probe error as "skip presence check". Decision: **skip the presence check when `snap.probe_error` is set** (no probe data ⇒ can't prove absence; the allow-list already gated, and load will surface the real failure). Add immediately after computing `snap`:

```python
        # When nvidia-smi itself failed we have no ground truth to check
        # against — don't block load on an absent probe (the allow-list
        # already gated, and a genuinely-missing GPU will fail at spawn).
        present = {g.index for g in snap.gpus}
        if snap.probe_error is None:
            missing = [i for i in model.gpu_indices if i not in present]
            if missing:
                raise HTTPException(
                    422,
                    detail={
                        "error_code": "gpu_index_missing",
                        "message": f"gpu_indices {missing} not present in nvidia-smi probe",
                        "available": sorted(present),
                        "probe_error": snap.probe_error,
                    },
                )
```

(Use this `probe_error`-guarded version; drop the unguarded block above. The new tests set `probe_error=None`, so they exercise the gate; the legacy tests hit a real probe with `probe_error="nvidia-smi unavailable"` and skip it.)

- [ ] **Step 4: Run the new + existing load tests, verify pass**

Run:
```bash
docker run --rm -v "$(pwd)":/app -w /app python:3.11-slim \
  sh -c "pip install -q -r requirements-dev.txt && pytest -v tests/unit/models/test_load_endpoint.py"
```
Expected: all PASS (new gate tests + all pre-existing load tests; the quarantined `test_on_exit_...` stays skipped).

- [ ] **Step 5: Lint**

Run: `make lint`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add app/models/routes_api.py tests/unit/models/test_load_endpoint.py
git commit -m "feat(load): 422 gpu_index_missing when a configured GPU is absent from the probe (#175)"
```

---

## Task 8: Full-suite verification + changelog

**Files:**
- Modify: `changelog.md` (prepend a 2026-05-26 entry; Keep a Changelog format)

- [ ] **Step 1: Run the full frontend suite**

Run:
```bash
docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v "$(pwd)":/work -w /work/frontend \
  node:20-alpine sh -c "npx vitest run && npx tsc --noEmit"
```
Expected: all suites PASS, no TS errors. Fix any regressions before continuing.

- [ ] **Step 2: Run the full backend unit suite**

Run: `make test-unit`
Expected: all PASS (modulo pre-existing skips).

- [ ] **Step 3: Update the changelog**

Read `changelog.md`; prepend under a `## [Unreleased]` / `2026-05-26` heading:

```markdown
### Added
- GPU selection in Settings (default GPU indices) and per-model settings is now
  checkbox-driven from the live system GPU inventory instead of a comma-text
  field. Configured GPUs that are no longer present render as a removable amber
  "not present" row with a warning banner. (#175)
- Model load now fails fast with `422 gpu_index_missing` when a configured GPU
  index is absent from the live nvidia-smi probe. (#175)

### Fixed
- Settings inputs for GPU indices, extra args, and extra env no longer wipe
  in-progress text (trailing comma / newline), so more than one value can be
  entered. (#175)
```

- [ ] **Step 4: Commit**

```bash
git add changelog.md
git commit -m "docs: changelog for GPU checkbox selector + freeform input fix (#175)"
```

---

## Task 9: Code review, UI verification, MR, issue comment

- [ ] **Step 1: Request code review** (superpowers:requesting-code-review)

```bash
BASE_SHA=$(git merge-base origin/develop HEAD)
HEAD_SHA=$(git rev-parse HEAD)
```
Dispatch the code-reviewer subagent with DESCRIPTION = this feature, PLAN_OR_REQUIREMENTS = the spec path, BASE_SHA, HEAD_SHA. Fix Critical/Important findings before proceeding.

- [ ] **Step 2: UI verification THROUGH the UI** (NOT a backend probe)

Per the standing constraint, drive the actual UI with Chrome DevTools MCP (creds admin / lollipop) on the running instance: open Settings → confirm Default GPU indices is checkboxes and that typing in extra-args keeps a trailing newline; open a model's settings → confirm gpu_indices checkboxes; if a model has a configured-but-absent index, confirm the ghost row + banner. Capture a screenshot as evidence. (If no live GPU instance is reachable, note that and rely on the component tests + a local `make docker-run` smoke; do not substitute a backend probe for the UI assertion.)

- [ ] **Step 3: Push the branch and open the MR to `develop`**

```bash
git push -u origin feat/gpu-checkbox-selector
```
Open an MR targeting `develop` (vllm-warden's default). Body: summary, "Closes #175", the change-type checklist, QA evidence (component test counts + UI screenshot), and a note that `int-list` is retained (used elsewhere) and got the typing fix too. Do **not** merge to main.

- [ ] **Step 4: Comment on issue #175**

Post a dev summary on podwarden/apps/vllm-warden#175: root cause (parse-on-change controlled-input), the checkbox redesign, missing-GPU ghost-row + 422 decisions, MR link.

---

## Self-Review (completed during planning)

**1. Spec coverage:**
- GpuChecklist component → Task 1 ✓
- gpu-set SettingField kind → Task 3 ✓
- Settings pages fetch inventory (general + per-model) → Tasks 5, 6 ✓
- Backend load pre-flight 422 → Task 7 ✓
- Missing-GPU ghost row + warning banner → Task 1 (component) + Task 6 (per-model wiring) ✓
- Freeform input fix (int-list/string-list/kv-map, local text + canonical resync) → Task 4 ✓
- Edge case "no GPUs detected / probe error" → GpuChecklist empty-state (Task 1) + load pre-flight skips on `probe_error` (Task 7) ✓
- Edge case "zero GPUs selected": `default_gpu_indices` may be empty (allowed — checkbox group with none ticked); per-model `gpu_indices` ≥1 enforcement — **NOTE:** the spec asks per-model Save be disabled when empty. Existing dirty-tracking + the backend allow-list/`gt` validators already reject an empty load; explicit Save-disable is **out of this plan's minimal scope** and flagged in the MR as a follow-up if the user wants the client-side guard. (Surfaced here rather than silently dropped.)
- Testing (Vitest component + pytest load 422) → Tasks 1,3,4,5,6 (frontend), Task 7 (backend) ✓
- Out of scope (multi-node, allow-list semantics, auto-drop) → respected ✓

**2. Placeholder scan:** No TBD/TODO/"handle edge cases" — every code step has concrete code. One deliberate adaptation point: existing settings/model-settings test suites may assert the old text-input contract; steps instruct reading the file first and updating assertions, with the exact new query shown.

**3. Type consistency:** `GpuInfo` defined once in `gpu-checklist.tsx`, imported everywhere (modal, setting-field, runtime-field, both pages). `GpuChecklist` props `{ gpus, selected, onChange, disabled }` consistent across Tasks 1/2/3. `gpu-set` props `{ kind, value, onChange, gpus }` consistent in Task 3 (definition) and Tasks 5/6 (usage). Backend envelope keys (`error_code`, `message`, `available`, `probe_error`) match the fit-preview envelope and the Task 7 test assertions.
