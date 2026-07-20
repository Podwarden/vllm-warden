# GPU checkbox selector + missing-GPU handling + freeform input fix

**Issue:** podwarden/apps/vllm-warden#175
**Date:** 2026-05-26

## Problem

The **Settings → Default GPU indices** field and the per-model settings **GPU indices**
field are comma-separated text inputs that reject commas. The control
(`SettingField` `kind: "int-list"`) re-derives its displayed value from the
*parsed* integer array on every keystroke, so a trailing `,` (and any
in-progress text) is wiped on the next render. You can never enter more than
one GPU.

The same root cause breaks the other parse-on-change controls:

- `string-list` (`extra_args`, one arg per line) — `parseStringList` drops empty
  lines, so pressing Enter to start a new entry is wiped.
- `kv-map` (env, `KEY=value` per line) — `parseKvMap` drops lines without `=`,
  so the same Enter-to-new-line is wiped.

Separately, a model can be saved with a GPU index that later disappears from the
system (card removed, driver re-index). Today the load route only checks
`gpu_indices ⊆ allowed_gpu_indices` (the operator allow-list), not physical
presence, so vLLM is handed a bad `CUDA_VISIBLE_DEVICES` and crashes opaquely.

## Goal

Replace comma-text GPU entry with checkboxes driven by the live system GPU
inventory; visibly flag and allow repair of GPUs missing from a saved config;
fail model-load fast with a clear message when a configured GPU is absent; and
fix the controlled-input bug that also breaks `extra_args`/env.

## Architecture & components

### Frontend

- **`GpuChecklist`** (`frontend/src/components/gpu/gpu-checklist.tsx`) — new
  shared presentational component.
  Props: `gpus: GpuInfo[]`, `selected: number[]`, `onChange(next: number[])`,
  `disabled?: boolean`.
  Renders one checkbox per present GPU showing `index · name · free VRAM`.
  Selection is order-independent; `onChange` emits a sorted ascending
  `number[]`. The add-model modal's inline GPU checkbox logic is refactored to
  render this component (single source of truth for the GPU-selection UX).
  - `GpuInfo` shape (already defined in the modal, to be lifted to a shared
    location): `{ index: number; name: string; memory_total_mib: number;
    memory_used_mib: number; utilization_pct: number }`.

- **New `SettingField` `kind: "gpu-set"`**
  (`frontend/src/components/settings/setting-field.tsx`).
  Props union member: `{ kind: "gpu-set"; value: number[];
  onChange(v: number[]): void; gpus: GpuInfo[] }`. Renders `GpuChecklist`
  inside the standard label/hint/restart-badge chrome. Replaces the `int-list`
  usage for `default_gpu_indices` (Settings page) and `gpu_indices`
  (per-model settings page).

- **Settings pages fetch the GPU inventory.**
  - `frontend/src/app/settings/page.tsx` (general tab path) and
    `frontend/src/components/settings/general-tab.tsx`: fetch
    `GET /api/system/gpus` once, pass the list into the `gpu-set` field for
    `default_gpu_indices`.
  - `frontend/src/app/models/[id]/settings/page.tsx`: same fetch, pass into the
    `gpu-set` field for `gpu_indices`; compute the missing-GPU warning.
  The endpoint already has a 2 s server-side cache, so re-fetch is cheap.

### Backend

- **Load pre-flight** in `POST /{model_id}/load`
  (`app/models/routes_api.py`, after the existing allow-list subset check and
  before `update_status(model_id, "loading")`): obtain the shared
  `gpu_probe_cache` snapshot (the same cache used by `/api/system/gpus` and the
  fit-preview check), and if any of `model.gpu_indices` is not present in the
  probe, raise:
  ```
  HTTPException(422, detail={
      "error_code": "gpu_index_missing",
      "message": f"gpu_indices {missing} not present in nvidia-smi probe",
      "available": sorted(present_indices),
      "probe_error": snap.probe_error,
  })
  ```
  This mirrors the existing fit-preview envelope so the frontend can reuse its
  handling.

## Missing-GPU handling (config UI)

`GpuChecklist` is given the configured `selected` indices and the present
`gpus`. For any index in `selected` that is **not** in `gpus`, it renders a
distinct **ghost row**: checkbox checked, amber styling, label
`GPU N — not present`. Unchecking removes it from `selected` (so the operator
can repair the config without hand-editing); configured indices are never
silently dropped. When ≥1 configured index is missing, a one-line amber warning
banner is shown above the group.

## Freeform controlled-input fix

The `int-list`, `string-list`, and `kv-map` render branches in
`setting-field.tsx` are each converted to a small internal stateful component
(`IntListInput`, `StringListTextarea`, `KvMapTextarea`) that:

1. Owns local text state: `const [text, setText] = useState(() => toText(value))`.
2. On input: `setText(raw); onChange(parse(raw))` — emits the parsed typed value
   upward but keeps the raw text locally so commas/newlines/partial entries
   survive.
3. Resyncs from the prop only when the **canonical parsed form** of `value`
   differs from the canonical form of the current local text, e.g. for int-list
   `if (value.join(",") !== parseIntList(text).join(",")) setText(value.join(","))`
   in a `useEffect([value])`. This adopts external resets (model switch, reset
   button) without clobbering in-progress typing (typing `"0,"` parses to `[0]`
   whose canonical `"0"` matches the prop, so no resync fires).

`gpu_indices`/`default_gpu_indices` move to the checkbox `gpu-set` kind, but
`int-list` remains (still used by `runtime-field.tsx`) and gets this fix too.

## Edge cases

- **No GPUs detected / probe error:** `GpuChecklist` shows "no GPUs detected" and
  keeps every configured index as a ghost row (config not lost). Save is allowed;
  the load pre-flight will 422 with `gpu_index_missing`.
- **Zero GPUs selected:** per-model `gpu_indices` requires ≥1 selection (Save
  disabled / validation message when empty). `default_gpu_indices` may be empty
  (= no preset pre-selected when adding a model).

## Testing

- **Frontend (Vitest + Testing Library, `frontend/tests/component/`):**
  - `gpu-set` renders one checkbox per stub GPU; toggling emits a sorted
    `number[]`.
  - A configured index absent from the stub gpu list renders the ghost row and
    is removable; the warning banner appears.
  - Regression: `int-list` accepts a typed `"0,1"` (comma survives), `string-list`
    accepts a trailing newline + second line, `kv-map` accepts Enter then a second
    `KEY=value`.
- **Backend (pytest, `tests/`):** `load_model` returns 422 `gpu_index_missing`
  when a configured index is absent from a stubbed probe, reusing the
  fit-preview probe-stub pattern. Existing allow-list 422 still passes when the
  index is present in the allow-list but absent from the probe (probe check runs
  after the allow-list check).

## Out of scope

- Multi-node / remote GPU inventory (only local `nvidia-smi` probe).
- Changing the `allowed_gpu_indices` allow-list semantics.
- Auto-repairing or auto-dropping missing GPUs (explicitly rejected: warn + block).
