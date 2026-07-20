# vllm-warden /settings Page Redesign

> Spec doc for issue **#154** ("Settings: full /settings page UI/UX redesign").
> Locked at MR open time; subsequent refinements should land as follow-up
> specs, not in-place edits.
>
> **Phase 1 of 2.** This MR ships the spec only — no code. A separate
> implementation pass will land the refactor + new field once the spec is
> approved.
>
> **Issue-number note:** the dispatch prompt referenced "#71" by ID, but
> #71 is the closed bench-truncation issue. The matching live issue by
> title (and by every field in scope) is **#154**, which explicitly says
> "Subsumes #151" — same Public-URL field the dispatch calls out. We
> proceed against #154. The dev-2 branch name (`dev-2/71-settings-redesign-spec`)
> retains the dispatched ID for traceability.

## Goal
Make `/settings` discoverable. Today every runtime knob is rendered in a
single 13-field vertical list under a generic "Runtime" tab; operators
land on the page looking for the HF token, the public landing-page
toggle, or a session TTL, and have to scroll past every unrelated knob
to find it. The redesign groups settings by **operator purpose** —
identity, networking, sessions, maintenance — and folds in two new /
recently-added fields (`public_url` from #151, `landing_page_enabled`
from !158) without bolting them onto the end of a flat list.

## Problem statement (what's wrong with current /settings)

1. **Flat list, no IA.** All 13 runtime fields render in `RUNTIME_HINTS`
   declaration order under one "Configuration" card. There is no visual
   grouping — `hf_token` sits next to `session_refresh_ttl_days` for no
   reason except authoring order.
2. **Mixed concerns under "Runtime".** The tab labelled "Runtime"
   contains identity (`admin_username`, `admin_password`), Hugging Face
   auth (`hf_token`, `hf_cache_dir`), default-for-new-models hints
   (`default_gpu_indices`), browser-session knobs (`session_*_ttl`),
   token-lifecycle defaults (`default_token_expiration_days`,
   `rotation_grace_hours`), and operational tunables (`vllm_version`,
   `log_retention_lines`, `landing_page_enabled`). "Runtime" describes
   the backend route, not the operator's mental model.
3. **No surface for `public_url` (#151)** yet — needed for
   reverse-proxy deployments so every "Try this curl" / OpenAI-config
   snippet shows the right base URL. The redesign delivers it.
4. **`landing_page_enabled` was placed in Runtime as a stop-gap** by
   !158 (#155 unified-port). The spec for #155 explicitly defers deeper
   UX to this issue. It belongs with the other public-surface knob
   (`public_url`), not next to TTL fields.
5. **No future-proofing.** #154 AC #5 calls for "~5–10 more settings to
   be added later without re-architecting" — a flat list grows to 25
   fields and breaks down completely.

## Design decisions (locked)

1. **Tab structure: five tabs by operator purpose** — *General* ·
   *Networking* · *Sessions & Tokens* · *Maintenance* · *Model*.
   Rationale: matches the four mental clusters identified above plus
   the existing *Model* navigation pivot. Five tabs is the upper bound
   of "scannable without overflow" on a 1280-wide viewport using the
   existing `Tabs` primitive; verified visually against the current
   nav-bar item count.

2. **Within each tab, content is a vertical stack of titled `Card`
   sections.** Each section has a plain-English title + one-sentence
   subtitle. Sections never span tabs. Rationale: tabs split by
   purpose, sections split by *sub-purpose*; together they give two
   levels of hierarchy without introducing a left-rail nav (which
   would need a new primitive and break mobile).

3. **Save behaviour: explicit Edit / Save / Cancel — kept.** #154 AC
   says "save on blur OR explicit Save — pick one and be consistent."
   We pick **explicit Save** because Runtime contains restart-affecting
   fields (`session_*_ttl`, `vllm_version`) where a stray focus-out
   would silently restart-flag the warden, and because the existing
   dirty-tracking + secret-sentinel logic in `runtime-tab.tsx` already
   assumes a snapshot vs. draft model. Save scope: per-tab (each tab
   has its own Edit/Save). Cross-tab edits are intentionally not
   merged into one Save because that would force a global draft and
   make "reset just this tab" hard.

4. **`public_url` placement:** Networking → Public access section,
   adjacent to `landing_page_enabled`. Single coherent story for any
   operator who reverse-proxied the deployment.

5. **`landing_page_enabled` placement:** moves from Runtime →
   Networking → Public access. Stops being orphaned beside TTL knobs.

6. **No new backend endpoints.** The redesign is a frontend IA change.
   `GET/PATCH /api/settings/runtime` continues to be the single
   read/write surface; the new `public_url` field is added to the
   existing `RUNTIME_KEYS` map + a new coercer; one migration seeds
   no default (absent = use `window.location.origin`).

7. **Restart-impact visibility unchanged at field level.** The
   existing per-field `Badge` ("requires model-reload" /
   "requires warden-restart") stays. We additionally render a
   section-level header chip "Some fields here require warden restart"
   when the section contains ≥1 such field, so the operator sees the
   warning before scrolling through hints.

8. **Frontend reorg of code.** `runtime-tab.tsx` is split into four
   per-tab components (`general-tab.tsx`, `networking-tab.tsx`,
   `sessions-tab.tsx`, `maintenance-tab.tsx`); model-tab unchanged.
   The shared scaffolding (SWR fetch, draft, dirty-tracking,
   PATCH-with-restart-banner) is extracted into a
   `useRuntimeSettings(tabKeys)` hook so each tab reads/writes only
   the slice it owns while still PATCHing the same KV-backed route.

9. **`getPublicBaseUrl()` helper.** New `frontend/src/lib/public-url.ts`
   exports a single function `getPublicBaseUrl(): string` that returns
   `settings.public_url || window.location.origin` (no trailing
   slash). Every existing call-site in `frontend/src/**` that reads
   `window.location.origin` for a user-facing snippet/clipboard
   payload is migrated to call this helper. Sites that legitimately
   need the browsing user's origin (CSRF same-origin checks, in-app
   navigation) are NOT migrated. Implementation pass enumerates the
   migrated call-sites.

10. **No DB schema upheaval.** The settings KV table stays a flat
    string-keyed bag. The "sections" are a frontend-only construct.
    Rationale: a sections column / per-section table would couple the
    DB to a UI choice, and every section is just a label anyway.

## Tab + section structure (canonical)

### Tab 1 — General

For day-to-day operator identity + Hugging Face + new-model defaults.

| Section | Subtitle | Fields |
|---|---|---|
| Identity | The single admin account used to log in. | `admin_username`, `admin_password` |
| Hugging Face | Credentials and cache path used when pulling weights. | `hf_token`, `hf_cache_dir` |
| Defaults for new models | Pre-fills the Add Model modal. Does not affect existing models. | `default_gpu_indices` |

### Tab 2 — Networking

Public-facing surfaces of the warden. Everything an operator needs when
the deployment is reachable from outside the host.

| Section | Subtitle | Fields |
|---|---|---|
| Public access | How clients outside the host see this warden. | `public_url` (NEW), `landing_page_enabled` |

(Single section today; tab exists as a home for the imminent expansion
called out in #154 AC #5 — TLS cert reload toggles, CORS allowlists,
trusted proxy ranges, etc. — without needing another IA shuffle.)

### Tab 3 — Sessions & Tokens

Browser auth lifetimes + API-token default policy. Two distinct audiences
(humans logging in vs. machine tokens) but the same operator concern
("how long do credentials live"), so one tab.

| Section | Subtitle | Fields |
|---|---|---|
| Browser session | How long a logged-in browser stays authenticated before re-login. | `session_access_ttl_minutes`, `session_refresh_ttl_days` |
| Token defaults | Pre-fills the Token Create dialog. Existing tokens are unaffected. | `default_token_expiration_days`, `rotation_grace_hours` |
| Streaming | One-shot tickets used to authenticate SSE connections. | `sse_ticket_ttl_seconds` |

### Tab 4 — Maintenance

Operational tunables that change rarely but matter when they do.

| Section | Subtitle | Fields |
|---|---|---|
| vLLM runtime | Version of the vLLM Python package baked into the image. | `vllm_version` |
| Logs | Per-model log buffer retention. | `log_retention_lines` |

### Tab 5 — Model

Unchanged from today. Navigation pivot to the per-model editor at
`/models/<id>/settings`. Reusing the existing component verbatim.

## Field-by-field placement (full table)

| Field | Current placement | New placement | Restart kind |
|---|---|---|---|
| `admin_username` | Runtime tab | General → Identity | none |
| `admin_password` | Runtime tab | General → Identity | none |
| `hf_token` | Runtime tab | General → Hugging Face | model-reload |
| `hf_cache_dir` | Runtime tab | General → Hugging Face | model-reload |
| `default_gpu_indices` | Runtime tab | General → Defaults for new models | none |
| **`public_url`** | _(new — #151)_ | **Networking → Public access** | **none** |
| `landing_page_enabled` | Runtime tab | Networking → Public access | none |
| `session_access_ttl_minutes` | Runtime tab | Sessions & Tokens → Browser session | warden-restart |
| `session_refresh_ttl_days` | Runtime tab | Sessions & Tokens → Browser session | warden-restart |
| `default_token_expiration_days` | Runtime tab | Sessions & Tokens → Token defaults | none |
| `rotation_grace_hours` | Runtime tab | Sessions & Tokens → Token defaults | none |
| `sse_ticket_ttl_seconds` | Runtime tab | Sessions & Tokens → Streaming | none |
| `vllm_version` | Runtime tab | Maintenance → vLLM runtime | warden-restart |
| `log_retention_lines` | Runtime tab | Maintenance → Logs | none |

## Wireframes

### Tab 1 — General

```
┌──────────────────────────────────────────────────────────────────┐
│  Settings                                                        │
├──────────────────────────────────────────────────────────────────┤
│  [ General ]  Networking   Sessions & Tokens   Maintenance   Model │
└──────────────────────────────────────────────────────────────────┘
                                                      [ Edit ]

  ┌─ Identity ─────────────────────────────────────────────────────┐
  │  The single admin account used to log in.                      │
  │                                                                │
  │  Admin username                                                │
  │  [ admin                                                    ]  │
  │  ↳ Used for the login page.                                    │
  │                                                                │
  │  Admin password                                                │
  │  [ ••••••••                                                 ]  │
  │  ↳ Updates the bcrypt hash. All sessions invalidated on save.  │
  │    (Leave blank to keep current value.)                        │
  └────────────────────────────────────────────────────────────────┘

  ┌─ Hugging Face   ⚠ Some fields here require model reload ───────┐
  │  Credentials and cache path used when pulling weights.         │
  │                                                                │
  │  Hugging Face token              [requires model-reload]       │
  │  [ ••••••••                                                 ]  │
  │  ↳ Required for gated repos. Applies to next model load.       │
  │                                                                │
  │  HF cache directory              [requires model-reload]       │
  │  [ /hfcache                                                 ]  │
  │  ↳ Must be persistent + have enough free space.                │
  └────────────────────────────────────────────────────────────────┘

  ┌─ Defaults for new models ──────────────────────────────────────┐
  │  Pre-fills the Add Model modal. Does not affect existing       │
  │  models.                                                       │
  │                                                                │
  │  Default GPU indices                                           │
  │  [ 0,1                                                      ]  │
  │  ↳ Comma-separated GPU IDs.                                    │
  └────────────────────────────────────────────────────────────────┘
```

### Tab 2 — Networking

```
   General  [ Networking ]  Sessions & Tokens   Maintenance   Model
                                                      [ Edit ]

  ┌─ Public access ────────────────────────────────────────────────┐
  │  How clients outside the host see this warden.                 │
  │                                                                │
  │  Public URL                                                    │
  │  [ https://vllm.protrener.com                               ]  │
  │  ↳ If set, used in API endpoint examples instead of the        │
  │    browser's address bar. Useful behind a reverse proxy.       │
  │                                                                │
  │  Public landing page                              [ ✓ enabled ] │
  │  ↳ Serves a public HTML page at "/". Disable for private       │
  │    deployments that should 404 at the root.                    │
  └────────────────────────────────────────────────────────────────┘
```

### Tab 3 — Sessions & Tokens

```
   General   Networking  [ Sessions & Tokens ]  Maintenance   Model
                                                      [ Edit ]

  ┌─ Browser session   ⚠ Some fields require warden restart ───────┐
  │  How long a logged-in browser stays authenticated.             │
  │                                                                │
  │  Session access TTL              [requires warden-restart]     │
  │  [ 15                                            ] minutes     │
  │  ↳ How long a login JWT stays valid before refresh.            │
  │                                                                │
  │  Session refresh TTL             [requires warden-restart]     │
  │  [ 7                                             ] days        │
  │  ↳ How long until forced re-login.                             │
  └────────────────────────────────────────────────────────────────┘

  ┌─ Token defaults ───────────────────────────────────────────────┐
  │  Pre-fills the Token Create dialog. Existing tokens are        │
  │  unaffected.                                                   │
  │                                                                │
  │  Default token expiration                                      │
  │  [ 365                                           ] days        │
  │                                                                │
  │  Rotation grace window                                         │
  │  [ 24                                            ] hours       │
  └────────────────────────────────────────────────────────────────┘

  ┌─ Streaming ────────────────────────────────────────────────────┐
  │  One-shot tickets used to authenticate SSE connections.        │
  │                                                                │
  │  SSE ticket TTL                                                │
  │  [ 60                                            ] seconds     │
  └────────────────────────────────────────────────────────────────┘
```

### Tab 4 — Maintenance

```
   General   Networking   Sessions & Tokens  [ Maintenance ]  Model
                                                      [ Edit ]

  ┌─ vLLM runtime   ⚠ Some fields require warden restart ──────────┐
  │  Version of the vLLM Python package baked into the image.      │
  │                                                                │
  │  vLLM version                    [requires warden-restart]     │
  │  [ 0.9.2                                                    ]  │
  │  ↳ Container rebuild required.                                 │
  └────────────────────────────────────────────────────────────────┘

  ┌─ Logs ─────────────────────────────────────────────────────────┐
  │  Per-model log buffer retention.                               │
  │                                                                │
  │  Log retention                                                 │
  │  [ 5000                                          ] lines       │
  └────────────────────────────────────────────────────────────────┘
```

## Save UX (per-tab, unchanged shape)

- Each tab renders its own `[ Edit ]` button when not editing.
- When editing: `[ Cancel ] [ Reset ] [ Save ]` row appears top-right.
- `Save` is disabled until `dirty.length > 0`.
- On 2xx response, the per-tab snapshot is re-seeded from `mutate()`
  and a `requires_restart` banner is shown if the PATCH response
  flagged any restart kinds. Banner styling, copy, and ARIA mirror
  the current Runtime tab — unchanged.
- On 4xx, the response `detail` is shown in an inline error banner;
  the draft is preserved.
- Mobile (<640px): tabs collapse to a horizontal scroll strip; section
  cards stack full-width with their own headers. Buttons wrap onto a
  second row inside the per-card action area. (Existing `Card` +
  `Tabs` primitives already handle this — no responsive work needed.)

## Backend API contract changes

### Net delta
- **One new key**: `public_url` (string, optional, restart kind:
  `none`).
- **One new SQL migration** (`0021_public_url_setting.sql`): does NOT
  seed a default — absent row means "use `window.location.origin`",
  which is the most-conservative no-op default for installations not
  behind a proxy.
- **Hints registry**: one new entry in `RUNTIME_HINTS`.
- **Coercer**: new `_url(v)` coercer enforcing
  - must be a non-empty string,
  - must parse with `urllib.parse.urlsplit` to scheme ∈ {`http`,
    `https`} and a non-empty netloc,
  - trailing slash stripped before persist (`url.rstrip("/")`).
- **No endpoint added.** `GET /api/settings/runtime` returns the new
  field automatically (because `RUNTIME_KEYS` is iterated). `PATCH`
  validates via the existing `_COERCERS` dispatch — no route changes.
- **No new HTTP status codes; no new response fields.**

### Field shape

| Property | Value |
|---|---|
| Key | `public_url` |
| Type (DB) | `TEXT NULL` in `settings` KV (absent row = unset) |
| Type (API) | `string` or `null` |
| Validation | parseable URL, scheme http(s), non-empty netloc, max length 2048 |
| Persist transform | strip trailing `/` |
| Restart kind | `none` (read on every snippet render; flips take effect on next page load) |
| GET returns | string verbatim (no masking — not secret), or absent key if unset |
| PATCH accepts | non-empty string (422 if invalid). Clearing not in scope for v1 — same posture as `hf_token`. |

(Clearing back to "use origin" is filed as a follow-up rather than
shipped in v1; the same posture as `hf_token`, which can't be cleared
either. Operators who set it wrong can `PATCH` again with a different
URL; the migration path for "actually I want origin again" is "rm the
row via SQL" — fine for a v1 escape hatch, captured in follow-ups.)

## Frontend code reorganisation

### File map

```
frontend/src/app/settings/
  page.tsx                          MODIFIED — tab list grows to 5; activeTab union grows.

frontend/src/components/settings/
  general-tab.tsx                   NEW — Identity / HF / Defaults sections.
  networking-tab.tsx                NEW — Public access section. Owns public_url + landing_page_enabled.
  sessions-tab.tsx                  NEW — Browser session / Token defaults / Streaming sections.
  maintenance-tab.tsx               NEW — vLLM runtime / Logs sections.
  model-tab.tsx                     UNCHANGED.
  runtime-tab.tsx                   DELETED — superseded by the four new tabs.
  setting-field.tsx                 UNCHANGED.
  setting-section.tsx               NEW — Card-wrapped section primitive with title + subtitle + optional restart chip. Pure presentation, no state.

frontend/src/components/settings/hooks/
  use-runtime-settings.ts           NEW — extracted from runtime-tab.tsx. Owns SWR fetch, draft, dirty-tracking, PATCH-with-restart-banner. Parameterised by the subset of RUNTIME_KEYS a tab cares about. Returns { draft, dirty, isLoading, error, save(), reset(), edit(), cancel(), editing, saveError, restartBanner }.

frontend/src/lib/
  settings-hints.ts                 MODIFIED — add public_url entry; group constants RUNTIME_GENERAL_KEYS / _NETWORKING_KEYS / _SESSIONS_KEYS / _MAINTENANCE_KEYS exported so each tab imports its own slice in declared order.
  public-url.ts                     NEW — exports getPublicBaseUrl(): string and (test only) _resolvePublicBaseUrl(s, origin).

app/db/sql/
  0021_public_url_setting.sql       NEW — no seed; this migration only documents the new key and serves as a marker for tests.

app/settings/
  routes_api.py                     MODIFIED — RUNTIME_KEYS gains 'public_url': 'none'; _COERCERS gains _url; no route changes.
```

### Hook contract (`useRuntimeSettings`)

```ts
function useRuntimeSettings(keys: readonly RuntimeKey[]): {
  draft: Partial<Draft> | null;     // null while loading
  snapshot: Partial<Draft> | null;
  dirty: RuntimeKey[];              // subset of `keys` that are dirty
  setField: <K extends RuntimeKey>(k: K, v: Draft[K]) => void;
  editing: boolean;
  saving: boolean;
  saveError: string | null;
  restartBanner: string[];
  edit(): void;
  cancel(): void;
  reset(): void;
  save(): Promise<void>;            // PATCHes ONLY dirty fields in `keys`
}
```

The hook holds the full `Draft` and `snapshot` internally (one SWR
key, one fetch shared across all four tabs via SWR cache dedupe);
the `keys` argument scopes which subset participates in
`dirty`/`setField`/`save`. This lets per-tab Save buttons PATCH only
their own slice without forcing a global draft.

## Test plan

### Backend (pytest, Docker)

1. **`tests/unit/settings/test_public_url_coercer.py`** — NEW.
   - Accepts `https://vllm.protrener.com` → stored verbatim.
   - Accepts `https://vllm.protrener.com/` → stored as
     `https://vllm.protrener.com` (trailing slash stripped).
   - Rejects `ftp://example.com` (scheme).
   - Rejects `http://` (empty netloc).
   - Rejects `not a url` (parse failure).
   - Rejects `""` (empty).
   - Rejects integer / null / dict (type).

2. **`tests/integration/test_runtime_settings_api.py`** — APPEND.
   - GET returns `public_url` as `None` on a fresh DB.
   - PATCH with valid URL → 200; subsequent GET round-trips the value.
   - PATCH with invalid URL → 422 with `detail` mentioning
     `public_url`.

3. **`tests/unit/db/test_migration_0021.py`** — NEW.
   - Migration applies cleanly on a 0020-seeded DB.
   - Migration is idempotent (re-apply is a no-op).

### Frontend (Vitest)

4. **`frontend/tests/component/settings.test.tsx`** — REWRITE.
   The existing file pins the old IA. Update the existing 7 cases to
   land in the right tab:
   - "Runtime tab renders all 12 fields" → split into four per-tab
     assertions, each verifying its sectioned cards + field set.
   - PATCH echo `requires_restart` banner cases stay, but assert that
     the banner appears in the tab where the user clicked Save (not
     across tabs).
   - PATCH 422 surfaces `detail` — keep, scope to one tab.
   - Secret sentinel + leave-blank — keep, in General tab.
   - Model tab no-loaded-model + with-loaded-model — keep, unchanged.

5. **`frontend/tests/contract/settings-hints.test.ts`** — NEW.
   Pins the contract between `RUNTIME_HINTS` and the four per-tab key
   constants:
   - Every key in `RUNTIME_HINTS` belongs to exactly one of the four
     per-tab arrays (no orphans, no duplicates).
   - The four per-tab arrays together equal `Object.keys(RUNTIME_HINTS)`
     as a set.
   - `public_url` exists in `RUNTIME_HINTS` with restart `'none'`.

6. **`frontend/tests/component/networking-tab.test.tsx`** — NEW.
   Field-level validation that #154 AC #5 specifically calls for:
   - Renders both `public_url` and `landing_page_enabled` fields.
   - Submits a valid `public_url` → PATCH body contains
     `public_url: "https://example.com"` (no trailing slash).
   - On 422 from the server (bad URL), shows the server's `detail`
     in an error banner; draft preserved.

7. **`frontend/tests/component/public-url-helper.test.ts`** — NEW.
   `getPublicBaseUrl()` unit tests:
   - Returns `settings.public_url` when set.
   - Returns `window.location.origin` when unset.
   - Both branches strip trailing slash.
   - Integration: a snippet component (e.g. token-mint modal `curl`
     example) calls the helper and reflects an updated `public_url`
     after mutate.

8. **Snapshot test of the new layout.** A single Vitest snapshot
   covering one rendered tab (Networking) — confirms the section
   structure is wired (titles, subtitles, restart chips), without
   over-pinning copy. We pick Networking because it's the most-
   changed surface; over-pinning every tab via snapshot is brittle
   and noisy.

### Verification gate (per `feedback_stage_click_before_user_facing_hotfix`)

Before opening the implementation MR, dev MUST:

- `make build && make start` against the worktree;
- click through each of the five tabs;
- edit one field per tab and Save;
- verify the restart banner appears for at least one warden-restart
  field (e.g. `session_access_ttl_minutes`) and one model-reload
  field (e.g. `hf_token`);
- verify `public_url` setting changes the URL shown in at least one
  user-facing snippet (token-mint modal `curl` example is the
  canonical surface to check).

Captured in the implementation plan as an explicit gate, NOT a
"verified later" claim.

## Anti-scope (what this redesign does NOT change)

1. **Authentication and RBAC.** Single-admin model unchanged. PATCH
   still requires JWT. No role gates added.
2. **Backend route shape.** Still one `GET/PATCH /api/settings/runtime`.
   No per-tab endpoints, no nested response shape.
3. **The `settings` KV table schema.** No new columns; new field is
   just another row.
4. **Per-model settings.** `/models/<id>/settings` is unchanged. The
   Model tab on `/settings` remains the existing navigation pivot.
5. **Tokens page (`/tokens`).** Already a separate top-nav item.
   Token list / create / rotate UI is out of scope.
6. **Stats / Cache / Chat / Models pages.** Untouched.
7. **CSRF, SSE ticket, and auth-fetch internals.** Read-only consumers
   of the new field (via the new helper) where relevant; no protocol
   change.
8. **vLLM-version selector UX.** Stays a free-form text input. The
   "list available versions" affordance is a separate UX ask.
9. **Theme / colour palette / typography.** Inherits existing.

## Follow-ups (filed during this spec, not blocking implementation)

- **Clearing `public_url` back to "use origin"** — v1 requires editing
  via SQL or PATCH with a new URL. A cleaner clear-affordance can
  follow when we also reconsider `hf_token` clearing (#TBD).
- **Networking tab expansion.** TLS cert reload toggle, CORS
  allowlist, trusted proxy ranges — none in scope here; the tab
  exists to host them when each lands.
- **Section-level "Restart pending" surfacing.** Today the banner
  fires only after a Save that returned a restart kind. A persistent
  "restart pending" indicator across page reloads (so an operator
  who edited then walked away can see "you still need to restart")
  is a worthwhile add; out of scope here.
- **No backend coercer regression test for `_url`** of the form "the
  coercer is registered for every key in `RUNTIME_KEYS`" — the
  existing 500 defensive check already covers this at runtime, but a
  unit test that walks `RUNTIME_KEYS` and asserts `_COERCERS.get(k)
  is not None or k in {'hf_token','admin_username','admin_password'}`
  would be a worthwhile guard. Filed for the implementation pass.

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `runtime-tab.tsx` deletion drops a hidden caller | low | medium | grep for `RuntimeTab` / `runtime-tab` before delete; the tab is only mounted in `page.tsx`. |
| `useRuntimeSettings` hook over-fits the existing dirty-tracker and breaks secret-sentinel handling for `admin_password` / `hf_token` | medium | high | port the existing logic verbatim into the hook; secret-handling test from `settings.test.tsx` must keep passing without edits to the assertion shape. |
| `public_url` migrated call-sites accidentally include CSRF-relevant `location.origin` reads | low | high | implementation pass enumerates each migrated site with rationale in the commit message; CSRF middleware tests gate the merge. |
| Tabstrip overflows at narrow viewport with 5 tabs | low | low | existing `Tabs` primitive already scrolls horizontally — verified via current screen with `Models / Chat / Tokens / Stats / Cache / Settings` (six items) in nav-bar. |

## Acceptance criteria (mirrors #154)

1. `frontend-design` skill invoked and design output recorded — see
   the wireframes section above.
2. All existing 13 settings preserved; placement table in this spec
   is the authoritative migration plan.
3. New `public_url` field present, validates URL, strips trailing
   slash, used by `getPublicBaseUrl()` everywhere a user-facing
   snippet references the base URL.
4. Sectioned layout per design principles — see canonical structure
   table and wireframes.
5. Tests:
   - Backend: `_url` coercer suite; runtime settings round-trip;
     migration test.
   - Frontend: rewritten `settings.test.tsx`; new `settings-hints`
     contract test; new networking-tab test; new public-url-helper
     test; one snapshot of Networking layout.
6. Docs updated — `docs/operating.md` settings section + a new
   "Public URL" subsection. Implementation pass owns the doc edits
   (not Phase 1).
7. Dev walks through the redesigned page in a local Docker stack
   before pushing the implementation MR.

## References

- Issue **#154** — Settings: full /settings page UI/UX redesign.
- Issue **#151** — Settings: add "Public URL" field. Subsumed.
- Issue **#155** — Unified-port architecture (introduced
  `landing_page_enabled`; punted UX to #154). Spec at
  `docs/superpowers/specs/2026-05-23-unified-port-architecture-design.md`.
- Existing settings code surface:
  `frontend/src/app/settings/page.tsx`,
  `frontend/src/components/settings/runtime-tab.tsx`,
  `frontend/src/components/settings/model-tab.tsx`,
  `frontend/src/components/settings/setting-field.tsx`,
  `frontend/src/lib/settings-hints.ts`,
  `app/settings/routes_api.py`.
- Existing settings migrations: `app/db/sql/0010_settings_expansion.sql`,
  `app/db/sql/0020_landing_page_setting.sql`.
- Existing tests this redesign rewrites:
  `frontend/tests/component/settings.test.tsx`,
  `tests/integration/test_runtime_settings_api.py`.
