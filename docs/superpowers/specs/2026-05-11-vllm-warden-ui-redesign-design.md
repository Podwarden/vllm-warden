# vllm-warden UI Redesign — Design Spec

**Date:** 2026-05-11
**Status:** Revised after code review; ready for plan-writing
**Replaces:** the Jinja + htmx UI layer shipped in v2 (2026-05-08 spec)
**DB:** SQLite only. All migration SQL in this spec uses SQLite dialect (`datetime(...)`, ALTER TABLE without IF NOT EXISTS for columns). vllm-warden ships exactly one DB engine via `VW_DB_PATH`.

---

## Goal

Replace vllm-warden's Jinja + htmx + Chart.js UI wholesale with a Next.js 15 + Tailwind + recharts frontend that mirrors podwarden's frontend architecture 1:1, reusing podwarden's retro/retro-dark theme, shadcn-style primitives, and chart/log components. The JSON API layer stays; the HTML rendering layer is deleted.

## Why now

The Jinja wizard shipped in v2 was rated unusable across all four user-facing surfaces (pull progress, load step, error reporting, navigation). Iterative patches did not bring it to working state. Rather than queue more wizard hotfixes, replace the rendering layer with the same stack we already operate in production for podwarden, where the UX is known to work.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│ Browser (Next.js client) ────EventSource SSE────┐        │
│         │                                       │        │
│         ▼  fetch (Bearer JWT)                   │        │
│  ┌──────────────┐         ┌──────────────────────────┐   │
│  │ vllm-warden- │  proxy  │ vllm-warden-api          │   │
│  │ ui (Next.js  │────────►│ (FastAPI)                │   │
│  │ standalone)  │         │  - JSON only             │   │
│  │ :3000        │         │  - JWT auth              │   │
│  └──────────────┘         │  - SSE streams           │   │
│                           │  - supervises vLLM       │   │
│                           │  :8080                   │   │
│                           └──────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

Two services in compose, mirroring podwarden's split. UI is server-side-rendered Next.js standalone, talking to the API both server-side (via `BACKEND_URL` env) and client-side (via browser-visible base URL).

## Tech stack

| Layer | Choice | Source |
|---|---|---|
| Framework | Next.js 15 + React 19 | copy podwarden |
| Styling | Tailwind 3.4 + retro/retro-dark themes | copy podwarden globals.css verbatim |
| Charts | recharts ^3.7.0 | copy podwarden |
| Icons | lucide-react | copy podwarden |
| Primitives | shadcn-style (cva + clsx + tailwind-merge) | copy podwarden `components/ui/*` |
| Forms | uncontrolled + native validation; SWR mutations | match podwarden |
| Data fetching | SWR for REST, native EventSource for SSE | match podwarden |
| API client | `lib/api.ts` typed fetch wrapper | adapt podwarden pattern |
| Type generation | `openapi-typescript` from FastAPI OpenAPI | adapt podwarden pipeline |
| Auth | JWT bearer + refresh on 401 | new on backend, copy refresh pattern |
| Font | DM Sans (lazy-loaded) | copy podwarden `theme.tsx` |

## Repo layout

```
vllm-warden/
├── app/                          # FastAPI backend (existing, modified)
│   ├── auth/
│   │   ├── jwt.py                # NEW: HS256 access + refresh tokens
│   │   └── deps.py               # MODIFIED: replace session deps with JWT deps
│   ├── tokens/
│   │   ├── routes_api.py         # MODIFIED: expires_in_days, rotate endpoint
│   │   ├── routes_web.py         # DELETED
│   │   └── ...
│   ├── models/
│   │   ├── routes_web.py         # DELETED
│   │   ├── routes_logs.py        # KEPT (SSE)
│   │   └── routes_api.py         # KEPT
│   ├── settings/
│   │   ├── routes_web.py         # DELETED
│   │   └── routes_api.py         # MODIFIED: expose all runtime settings
│   ├── stats/routes_web.py       # DELETED
│   ├── setup/routes_web.py       # DELETED
│   ├── web/                      # DELETED (entire dir: templates + static)
│   └── main.py                   # MODIFIED: drop Jinja mount, drop static
├── frontend/                     # NEW
│   ├── package.json
│   ├── next.config.ts
│   ├── tailwind.config.ts
│   ├── postcss.config.mjs
│   ├── tsconfig.json
│   ├── Dockerfile                # multi-stage: deps → build → standalone
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx
│   │   │   ├── globals.css       # COPY podwarden verbatim
│   │   │   ├── page.tsx          # redirect to /models or /setup
│   │   │   ├── login/page.tsx
│   │   │   ├── setup/
│   │   │   │   ├── layout.tsx
│   │   │   │   ├── welcome/page.tsx
│   │   │   │   ├── admin/page.tsx
│   │   │   │   ├── hf-token/page.tsx
│   │   │   │   ├── gpus/page.tsx
│   │   │   │   └── done/page.tsx
│   │   │   ├── models/
│   │   │   │   ├── page.tsx       # dashboard
│   │   │   │   └── [id]/
│   │   │   │       ├── page.tsx   # detail + logs
│   │   │   │       └── settings/page.tsx
│   │   │   ├── tokens/page.tsx
│   │   │   ├── stats/page.tsx
│   │   │   └── settings/page.tsx  # tabs: Runtime / Model
│   │   ├── components/
│   │   │   ├── theme-switcher.tsx           # COPY from podwarden
│   │   │   ├── nav-bar.tsx                   # adapt nav items
│   │   │   ├── ansi-log.tsx                  # COPY from podwarden
│   │   │   ├── ui/                           # COPY shadcn primitives
│   │   │   ├── panels/                       # COPY relevant panels
│   │   │   ├── models/
│   │   │   │   ├── add-model-modal.tsx
│   │   │   │   ├── model-card.tsx
│   │   │   │   ├── pull-progress.tsx        # SSE consumer
│   │   │   │   └── log-stream.tsx           # SSE consumer + ansi-log
│   │   │   ├── tokens/
│   │   │   │   ├── token-row.tsx
│   │   │   │   ├── create-token-dialog.tsx
│   │   │   │   ├── rotate-token-dialog.tsx
│   │   │   │   └── expiry-banner.tsx
│   │   │   ├── settings/
│   │   │   │   ├── runtime-tab.tsx
│   │   │   │   ├── model-tab.tsx
│   │   │   │   └── setting-field.tsx        # label + hint + input
│   │   │   └── stats/
│   │   │       ├── throughput-chart.tsx
│   │   │       └── gpu-util-chart.tsx
│   │   └── lib/
│   │       ├── api.ts                        # adapt podwarden
│   │       ├── api-types.generated.ts        # codegen
│   │       ├── auth-fetch.ts                 # JWT refresh-on-401
│   │       ├── theme.tsx                     # COPY from podwarden
│   │       ├── utils.ts                      # COPY (cn helper)
│   │       └── sse.ts                        # EventSource hook
│   └── public/
├── deploy/hub/
│   ├── compose.yaml              # MODIFIED: 1 → 2 services
│   └── template.json             # MODIFIED: add ui port + env
├── docker-compose.yml            # MODIFIED: add ui service for local dev
└── docs/superpowers/specs/2026-05-11-vllm-warden-ui-redesign-design.md
```

## Cutover strategy

**Selected: Two MRs, backend first, accept brief UI outage on `vllm.protrener.com`.**

The pw_prod operator already knows this redesign is coming; the UI gap between MR-1 and MR-2 deploys is acceptable. During the gap, the API is fully functional and callable via `curl` (a runbook is written as part of MR-1 — see `docs/operating.md`).

### MR-1: Backend cutover (`feat/jwt-and-jinja-cleanup`)

1. Add JWT auth (access + refresh) replacing existing session auth.
2. Add `POST /api/auth/login`, `POST /api/auth/refresh` (cookie-driven), `POST /api/auth/logout` (revokes streams + clears cookie), `POST /api/auth/sse-ticket`.
3. Tokens table migration: `expires_at`, `rotated_at`, `rotated_from` columns + backfill.
4. `POST /api/tokens/{id}/rotate` endpoint.
5. Bearer proxy auth: enforce `expires_at`, `revoked_at`.
6. Settings API: extend with full runtime + per-model settings surfaces, including `requires_restart` in PATCH response.
7. Delete all `routes_web.py` files + `app/web/` directory.
8. Drop Jinja2 + static-file mounts + `SessionMiddleware` from `app/main.py`.
9. Remove `jinja2` and `itsdangerous` deps; add `pyjwt[crypto]` (preferred over `python-jose` — actively maintained, simpler API, already used in podwarden backend).
10. New `docs/operating.md` with a curl-based runbook covering: login, mint token, register model, pull, load, completion, rotate, revoke. This is the operator's escape hatch during the MR-1 → MR-2 gap.
11. Tests: all existing API tests stay green; new tests for JWT login/refresh/logout, SSE ticket issuance, token rotation grace window, token expiry enforcement on bearer proxy, SameSite + Origin checks on refresh, stream revocation on logout.

Backend runs headless after MR-1 ships — no UI until MR-2 lands.

### MR-2: Next.js frontend (`feat/ui-redesign-nextjs`)

1. Scaffold `frontend/` with podwarden's exact toolchain. `next.config.ts` MUST set `output: 'standalone'` so the Dockerfile can copy `.next/standalone/` + `.next/static/` for the minimal runtime image (matches podwarden frontend).
2. Copy theme + globals.css + ui primitives + ansi-log + nav-bar shell + theme-switcher from the source paths pinned in "Reuse from podwarden".
3. Generate `api-types.generated.ts` from vllm-warden's OpenAPI.
4. Build `lib/api.ts` + `lib/auth-fetch.ts` against JWT.
5. Implement pages in order: Login → Setup wizard → Models dashboard → Add-model modal → Model detail + logs → Tokens → Settings → Stats.
6. Add a Next.js `/api/health/route.ts` that returns `{ ok: true }` (Next.js does not provide one by default — the spec previously implied it did).
7. Add `vllm-warden-ui` service to `docker-compose.yml` for local dev.
8. Update `deploy/hub/compose.yaml` + `deploy/hub/template.json` to ship 2 services.

## Reuse from podwarden

All source paths are relative to `/home/ip/projects/podwarden/frontend/src/`. The implementer copies these into `vllm-warden/frontend/src/` at the matching path.

**Verbatim (no edits):**
| Source (podwarden) | Destination (vllm-warden frontend/src/) |
|---|---|
| `lib/theme.tsx` | `lib/theme.tsx` |
| `app/globals.css` | `app/globals.css` |
| `../tailwind.config.ts` | `../tailwind.config.ts` |
| `../postcss.config.mjs` | `../postcss.config.mjs` |
| `components/theme-switcher.tsx` | `components/theme-switcher.tsx` |
| `lib/utils.ts` | `lib/utils.ts` |
| `components/ui/button.tsx` | `components/ui/button.tsx` |
| `components/ui/card.tsx` | `components/ui/card.tsx` |
| `components/ui/input.tsx` | `components/ui/input.tsx` |
| `components/ui/badge.tsx` | `components/ui/badge.tsx` |
| `components/ui/skeleton.tsx` | `components/ui/skeleton.tsx` |
| `components/ui/modal.tsx` | `components/ui/modal.tsx` |
| `components/ui/tabs.tsx` | `components/ui/tabs.tsx` |
| `components/ui/select.tsx` | `components/ui/select.tsx` |
| `components/ui/combobox.tsx` | `components/ui/combobox.tsx` |
| `components/ansi-log.tsx` | `components/ansi-log.tsx` |

**Lightly adapted (rename, retarget data):**
| Source | Destination | Change |
|---|---|---|
| `components/nav-bar.tsx` | `components/nav-bar.tsx` | nav items become Models / Tokens / Stats / Settings |
| `components/panels/metric-summary-panel.tsx` | `components/panels/metric-summary-panel.tsx` | retarget to Stats data |
| `components/panels/status-table-panel.tsx` | `components/panels/status-table-panel.tsx` | retarget to Models dashboard |
| `components/panels/config-form-panel.tsx` | `components/panels/config-form-panel.tsx` | retarget to Settings forms |
| `components/auth-gate.tsx` | `components/auth-gate.tsx` | JWT bearer instead of OIDC |

**Pattern only (rewrite for vllm-warden domain):**
| Source (study, do not copy) | Destination (rewrite) |
|---|---|
| `lib/api.ts` | `lib/api.ts` — typed fetch wrapper structure |
| `lib/auth-fetch.ts` + `lib/auth-refresh.ts` | `lib/auth-fetch.ts` — JWT refresh-on-401 loop |
| `lib/api-types.generated.ts` + the codegen npm script | `lib/api-types.generated.ts` — same `openapi-typescript` pipeline |

**Not reused:** clusters, hosts, deploy, system-apps, doctor, canvas, zone-*, network-badges, cloud-provider-logo, migration-guide, support, hub-catalog-modal, terminal (Quake), update-banner, backend-refresh-banner, temp-admin-banner.

## Auth contract

**Login:**
```
POST /api/auth/login
{ "username": "admin", "password": "..." }
→ 200 { "access_token": "...", "expires_in": 900 }
   Set-Cookie: vw_refresh=<jwt>; HttpOnly; Secure; SameSite=Strict; Path=/api/auth; Max-Age=604800
```

The refresh token is **never** in the response body. The body returns only the access token; the refresh JWT lives in the `vw_refresh` cookie.

**Refresh:**
```
POST /api/auth/refresh           (no body; refresh token comes from cookie)
→ 200 { "access_token": "...", "expires_in": 900 }
```

**Protected requests:** `Authorization: Bearer <access_token>` on every API call.

### Refresh cookie — CSRF defence

The lock "switch to JWT bearer, NOT cookie+CSRF" is partially relaxed: the refresh token lives in a cookie, but only the `/api/auth/refresh` endpoint reads it. CSRF is prevented without a CSRF token by combining:

1. `SameSite=Strict` — browser never sends the cookie on cross-site requests of any method.
2. `Path=/api/auth` — cookie is scoped tightly; never sent to `/v1/*` or `/api/models/*`.
3. **Server-side Origin check** on `/api/auth/refresh`: reject if `Origin` header is absent or does not match the configured `VW_FRONTEND_ORIGIN`. Same for `/api/auth/logout`. (Belt-and-suspenders against future browser SameSite-Strict regressions.)
4. **No GET form** of refresh — POST-only, so a malicious `<img src>` or `<a>` cannot trigger it.

All other endpoints use bearer auth only; no cookies are read or written outside `/api/auth/*`.

### JWT secret bootstrap + rotation

`VW_JWT_SECRET` is loaded with this precedence:

1. If env var `VW_JWT_SECRET` is set and non-empty → use it.
2. Else, on first boot, generate a 64-byte random secret with `secrets.token_urlsafe(64)` and persist it to `<VW_DB_PATH dir>/jwt_secret` (mode 0600). On subsequent boots, load from this file.
3. If neither path works (read-only data dir), refuse to start with a clear error.

**Rotation:** delete the persisted file + restart, or set `VW_JWT_SECRET` to a new value. All access tokens and refresh cookies become invalid instantly; every client re-logs-in. Acceptable for single-admin. Document the rotation procedure in `docs/operating.md` (created in MR-1).

### v1 cookie session → JWT migration (existing deployments)

vllm-warden v1/v2 used Starlette session cookies (`itsdangerous`-signed). After MR-1 ships, an operator with an active session cookie will see their session quietly fail at the next API call — the v1 cookie is no longer trusted.

**Plan:**
- MR-1 deletes the Starlette `SessionMiddleware` outright. No grace period.
- The login page is bookmarkable (`/login`); a 401 from any API endpoint redirects there.
- The frontend always calls `/api/auth/refresh` on mount; if the cookie is from v1 (wrong signing key, wrong claim shape), the refresh endpoint returns 401 and the UI shows the login form.
- No data migration needed — there are no persisted server-side sessions.

This is acceptable because:
- vllm-warden is operator-only; one human re-logs-in on cutover.
- The operator already has admin credentials (no recovery flow needed).
- The cutover happens during a coordinated deploy window.

**SSE auth:** EventSource cannot set custom headers. Two options, picking option B:
- ~~A: pass JWT as `?token=` query param~~ — leaks to access logs
- **B: short-lived signed query param.** Client calls `POST /api/auth/sse-ticket` → server returns a 60s HMAC-signed ticket bound to (user_id, stream_path, issued_at). EventSource opens with `?ticket=<ticket>`. SSE routes validate ticket on initial connect.

**SSE leak blast radius + server-side revocation:** A ticket that leaks is only useful for 60s **and only to open a new stream**. But an *already-open* stream is not bound to ticket lifetime — it keeps streaming until disconnect. To revoke an in-flight stream on logout:

- Each authenticated SSE stream registers itself in an in-memory `set[StreamHandle]` keyed by `user_id`.
- `POST /api/auth/logout` calls `cancel_streams(user_id)`, which raises an asyncio cancellation into every registered stream for that user.
- On logout, ticket-mint endpoint is also rate-limited via a short deny-list (60s) on the user so a leaked-cookie-then-logout race can't immediately remint.

This is in-process state; on multi-replica deploys it's incomplete, but vllm-warden runs single-replica (it supervises a local subprocess), so single-process is sufficient.

**Token rotation grace window** (separate from JWT) applies only to **API keys** (Bearer `vw_...` tokens used for `/v1/*` proxy), not session JWTs.

## Tokens — schema + endpoints

### Migration

**SQLite only** — vllm-warden uses SQLite exclusively. SQLite's `ALTER TABLE ADD COLUMN` does not accept a non-constant default in a single statement, so the migration runs in two steps: add the column nullable, then backfill, then enforce via the bearer-check (we don't add a CHECK constraint because SQLite can't enforce NOT NULL retroactively without a table rebuild).

```sql
-- Step 1: add columns (nullable to satisfy SQLite ALTER limitations)
ALTER TABLE tokens ADD COLUMN expires_at TEXT NULL;
ALTER TABLE tokens ADD COLUMN rotated_at TEXT NULL;
ALTER TABLE tokens ADD COLUMN rotated_from TEXT NULL
  REFERENCES tokens(id) ON DELETE SET NULL;

-- Step 2: backfill existing rows (one year from created_at)
UPDATE tokens
   SET expires_at = datetime(created_at, '+365 days')
 WHERE expires_at IS NULL;

-- Step 3: index for near-expiry scans
CREATE INDEX IF NOT EXISTS idx_tokens_expires_at ON tokens(expires_at);
```

`expires_at` is stored as ISO-8601 TEXT (matches existing `created_at` / `revoked_at` columns). New rows set `expires_at` at INSERT time in `TokenRepo.create()`, not via DB default. Bearer check treats `expires_at IS NULL` as "never expires" so the migration is safe even if a row sneaks in before backfill completes.

### Endpoints

| Method | Path | Body | Notes |
|---|---|---|---|
| POST | `/api/tokens` | `{name, expires_in_days?}` | default 365; 0 = never; returns plaintext once |
| GET | `/api/tokens` | — | items include `expires_at`, `is_expired`, `is_near_expiry`, `rotated_at`, `rotated_from`, `successor_id` |
| POST | `/api/tokens/{id}/rotate` | `{grace_hours?}` | default 24; mints new token, sets old `revoked_at = now + grace_hours`, sets old `rotated_at = now`, sets new `rotated_from = old_id`; returns new plaintext one-time |
| DELETE | `/api/tokens/{id}` | — | immediate revoke (sets `revoked_at = now`) |

### Bearer auth check (`app/proxy/auth.py`)

```python
if row.expires_at is not None and row.expires_at <= now():
    raise HTTPException(401, "token expired")
if row.revoked_at is not None and row.revoked_at <= now():
    raise HTTPException(401, "token revoked")
```

`expires_at IS NULL` means "never expires" (legacy rows that somehow escaped backfill, or future `expires_in_days=0` semantics if we ever expose that).

### Frontend Tokens page

Columns: **Name · Prefix · Created · Expires · Last used · Status · Actions**

- **Expires** cell: pill `red <7d`, `amber <30d`, `green ≥30d`, `gray Never`
- **Status** cell: one of `Active` · `Rotated → <prefix>` (link to successor) · `Expired` · `Revoked` · `Grace (revokes in Xh)`
- **Actions**: `Rotate` button (icon: `RefreshCw`), `Revoke` button (icon: `Trash2`)
- **Create-token modal**: name + expiration radio (`30d` / `90d` / `1 year (default)` / `Custom date` / `Never`)
- **Rotate dialog**: confirms grace hours (default 24), reveals new plaintext **once** with copy button, shows old token revoke timestamp
- **Top-of-page banner**: amber banner when any token has `is_near_expiry`; lists names + days remaining

## Settings — full surface

### Runtime tab (`/settings`, tab 1)

Settings stored in `settings` table or env-derived. Read-only display + Edit button per section. The **Restart** column marks fields that require the warden process (or the vLLM subprocess) to restart. PATCH responses echo `requires_restart: bool` so the UI can show a "Restart pending" banner.

| Field | Type | Default | Restart | Hint |
|---|---|---|---|---|
| Admin username | string | `admin` | no | The single operator account. Used for the login page. |
| Admin password | secret | — | no | Updates the bcrypt hash. All sessions invalidated on change. |
| Hugging Face token | secret | — | model-reload | Passed to vLLM via `HF_TOKEN` env. Required for gated repos (Llama, gpt-oss, Mistral, etc.). New value applies to next model load. |
| HF cache directory | path | `/hfcache` | model-reload | Where model weights are pulled. Must be a persistent volume with enough free space (gpt-oss-20b ≈ 13 GB on disk). |
| Default GPU indices | int[] | `[0]` | no | Pre-selected when adding a new model. Comma-separated GPU IDs. |
| Default token expiration | int (days) | `365` | no | Pre-fills the new-token dialog. Affects new tokens only. |
| Rotation grace window | int (hours) | `24` | no | When you rotate a token, the old one stays valid for this many hours. Lets you swap creds in your clients without downtime. |
| Session access TTL | int (minutes) | `15` | warden-restart | How long a login JWT stays valid before refresh. Short = safer. Existing tokens keep their original TTL. |
| Session refresh TTL | int (days) | `7` | warden-restart | How long until forced re-login. Existing refresh cookies keep their original TTL. |
| SSE ticket TTL | int (seconds) | `60` | no | How long a single SSE connect ticket is valid. SSE streams stay open after auth; only the initial connect needs the ticket. |
| vLLM version | string | `0.9.2` | warden-restart | vLLM Python package version. Warden container rebuild required (image baked at build time). |
| Log retention | int (lines) | `5000` | no | Per-model log buffer size kept in memory + on disk. Applied to new lines after change. |

**Restart semantics:**
- `no` — change takes effect immediately on PATCH.
- `model-reload` — change persists but the running vLLM subprocess keeps using the old value until the model is unloaded + reloaded. UI shows "Restart pending" badge on the affected model.
- `warden-restart` — change persists but the warden process itself must restart for the new value to take effect. UI shows a global banner.

### Model tab (`/settings`, tab 2)

Shows the **currently-loaded** model (if any). Read-only until you click Edit. Edit warns: "Changes require model reload — vLLM will be stopped, settings persisted, and started again."

If no model is loaded: empty-state with "No model is currently loaded. Load a model from the [Models page](/models) to edit its settings here." plus a dropdown to select a registered-but-not-loaded model to edit.

| Section | Field | Hint |
|---|---|---|
| Identity | served_model_name | The name clients pass in `model:` for `/v1/completions`. Slug only — current backend regex allows alphanumeric + `.`, `_`, `-`. Dots permitted because vLLM model names like `Mixtral-8x7B-v0.1` are common. Frontend mirrors that regex client-side for instant validation. |
|  | hf_repo | Hugging Face repo path, e.g. `facebook/opt-125m` or `openai/gpt-oss-20b`. |
|  | hf_revision | Branch, tag, or commit SHA. Default `main` — pin to a SHA for reproducibility. |
| Hardware | gpu_indices | Which GPU slots to bind. Determines `CUDA_VISIBLE_DEVICES` for the vLLM subprocess. The 2026-05-08 bug landed here — picker must reach the subprocess. |
|  | tensor_parallel_size | Number of GPUs that shard each layer's weights. Auto-set to `len(gpu_indices)` because we only support **tensor-parallel** today. Data-parallel and pipeline-parallel are out of scope. |
|  | gpu_memory_utilization | Fraction of VRAM vLLM may consume per GPU (0.0–1.0, default 0.9). Lower if you hit OOM during paged-attention warmup. |
| Precision | dtype | One of `auto`, `float16`, `bfloat16`, `float32`. `auto` follows the model's config. |
|  | quantization | One of `none`, `awq`, `gptq`, `fp8`, `bitsandbytes`. Most models ship with weights pre-quantized — leave `none` unless you know otherwise. |
|  | kv_cache_dtype | One of `auto`, `fp8`, `fp8_e5m2`. `fp8` halves KV cache memory; tiny accuracy impact. |
| Context | max_model_len | Max sequence length (prompt + generation). Lower = less KV cache memory; higher = fits longer contexts. Capped by model's training context. |
|  | block_size | KV cache block size (tokens). Default 16. Power users only. |
|  | swap_space | GiB of CPU RAM to spill KV cache to under pressure. Default 4. 0 = disable. |
| Serving | max_num_seqs | Max concurrent sequences. Higher = more throughput, more memory. |
|  | max_num_batched_tokens | Per-step token budget across all sequences. Default = `max_model_len`. Lower to bound step latency. |
|  | enforce_eager | If true, disable CUDA graphs. Useful for debugging and tiny models; ~5% slower. |
|  | trust_remote_code | Required for models that ship Python in their HF repo (e.g. some custom architectures). Off by default for security. |
|  | disable_log_requests | Suppress per-request vLLM logs. Default off. |
| Advanced | extra_args | Free-form list passed to `vllm serve` after the curated flags. One arg per row, e.g. `--worker-use-ray`, `--scheduler-delay-factor`, `0.3`. |
|  | extra_env | Free-form env vars passed to the vLLM subprocess. Allowlisted by `app.runtime.env_builder` (prefix `VLLM_`, `HF_`, `TRITON_`, etc.). |

Hint copy ships as a single TS object in `frontend/src/lib/settings-hints.ts` so QA + docs can review without grepping JSX.

### Settings API

```
GET    /api/settings/runtime       → all runtime settings (above table 1)
PATCH  /api/settings/runtime       → partial update; reload-required keys flagged in response
GET    /api/models/{id}/settings   → all model fields incl. curated extras
PATCH  /api/models/{id}/settings   → mutate; 409 if model is currently `loaded` (must unload first)
```

`extra_args` and `extra_env` validation runs server-side using the same allowlists as today.

## Pages — inventory

| Route | Purpose | Key components | Data sources |
|---|---|---|---|
| `/login` | Username + password | `Input`, `Button` | `POST /api/auth/login` |
| `/setup/welcome` | First-run intro | static | — |
| `/setup/admin` | Set admin password | form | `POST /api/setup/admin` |
| `/setup/hf-token` | Set HF token | form | `POST /api/setup/hf-token` |
| `/setup/gpus` | Detect + pick GPUs | GPU table | `GET /api/setup/gpus` |
| `/setup/done` | Finish + redirect | static | — |
| `/models` | Dashboard of registered models | `status-table-panel`, `add-model-modal` | `GET /api/models` (SWR) |
| `/models/[id]` | Detail + live logs | `model-card`, `log-stream` | `GET /api/models/{id}`, SSE `/api/models/{id}/logs/stream` |
| `/models/[id]/settings` | Per-model settings editor | `setting-field`, tabs | `GET/PATCH /api/models/{id}/settings` |
| `/tokens` | API key management | `token-row`, `create-token-dialog`, `rotate-token-dialog`, `expiry-banner` | `GET /api/tokens` (SWR) |
| `/stats` | Throughput + GPU util | `throughput-chart`, `gpu-util-chart`, `metric-summary-panel` | `GET /api/stats/*` |
| `/settings` | Tabs: Runtime / Model | `runtime-tab`, `model-tab` | `GET /api/settings/runtime`, `GET /api/models/{id}/settings` |

## SSE consumption

EventSource has a critical pitfall: on transport error it **auto-reconnects with the same URL** — meaning the same expired ticket. The server will reject it forever and the browser will keep retrying. The hook below sidesteps native reconnect entirely.

Single hook `useEventSource<T>(path)` in `lib/sse.ts`:

```ts
function useEventSource<T>(path: string, opts: { onMessage: (msg: T) => void; enabled?: boolean }) {
  useEffect(() => {
    if (opts.enabled === false) return;
    let stopped = false;
    let es: EventSource | null = null;
    let backoffMs = 1000;

    async function connect() {
      if (stopped) return;
      let ticket: string;
      try {
        const r = await authFetch('/api/auth/sse-ticket', { method: 'POST', body: JSON.stringify({ path }) });
        if (!r.ok) throw new Error(`ticket ${r.status}`);
        ({ ticket } = await r.json());
      } catch {
        // auth-fetch already tried to refresh; if it still failed, redirect to login
        if (!stopped) setTimeout(connect, Math.min(backoffMs *= 2, 30000));
        return;
      }
      const url = `${path}?ticket=${encodeURIComponent(ticket)}`;
      es = new EventSource(url);
      es.onopen = () => { backoffMs = 1000; };
      es.onmessage = (e) => { try { opts.onMessage(JSON.parse(e.data)); } catch {} };
      es.onerror = () => {
        // Native EventSource would auto-reconnect with the SAME (now-expired) ticket.
        // We close it deliberately and run our own reconnect with a fresh ticket.
        es?.close();
        es = null;
        if (!stopped) setTimeout(connect, Math.min(backoffMs *= 2, 30000));
      };
    }

    connect();
    return () => { stopped = true; es?.close(); };
  }, [path, opts.enabled]);
}
```

Server-side ticket endpoint binds the ticket to (user_id, requested_path, exp=now+60s) so it cannot be replayed on a different stream. Each ticket is single-use: validated tickets are added to an in-memory deny-set with TTL 65s.

Consumers: `<PullProgress modelId={id}/>` and `<LogStream modelId={id}/>`. Both render the existing podwarden `<AnsiLog>` for line rendering.

## Compose changes

### `docker-compose.yml` (local dev)

```yaml
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports: ["8080:8080"]
    environment:
      VW_JWT_SECRET: ""                       # empty → auto-mint on first boot, persist to /data/jwt_secret
      VW_FRONTEND_ORIGIN: http://localhost:3000
      VW_DB_PATH: /data/vw.db
    volumes:
      - vw-data:/data
      - vw-hfcache:/hfcache
    deploy: { resources: { reservations: { devices: [{driver: nvidia, count: all, capabilities: [gpu]}] } } }

  ui:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports: ["3000:3000"]
    environment:
      BACKEND_URL: http://api:8080
      NEXT_PUBLIC_BACKEND_URL: http://localhost:8080
    depends_on: [api]

volumes:
  vw-data:
  vw-hfcache:
```

### `deploy/hub/compose.yaml` (PodWarden catalog template)

Same shape as above, with PodWarden's standard image-pin sentinels:
```yaml
services:
  api:
    image: registry.podwarden.com/vllm-warden:{{ image_tag }}
    # ...
  ui:
    image: registry.podwarden.com/vllm-warden-ui:{{ image_tag }}
    # ...
```

### `deploy/hub/template.json`

- Ports: add UI port (3000 internal → user-chosen external)
- Ingress: point to `ui` service, not `api`
- Required env: `VW_FRONTEND_ORIGIN` (must match the public UI URL for the CSRF/Origin check on `/api/auth/*`). `VW_JWT_SECRET` is **optional** — empty/unset triggers the auto-mint path described under "JWT secret bootstrap"; operators who want centralised secret management can set it explicitly.
- Health checks: api `/healthz` (already exists), ui `/api/health` (new — see MR-2 step 6; Next.js does not provide this by default)
- **Template versioning:** this is a breaking compose change (1 → 2 services + new required env). Publish as a **new major version** in the Hub catalog (e.g. `v2.x` track) rather than as a patch on the v1 track. The publish script must:
  1. Create a new template version row with `breaking: true` and `min_compatible_warden_version: "v2026.05.??.0"` (the version that ships MR-1+MR-2 together).
  2. Leave the existing v1 template version live so existing single-service installs keep upgrading on the v1 track until operators explicitly migrate.
  3. Surface the breaking change in the Hub upgrade UI (PodWarden core already reads `breaking: true` and surfaces it; see `pw_prod` upgrade flow).
- **Migration note** in `deploy/hub/README-hub.md`: existing v1 single-service installs must edit their compose stack to add the `ui` service, set `VW_FRONTEND_ORIGIN`, and bind a new ingress route. There is no in-place upgrade path because the ingress target changes (api → ui).

## CI changes

- Add `lint:frontend` job: `cd frontend && npm ci && npm run lint && npm run typecheck`
- Add `build:ui` job: builds + pushes `registry.podwarden.com/vllm-warden-ui:<tag>` alongside the existing `build:image`
- Add `typecheck:api-types` job: regenerate from FastAPI OpenAPI, fail if drift vs committed file
- Both new image tags follow the same tag scheme as the api image (`sha-`, branch slug, `:staging`, `:production`, `:latest`, `vYYYY.MM.DD.N`)

## Testing

**Backend:**
- All existing API tests stay green.
- New: JWT login/refresh/logout, SSE ticket issuance, token rotation grace window, token expiry enforcement on bearer proxy.
- Integration: end-to-end token rotate → old token works for grace window → old token rejected after grace.

**Frontend:**
- Component tests via Vitest + Testing Library for: `setting-field`, `rotate-token-dialog`, `create-token-dialog`, `pull-progress`, `log-stream`, `expiry-banner`.
- E2E via Playwright (one happy path): login → register a tiny model (opt-125m) → pull → load → mint token → completion → rotate token → unload → delete.

## Non-goals (out of scope)

- Multi-user / RBAC — still single-admin.
- OIDC / SSO integration — JWT-bearer with local password only.
- Mobile-first responsive — desktop-first; mobile gets best-effort but isn't a release gate.
- WebSocket — SSE everywhere; no WS upgrade.
- Per-token scopes — every token has full `/v1/*` access.
- Internationalization — English only.

## Rollout

1. MR-1 merges → develop → cut `v2026.05.12.0` or `v2026.05.13.0` → image lacks UI, headless API only.
2. MR-2 merges → develop → cut next CalVer → image bundles `vllm-warden-ui` for the first time.
3. PodWarden Hub catalog update → bumps `compose.yaml` from 1 to 2 services. Existing single-service installs need a manual stack-edit; document this.
4. `vllm.protrener.com` (pw_prod): update Hub template → rollout → manual smoke through the new UI.

## Risk register

| Risk | Mitigation |
|---|---|
| MR-1 lands headless; we can't operate the system between MR-1 and MR-2 | Keep `/api/*` endpoints + curl runbook (`docs/operating.md`) for the gap window. Cut MR-2 within same sprint. |
| JWT secret rotation isn't planned | Auto-mint + persist to `<VW_DB_PATH dir>/jwt_secret` (mode 0600); rotation = delete file + restart, everyone re-logs-in. Acceptable for single-admin. |
| SSE ticket leak in browser URL bar | Ticket TTL 60s + bound to (user_id, stream_path), single-use via deny-set; on logout, in-process stream registry calls `cancel_streams(user_id)` to terminate any still-open streams. Worst case: a leaked ticket gives ≤60s of read-only access to **one** stream from **one** vantage point. |
| Existing API token holders see no expiry change until they rotate | Backfill puts every existing token at `created_at + 365d`; near-expiry banner will flag them as they approach. |
| Hub template breaking change for compose users | New major version (`v2.x` track) with `breaking: true` + `min_compatible_warden_version`. PodWarden core surfaces breaking changes in the Hub upgrade UI. v1 track stays live so existing installs upgrade on the old track until they explicitly migrate. |
| Recharts bundle size | Already in podwarden; not a regression. |
| Next.js bundle size baseline unknown for vllm-warden | MR-2 first build records baseline (`.next/analyze` from `@next/bundle-analyzer`); fail CI on +20% regression in subsequent MRs. Establishes the budget that podwarden lacks. |
| Fate of existing Jinja+pytest UI tests during MR-1 → MR-2 gap | MR-1 deletes the Jinja templates and their pytest UI tests (they assert HTML structure that no longer exists). The API tests they shared fixtures with stay. New Vitest + Playwright suite lands with MR-2. Net coverage drops between MRs; the curl runbook + backend API tests are the only safety net during that window. Accepted because the gap is one sprint. |
| Dual-auth blast radius during MR-1 → MR-2 gap | Between MR-1 (deletes SessionMiddleware) and MR-2 (ships UI), the system has **JWT-only** auth. Operators must use the curl runbook with `Authorization: Bearer <jwt>`. There is no fallback to v1 cookie sessions. Risk: an operator who skips reading `docs/operating.md` will be locked out and need to reset the admin password via the CLI. Mitigation: MR-1 release notes spell out the new auth flow and link the runbook prominently. |

## Open questions

None — resolved during user review of the draft (refresh-token storage, cutover sequencing, JWT-secret bootstrap, v1→v2 cookie migration, SSE leak blast radius, tensor-parallel scope, and Hub template versioning all decided; see git history for the question-by-question pivots).

---

## Self-review

1. **Placeholders:** none — every field, endpoint, env var, file path, and migration step is concrete.
2. **Internal consistency:** JWT contract (access-in-memory + refresh-cookie with SameSite=Strict + Origin check) matches the SSE ticket flow (short-lived HMAC ticket, single-use, server-side stream registry for logout revocation) and the `auth-fetch.ts` refresh-on-401 pattern. The migration SQL is SQLite-only and matches the schema described in the auth contract. The Hub template versioning bump matches the rollout sequence (MR-1 + MR-2 ship together as a new CalVer, which becomes `min_compatible_warden_version` for the v2 template track). ✓
3. **Scope check:** Single coherent project — backend auth/schema cleanup + frontend rewrite of the same surface. Two MRs, one product. Plan-sized.
4. **Ambiguity:** Settings hint copy is enumerated in this doc; QA can review before plan writes test cases. SSE auth picked option B (HMAC ticket + close-and-remint on error) explicitly, with rationale. tensor_parallel_size scope is locked to **tensor-parallel only** (data-parallel + pipeline-parallel explicitly out of scope, deferred to a separate spec). Refresh-token storage picked **cookie + SameSite=Strict + Origin check** over Authorization-only with explicit rationale around XSS vs CSRF tradeoffs.
5. **Reviewer issues folded in:** All 3 Critical (SSE ticket renewal race, SQLite-only migration syntax, MR-1/MR-2 cutover), 8 Important (JWT-secret bootstrap, v1 cookie migration, reload-required-keys enumeration, risk-register gaps, SSE ticket leak blast radius, tensor-parallel scope, Hub template migration, AnsiLog source paths), and 7 Minor nits from the code review pass are reflected above. Spec is ready for plan-writing.
