# Setup Wizard — Design Spec (#441)

**Status:** Design — pending implementation
**Issue:** [vllm-warden#441](https://git.mediablade.net/podwarden/apps/vllm-warden/-/issues/441)
**Date:** 2026-05-02
**Authors:** brainstorming session output

---

## 1. Problem

vllm-warden currently boots into an authenticated dashboard that assumes a populated `config.json`. There is no first-run path. Operators provisioning a fresh host must hand-author `config.json` (admin password hash, HF token, model spec, vLLM args) before the UI is usable. This blocks the "single-binary, paste-into-systemd" install story and pushes new users straight into the docs.

We need a guided 5-step wizard that runs at `/setup/step/{1..5}` when the binary detects an empty config, walks the operator through GPU detection → HF auth → model selection → vLLM tuning → launch, and atomically promotes the result into `config.json` once the first `/health` 200 lands. After promotion, `/setup/*` becomes inaccessible and the operator is redirected to `/login`.

The wizard must:
- Survive interrupts: closing the browser mid-step preserves work in `wizard-draft.json`.
- Single-flight launch: only one wizard goroutine ever runs per process.
- Live progress: SSE feed shows phase transitions and a 200-line log tail to the browser.
- Promote atomically: either `config.json` is written and `wizard-draft.json` is deleted, or neither side mutates.
- Block re-entry: once `admin_password_hash` is set in config, `/setup/*` returns 303 to `/login`.

## 2. Scope

**In scope:**
- 5-step wizard + welcome page at `/setup/step/{1..5}`.
- HTMX 2.0.4 + Alpine 3.14.8 + Tailwind CDN (no Node bundler).
- New packages `internal/wizard/` and `internal/auth/`.
- Atomic draft store (`wizard-draft.json`) mirroring the existing `config.Manager` write pattern.
- `progress.Bus` SSE multiplexer with snapshot-then-delta semantics, drop-oldest ring discipline (cap 64).
- `wizard.Launcher` in-memory single-flight bound to `context.Background()` (lesson from #9, commit `d6de76d`).
- bcrypt cost 12 password hashing.
- In-memory session store (7-day absolute / 24-hour idle TTL).
- Per-session double-submit CSRF token.
- Per-IP login rate limiter (5 attempts/minute).
- `trust_proxy_auth` config-flag-driven proxy header trust (no runtime auto-detect).
- `github.com/google/shlex` for `extra_args` parsing.
- Smoke-test runbook wiki page (`Runbooks/Smoke-Testing`).

**Out of scope:**
- Multi-admin / RBAC beyond single admin user.
- Audit logging.
- Telemetry.
- Orphan-draft sweep job (deferred — see §10).
- 70B-class model timeouts (deferred — see §10).
- OIDC / external IdP integration (deferred).

## 3. State model

The wizard goroutine has six observable phases. Each transition publishes a `phase` event on the `progress.Bus`; the terminal `done` event always fires on goroutine exit (LIFO defer guarantee — see §8.5).

| Phase | Trigger | Exit |
|-------|---------|------|
| `pending` | `Launcher.Start` returns | first defer registers |
| `fetching` | HF token + model files validated | all model files on disk |
| `starting` | vLLM subprocess spawn | first `/health` poll begins |
| `polling` | poll loop start | first 200 OR timeout |
| `promoting` | first 200 received | `configMgr.Save` returns |
| `healthy` | Save succeeded, `drafts.Delete()` attempted | `done` published with `{redirect:"/dashboard"}` |
| `failed` | panic, timeout, or Save error | `done` published with `{}` |

**Delete-failure handling:** if `drafts.Delete()` fails post-Save, the goroutine logs a warning and STILL transitions to `healthy` with the redirect payload. The completed config wins; orphan draft is benign and a dashboard banner sweeps later (§10 follow-up #10.1).

**`done` event payload contract:**
- Healthy: `{"redirect": "/dashboard"}`
- Failed/cancelled: `{}` (empty object, not absent)

## 4. Authentication

`/internal/auth/` is a new package. It is intentionally minimal and does NOT integrate with Keycloak — vllm-warden ships standalone.

**Password storage:** bcrypt cost 12, hash written to `config.admin.password_hash`. Plaintext never touches disk. `config.HasAdminPassword()` returns `len(hash) > 0` under read lock.

**Session store:** in-memory map keyed by random 32-byte session ID. 7-day absolute TTL (creation+7d), 24-hour idle TTL (last touch+24h). Background sweeper every 5 minutes reaps expired entries. Sessions DO NOT survive process restart — by design; reboot forces re-login.

**CSRF:** per-session double-submit token, generated on session create. Token accepted from `X-CSRF-Token` header OR `csrf_token` form field. Constant-time compare. Required on all state-changing routes (POST/PUT/DELETE/PATCH).

**Rate limit:** per-IP sliding window, 5 attempts/minute. IP source: `RemoteAddr` by default; if `config.trust_proxy_auth` is true, first hop from `X-Forwarded-For`.

**`next` param sanitization:** `/login?next=<path>` redirects post-login. Sanitization: must start with `/`, must not start with `//`, must not contain `:`. Failed sanitization → `/dashboard`.

**Middleware composition (chi):**
- Open group: `/login`, `/setup/*` (when no admin password), `/healthz`, static assets.
- Protected group: everything else, including `/dashboard`, `/api/v1/*` (except wizard endpoints).
- `wizardOnlyWhenUnsetup` middleware: if `HasAdminPassword()` returns true, `/setup/*` returns **303 See Other** to `/login`. 303 (not 301) so browsers don't cache the redirect against future config changes.

## 5. Backend surface

### 5.1 Routes

| Method | Path | Group | Purpose |
|--------|------|-------|---------|
| GET | `/setup/step/{n}` | open (when unsetup) | Render step partial. |
| POST | `/setup/step/{n}` | open (when unsetup) | Validate + persist draft fragment. |
| POST | `/api/v1/wizard/launch` | open (when unsetup) | Call `Launcher.Start`. |
| GET | `/api/v1/wizard/progress` | open (when unsetup) | SSE stream from `Bus.Subscribe`. |
| POST | `/api/v1/wizard/reset` | protected (auth+CSRF) | Delete draft, return 204. |
| GET | `/login` | open | Render login form. |
| POST | `/login` | open | Authenticate; rate-limited. |
| POST | `/logout` | protected | Destroy session. |

### 5.2 `Launcher.Start`

Single-flight. If a goroutine is already in flight, returns the existing token; never spawns a second. Goroutine bound to `context.Background()` (NOT request context — lesson from #9 / `d6de76d`).

### 5.3 `Bus.Subscribe`

Returns `(snapshot []Event, sub *Subscription)`. Snapshot is a synchronous slice copy of the internal ring buffer at subscribe time. `sub.C` is `chan Event` (cap 64) carrying ONLY post-subscribe deltas. Drop-oldest discipline: when `sub.C` is full, oldest queued event is discarded to make room (slow consumer sees "what's happening now").

### 5.4 `extra_args` parsing

Step 4's `extra_args` text field is parsed via `github.com/google/shlex` into `[]string`. Quoted strings honoured; backslash escapes honoured; unmatched quotes return validation error.

## 6. Frontend mechanics

HTMX 2.0.4 (CDN) + Alpine 3.14.8 (CDN) + Tailwind CDN. No Node bundler. All templates server-rendered Go HTML.

**Step navigation:** `hx-post` on form submit → server validates → returns the next step's body partial → swaps `#wizard-body`. Browser history via `hx-push-url`.

**Live validation:** Alpine `x-on:blur` for client-side hints; authoritative validation server-side on POST.

**SSE subscription (step 5):** Alpine sets up `EventSource('/api/v1/wizard/progress')` on launch. Three event types:
- `phase`: updates phase indicator.
- `log`: appends to the 200-line log tail (drop-oldest).
- `done`: closes the EventSource; if `redirect` present, `htmx.ajax('GET', redirect)`; else show retry button.

**Browser refresh resilience:** `wizard-draft.json` is reloaded on every GET `/setup/step/{n}` so a refresh re-populates form fields.

## 7. Per-step contracts

### Step 1 — GPU detection

Server enumerates GPUs via `nvidia-smi --query-gpu=...`. Each GPU rendered with a `fit_badge` partial (green/yellow/red based on VRAM thresholds). Operator selects target GPU(s) for tensor-parallel split. Persisted to draft as `gpu_indices: []int`.

### Step 2 — HuggingFace token

Operator pastes HF token. "Test connection" button fires `POST /setup/step/2/test` → server calls HF whoami API → returns username on success or error band on failure. Validated token stored in draft (plaintext in draft is acceptable; promote writes it to `config.json` with file mode 0600).

### Step 3 — Model selection

Operator enters repo (e.g. `meta-llama/Llama-3.1-8B-Instruct`) and revision (defaults to `"main"`, matches `cmd/sidekick/main.go`'s hardcoded literal — §10 follow-up #10.11). Server lists files via HF API; per-file fit badge based on selected GPU VRAM. Selected files persisted to draft.

### Step 4 — vLLM args

Form fields: `max-model-len`, `dtype`, `tensor-parallel-size`, `extra_args` (free-text, shlex-parsed). Live-validate on blur. Persisted to draft.

### Step 5 — Launch

Review pane summarises all draft fields. Launch button → `POST /api/v1/wizard/launch` → SSE subscription begins → log tail + phase indicator render. On `done {redirect:"/dashboard"}`: HTMX navigates. On `done {}`: retry button + error band.

## 8. Step 5 deep dive

### 8.1 Atomic draft write

`DraftStore.Save` mirrors `config.Manager.Save` (see `internal/config/config.go:267-334`):

1. `os.CreateTemp(dir, "wizard-draft.*.tmp")`
2. `f.Chmod(0600)`
3. `f.Write(json)`
4. `f.Sync()`
5. `f.Close()`
6. `os.Rename(tmp, target)`

All under a `sync.Mutex` held for the full sequence. Schema version 1 (`{"version":1, ...}`); future schema bumps reject older drafts at boot.

### 8.2 `done` event payload contract

Restated for clarity:

| Path | `done` payload |
|------|----------------|
| Healthy promote (Save OK, Delete OK) | `{"redirect": "/dashboard"}` |
| Healthy promote (Save OK, Delete FAILED) | `{"redirect": "/dashboard"}` (Delete-failure is benign; logged as warning; orphan draft swept later) |
| Save error | `{}` |
| Health timeout (10min, no /health 200) | `{}` |
| Panic in poll loop | `{}` |
| `Launcher.Cancel` invoked | `{}` |

`done` always fires (LIFO defer guarantee, §8.5).

### 8.3 Promote sequence

```
build cfg from draft     → pure function, no side effects
configMgr.Save(cfg)      → atomic write of config.json
if err != nil → failed
drafts.Delete(ctx)       → if err: logger.Warn("orphan draft", err); continue
publish phase=healthy
publish done {redirect:"/dashboard"}
```

### 8.4 `progress.Bus` mechanics

```go
type Subscription struct {
    C    chan Event
    done chan struct{}
}

func (b *Bus) Subscribe() (snapshot []Event, sub *Subscription) {
    b.mu.Lock()
    defer b.mu.Unlock()
    snapshot = append([]Event(nil), b.ring...)   // synchronous slice copy
    sub = &Subscription{C: make(chan Event, 64), done: make(chan struct{})}
    b.subs = append(b.subs, sub)
    return
}

func (b *Bus) Publish(e Event) {
    b.mu.Lock()
    defer b.mu.Unlock()
    b.ring = appendRing(b.ring, e, RingCap)
    for _, s := range b.subs {
        select {
        case s.C <- e:
        default:
            // drop-oldest: pop one, push new
            select { case <-s.C: default: }
            select { case s.C <- e: default: }
        }
    }
}
```

Snapshot is returned as a synchronous slice; channel only carries post-subscribe deltas. No deadlock risk between snapshot and channel.

### 8.5 Defer ordering

Goroutine entry:

```go
go func() {
    defer publishDone()           // registered FIRST
    defer recoverAndMarkFailed()  // registered SECOND

    // ... poll, promote, etc.
}()
```

LIFO execution: `recoverAndMarkFailed` runs first (catches panic, marks state failed), THEN `publishDone` fires the terminal event. This guarantees `done` always fires regardless of panic, with the correct payload (failed → `{}`, healthy → `{redirect:"/dashboard"}`).

## 9. Testing

### 9.1 `internal/wizard/bus_test.go`

- `TestBus_SnapshotThenDelta` — publish 3 events, subscribe, assert snapshot = `[e1,e2,e3]`, channel empty. Publish e4, assert channel receives e4 only.
- `TestBus_ChannelCapDrop` — subscribe with slow consumer (no reads). Publish 200 events. Drain channel. Assert ordinals 137-200 received (drop-oldest: oldest 136 evicted, newest 64 retained).

### 9.2 `internal/wizard/launcher_test.go`

- `TestLauncher_HealthyPromote` — fakehf returns files, fakevllm /health 200 after 3 polls, configMgr.Save succeeds, drafts.Delete succeeds → state transitions through `polling → promoting → healthy`, `done` payload `{redirect:"/dashboard"}`.
- `TestLauncher_DeleteFailureBenign` — same as above but `drafts.Delete` returns error → state STILL `healthy`, `done` payload STILL `{redirect:"/dashboard"}`, log entry contains "orphan draft" warning.
- `TestLauncher_PanicRecover` — fakevllm panics during poll → recover catches → state `failed` → `done` payload `{}`.
- `TestLauncher_HealthTimeout` — fakevllm never returns 200, fast-clock advances 10min → state `failed` → `done` payload `{}`.
- `TestLauncher_KillWindow` — context cancelled mid-poll (via `Launcher.Cancel`) → state `failed` → `done` payload `{}`.

### 9.3 `internal/wizard/promote_test.go`

- `TestPromote_BuildConfigFromDraft` — golden draft → expected `config.Config` struct equality.
- `TestPromote_SavePropagatesError` — configMgr.Save returns sentinel error → `Promote` returns same error unchanged.

### 9.4 `internal/wizard/store_test.go`

- `TestDraftStore_Save_AtomicTempRename` — uses directory-listing pattern (per `internal/config/config_test.go:133-162` — verified no `WithSaveHook` exists in `config.Manager` and none required here either). Save a draft, list dir, assert ONLY `wizard-draft.json` present (no `*.tmp` orphans).
- `TestDraftStore_Load_MissingReturnsNil` — Load on empty dir returns `(nil, nil)`, not error.
- `TestDraftStore_Load_BadJSON` — Load on corrupt file returns error.
- `TestDraftStore_Delete_Idempotent` — Delete on missing file returns nil.

### 9.5 `internal/auth/*_test.go`

- `TestRateLimit_FifthAttemptBlocked`, `TestRateLimit_WindowSlides`, `TestRateLimit_PerIPIsolation`.
- `TestCSRF_HeaderAccepted`, `TestCSRF_FormFieldAccepted`, `TestCSRF_WrongTokenRejected`, `TestCSRF_ConstantTimeCompare`.
- `TestLogin_Success`, `TestLogin_BadPassword`, `TestLogin_RateLimited`, `TestLogin_NextParamSanitized`, `TestLogout_DestroysSession`.

### 9.6 `internal/web/wizard_integration_test.go`

End-to-end: empty config → GET `/` → 303 to `/setup/step/1` → POST through 1-5 → POST `/api/v1/wizard/launch` → SSE subscriber observes phase events and `done {redirect:"/dashboard"}` → GET `/` now returns 303 to `/login`. Uses `launchertest/fakevllm.go` for deterministic /health timing.

### 9.7 Smoke-test runbook

New wiki page `Runbooks/Smoke-Testing` (verified empty via `glab api projects/podwarden%2Fapps%2Fvllm-warden/wikis`). Created via GitLab API as part of #441's MR-companion deliverables. Covers manual smoke procedure on a real GPU host: empty config → wizard walkthrough → first `/health` 200 → `/dashboard` reachable → /login flow.

## 10. Known follow-ups (out of scope for #441)

These are tracked separately and intentionally NOT addressed in this MR:

| ID | Theme | Scope |
|----|-------|-------|
| 10.1 | Orphan draft sweep | Background job that detects `wizard-draft.json` present alongside `admin_password_hash` set; renders dashboard banner offering one-click delete. |
| 10.2 | Stale session token revocation | When admin password is changed (future feature), invalidate existing sessions. |
| 10.3 | Full-draft wipe endpoint | Operator-facing "start over" button that deletes draft and returns to step 1. |
| 10.4 | 70B-class model timeout tuning | Current 10min `/health` timeout may not suffice for 70B+ models on slow hosts; needs adaptive timeout based on model size. |
| 10.5 | Multi-admin / RBAC | Currently single admin user; future may want operator/viewer roles. |
| 10.6 | Audit log | Login attempts, config changes, wizard launches — log to file or external sink. (Not commit to scope yet; may live as separate concern.) |
| 10.7 | POST `/api/v1/auth/change-password` | Self-service password change once logged in. (POST, not PUT — REST convention for sensitive state-mutating operations on the current session subject.) |
| 10.8 | Browser-side resume of in-progress launch | If operator refreshes during step 5 launch, currently re-subscribes to SSE; should also reconstruct phase indicator state from snapshot. |
| 10.9 | OIDC / external IdP | Future integration with Keycloak so vllm-warden can join the imi realm. |
| 10.10 | Telemetry | Anonymous usage metrics (opt-in). |
| 10.11 | Remove hardcoded `"main"` revision | `cmd/sidekick/main.go` calls `store.GetModelByRepoRevision(ctx, repo, "main")` with literal; should read revision from config. |

## 11. Edit list

The exhaustive enumeration of files created/modified for #441, grouped by package, with a one-line purpose for each. A developer should be able to scan this section, open each file, and know what they're touching and why.

### `internal/wizard/` (new package)

| File | Status | Purpose |
|------|--------|---------|
| `internal/wizard/constants.go` | new | Package-level constants: `LaunchHealthTimeout = 10*time.Minute`, `LaunchPollInterval = 2*time.Second`, `RingCap = 64`, `LogTailLines = 200`. |
| `internal/wizard/drafts.go` | new | `DraftStore` with `Load(ctx) (*Draft, error)`, `Save(ctx, *Draft) error`, `Delete(ctx) error`. Mirrors `config.Manager` atomic write: CreateTemp → Chmod 0600 → Write → Sync → Close → Rename, all under a `sync.Mutex`. Path: `<dataDir>/wizard-draft.json`. Schema version 1. |
| `internal/wizard/bus.go` | new | `progress.Bus` SSE multiplexer. `Subscribe() (snapshot []Event, sub *Subscription)` returns the snapshot synchronously; `sub.C` is a `chan Event` (cap 64) carrying live deltas only. Drop-oldest ring discipline: when full, the oldest queued event is dropped to make room. `Publish(Event)` appends to internal ring + fans out to all subscribers. |
| `internal/wizard/launcher.go` | new | `Launcher` in-memory single-flight. `Start(cfg LaunchConfig) (token string, err error)` spawns one goroutine bound to `context.Background()` (lesson from #9 / `d6de76d`: never use request-context). Goroutine ordering: `defer publishDone()` registered FIRST, `defer recoverAndMarkFailed()` registered SECOND (LIFO so recover runs first on panic, marks state failed, then publishDone fires the terminal event with `{}` payload). Healthy path: poll first `/health` 200 → `configMgr.Save(cfg)` → `if err := drafts.Delete(ctx); err != nil { logger.Warn(...) }` (Delete-failure is benign, no recover, goroutine still transitions to healthy and publishes `done` with `{redirect: "/dashboard"}`). |
| `internal/wizard/promote.go` | new | Atomic promote helper called by `launcher.go`: builds `config.Config` from `Draft`, calls `configMgr.Save`, returns error. Pure function, no goroutine state. Kept separate so `promote_test.go` can exercise the build-cfg path without spinning a launcher. |
| `internal/wizard/bus_test.go` | new | `TestBus_SnapshotThenDelta` (snapshot returned synchronously, channel only carries post-subscribe deltas), `TestBus_ChannelCapDrop` (publish 200 events, slow subscriber receives ordinals 137-200 — drop-oldest verified). |
| `internal/wizard/launcher_test.go` | new | `TestLauncher_HealthyPromote` (first-200 → Save → Delete-success → done `{redirect:"/dashboard"}`), `TestLauncher_DeleteFailureBenign` (Delete returns error → state still healthy, `done` payload still has redirect, log line emitted), `TestLauncher_PanicRecover` (poll panics → recover marks failed → `done` payload `{}`), `TestLauncher_HealthTimeout` (no /health 200 in 10min → failed → `done` `{}`), `TestLauncher_KillWindow` (cancel during poll → failed → `done` `{}`). |
| `internal/wizard/promote_test.go` | new | `TestPromote_BuildConfigFromDraft` (golden draft → expected `config.Config`), `TestPromote_SavePropagatesError` (configMgr.Save returns error → bubbles up unchanged). |
| `internal/wizard/store_test.go` | new | `TestDraftStore_Save_AtomicTempRename` uses directory-listing pattern (per `config_test.go:133-162` — verified no test-hook exists in `config.Manager`, same pattern applies here). Asserts only `wizard-draft.json` present post-Save (no `*.tmp` orphans). `TestDraftStore_Load_MissingReturnsNil`, `TestDraftStore_Load_BadJSON`, `TestDraftStore_Delete_Idempotent`. |
| `internal/wizard/launchertest/fakehf.go` | new | Test-only fake HF fetcher: implements the model-fetch interface, returns deterministic file lists, programmable failure mode. |
| `internal/wizard/launchertest/fakevllm.go` | new | Test-only fake vLLM process: serves `/health` after a configurable delay, programmable to never-200, panic, or stay-healthy. |

### `internal/auth/` (new package)

| File | Status | Purpose |
|------|--------|---------|
| `internal/auth/bcrypt.go` | new | `Hash(password string) (string, error)` and `Verify(hash, password string) bool` — wrappers around `golang.org/x/crypto/bcrypt` at cost 12. |
| `internal/auth/session.go` | new | In-memory session store: `Create(userID) (id string)`, `Get(id) (*Session, bool)`, `Touch(id)`, `Destroy(id)`. 7-day absolute TTL, 24-hour idle TTL. Background sweeper every 5min. |
| `internal/auth/csrf.go` | new | Per-session double-submit token: `IssueToken(sessionID) string`, `Validate(sessionID, presented string) bool` (constant-time compare). Token accepted from `X-CSRF-Token` header OR form field `csrf_token`. |
| `internal/auth/ratelimit.go` | new | Per-IP login limiter: 5 attempts/minute, sliding window, in-memory map keyed by `RemoteAddr` (or first hop from `X-Forwarded-For` if `config.trust_proxy_auth` is on). |
| `internal/auth/middleware.go` | new | `RequireAuth(next http.Handler)` redirects to `/login?next=<original>` if no session. `RequireCSRF(next)` for state-changing routes. `LoadSession(next)` populates request context. |
| `internal/auth/handlers.go` | new | `HandleLogin(w, r)` (POST `/login`: rate-limit check → bcrypt verify → session create → 302 to `next` or `/dashboard`), `HandleLogout(w, r)` (POST `/logout`: destroy session → 302 to `/login`). |
| `internal/auth/ratelimit_test.go` | new | `TestRateLimit_FifthAttemptBlocked`, `TestRateLimit_WindowSlides`, `TestRateLimit_PerIPIsolation`. |
| `internal/auth/csrf_test.go` | new | `TestCSRF_HeaderAccepted`, `TestCSRF_FormFieldAccepted`, `TestCSRF_WrongTokenRejected`, `TestCSRF_ConstantTimeCompare` (timing parity smoke). |
| `internal/auth/handlers_test.go` | new | `TestLogin_Success`, `TestLogin_BadPassword`, `TestLogin_RateLimited`, `TestLogin_NextParamSanitized` (open-redirect defence), `TestLogout_DestroysSession`. |

### `internal/web/` (modified)

| File | Status | Purpose |
|------|--------|---------|
| `internal/web/server.go` | modified | Add `WithWizard(launcher *wizard.Launcher, drafts *wizard.DraftStore, bus *wizard.Bus) Option` and `WithAuth(authmw *auth.Middleware, sessions *auth.SessionStore, csrf *auth.CSRF) Option`. Wire into chi router as middleware groups: open group (`/login`, `/setup/*` when no admin password set, `/healthz`) and protected group (everything else). Mount wizard and auth route groups in chi BEFORE the existing vLLM API proxy handler so wizard/auth prefix matches resolve locally first. No new proxy file required. |
| `internal/web/wizard_routes.go` | new | Mounts wizard routes: `GET /setup/step/{1..5}` (renders step partial), `POST /setup/step/{n}` (validates + persists draft fragment), `POST /api/v1/wizard/launch` (calls `Launcher.Start`), `GET /api/v1/wizard/progress` (SSE, calls `Bus.Subscribe`), `POST /api/v1/wizard/reset` (auth+CSRF, deletes draft, 204). |
| `internal/web/auth_routes.go` | new | Mounts auth routes: `GET /login` (renders login.html), `POST /login` (delegates to `auth.HandleLogin`), `POST /logout` (delegates to `auth.HandleLogout`). |
| `internal/web/middleware.go` | new | Glue: `wizardOnlyWhenUnsetup` middleware (303s `/setup/*` to `/login` once `config.HasAdminPassword()` returns true; 303 not 301 so browsers don't cache against future config changes), `csrfForStateChanges` wrapper, `trustProxyAuthOnce`: caches the `config.trust_proxy_auth` boolean at first request via `sync.Once` so the proxy-header decision avoids per-request config-mutex contention. No runtime auto-detection — config flag is authoritative. |
| `internal/web/templates.go` (or wherever loader lives) | modified | Replace single-level glob with multi-pattern call: `template.ParseFS(templatesFS, "templates/*.html", "templates/auth/*.html", "templates/setup/*.html", "templates/partials/*.html")`. (Go's `embed.FS` glob is `path.Match`-based and does not honour `**`.) Required so `templates/setup/*.html` and `templates/auth/*.html` are discovered. |
| `internal/web/wizard_integration_test.go` | new | End-to-end test: empty config → GET `/` redirects to `/setup/step/1` → POST through steps 1-5 → POST `/api/v1/wizard/launch` → SSE subscriber receives `phase` events and `done {redirect:"/dashboard"}` → GET `/` now redirects to `/login`. Uses `launchertest/fakevllm.go` for deterministic /health timing. |

### `cmd/sidekick/main.go` (modified)

| File | Status | Purpose |
|------|--------|---------|
| `cmd/sidekick/main.go` | modified | Root precedence: load `config.json`; if `HasAdminPassword()` is false, mount wizard routes and skip protected-route middleware on `/setup/*`. Construct singletons once: `auth.SessionStore`, `auth.CSRF`, `auth.Middleware`, `wizard.DraftStore` (loads existing draft at boot), `wizard.Bus`, `wizard.Launcher`. Pass into `web.Server` via `WithAuth` + `WithWizard` options. Hardcoded `"main"` revision at `store.GetModelByRepoRevision(ctx, repo, "main")` stays (matches draft revision); §10 follow-up #10.11 tracks removal. |

### `internal/config/` (modified)

| File | Status | Purpose |
|------|--------|---------|
| `internal/config/manager.go` | modified | Add `HasAdminPassword() bool` helper: returns `m.cfg.Admin.PasswordHash != ""` under read lock. No test hook added — `TestDraftStore_Save_AtomicTempRename` uses directory-listing pattern (per `config_test.go:133-162`), confirmed no `WithSaveHook` exists in `Manager` and none required. |

### `go.mod` / `go.sum` (modified)

| File | Status | Purpose |
|------|--------|---------|
| `go.mod` | modified | `+ github.com/google/shlex` (parses `extra_args` in step 4). `+ golang.org/x/crypto` (already transitive, declared explicit for `bcrypt`). |
| `go.sum` | modified | Pinned hashes for the above. |

### Templates (new)

| File | Status | Purpose |
|------|--------|---------|
| `web/templates/auth/login.html` | new | Full-page login form (Tailwind CDN, Alpine for client-side validation, CSRF token hidden field). |
| `web/templates/setup/step.html` | new | Wizard chrome: 5-step progress bar partial, step body slot, prev/next button row, error band slot. Renders the step number's partial via `{{ template "step1_gpu" . }}` etc. |
| `web/templates/setup/step1_gpu.html` | new | Step 1 body: GPU detection results, "fit badge" partial per detected GPU, continue button. |
| `web/templates/setup/step2_hf.html` | new | Step 2 body: HF token input, "test connection" htmx button, masked-token display once validated. |
| `web/templates/setup/step3_model.html` | new | Step 3 body: model repo input, revision (defaults to `main`), per-file fit badges as files are listed. |
| `web/templates/setup/step4_config.html` | new | Step 4 body: form for vLLM args (max-model-len, dtype, tensor-parallel, extra_args via shlex). Live-validate on blur. |
| `web/templates/setup/step5_launch.html` | new | Step 5 body: review pane + launch button. On submit → POST `/api/v1/wizard/launch` → Alpine subscribes to SSE → renders log tail + phase indicator → on `done {redirect}` triggers `htmx.ajax('GET', redirect)`. |
| `web/templates/partials/fit_badge.html` | new | Reusable green/yellow/red fit indicator. |
| `web/templates/partials/progress_bar.html` | new | 5-step header progress bar. |
| `web/templates/partials/error_band.html` | new | Top-of-step inline error display, dismissible via Alpine. |

### Documentation

| File | Status | Purpose |
|------|--------|---------|
| `docs/superpowers/specs/2026-05-02-441-setup-wizard-design.md` | new | The consolidated spec produced by this brainstorming session. Lives in vllm-warden checkout. |
| Wiki page `Runbooks/Smoke-Testing` | new (off-tree) | Smoke-test runbook for the wizard. Created via `glab api projects/podwarden%2Fapps%2Fvllm-warden/wikis -X POST` (verified empty wiki). NOT a git commit — landed via GitLab API as part of #441's MR-companion deliverables. |

### Implementation order

1. **`internal/auth/`** lands first. Bcrypt, session store, CSRF, rate limiter, middleware, handlers — wire `RequireAuth` so the protected chi group has something real to gate against. Without auth in place, wizard's `/api/v1/wizard/reset` (auth+CSRF) has nothing to attach to.
2. **`internal/wizard/`** next. DraftStore (atomic write mirror), Bus (snapshot+delta), Launcher (single-flight goroutine with defer ordering and Delete-failure tolerance), promote helper. Each file gets its test in the same commit; `launchertest/` fakes shared across them.
3. **`internal/web/`** route wiring + templates. Add `WithWizard`/`WithAuth` options to `server.go`, mount routes via new `wizard_routes.go` / `auth_routes.go`, switch template loader to multi-pattern call, drop in all 9 templates. `wizard_integration_test.go` is the gate that proves everything composes.
4. **`cmd/sidekick/main.go`** integration. Construct singletons, wire them via `web.Server` options, add `config.HasAdminPassword()` precedence at boot. Smallest diff of any package — should be a clean composition step.
5. **Documentation last.** Spec doc is committed alongside the feature branch (already part of this brainstorming output). Wiki page lands via `glab api` post-MR-merge so the runbook references the merged code.
