# vllm-warden 2026-05 Overhaul — Master Plan

**Status:** APPROVED v1 · **Owner:** CTO (info@protrener.com) · **Date:** 2026-05-22 · **Target window:** 2026-05-26 → 2026-07-10 (6–7 weeks, 9 MRs)

## Goal

Strip vllm-warden back to its core competency — running and operating vLLM workloads safely from a friendly UI — by (a) excising the in-product Bench v2 subsystem (~23k LOC, ~30% of the codebase, single largest source of code-review surface and pre-merge failures), (b) closing all 23 open `~bug` issues in coordinated thematic slices rather than per-issue MRs, and (c) executing a deliberate UX overhaul along the "model lifecycle" user pathway (find → pull → load → tune → use → measure → cache-cleanup) so the product reads as one coherent tool instead of a collection of disjoint admin pages. By the end of the epic, a brand-new operator should be able to add, load, and chat with a model without leaving the default tab, and a returning operator should see live GPU/VRAM in the header on every page, manage tokens with per-key rate limits and priorities, and trust that Stats v2 numbers reconcile with proxy reality.

## CTO-locked decisions (the 8 open questions resolved before S1 opens)

| # | Question | Decision |
|---|---|---|
| 1 | JWT re-login batching with S1? | **Moot.** `jwt_secret_v2` rename already merged on `feature/auth-multiuser-rbac` (commit `2911a76`); ships independently. S1 makes no further auth change. |
| 2 | Preset format — built-in JSON vs DB? | **Built-in JSON in repo for S4. Defer user-defined custom presets to a future slice.** |
| 3 | Priority scheduler semantics? | **STRICT.** "9 always first" verbatim — starvation possible, document the risk. |
| 4 | Header widget refresh rate? | **2s default, override via env var `VW_HEADER_METRICS_INTERVAL_S`.** |
| 5 | Chat playground token model? | **Auto-create `vw-playground` system token on first chat visit.** Inherits S5 rate-limit defaults. |
| 6 | Power sampling cadence? | **Match GPU util cadence — 5s collected, 60s bucketed.** Single NVML pass. |
| 7 | Stats v1 deprecation? | **Keep both for the S9 release. Delete v1 endpoints one release after S9.** |
| 8 | Worktree CI capacity for parallel S2/S5/S6? | **Start parallel; serialize on first pipeline backup.** Empirical check, easy fallback. |

## Bug triage — all 23 issues

Verdict column: **FIX-IN-SLICE** (closed by an MR in this epic), **DROPPED-WITH-BENCH** (only meaningful while bench exists; closes for free when bench is deleted), **CLOSE-WONTFIX** (operator question, not code; resolve with a doc/comment and close), **PRE-MERGE-FIX** (one-line; opportunistic fix in whichever slice touches that file).

| # | Title (1-line) | Verdict | Target slice |
|---|---|---|---|
| 11 | `mark_runtime_dead_on_startup` doesn't reset `pulling` rows; orphan pull-progress survives restart | FIX-IN-SLICE | S3 (model lifecycle hardening) |
| 14 | `bench pause` returns 204 on no-op; should be 200 with body or 409 | DROPPED-WITH-BENCH | S1 |
| 15 | `bench resume` 204 vs body/409 mirror of #14 | DROPPED-WITH-BENCH | S1 |
| 16 | `bench internal_routes` lacks loopback-only bind guard (defense-in-depth) | DROPPED-WITH-BENCH | S1 |
| 17 | bench `run_id = uuid4()` → `secrets.token_hex(16)` (unguessable callback secret) | DROPPED-WITH-BENCH | S1 |
| 20 | BenchTab "pinned run" doesn't survive route change | DROPPED-WITH-BENCH | S1 |
| 24 | CI lint jobs skip on `merge_request_event` pipelines | FIX-IN-SLICE | S1 (CI is touched by removing bench jobs anyway) |
| 25 | `integration-tests` job pre-existing failure | FIX-IN-SLICE | S1 |
| 29 | `model_runtime.health_ok` column declared but never written | FIX-IN-SLICE | S3 |
| 31 | GitLab runner wedges on `integration-tests` finalization | FIX-IN-SLICE | S1 (CI cleanup) |
| 39 | `/login` route-guard pattern matching too loose | FIX-IN-SLICE | S2 (auth/header refresh slice) |
| 46 | `VW_CONTAINER_GPU_COUNT=1` but host has 3 physical GPUs (operator question) | CLOSE-WONTFIX | n/a — doc note in S3 release notes |
| 47 | `docs/releasing.md` references missing `scripts/publish_to_hub.py` | PRE-MERGE-FIX | S1 (releasing docs are touched when bench is yanked) |
| 55 | Flaky JWT auth fixture | FIX-IN-SLICE | S2 |
| 67 | Frontend Dockerfile missing `ARG VW_BUILD_VERSION` / `VW_BUILD_SHA` | PRE-MERGE-FIX | S2 (nav-bar/version slice) |
| 83 | Nav-bar shows `vv2026.05.19.1` (double-v) | PRE-MERGE-FIX | S2 |
| 99 | `test_start_run_waits_for_health_ok` asserts stale `30s` (supervisor default now 600s) | FIX-IN-SLICE | S3 |
| 101 | Model discovery should flag non-vLLM-supported GGUF architectures | FIX-IN-SLICE | S4 (add-model UX) |
| 104 | Flaky `test_rotate_endpoint` sqlite/`tmp_data_dir` fixture race | FIX-IN-SLICE | S5 (tokens slice) |
| 105 | Delete model leaves orphan files in HF cache | FIX-IN-SLICE | S6 (cache page) — partially superseded by GC endpoint, but per-model delete still needs the "and free its cache" checkbox |
| 110 | `PATCH /models/:id` silently drops `hf_config_repo`/`tokenizer_repo` | FIX-IN-SLICE | S3 |
| 112 | File picker lists N rows for N-shard safetensors (UI) | FIX-IN-SLICE | S4 |
| 113 | Worker fails to load AWQ-INT4 27B on 2× A4000 | FIX-IN-SLICE | S3 (presets + better diagnostic; root-cause is OOM on KV — needs preset for tight-VRAM AWQ) |

**Bug coverage:** 23/23 (16 FIX-IN-SLICE, 5 DROPPED-WITH-BENCH, 1 CLOSE-WONTFIX, 1 PRE-MERGE-FIX overlaps two categories — counted once).

## Code review findings (NEW bugs surfaced during planning)

These are net-new defects identified while reading the codebase for the overhaul; each is bound to a slice rather than filed as a separate issue (to keep the epic self-contained):

1. **Settings PATCH allowlist drift (companion to #110).** `_PATCHABLE_MODEL_FIELDS` in `app/settings/routes_api.py` is a hardcoded `frozenset` that has drifted from the `ModelRow` schema four times historically. The fix should derive the allowlist from `ModelRow.__fields__` minus an explicit blocklist (`id`, `status`, `created_at`, etc.) so the next added column doesn't silently fail to PATCH. → **S3**.
2. **Proxy ↔ bench coupling via `envelope_hint.py`.** `app/proxy/routes.py::enrich_5xx_from_db` imports `app/proxy/envelope_hint.py` (176 LOC) which reads bench tables to enrich error responses with "last successful bench config was X". When bench is removed, the proxy must not regress to bare 500s — replace with a simpler "last-known-good extra_args from this model's own load history" hint sourced from `model_runtime` rows. → **S1** (must ship in the same MR as bench deletion or proxy 5xx UX regresses).
3. **Settings page imports `FromBenchmarkChip` and `BenchmarkBestResponse`.** `frontend/src/app/models/[id]/settings/page.tsx` pre-fills `max_model_len` and `gpu_memory_utilization` from bench best-of. After bench removal the pre-fill source disappears; replace with a "Suggest values" button that calls a new `/api/models/{id}/suggest-config` endpoint computing safe defaults from `gpu_samples.memory_total_mib` and model_config heuristics. → **S1** must stub the button (calls endpoint that returns "not implemented" with friendly message); **S3** implements the suggester.
4. **Supervisor state machine `mark_warming` / `mark_ready` orphaned API.** These were added for bench's load runner. Audit shows non-bench callers do use them (the main `POST /api/models/{id}/load` path), so they survive — but the docstring referencing bench needs scrubbing. → **S1**.
5. **Counter increments in proxy are unbounded in memory.** `app/proxy/routes.py` keeps an in-process Counter that's never flushed when models are unloaded. Low priority but visible in `/stats` accuracy. → **S7** (Stats v2).
6. **`/api/cache/models` is already shipped but unused by frontend route `/cache`.** The endpoint exists (379 LOC `routes_api.py`); the UI currently surfaces it inside the Stats Storage section only. Re-homing is purely frontend work — no backend churn needed. → **S6**.
7. **JWT `jwt_secret_v2` rename in Unreleased changelog** forces re-login on upgrade. The overhaul will ship multiple times during this epic; operators will be re-logged out at v1 only. Document in S1 release notes. → **S1 docs**.
8. **`extra_args` parser fragility (from changelog Qwen3-VL fix).** The recent fix forwards extra_args correctly, but there's no validation/preview UI showing the operator what argv vLLM will actually receive. → **S4 stretch** (add a read-only "Effective argv" panel in model settings).

## UX review findings (NEW UX issues)

Per-route, observed by reading current pages:

- **`/models` list:** No "add model" call-to-action above the fold; new users hit a blank table. → S4 hero CTA.
- **`/models/[id]`:** Overview + Benchmark tabs; once Benchmark is removed, the tab strip collapses to a single tab which looks broken. → S1 removes tab strip entirely; S3 reintroduces tabs as Overview · Logs · Effective config.
- **`/models/[id]/settings`:** 17 form fields, no visual grouping, hint text only present on some. `settings-hints.ts` already has the copy for all of them — just unused. → S4 wraps fields into Memory · Compute · Concurrency · Advanced sections each with collapsible (i) tooltips.
- **`/tokens`:** Plain table with no usage stats per row; rate/priority columns to be added. No "test this token" affordance (operators currently use `curl`). → S5.
- **`/stats`:** Polls every 5s including Storage section (30s) — fine. But range selector resets on tab switch (not persisted). Charts have no Y-axis units in some rendering paths. → S7.
- **No `/chat` route exists.** Operators currently smoke-test loaded models via `curl` to `/v1/chat/completions`. → S8.
- **Nav-bar has no live metrics.** Operators tab back-and-forth between page and /stats to know if a GPU is hot. → S2 header widget.
- **`/login` and `/setup`** hide nav-bar (correct), but show no product version → users don't know which build they're authenticating against during upgrades. → S2.
- **No "presets" anywhere.** Each model's settings start from cmd_builder defaults; no way to say "use my A4000-tight preset". → S4.

## Scope removed (Slice 1 — `bench` excision)

**Delete:**
- `app/bench/` (7,694 LOC)
- `app/db/repos/bench.py` (478 LOC)
- `app/db/sql/0012_bench.sql` (NOT deleted — superseded by new migration `0017_drop_bench.sql` that drops bench tables idempotently)
- `app/proxy/envelope_hint.py` (176 LOC) — replaced with 30-LOC stub reading `model_runtime.last_error`
- `frontend/src/components/bench/` (3,720 LOC)
- `frontend/src/app/models/[id]/page.tsx` Benchmark tab + imports
- `frontend/src/app/models/[id]/settings/page.tsx` FromBenchmarkChip usages
- Bench-related tests (~11,632 LOC)
- CI jobs: `bench-integration`, `bench-fuzz`
- Docs: `docs/bench/`, `docs/specs/2026-*-bench-*.md`

**Net delta:** ≈ −23,000 LOC source + −11,600 LOC tests. Wins: removes #14, #15, #16, #17, #20; collapses proxy coupling; halves migration count touchpoints; closes a CI flake source (#25, #31 in same hit because bench-integration was the wedging job).

**Migration rule:** `0017_drop_bench.sql` MUST `DROP TABLE IF EXISTS` (idempotent) so re-running against a fresh DB without bench history is a no-op.

## Scope added

| Capability | Where | Slice |
|---|---|---|
| Header live-metrics widget (VRAM% + GPU% + active-model name; SSE polled 2s, `VW_HEADER_METRICS_INTERVAL_S` overridable) | `frontend/src/components/header-metrics.tsx` + `/api/header/metrics` endpoint | S2 |
| Chart range persistence in localStorage | Stats page + (later) header widget | S7 |
| Standalone `/cache` page (re-home from /stats Storage) | `frontend/src/app/cache/page.tsx` + nav entry | S6 |
| Cache-aware "Delete model" (the "free cache too" checkbox) | Model row action; closes #105 properly | S6 |
| vLLM options panel with (i) hints exposed on every control | Settings page refactor; settings-hints.ts now consumed | S4 |
| "Suggest values" button → `/api/models/{id}/suggest-config` | Settings page + new endpoint | S3 (endpoint), S4 (button) |
| Model presets ("A4000-tight 27B AWQ", "H100 80GB single-shot", "Dev: tiny + fast", "MoE-balanced") — built-in JSON in repo, user-defined deferred | `app/presets/` module + Settings dropdown | S4 |
| Tokens v2: per-token `rate_limit_tps` + `priority` (0-9, **STRICT** — 9 always-first, starvation documented) | DB migration 0018 + Token form + proxy scheduler | S5 |
| Token usage stats per row | `/api/tokens/{id}/usage` + table column | S5 |
| Stats v2: per-API-key tokens, VRAM, power, GPU load historical+current, aggregate (power sampled 5s/60s bucketed) | New `power_samples` table + `/api/stats/v2/*` endpoints + redesigned page | S7 |
| Chat playground (session-only, no persistence; auto-creates `vw-playground` token on first visit) | `frontend/src/app/chat/page.tsx` + reuse existing `/v1/chat/completions` proxy | S8 |
| Model discovery quality: flag unsupported GGUF arches | `app/models/discovery.py` warnings list; surfaced in add-model modal | S4 (closes #101) |
| Add-model modal: shard-aware safetensors picker | `frontend/src/components/add-model-modal.tsx` collapse by shard family | S4 (closes #112) |

## Slice breakdown — 9 MRs

Each slice has: **branch** (worktree slug), **owner**, **scope**, **closes** (issues), **key files**, **risks**, **test plan**, **dependencies**, **LOC budget** (delta, capped <2000), **skill invocation note**.

All frontend-touching slices invoke the `frontend-design` skill at MR-open time per house policy.

---

### S1 — `bench-removal` (foundation)

- **Branch:** `epic/overhaul/01-bench-removal` · **Worktree:** `dev-1-overhaul-01-bench` · **Owner:** dev-1 · **LOC:** −23k source / −11.6k tests / +500 (new `envelope_hint` stub, migration, docs) → **net −34k, but +500 net-new** code budget.
- **Scope:** Delete bench subsystem end-to-end; add `0017_drop_bench.sql`; replace `envelope_hint.py` with `model_runtime.last_error` lookup (30 LOC); remove BenchTab from `models/[id]/page.tsx`; remove FromBenchmarkChip from settings page (replace with TODO comment for S4's preset button); strip CI bench jobs; update `docs/releasing.md` to drop `publish_to_hub.py` reference (#47); fix #24 by adding `merge_request_event` to lint rules; fix #25 / #31 by removing the wedging integration-tests job (bench-integration was the wedger).
- **Closes:** #14, #15, #16, #17, #20, #24, #25, #31, #47.
- **Key files:** `app/bench/` (delete), `app/db/repos/bench.py` (delete), `app/proxy/envelope_hint.py` (rewrite as stub), `app/proxy/routes.py` (update import), `app/db/sql/0017_drop_bench.sql` (new), `frontend/src/app/models/[id]/page.tsx` (collapse tabs), `frontend/src/app/models/[id]/settings/page.tsx` (drop bench imports), `.gitlab-ci.yml`, `docs/releasing.md`.
- **Risks:**
  - **HIGH:** Proxy 5xx UX regression if envelope_hint stub returns less useful messages. **Mitigation:** Snapshot tests of 5xx response bodies before/after.
  - **MED:** Settings page broken for in-flight branches that reference FromBenchmarkChip. **Mitigation:** Branch-protection check.
  - **LOW:** Operators with active bench runs lose them. **Mitigation:** Release note + `0017` migration logs row counts before drop.
- **Test plan:** Full test suite green; manual proxy 5xx smoke (load broken model, observe error); migration up/down test; `/models/[id]` renders without tabs cleanly.
- **Deps:** None (first MR).
- **Skill:** `frontend-design` not invoked (this slice only deletes UI, doesn't design new).

---

### S2 — `header-metrics` (live widget + auth fixes)

- **Branch:** `epic/overhaul/02-header-metrics` · **Worktree:** `dev-2-overhaul-02-header` · **Owner:** dev-2 · **LOC:** +900.
- **Scope:** Build `header-metrics.tsx` component (small VRAM% + GPU% + active-model badge, SSE 2s default, env-overridable per Q4); new `/api/header/metrics` SSE endpoint reading the same `gpu_samples` + `model_runtime` source as `/stats`; fix #83 (nav-bar double-v, single-char patch at line 64); fix #67 (Dockerfile ARG declarations); fix #39 (`/login` guard exact-match); fix #55 (JWT auth fixture race — use module-scoped fixture or barrier).
- **Closes:** #39, #55, #67, #83.
- **Key files:** `frontend/src/components/nav-bar.tsx` (line 64 fix + widget mount), `frontend/src/components/header-metrics.tsx` (new), `app/header/routes_api.py` (new, ~60 LOC), `frontend/Dockerfile` (ARG), `frontend/src/middleware.ts` or wherever `/login` guard lives, `tests/conftest.py` (JWT fixture).
- **Risks:**
  - **MED:** Adding SSE to every page increases server load; cap at one stream per browser via `EventSource` reuse hook.
  - **LOW:** Widget visible on `/login` and `/setup` would leak GPU info pre-auth — must respect nav-bar's existing hide-on-login logic.
- **Test plan:** Vitest for widget; Playwright login flow; manual: open 5 tabs, confirm only one SSE connection on the network panel.
- **Deps:** S1 merged (CI must be green for fast iteration; not a code dep).
- **Skill:** **`frontend-design`** invoked — widget design must match Stats chart palette and density.

---

### S3 — `model-lifecycle-hardening`

- **Branch:** `epic/overhaul/03-lifecycle` · **Worktree:** `dev-1-overhaul-03-lifecycle` · **Owner:** dev-1 · **LOC:** +1,200.
- **Scope:** Fix #11 (add `pulling` to `mark_runtime_dead_on_startup` WHERE clause + zero `pulled_bytes`/`pulled_total`); fix #29 (write `health_ok` column in `supervisor.wait_for_health` success path); fix #99 (update test assertion to current supervisor default of `600s` OR re-spec to use injected timeout — design decision: inject); fix #110 + companion (derive PATCH allowlist from `ModelRow.__fields__`); add `/api/models/{id}/suggest-config` endpoint (heuristic: total VRAM × 0.92 → `gpu_memory_utilization`; model_config max_position_embeddings → `max_model_len`; flag AWQ-INT4 models for `kv_cache_dtype=fp8` recommendation — addresses #113 root cause); add release note for #46 (`VW_CONTAINER_GPU_COUNT` operator guidance).
- **Closes:** #11, #29, #99, #110, #113, #46 (doc-only).
- **Key files:** `app/db/repos/models.py` (#11 fix), `app/runtime/supervisor.py` (#29 fix), `tests/test_supervisor.py` (#99), `app/settings/routes_api.py` (#110 derivation), `app/models/suggest.py` (new), `app/models/routes_api.py` (mount suggest endpoint), `docs/operating.md` (#46 note).
- **Risks:**
  - **MED:** Derived PATCH allowlist might let through fields we *want* to block (e.g. `status`). **Mitigation:** Explicit `_NEVER_PATCH` blocklist tested at startup.
  - **LOW:** suggest-config heuristic misfires on exotic models. **Mitigation:** "These are starting points" copy in UI; never auto-apply.
- **Test plan:** Restart-with-pulling-row unit test; `health_ok` written after warmup probe; suggest-config returns sane numbers for the 4 model archetypes (dense 7B, MoE 35B, AWQ 27B, GGUF tiny).
- **Deps:** S1 (settings page must not still import bench chip).
- **Skill:** N/A (backend-only).

---

### S4 — `add-and-tune-ux` (find/pull + presets + options panel)

- **Branch:** `epic/overhaul/04-add-tune-ux` · **Worktree:** `dev-2-overhaul-04-tune` · **Owner:** dev-2 (FE lead) + dev-1 BE support · **LOC:** +1,800.
- **Scope:** Hero CTA on `/models`; redesigned add-model modal (shard-aware safetensors grouping closes #112; GGUF arch warning closes #101); redesigned settings page with Memory/Compute/Concurrency/Advanced sections, every field surfaces its `MODEL_HINTS` entry via (i) tooltip; **presets**: ship 4 built-in presets (A4000-tight-AWQ, H100-single-shot, Dev-tiny, MoE-balanced) as a JSON file in `app/presets/builtin.json` + `GET /api/presets` + dropdown in settings; "Suggest values" button consumes S3's endpoint; "Effective argv" read-only panel.
- **Closes:** #101, #112.
- **Key files:** `frontend/src/app/models/page.tsx` (hero CTA), `frontend/src/components/add-model-modal.tsx` (shard grouping + arch warnings), `frontend/src/app/models/[id]/settings/page.tsx` (full restructure), `frontend/src/components/settings-section.tsx` (new), `app/presets/__init__.py`, `app/presets/builtin.json`, `app/presets/routes_api.py`, `app/models/discovery.py` (GGUF arch list).
- **Risks:**
  - **HIGH:** Settings page refactor is the largest UI change in the epic; risk of regressing the 409-on-loaded-edit flow. **Mitigation:** Playwright test covering "edit while loaded" path explicitly.
  - **MED:** Preset application UX: applying a preset to a loaded model needs explicit "this will require restart" warning.
  - **LOW:** Shard grouping algorithm needs to handle `model-00001-of-00007.safetensors` style — `SHARD_NAME_RE` already exists in `app/models/sharding.py`, reuse it client-side via a shared regex constant emitted in build.
- **Test plan:** Vitest for modal + settings; Playwright for add-load-tune happy path; preset application snapshot tests.
- **Deps:** S3 (suggest-config endpoint), S1 (no FromBenchmarkChip leftover).
- **Skill:** **`frontend-design`** — this slice IS the UX overhaul keystone.

---

### S5 — `tokens-v2` (rate-limit + priority + usage)

- **Branch:** `epic/overhaul/05-tokens-v2` · **Worktree:** `dev-1-overhaul-05-tokens` · **Owner:** dev-1 · **LOC:** +1,400.
- **Scope:** Migration `0018_tokens_rate_priority.sql` adds `rate_limit_tps INTEGER NULL` (NULL = unlimited) and `priority INTEGER NOT NULL DEFAULT 5 CHECK(priority BETWEEN 0 AND 9)`; extend `TokenCreate` Pydantic; proxy gains a per-token sliding-window rate limiter (10-second window, configurable) + a **STRICT priority-based scheduler queue** in front of vLLM (priority 9 always first, document starvation risk); per-token usage rollup `token_usage_minute` table + `GET /api/tokens/{id}/usage?range=24h`; UI: new columns Rate (tps), Priority, Last 24h tokens; "Test token" affordance issues a 1-token completion request; fix #104 (`tmp_data_dir` fixture race — use `tmp_path_factory` with session scope or proper isolation).
- **Closes:** #104.
- **Key files:** `app/db/sql/0018_tokens_rate_priority.sql`, `app/db/repos/tokens.py`, `app/tokens/routes_api.py`, `app/proxy/scheduler.py` (new), `app/proxy/routes.py` (call scheduler before forward), `frontend/src/app/tokens/page.tsx`, `frontend/src/components/token-create-modal.tsx`, `tests/test_token_rotation.py` (#104 fixture).
- **Risks:**
  - **HIGH:** Priority queue in proxy adds latency variance; needs benchmark before/after on baseline workload (P50/P99 to-first-token).
  - **MED:** Rate limiter must reject with `429` not `503` to match OpenAI client retry semantics.
  - **LOW:** Backfill of `priority=5` for existing tokens via migration default — verified safe.
- **Test plan:** Migration up/down; unit tests for scheduler ordering; integration test fires 10 priority-0 + 1 priority-9 request simultaneously and asserts priority-9 served first; rate-limit returns 429 after burst.
- **Deps:** S1 (clean proxy, no envelope_hint coupling).
- **Skill:** **`frontend-design`** for the token row redesign.

---

### S6 — `cache-page` (re-home + per-model delete-with-cache)

- **Branch:** `epic/overhaul/06-cache-page` · **Worktree:** `dev-2-overhaul-06-cache` · **Owner:** dev-2 · **LOC:** +700.
- **Scope:** New `/cache` route consuming existing `/api/cache/*` endpoints; nav entry; remove Storage section from `/stats`; add "Free cache too" checkbox to model delete confirmation (closes #105 properly — calls `DELETE /api/models/{id}` then `DELETE /api/cache/models/{repo}?force=true` if checked); cache page surfaces GC dry-run preview in a confirm modal.
- **Closes:** #105.
- **Key files:** `frontend/src/app/cache/page.tsx` (new, lifted from `/stats` Storage section), `frontend/src/app/stats/page.tsx` (remove Storage section), `frontend/src/components/nav-bar.tsx` (add Cache item), `frontend/src/components/delete-model-modal.tsx`.
- **Risks:**
  - **LOW:** Deleting a model row while its cache is being deleted async — order matters: delete row first, then cache (cache delete refuses if row active, but row is already gone).
  - **LOW:** Operators may not find Storage if no migration tip — add a one-time toast "Storage moved to Cache".
- **Test plan:** Vitest; Playwright happy path (delete model + cache); manual: GC dry-run shows correct preview.
- **Deps:** S7 (Stats page restructure happens after Storage removal; can be parallel if careful with conflicts — sequence is safer).
- **Skill:** **`frontend-design`**.

---

### S7 — `stats-v2` (VRAM, power, per-key tokens, history)

- **Branch:** `epic/overhaul/07-stats-v2` · **Worktree:** `dev-1-overhaul-07-stats` · **Owner:** dev-1 (BE) + dev-2 (FE) · **LOC:** +1,900.
- **Scope:** Migration `0019_power_samples.sql` (per-GPU watts, minute-bucketed); collector reads NVML power draw alongside util/mem (single pass, 5s collected / 60s bucketed per Q6); new `/api/stats/v2/overview` returns aggregated VRAM/power/tokens summary; per-API-key tokens via join on `token_usage_minute`; chart range persisted in `localStorage.statsRange`; Stats page rebuild: Top cards (current VRAM%, current power W, current TPS, active models) + 4 charts (GPU util, VRAM, power, tokens) + per-token leaderboard; cap counter memory bloat (code-review finding #5).
- **Closes:** none (pure feature).
- **Key files:** `app/db/sql/0019_power_samples.sql`, `app/collectors/gpu_collector.py` (add power), `app/stats/routes_api.py` (v2 endpoints), `frontend/src/app/stats/page.tsx` (rebuild), `frontend/src/components/stat-card.tsx` (new), `frontend/src/lib/use-persisted-range.ts` (new hook).
- **Risks:**
  - **MED:** NVML power query latency on some driver versions — wrap in `asyncio.to_thread`, cap at 100ms.
  - **MED:** v1 endpoints kept side-by-side until v2 proven, then deprecated one release after S9 (per Q7).
  - **LOW:** localStorage range key collides with header widget's own range — namespace under `vw.stats.range` vs `vw.header.range`.
- **Test plan:** Migration; collector unit test with mocked NVML; v2 endpoint snapshot tests; localStorage persistence test.
- **Deps:** S5 (per-key token data source); S6 (Storage section already gone from /stats).
- **Skill:** **`frontend-design`** — primary UX surface.

---

### S8 — `chat-playground`

- **Branch:** `epic/overhaul/08-chat-playground` · **Worktree:** `dev-2-overhaul-08-chat` · **Owner:** dev-2 · **LOC:** +1,100.
- **Scope:** New `/chat` route with model picker (only `loaded` models), token picker (auto-creates `vw-playground` system token on first visit per Q5; rate-limit inherits S5 defaults), streaming chat via existing `/v1/chat/completions` proxy with SSE; session-only history (in-memory React state, lost on refresh — explicitly NOT persisted per requirement); copy-message, regenerate-last, stop-streaming controls; temperature/max_tokens sliders.
- **Closes:** none.
- **Key files:** `frontend/src/app/chat/page.tsx`, `frontend/src/components/chat/message-list.tsx`, `frontend/src/components/chat/composer.tsx`, `frontend/src/lib/use-chat-stream.ts`, `frontend/src/components/nav-bar.tsx` (Chat nav item).
- **Risks:**
  - **MED:** SSE stream cancellation on tab close must not leak server-side request → AbortController + matching proxy cleanup; verify with `/api/admin/active-requests` (if not present, add lightweight diagnostic in S7).
  - **LOW:** Default token picker reveals secret? No — uses session JWT, not a bearer token in browser memory.
- **Test plan:** Vitest for composer; Playwright happy path streaming a short completion; abort mid-stream test.
- **Deps:** S5 (token integration for "playground token" concept); S2 (header widget shows live GPU during chat).
- **Skill:** **`frontend-design`**.

---

### S9 — `polish-and-release`

- **Branch:** `epic/overhaul/09-polish` · **Worktree:** `dev-1-overhaul-09-polish` · **Owner:** rotating · **LOC:** +400.
- **Scope:** Documentation pass (`docs/index.md`, `docs/quickstart.md`); changelog finalization; version bump; release notes covering JWT secret rotation + bench removal + migration path; integration test additions for the full new happy path (add → pull → load → chat → stats sees the request); fix any code-review fallout from S1-S8 reviews.
- **Closes:** none net-new.
- **Key files:** `docs/*`, `changelog.md`, `app/__init__.py` (version), `tests/integration/test_full_lifecycle.py`.
- **Risks:** Pure polish; risk is scope creep — strict <400 LOC cap.
- **Deps:** S1-S8 all merged.
- **Skill:** N/A.

---

## Sequencing diagram

```
S1 bench-removal ────┬─► S2 header-metrics ───┐
                     │                        │
                     ├─► S3 lifecycle ───► S4 add-tune-ux ─┐
                     │                                     │
                     ├─► S5 tokens-v2 ─────────────────────┼─► S7 stats-v2 ─► S9 polish
                     │                                     │
                     └─► S6 cache-page ────────────────────┘
                                                           │
                                            S8 chat ◄──────┘ (deps on S5+S2)
```

**Critical path:** S1 → S3 → S4 → S7 → S9 (6 MRs, 5–6 weeks).
**Parallelizable:** S2, S5, S6 can land in any order after S1.
**Last-in:** S8 chat depends on S2 (header) + S5 (tokens) being merged.

## Out of scope

- **Multi-tenancy / org accounts.** RBAC was just shipped (Unreleased); no further account model work.
- **Distributed serving / multi-node vLLM.** Single-host only.
- **Model fine-tuning / training.** vllm-warden is inference-only by charter.
- **External benchmarking tool.** Will become a separate repo (`vllm-warden-bench`) consuming the public API; not part of this epic.
- **Mobile responsive design.** Desktop-first; phone is a non-goal.
- **Audit log UI.** Operations are logged server-side; UI for browsing them is future work.
- **Cost dashboards / billing.** Out of charter.
- **Themes other than current.** Design system kept stable; only layout/density changes per slice.
- **Migrations 0001-0011 cleanup.** Risk-of-loss not worth the win; only 0012 dropped via 0017.

## Top 3 risks across the epic

1. **(S1) Proxy 5xx UX regression** when `envelope_hint.py` is replaced. Mitigation: snapshot-test 5xx responses pre/post; gate merge on green snapshots.
2. **(S4) Settings page refactor regression** of the "edit-while-loaded → 409" flow. Mitigation: Playwright test covers the path explicitly before refactor begins; refactor is field-grouping + tooltip wiring only, no semantics change.
3. **(S5) Priority scheduler latency variance** in the proxy hot path. Mitigation: P50/P99 to-first-token benchmark before/after on a baseline workload; rate limiter implemented in pure-Python sliding-window first, optimize only if benchmark regresses >5%.

## Tracking

- **Bug coverage:** 23/23 (16 fixed, 5 dropped, 1 doc-only, 1 pre-merge)
- **LOC sum:** S1 −34k delete / +500 new = net -33,500 (bench excision); S2..S9 +9,400 → epic net **−23,600 LOC**
- **Per-slice forward delta:** S2 +900, S3 +1,200, S4 +1,800, S5 +1,400, S6 +700, S7 +1,900, S8 +1,100, S9 +400 — all under 2,000 LOC cap.
