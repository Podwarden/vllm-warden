# Engine Templates + Universal-Engine Delivery Plan & d5 E2E Runbook

**Status:** Draft (delivery plan + ops runbook)
**Date:** 2026-05-25
**Companion spec:** `2026-05-25-universal-engine-driver-design.md` (the driver architecture — engine-sidecar split, `EngineDriver` interface, DockerSocketDriver/DooD, K8sApiDriver, channels). This doc does NOT re-specify the drivers; it sequences the work, designs the user-facing **Template** feature, and defines the destructive E2E on host d5 (10.10.0.187).

---

## 1. Why this doc exists (scope honesty)

The originating request bundles three things that cannot share one MR or one session:

1. **Six architecture issues (#160–#165)** — multi-week, partly event-driven on upstream vLLM.
2. **A new user-facing "vLLM+CUDA template" feature + UI** — depends on (1) existing.
3. **A destructive, deployment-dependent E2E on d5** — depends on (1) and (2) being *built AND deployed* to d5, which today runs the OLD in-container subprocess supervisor.

Per the brainstorming skill, an oversized request is decomposed before designing. The decomposition and the autonomous decisions below are the design.

---

## 2. Locked autonomous decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | **Critical path = Docker path only.** Deliver #160 (engine-sidecar + DockerSocketDriver), #161-minimal (channel→image resolution), and the Template feature (folds #162's stack-resolver + failure-classifier). | d5 is a single Docker host with 2× A4000; the K8s/PodWarden drivers add zero E2E value here. |
| D2 | **Defer #163 (K8sApiDriver), #164 (PodWardenDriver), #165 (GGUF-sed removal).** Documented, not dropped. | #163/#164 are unexercised by the d5 E2E; #165 is event-driven on upstream vllm#38140 (OPEN). |
| D3 | **Extend the existing `ModelTemplate` concept; do NOT invent a separate "Engine Template" type.** Add an `engine` axis (vLLM version + CUDA/channel + engine image) to templates, and make templates **user-definable + DB-stored** alongside the built-in registry presets. | `app/templates/registry.py` already owns the "battle-tested combo users pick from a dropdown" concept. Adding a parallel type would collide and confuse. |
| D4 | **Use upstream `vllm/vllm-openai` tags directly via DooD first.** Our patched per-channel images come later (#161 full). | Unblocks the E2E without first building all six channel images. |
| D5 | **"qwen 3.6 25b" is treated as the available Qwen3.6 checkpoint on HF.** Exact repo pinned at bring-up and recorded in the saved template. | Likely a typo for the 27B already on d5 or a quant variant; the simplest safe reading is "the Qwen3.6 we can fetch." |
| D6 | **Irreversible steps (delete models on d5, push to main) are pre-authorized in the originating transcript.** Announce before executing; do not block. | User explicitly authorized; auth lives in the executing transcript. |

---

## 3. The Template feature (extends `ModelTemplate`)

### 3.1 Data model

`ModelTemplate` gains an **engine** axis. New optional fields (backward-compatible — existing built-ins keep working with defaults):

```python
@dataclass(frozen=True)
class EngineSpec:
    channel: str              # "cuda-stable" | "cuda-edge" | "cuda-legacy" | "rocm" | "xpu" | "cpu"
    vllm_version: str         # e.g. "0.20.0"
    image: str | None = None  # explicit override; else resolved from (channel, vllm_version)

@dataclass(frozen=True)
class ModelTemplate:
    ...                       # all existing fields unchanged
    engine: EngineSpec | None = None   # None => legacy in-container engine (current behaviour)
    source: str = "builtin"            # "builtin" | "user"
```

- **Built-in presets** (registry.py) — read-only, shipped with the image. Existing `_GPT_OSS_20B` gains an `engine=EngineSpec("cuda-stable","0.20.0")`.
- **User-defined templates** — stored in a new DB table `engine_templates` (id, label, json payload of the dataclass, `source='user'`, created_at). Surfaced by the same `list_templates()`/`get_template()` so the rest of the app is agnostic to origin.

### 3.2 Backend

- `app/templates/registry.py` — add `EngineSpec`, extend `ModelTemplate`.
- New `app/templates/store.py` — DB-backed CRUD for user templates; `list_templates()` merges builtin + user.
- New migration `NNN_engine_templates.sql` — `engine_templates` table.
- Resolver `app/templates/resolver.py` — `(channel, vllm_version) -> image` (the #161-minimal mapping; starts as a static dict of upstream tags).
- `app/models/routes_api.py` — extend create-model to accept a `template_id`; add `GET/POST/DELETE /api/templates` for user templates; add `POST /api/models/{id}/try-stack` (#162) that records a trial-and-error attempt and its outcome.
- Failure classifier `app/models/stack_classifier.py` — maps engine boot failures (CUDA arch unsupported, OOM, quant unsupported, version mismatch) to a suggested next combo.

### 3.3 UI

Single new screen **Templates** + an extension of the model-create flow:

- **Create-model form:** a "Template" dropdown (built-in + user) that prefills all knobs incl. the engine combo; "Custom" lets the user fill the engine axis manually.
- **Templates manager page:** list builtin (badge: prepared) + user templates; "Save current config as template" from a running/known-good model; delete user templates.
- **Try-stack panel** (#162): shows trial-and-error history for a model — each attempt's (vLLM version, channel, result) and the classifier's suggested next combo; one-click "save working combo as template".

---

## 4. Delivery phases (sequenced reversible → irreversible)

| Phase | Content | Issue | Reversible? | MR |
|-------|---------|-------|-------------|-----|
| **P1** | Engine-sidecar split + `EngineDriver` interface + `DockerSocketDriver` (DooD). Supervisor delegates spawn/observe/kill to the driver behind a flag; legacy in-container path remains default. | #160 | yes | feat→develop |
| **P2** | Channel→image resolver (`resolver.py`) using upstream tags; `EngineSpec` plumbed through cmd_builder/supervisor. | #161 (min) | yes | with P1 or follow-up |
| **P3** | Template feature: DB table + store + routes + UI; stack-classifier + try-stack route. | #162 | yes | feat→develop |
| **P4** | Build slim control-plane image + deploy to d5 via PodWarden REST (per `reference_vllm_warden_bonus_deploy`). | — | reversible (redeploy) | n/a |
| **P5** | **DESTRUCTIVE E2E on d5** — section 5. | — | NO | n/a |
| **P6** | MR develop→main + push; required main→develop back-merge sync (`process_squash_backmerge_required`). | — | NO | develop→main |

Each phase: code review before MR (don't self-trust), QA where testable, issue comment summarizing.

---

## 5. d5 E2E runbook (Phase 5 — destructive, pre-authorized)

**Host:** d5 = 10.10.0.187, root SSH via `~/.ssh/id_ed25519`. 2× A4000 (sm_86 Ampere, ~32 GB total VRAM, fp8 emulated). Drives the *deployed* Warden + DooD engine.

**Hardware reality that bounds the trials:** dense ≥25 B will not fit fp16 on 32 GB — expect to need AWQ/GPTQ/GGUF quant or TP=2 with reduced ctx. Ampere fp8 is emulated (nvidia-smi %util undercounts).

**Per-model loop** (gpt-oss-20b → nemotron → mistral → llama → qwen3.6):

1. Pull the model (HF), record repo + revision.
2. Trial-and-error the engine combo via `POST /api/models/{id}/try-stack`: pick a (channel, vllm_version) candidate; on boot failure the classifier suggests the next; iterate until the model **answers a prompt coherently** (the success gate — send 2–3 probes, eyeball the answers, not just HTTP 200).
3. On success: **save the working combo as a user Template** (label = model + combo).
4. Unload the model; delete its files from the HF cache to free VRAM/disk for the next.

**Pre-step (once):** delete the existing Qwen3.6-27B model row + its cached files on d5 to free space.

**Coherence gate is human-judged on the answer text** — a model that boots but emits garbage is a FAILED combo, per `feedback_vllm_warden_probe_not_ux` (a boot/probe success ≠ a working model).

---

## 6. Non-goals

- No K8s operator / CRDs (see architecture spec).
- No #163/#164/#165 in this effort (D2).
- No building all six channel images upfront (D4).
- No forking vLLM.

---

## 7. Risks

| Risk | Mitigation |
|------|-----------|
| New engine path not deployed to d5 → E2E tests old code | P4 gates P5; verify `/health` build_sha on d5 before any trial. |
| A4000 can't fit a given model at all | Classifier records "infeasible-on-hardware" as a terminal outcome; that model's template documents the constraint rather than blocking the run. |
| Destructive deletes hit the wrong files | Delete by HF cache path scoped to the exact repo id; list before `rm`. |
| develop→main squash conflict | Mandatory back-merge sync MR per `process_squash_backmerge_required`. |
