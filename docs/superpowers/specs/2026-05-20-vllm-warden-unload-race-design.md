# vllm-warden — Eliminate the unload-during-warmup race

**Status:** Draft
**Date:** 2026-05-20
**Author:** ip + Claude (Opus 4.7)
**Related incident:** 2026-05-20 Qwen3.6-35B-A3B-AWQ-4bit (Qwen3-VL) crash loop on warden 10.10.0.187 — vLLM subprocess SIGTERMed during `_warmup_mm_processor`, leaving GPUs stuck and DB state inconsistent.

## Background

The supervisor's `unload()` method (`app/runtime/supervisor.py:144`) sends `os.killpg(pid, SIGTERM)` unconditionally. vLLM's `signal_handler` (`api_server.py:564`) catches SIGTERM and raises `KeyboardInterrupt("terminated")`, which aborts the subprocess. If unload is invoked while the subprocess is still in startup — specifically, during Qwen3-VL's `_warmup_mm_processor` phase — the subprocess dies mid-warmup and the model never becomes serviceable.

### Why the existing guards do not catch this

The HTTP unload route (`app/models/routes_api.py:687`) does check `model.status in ("loaded", "failed")` and 409s on `loading`. But there are three concrete paths that bypass or invalidate that guard:

1. **`/health` returns 200 before warmup completes.** vLLM's engine reports healthy once core init is done. Qwen3-VL's multimodal warmup (`_warmup_mm_processor`) runs after `/health` goes green. The load runner (`routes_api.py:668-677`) flips status to `loaded` as soon as `wait_for_health` returns, opening an ~8-second window where the DB says "loaded", the UI shows an unload button, but a click of that button SIGTERMs an actively-warming subprocess.

2. **Load-route health-timeout self-unload.** `routes_api.py:680` calls `sup.unload()` itself when `wait_for_health` times out, without checking whether the subprocess might still be in slow warmup. With a heavy multimodal model + cold HF Hub revalidation, warmup can exceed the default 600s.

3. **Supervisor accepts unload from any caller.** Bench supervisor, future internal cleanup code, and route-level callers all share the same unguarded `sup.unload()`. The DB-status check is a route-level guard, not a supervisor-level invariant.

### Crash timeline (2026-05-20)

| Time | Event |
|---|---|
| 02:52:47 | vLLM subprocess PID 28635 spawned |
| 02:54:47 | Weights loaded (79.34s, local from HF cache) |
| 02:55:08 | KV cache allocated |
| 02:55:19 | CUDA graphs compiled |
| 02:55:31 | `/health` returned 200; load runner flipped DB row to `loaded` |
| 02:55:34 | Chat template detected — multimodal warmup begins |
| 02:55:39 | `_warmup_mm_processor` running — SIGTERM caught, `KeyboardInterrupt("terminated")` raised |

The 8-second gap between `loaded` status and SIGTERM is the race window.

## Goal

Eliminate every code path that can send SIGTERM to a vLLM subprocess that is not in a fully-warmed serving state, except through an explicit operator force-unload.

## Non-goals

- Patching vLLM's signal handler. We do not modify vLLM; we modify when we send signals.
- Changing the model-row DB schema. The fix lives in the supervisor and the load/unload routes.
- Process adoption across api-container restarts. vLLM children die with the api container (PID 1 = api process); orphan reconciliation is not needed.

## Design

### Supervisor as source of truth

Add a per-model lifecycle state inside `Supervisor`:

```python
from enum import Enum

class ModelState(str, Enum):
    LOADING = "loading"      # subprocess spawned, /health not yet 200
    WARMING = "warming"      # /health green, warmup probe not yet succeeded
    READY = "ready"          # warmup probe succeeded, safe to unload
    UNLOADING = "unloading"  # SIGTERM sent, awaiting exit

class UnloadRefused(Exception):
    """Raised when unload() is called on a model not in READY state without force=True."""
```

`Supervisor` gains `self._state: dict[str, ModelState]` updated under `self._lock`:

- `load()` sets `LOADING` when the subprocess is spawned.
- Load runner calls `sup.mark_warming(model_id)` after `wait_for_health` returns true.
- Load runner calls `sup.mark_ready(model_id)` after the warmup probe succeeds.
- `unload(model_id, *, force=False)` — if state is not `READY` and `force=False`, raise `UnloadRefused`. Otherwise set `UNLOADING` and SIGTERM as today.
- `_watch_exit` clears state on subprocess exit.

The HTTP unload route catches `UnloadRefused` and returns 409 with a body that names the current supervisor state, so the UI can show a clear message ("model is still warming up; use force-unload to terminate anyway").

### Warmup verification probe

After `wait_for_health` returns true, the load runner sends one probe request to confirm the engine is actually serving before flipping DB status to `loaded`:

```python
POST http://127.0.0.1:{port}/v1/completions
{
  "model": model.served_model_name,
  "prompt": " ",
  "max_tokens": 1,
  "stream": false
}
```

- Timeout: 60s.
- Success: HTTP 200 with a `choices` array → `sup.mark_ready(model_id)`, DB row → `loaded`, runtime row upserted.
- Failure (5xx, timeout, or non-200): DB row → `failed` with `last_error="warmup probe failed: <detail>"`. **Subprocess is left running.** Operator must explicitly force-unload to free the GPUs.

`/v1/completions` was chosen over `/v1/chat/completions` because it does not require chat-template handling and works uniformly across instruct/base/multimodal models.

### Load-route changes

The load runner in `routes_api.py:660-681` is restructured:

```
spawn subprocess (sup.load)
  → wait_for_health
    → on timeout: DB→failed, DO NOT call sup.unload
    → on success: sup.mark_warming
      → warmup probe
        → on failure: DB→failed, DO NOT call sup.unload
        → on success: sup.mark_ready, DB→loaded, RuntimeRepo upsert
```

The two `await sup.unload(model_id)` calls in the load runner are removed. Both failure paths leave the subprocess running, holding its GPUs, with status `failed`. Cleanup is the operator's explicit choice.

### Force-unload endpoint

`POST /api/models/{model_id}/unload?force=true` is the only path that passes `force=True` to `sup.unload()`. Behavior:

- Skips the supervisor's state-gate check.
- Same JWT auth as the regular unload route.
- Writes a marker to `last_error` so audit trail captures forced operations: `last_error="forced unload from state=<state> by user=<sub>"`.
- Otherwise identical to the standard unload path (SIGTERM → 30s grace → SIGKILL).

### State transition diagram

```
                   load()                  health 200
   (absent) ───────────────► LOADING ─────────────────► WARMING
                              │                            │
                              │ subprocess exit            │ warmup probe ok
                              ▼                            ▼
                           (absent)                      READY
                                                          │
                                                          │ unload()
                                                          ▼
                                                      UNLOADING ───► (absent)

   force-unload from LOADING/WARMING:
     LOADING ─── unload(force=True) ───► UNLOADING ───► (absent)
     WARMING ─── unload(force=True) ───► UNLOADING ───► (absent)
```

### Error handling

| Scenario | Behavior |
|---|---|
| `unload()` from `LOADING`/`WARMING` without `force` | `UnloadRefused` → route returns 409 with current state in body |
| `unload()` from `READY` | SIGTERM, 30s grace, SIGKILL fallback (unchanged) |
| Warmup probe 5xx | DB→failed, subprocess left running, operator force-unloads |
| Warmup probe timeout | Same as 5xx |
| Subprocess crash during warmup | `_watch_exit` fires `on_exit(rc)`, DB row → failed, supervisor state cleared |
| API container restart mid-load | vLLM dies with container; `mark_runtime_dead_on_startup` flips DB to failed (existing behavior) |

### Testing

Unit tests in `tests/runtime/test_supervisor.py`:

- `unload()` from `LOADING` raises `UnloadRefused`.
- `unload()` from `WARMING` raises `UnloadRefused`.
- `unload(force=True)` from `LOADING`/`WARMING`/`READY` succeeds.
- `unload()` from `READY` succeeds without force.
- `mark_warming` / `mark_ready` transitions update state correctly.
- `_watch_exit` clears state on subprocess exit.

Integration test (mock vLLM subprocess) in `tests/integration/test_load_lifecycle.py`:

- Slow-warmup simulator: mock binary that opens port, returns `/health` 200 immediately, but `/v1/completions` 503 for 5s then 200.
- Verify DB status stays `loading` (not `loaded`) until probe succeeds.
- Verify UI-style unload during the 5s window returns 409.
- Verify post-probe unload succeeds.

Manual smoke on warden 10.10.0.187:

- Load Qwen3-VL.
- During the warmup window (between `/health` 200 and serving), click unload in UI.
- Verify 409 response, no SIGTERM in vLLM log, model proceeds to `loaded`.

## Migration / rollout

No DB migration required. Roll-out is a single image rebuild:

1. Patch `app/runtime/supervisor.py` (state machine, `UnloadRefused`, `mark_warming`, `mark_ready`, `unload(force=)`).
2. Patch `app/models/routes_api.py` (load runner restructure, force-unload query param, 409 translation of `UnloadRefused`).
3. Add the UI affordance for force-unload (separate ticket — not required for the race fix itself; operator can hit the endpoint directly).
4. Add tests.
5. Build image, deploy to warden, smoke-test with Qwen3-VL load + unload during warmup window.

Backward compatibility: existing unload calls without `?force=true` continue to work for ready models. Calls during warmup that previously SIGTERMed now 409 — this is the intended behavior change.

## Risks

- **Operator confusion on `failed`-with-running-subprocess state.** A model row showing `failed` but still holding GPUs is a new state the UI must surface clearly. Mitigation: the failed row's `last_error` field will say "warmup probe failed: ...; subprocess still holding GPUs N,M — force-unload to release". UI shows a force-unload button when `last_error` contains `subprocess still holding`.
- **Stuck `WARMING` state if the load runner crashes between `mark_warming` and `mark_ready`.** The subprocess is still alive but the supervisor state is `WARMING` forever. Mitigation: `_watch_exit` is the canonical state-clearer — if the subprocess dies, state clears. If the runner task itself dies (asyncio exception in the load route), the state stays stuck and unload requires force. Acceptable: this is rare and the force path exists.
- **Probe failure on legitimate slow models.** Some models may genuinely take longer than 60s for the first completion. Mitigation: the probe timeout is a per-model setting in `app/config.py` (default 60s, configurable per model in a follow-up).

## What is explicitly NOT changing

- The supervisor's SIGTERM → SIGKILL grace logic (`UNLOAD_GRACE_SECONDS = 30.0`) — unchanged.
- The `mark_runtime_dead_on_startup` startup reconciliation — unchanged.
- The DB schema and the `ACTIVE_STATUSES` tuple — unchanged.
- vLLM's signal handling — we are not patching upstream.
