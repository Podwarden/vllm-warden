# vLLM Warden v2 — Multi-Stage Management App Design

**Status:** Approved design (2026-05-08)
**Supersedes:** docs/superpowers/specs/2026-05-02-441-setup-wizard-design.md (Phase 1 wizard)
**Owner:** CTO
**Drives from issue:** vllm-warden Phase 1 wizard does not propagate `gpu_indices` to the vLLM subprocess (`CUDA_VISIBLE_DEVICES` is never set), and has no concept of multiple models or pre-load validation. Reproduced 2026-05-08 on pw_prod bonus node: Qwen/Qwen3.5-9B@main TP=2 on GPUs[1,2] crashed with `torch.OutOfMemoryError` because vLLM saw all three GPUs — including the 8 GiB Quadro RTX 4000 — and tried to load there.

## Goals

1. Replace the single-shot Phase 1 setup wizard with a **multi-stage management app**: setup → models → tokens → stats.
2. **Fix** the GPU-env-propagation bug **by construction** (not as a patch).
3. Support **multiple loaded models simultaneously** on a single node, each with its own GPU subset.
4. Provide **pull-test** (HF cache + disk-space check) and **load-test** (real GPU load) as discrete, observable operations.
5. Expose an **OpenAI-compatible API** that routes by `model` field across all loaded models.
6. Provide **API token management** with per-token model allow-lists.
7. Provide **stats**: per-model usage counters, live GPU state, and 24h time-series charts.

## Non-goals

- Auto-restart on crash (explicit in v2 — failures surface to operator).
- Multi-replica deployments (single-pod-single-vLLM is the unit of deploy).
- Cross-node GPU pooling.
- LoRA adapter management (could be added in v3 without architectural changes).
- Audit log of every API call (counters only).
- Prometheus `/metrics` endpoint (in-app charts only).

## Architecture

**One container, one Python process tree.** All four stages of the user-facing app live in a single FastAPI application; vLLM processes are spawned as supervised children.

```
┌─ vllm-warden container ──────────────────────────────────┐
│                                                          │
│  ┌────────────────── FastAPI app ─────────────────────┐  │
│  │                                                    │  │
│  │  Web tier (Jinja + htmx)                           │  │
│  │   /setup/*        first-run wizard                 │  │
│  │   /models         catalog + add/pull/load/unload   │  │
│  │   /tokens         API token mgmt                   │  │
│  │   /stats          per-model + GPU live + charts    │  │
│  │                                                    │  │
│  │  Admin API (JSON)            /api/v1/...           │  │
│  │  OpenAI-compat proxy         /v1/...   (token)     │  │
│  └────────────────────────────────────────────────────┘  │
│                          │                               │
│                          ▼                               │
│  ┌───────── runtime/ supervisor module ───────────────┐  │
│  │   ProcessPool:  model_id → asyncio.subprocess      │  │
│  │   - spawn / health-poll / log-tail / kill          │  │
│  │   - GPU registry (which GPUs are owned)            │  │
│  │   - upstream URL map (model name → 127.0.0.1:port) │  │
│  └────────────────────────────────────────────────────┘  │
│                          │                               │
│                          ▼                               │
│  vLLM-process-A   vLLM-process-B   vLLM-process-C  ...   │
│  (GPUs [0])       (GPUs [1,2])     (GPUs [3])            │
│                                                          │
│  ┌─ persistence ──────────────────────────────────────┐  │
│  │  /data/vllm-warden.db    SQLite                    │  │
│  │  /data/cache/            HF weights cache          │  │
│  │  /data/logs/             per-model vLLM logs       │  │
│  │  /data/secrets/          HF token, admin pw hash   │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### Components

Each module has one purpose and a small public surface.

| Module | Purpose |
|---|---|
| `app/web/` | Jinja templates + htmx fragments, route handlers |
| `app/api/` | JSON routes for the UI to call (htmx targets) |
| `app/proxy/` | OpenAI-compat router; reads `model` field, looks up upstream, streams response |
| `app/runtime/supervisor.py` | Owns the ProcessPool; only place that knows how to fork vLLM |
| `app/runtime/probe.py` | Pull-test (HF Hub SDK + disk check), no GPU |
| `app/runtime/log_tailer.py` | Per-process log ring buffer + SSE channel |
| `app/auth/` | Admin login (bcrypt sessions) + API tokens (bearer, allow-list) |
| `app/stats/` | Counters in SQLite, GPU/VRAM gauges polled from `nvidia-smi`, minute-bucket sampler |
| `app/db/` | SQLite migrations + repos |
| `app/setup/` | Wizard state machine (welcome / GPUs / HF token / admin / done) |

### Key invariants

- Each loaded model **owns its GPU set exclusively**; the supervisor refuses overlap.
- `served_model_name` is the routing key for the OpenAI proxy; the request body's `model` field must match.
- Setup-wizard-mode and managed-app-mode are **mutually exclusive**: app boots into wizard if `/data/secrets/admin-password` doesn't exist; otherwise login screen.
- The supervisor is the **only** module that knows how to construct vLLM's env and argv. Web/API/UI never spawn subprocesses themselves.

## Data model

SQLite at `/data/vllm-warden.db`, sized for a single-tenant operator deployment.

```
users
─────────────────────────────────────────────────────────
id            TEXT PK            'admin'  (only one for now)
password_hash TEXT NOT NULL      bcrypt
created_at    TIMESTAMPTZ

api_tokens
─────────────────────────────────────────────────────────
id            TEXT PK            uuid
name          TEXT NOT NULL      'studio-prod'
token_hash    TEXT UNIQUE NOT NULL   sha256 of bearer (only display secret on create)
prefix        TEXT NOT NULL      first 8 chars, for UI ("vw_a1b2c3d4…")
allow_all     BOOLEAN NOT NULL   default TRUE
created_by    TEXT NOT NULL      → users.id
created_at    TIMESTAMPTZ
last_used_at  TIMESTAMPTZ NULL
revoked_at    TIMESTAMPTZ NULL

api_token_models                 -- only used when allow_all = FALSE
─────────────────────────────────────────────────────────
token_id      TEXT NOT NULL      → api_tokens.id ON DELETE CASCADE
model_id      TEXT NOT NULL      → models.id     ON DELETE CASCADE
PRIMARY KEY (token_id, model_id)

models                           -- catalog
─────────────────────────────────────────────────────────
id                  TEXT PK             slug, e.g. 'qwen3.5-9b'
display_name        TEXT NOT NULL
hf_repo             TEXT NOT NULL       'Qwen/Qwen3.5-9B'
hf_revision         TEXT NOT NULL       'main' or commit sha
dtype               TEXT NOT NULL       'auto' | 'float16' | 'bfloat16'
tensor_parallel_size INTEGER NOT NULL   1, 2, 4
gpu_indices         TEXT NOT NULL       JSON: [1,2]
served_model_name   TEXT NOT NULL       what clients pass in OpenAI `model` field (defaults to id)
extra_args          TEXT NOT NULL       JSON: arbitrary --flag=value list
status              TEXT NOT NULL       'registered' | 'pulling' | 'pulled' |
                                        'loading' | 'loaded' | 'unloading' |
                                        'failed'
disk_bytes          BIGINT NULL         resolved after pull
last_error          TEXT NULL
created_at          TIMESTAMPTZ
updated_at          TIMESTAMPTZ

model_runtime                    -- only populated while status='loaded'
─────────────────────────────────────────────────────────
model_id      TEXT PK            → models.id ON DELETE CASCADE
pid           INTEGER NOT NULL
port          INTEGER NOT NULL   bound on 127.0.0.1
started_at    TIMESTAMPTZ
ready_at      TIMESTAMPTZ NULL   first /health=200

usage_counters                   -- per model, monotonic
─────────────────────────────────────────────────────────
model_id        TEXT PK          → models.id ON DELETE CASCADE
request_count   BIGINT NOT NULL DEFAULT 0
error_count     BIGINT NOT NULL DEFAULT 0
tokens_in       BIGINT NOT NULL DEFAULT 0
tokens_out      BIGINT NOT NULL DEFAULT 0
last_used_at    TIMESTAMPTZ NULL

model_samples                    -- 1 row per model per minute (chart data)
─────────────────────────────────────────────────────────
ts            TIMESTAMPTZ NOT NULL    bucketed to minute
model_id      TEXT NOT NULL           → models.id ON DELETE CASCADE
requests      INTEGER NOT NULL        delta in this minute
tokens_in     INTEGER NOT NULL        delta
tokens_out    INTEGER NOT NULL        delta
errors        INTEGER NOT NULL        delta
PRIMARY KEY (ts, model_id)

gpu_samples                      -- 1 row per GPU per minute (chart data)
─────────────────────────────────────────────────────────
ts            TIMESTAMPTZ NOT NULL
gpu_index     INTEGER NOT NULL
vram_used_mib INTEGER NOT NULL
util_percent  INTEGER NOT NULL
PRIMARY KEY (ts, gpu_index)

setup_state                      -- wizard progress (single row)
─────────────────────────────────────────────────────────
id          TEXT PK    'singleton'
step        TEXT NOT NULL    'welcome' | 'gpus' | 'hf-token' | 'admin' | 'done'
draft       TEXT NOT NULL    JSON blob of in-progress answers
updated_at  TIMESTAMPTZ
```

After `step='done'`, `setup_state.draft.allowed_gpu_indices` is the persistent source of truth for which GPUs are available to the app. The Add-model modal's GPU multi-select reads it (so disabled GPUs aren't selectable), and `Supervisor.load()` re-validates `model.gpu_indices ⊆ allowed_gpu_indices` at load time and rejects with HTTP 400 otherwise (Settings page edits this list).

Sample-table retention: 7 days, pruned by the `app/stats/sampler.py` background task. With 5 GPUs and 5 models the disk cost is ~14k rows/day — trivial.

### Files on disk (not in DB)

- `/data/secrets/hf_token` — chmod 600, plain text (filesystem perms only)
- `/data/cache/` — HF Hub default cache (`HF_HOME`)
- `/data/logs/<model_id>.log` — vLLM stdout+stderr, rotated by size
- `/data/secrets/admin-password` — sentinel file; presence indicates setup is complete

## UI flow & pages

Top nav for the managed app: **Models · Tokens · Stats · Settings · Logout**. The setup wizard is a separate full-screen flow before the app exists.

### Setup wizard (first run only)

5 steps, server-rendered, one screen each. State persisted in `setup_state.draft` after every Next.

1. **Welcome** — what this wizard does, list detected GPUs (probed via `nvidia-smi --query-gpu=index,name,memory.total,compute_cap`).
2. **GPUs** — checkboxes; user picks which GPUs the app may use. *Critical: persisted to `setup_state.draft` and propagated as `CUDA_VISIBLE_DEVICES` to every spawned vLLM (this fixes today's bug).*
3. **HF token** — paste field, validates against `https://huggingface.co/api/whoami-v2`. On success, written to `/data/secrets/hf_token`.
4. **Admin account** — email + password (≥12 chars). Bcrypt-hashed into `users`.
5. **Done** — summary screen, "Continue to Models →" sends to `/models`, marks `setup_state.step='done'` and writes `/data/secrets/admin-password` sentinel.

### Models page

The catalog. Shows GPU usage banner across the top, "Add model" button, and a card per catalogued model with verbs appropriate to its current status.

```
┌── Models ─────────────────────────────────────────────────────────────────┐
│  GPU usage  ████░░░  GPU 1: 14.1 / 16 GiB        ●●○○ 3 GPUs              │
│             ████░░░  GPU 2: 14.1 / 16 GiB                                 │
│             ░░░░░░░  GPU 0: idle                                          │
│                                                          [+ Add model]    │
│                                                                           │
│  qwen3.5-9b           Qwen/Qwen3.5-9B@main   GPUs[1,2]  TP=2              │
│  ● loaded · 14.1 GiB · 1,043 reqs · last used 2m ago                      │
│  [ Unload ]   [ Logs ]   [ Edit ]   [ Remove ]                            │
│                                                                           │
│  llama-3.1-8b        meta-llama/Llama-3.1-8B@main   GPUs[0]  TP=1         │
│  ○ pulled · 16.0 GiB on disk                                              │
│  [ Load ]   [ Logs ]   [ Edit ]   [ Remove ]                              │
│                                                                           │
│  mistral-7b          mistralai/Mistral-7B-v0.3@main   (not pulled)        │
│  ○ registered                                                             │
│  [ Pull ]   [ Edit ]   [ Remove ]                                         │
└───────────────────────────────────────────────────────────────────────────┘
```

**Add model** modal: HF repo, revision (default `main`), dtype, tensor_parallel_size, GPU indices (multi-select with live "would fit / wouldn't fit" indicator computed from `nvidia-smi` free memory + a model-size heuristic), `served_model_name`, extra args (textarea, `--key=value` per line). Submit creates a row with `status='registered'`.

**Pull** kicks off a background task (`runtime/probe.py` → `huggingface_hub.snapshot_download`); the page polls via htmx `hx-get` every 2s for progress text + bytes-on-disk. On completion: `status='pulled'`, `disk_bytes` populated. Pre-flight: free disk must be ≥ 1.5× estimated model size or the Pull button is disabled with an explanation.

**Load** asks the supervisor to spawn vLLM with the right env (see Supervisor section). UI shows progressive states: `loading → port-bound → warming → ready`. Failure flips status to `failed` and surfaces the last-error tail.

**Unload** calls supervisor.unload() — SIGTERM, 30s, then SIGKILL. Frees the GPUs.

**Edit** is only allowed when status ∈ `{registered, pulled, failed}` (i.e., not while loaded).

### Tokens page

Lists tokens (name, prefix, allow-list summary, created/used/revoked timestamps). **New token** modal: name, "Allow all loaded models" toggle (default on); when off, checkbox list of catalog models. On submit, the raw bearer (`vw_<32-byte-base32>`) is shown ONCE; thereafter only `prefix` survives.

Revoke is a soft delete (`revoked_at` set); revoked tokens never authenticate.

### Stats page

Two stacked panels: live snapshot ("Right now") and 24h time-series.

```
┌── Stats ──────────────────────────────────────────────────────────────────────┐
│                                                                               │
│  ┌─ Right now ────────────────────────────────────────────────────────────┐   │
│  │  GPU 0  Quadro RTX 4000   idle                          0 % util       │   │
│  │  GPU 1  RTX A4000      14.1 / 16 GiB  ┃ qwen3.5-9b    73 % util        │   │
│  │  GPU 2  RTX A4000      14.1 / 16 GiB  ┃ qwen3.5-9b    71 % util        │   │
│  │                                                                        │   │
│  │  Models loaded: 2     Active requests: 4     Tokens/s now: 142         │   │
│  └────────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
│  ┌─ Last 24h ─────────────────────────────────────────────────────────────┐   │
│  │  Requests / min                              [1h] [6h] [24h]           │   │
│  │  …                                                                     │   │
│  │  Tokens / sec   (in: dashed, out: solid, stacked)                      │   │
│  │  GPU VRAM used (per GPU, stacked)                                      │   │
│  │  GPU utilization % (per GPU, line)                                     │   │
│  └────────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
│  ┌─ Per-model totals ─────────────────────────────────────────────────────┐   │
│  │ Model         Reqs    Tokens in   Tokens out   Errors   Last           │   │
│  │ …                                                                      │   │
│  └────────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────────────┘
```

**Right now** refreshes every 2s via htmx polling. **Charts** rendered with Chart.js (~70 KB, no SPA). 24h auto-refresh every 30s. Time-window toggle is a query param.

Charts shown:

1. Requests / min — per model, stacked area
2. Tokens / sec — input dashed, output solid, stacked per model
3. GPU VRAM used — per GPU, stacked area
4. GPU utilization % — per GPU, line

API: `GET /api/v1/stats/now` (snapshot) and `GET /api/v1/stats/timeseries?window=1h|6h|24h` (chart data, pre-aggregated by minute).

### Settings (small)

Three cards: **HF token** (masked, "Replace" button), **Admin password** (Change form), **About** (version, build sha).

### Authentication flow

- Unauthenticated → `/login`
- Setup not done → all routes redirect to `/setup/<step>`
- `/v1/*` requires `Authorization: Bearer vw_…` (admin UI uses session cookies)

## Supervisor & vLLM lifecycle

The supervisor is the safety-critical module. It is the **only** place that constructs a vLLM env, and the env is derived 1:1 from the model row.

### State (in-memory, single instance)

```python
@dataclass
class RuntimeProcess:
    model_id: str
    process: asyncio.subprocess.Process
    port: int
    gpus: list[int]
    started_at: datetime
    ready_at: datetime | None
    log_path: Path
    log_tail: deque[str]          # ring buffer, last 500 lines
    health_task: asyncio.Task     # poll /health until ready
    waiter_task: asyncio.Task     # awaits process.wait() to detect crash

class Supervisor:
    def __init__(self, db: Database, settings_dir: Path) -> None:
        self._pool: dict[str, RuntimeProcess] = {}     # model_id -> proc
        self._gpu_owner: dict[int, str] = {}           # gpu_idx -> model_id
        self._port_seq = itertools.count(18000)
        self._lock = asyncio.Lock()

    async def load(self, model: ModelRow) -> None: ...
    async def unload(self, model_id: str) -> None: ...
    def upstream_for(self, served_name: str) -> str | None: ...
    def gpu_state(self) -> list[GpuLive]: ...
```

The pool is in-memory only. On container restart, `model_runtime` rows are wiped at startup and every model returns to `pulled`/`registered` state — matches the chosen `Add → Pull → Load → Unload → Remove` lifecycle (no auto-resume).

### `load()` — the critical path

```python
async def load(self, model: ModelRow) -> None:
    async with self._lock:
        # 1. Validate exclusive GPU ownership
        gpus = json.loads(model.gpu_indices)
        for g in gpus:
            if g in self._gpu_owner and self._gpu_owner[g] != model.id:
                raise GPUInUse(g, self._gpu_owner[g])

        # 2. Reserve port + claim GPUs (so a parallel load() can't race)
        port = next(self._port_seq)
        for g in gpus:
            self._gpu_owner[g] = model.id

        await self._db.update_model_status(model.id, 'loading')

    # 3. Build env — THE FIX
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in gpus)
    env['HF_HOME'] = '/data/cache'
    env['HF_TOKEN'] = (Path('/data/secrets/hf_token').read_text().strip())
    env['VLLM_LOGGING_LEVEL'] = 'INFO'

    # 4. Build argv — vLLM serves what it owns
    argv = [
        'vllm', 'serve', model.hf_repo,
        '--revision', model.hf_revision,
        '--served-model-name', model.served_model_name,
        '--tensor-parallel-size', str(model.tensor_parallel_size),
        '--port', str(port),
        '--host', '127.0.0.1',
    ]
    if model.dtype and model.dtype != 'auto':
        argv += ['--dtype', model.dtype]
    argv += json.loads(model.extra_args or '[]')

    # 5. Spawn, redirect logs to /data/logs/<model_id>.log
    log_path = Path(f'/data/logs/{model.id}.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fd = log_path.open('ab', buffering=0)
    proc = await asyncio.create_subprocess_exec(
        *argv, env=env,
        stdout=log_fd, stderr=asyncio.subprocess.STDOUT,
        stdin=asyncio.subprocess.DEVNULL,
        start_new_session=True,    # own process group
    )

    rp = RuntimeProcess(model.id, proc, port, gpus, utcnow(), None,
                        log_path, deque(maxlen=500),
                        health_task=None, waiter_task=None)
    rp.health_task = asyncio.create_task(self._await_ready(rp, model))
    rp.waiter_task = asyncio.create_task(self._await_exit(rp, model))
    self._pool[model.id] = rp

    await self._db.upsert_runtime(model.id, proc.pid, port)
```

### Health-readiness probe

```python
async def _await_ready(self, rp: RuntimeProcess, model: ModelRow) -> None:
    deadline = utcnow() + timedelta(seconds=600)
    url = f'http://127.0.0.1:{rp.port}/health'
    async with httpx.AsyncClient() as client:
        while utcnow() < deadline:
            try:
                r = await client.get(url, timeout=2.0)
                if r.status_code == 200:
                    rp.ready_at = utcnow()
                    await self._db.update_model_status(model.id, 'loaded')
                    await self._db.set_runtime_ready(model.id, rp.ready_at)
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2)
    await self._fail(rp, model, "health-check timeout after 600s")
```

### Crash detection

```python
async def _await_exit(self, rp: RuntimeProcess, model: ModelRow) -> None:
    rc = await rp.process.wait()
    if rp.model_id not in self._pool:    # we already unloaded; expected exit
        return
    tail = "\n".join(list(rp.log_tail)[-50:])
    await self._fail(rp, model, f"vllm exited rc={rc}\n{tail}")
```

### `unload()`

```python
async def unload(self, model_id: str) -> None:
    rp = self._pool.pop(model_id, None)
    if not rp:
        return
    await self._db.update_model_status(model_id, 'unloading')
    rp.health_task.cancel()
    try:
        rp.process.terminate()
        await asyncio.wait_for(rp.process.wait(), timeout=30)
    except asyncio.TimeoutError:
        rp.process.kill()
        await rp.process.wait()
    for g in rp.gpus:
        self._gpu_owner.pop(g, None)
    await self._db.delete_runtime(model_id)
    await self._db.update_model_status(model_id, 'pulled')
```

### Log tailing

A single `app/runtime/log_tailer.py` task per `RuntimeProcess` streams the log file via `aiofiles`, appends to `rp.log_tail` (the ring buffer), and pushes increments to a per-model SSE channel that the **Logs** modal in the UI subscribes to.

### Concurrency model

- One `Supervisor` instance, owned by FastAPI app state, lives for the process lifetime.
- All public methods take `self._lock` only for the brief GPU-claim/release windows. Subprocess spawning + readiness polling happen *outside* the lock so a slow load doesn't block parallel ops on other models.
- Container shutdown handler: `await asyncio.gather(*(self.unload(mid) for mid in list(self._pool)))` — graceful TERM, 30s budget, then KILL.

### What this fixes vs Phase 1

| Phase 1 bug | v2 behavior |
|---|---|
| `gpu_indices` from wizard never reach the vLLM env | `CUDA_VISIBLE_DEVICES` built from `model.gpu_indices` literally one line above the spawn |
| Pod relies on `NVIDIA_VISIBLE_DEVICES=all` + node selector | Same container env still works (we don't fight the operator), but the *process* sees only the GPUs we asked for |
| Crash → infinite respawn loop, no error to user | Crash detection flips status to `failed` with last 50 log lines surfaced in UI |
| Single supervisor, single model assumed | ProcessPool with explicit GPU registry; refuses overlap |

## API surface

Two distinct surfaces with separate auth:

| Surface | Routes | Auth | Audience |
|---|---|---|---|
| Web UI (HTML + htmx) | `/setup/*`, `/models`, `/tokens`, `/stats`, `/settings`, `/login`, `/logout` | session cookie | admin in browser |
| Admin REST | `/api/v1/*` | session cookie OR admin bearer | UI + scripts |
| OpenAI-compat proxy | `/v1/*` | `Authorization: Bearer vw_…` (api_tokens) | model clients |
| Health | `/healthz` | none | k8s probe |

### Admin REST

```
GET    /api/v1/models                       list catalog
POST   /api/v1/models                       add (status=registered)
GET    /api/v1/models/{id}                  detail
PATCH  /api/v1/models/{id}                  edit (only when not loaded)
DELETE /api/v1/models/{id}?delete_cache=    remove
POST   /api/v1/models/{id}/pull             start pull (returns 202)
POST   /api/v1/models/{id}/load             start load (returns 202)
POST   /api/v1/models/{id}/unload           graceful unload
GET    /api/v1/models/{id}/logs/stream      SSE: live log tail
GET    /api/v1/models/{id}/logs?lines=500   one-shot last N lines

GET    /api/v1/tokens                       list (no secrets, prefix only)
POST   /api/v1/tokens                       create (returns raw bearer ONCE)
PATCH  /api/v1/tokens/{id}                  edit (rename, allow-list)
DELETE /api/v1/tokens/{id}                  revoke

GET    /api/v1/stats/now                    current snapshot
GET    /api/v1/stats/timeseries?window=24h  chart data

GET    /api/v1/system/gpus                  detected GPUs (nvidia-smi)
GET    /api/v1/system/disk                  free bytes on /data

POST   /api/v1/setup/<step>                 advance wizard
GET    /api/v1/setup/state                  current step + draft
```

`/load` and `/pull` are **always 202 Accepted** — work runs in the background, the UI polls or watches the SSE log stream. Idempotent if the model is already in the target state (`load` while loaded → 200 no-op).

### OpenAI-compat proxy

Pass-through proxy with a routing layer. We do not parse vLLM's response shape (other than for token counting); we stream bytes back to the client unchanged.

```
GET    /v1/models                           union of *loaded* models, vLLM-shaped
POST   /v1/chat/completions                 proxy by `model` field
POST   /v1/completions                      proxy by `model` field
POST   /v1/embeddings                       proxy by `model` field
```

```python
# app/proxy/router.py

@app.post("/v1/{path:path}")
async def proxy(path: str, request: Request,
                token: ApiToken = Depends(verify_bearer)):
    body = await request.body()
    payload = json.loads(body)               # only to read `model`
    requested = payload.get("model")
    if not requested:
        raise HTTPException(400, "model field required")

    upstream = supervisor.upstream_for(requested)
    if not upstream:
        raise HTTPException(404, f"model '{requested}' not loaded")
    if not token.allows(requested):
        raise HTTPException(403, "token not authorized for this model")

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            request.method, f"{upstream}/v1/{path}",
            content=body, headers={k: v for k, v in request.headers.items()
                                   if k.lower() not in {"host", "authorization",
                                                        "content-length"}},
        ) as upstream_resp:
            return StreamingResponse(
                _accounting_stream(upstream_resp, requested, token),
                status_code=upstream_resp.status_code,
                headers=_proxy_headers(upstream_resp.headers),
                media_type=upstream_resp.headers.get("content-type"),
            )
```

**Token counting.** `_accounting_stream` peeks at chunks:

- Non-streaming JSON: parse the final `usage` object, write deltas to in-memory minute bucket.
- SSE streaming: parse `data:` chunks, accumulate `delta` text, count tokens via the `transformers` tokenizer cache for that `served_model_name` (loaded once per model, reused). The `[DONE]` chunk triggers the final write.
- Errors: increment `errors` counter, no usage delta.

**Latency.** All proxying is streaming; we never buffer a full response in memory. The token-counter sits in the byte stream; for chunked SSE it does its work between yields, never blocking the client.

### Auth implementation

```python
async def verify_bearer(request: Request, db: Database) -> ApiToken:
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer vw_"):
        raise HTTPException(401, "missing token")
    raw = auth.split(" ", 1)[1]
    h = sha256(raw.encode()).hexdigest()
    row = await db.fetch_token(h)
    if not row:
        raise HTTPException(401, "invalid token")
    await db.touch_token_used(row.id)
    return row

class ApiToken:
    def allows(self, served_name: str) -> bool:
        if self.allow_all:
            return True
        return served_name in self.allowed_models
```

Session auth for the UI: standard `itsdangerous`-signed cookie with `user_id`, 8h sliding expiry, CSRF-protected POSTs (htmx sends `X-CSRF-Token` header from a meta tag).

### Rate limiting

- `/login`: 5 attempts / 15 min / IP (in-memory bucket; container is single-replica)
- `/v1/*`: no app-level limit (vLLM's own queue is the limiter)

### Health endpoint

```python
@app.get("/healthz")
def healthz():
    return {"status": "ok", "loaded_models": len(supervisor._pool)}
```

The container is "healthy" even with zero loaded models, so an empty deployment doesn't CrashLoop.

## Error handling & recovery

### Pull failures

| Failure | Detection | UI / state |
|---|---|---|
| HF token invalid / 401 | `huggingface_hub.errors.RepositoryNotFoundError` or 401 | status=`failed`, `last_error="Hugging Face authentication failed — check your token in Settings"` |
| Repo or revision not found / gated | 403/404 from HF | status=`failed`, surface specific message ("model is gated — request access at huggingface.co/…") |
| Disk full | pre-flight check or `OSError errno=ENOSPC` | Pre-flight: Pull button disabled when free < 1.5× estimated model size. Mid-pull ENOSPC: status=`failed`, partial files preserved (HF Hub resumes on retry). |
| Network blip | `httpx.ConnectError` mid-snapshot | retry up to 3× exponential backoff inside the pull task |
| Process crash mid-pull | parent watches pull task | task wrapped so cancellation/exception flips status to `failed` |

Pull is **resumable** — re-clicking Pull on a `failed` model re-enters `huggingface_hub.snapshot_download`, which picks up where it left off via the cache.

### Load failures

| Failure | Detection | Behavior |
|---|---|---|
| GPU already owned by another model | `Supervisor.load()` pre-flight | reject with HTTP 409 + "GPU N is in use by `<model_id>`. Unload it first." — vLLM is not started |
| GPU not in setup-wizard's allowed list | startup check — `model.gpu_indices ∩ allowed` must equal `model.gpu_indices` | reject with HTTP 400; "GPU N is not enabled — go to Settings → GPUs" |
| OOM during model load | vLLM exits with `torch.OutOfMemoryError`; `_await_exit` fires before `_await_ready` succeeds | status=`failed`, `last_error` = last 50 log lines, GPUs released |
| Model architecture not supported by installed vLLM | vLLM exits early | status=`failed`, last_error has the message |
| Hangs during weight load | `_await_ready` 600s deadline | supervisor calls `_fail()` → SIGKILL, GPUs released, status=`failed` |
| vLLM serves but `/health` never returns 200 | same deadline | same path |

The **GPU registry release** is in `_fail()` and is the same path as a clean unload — that means a failed load never strands GPUs.

### Runtime failures (model was loaded, then crashed)

| Failure | Detection | Behavior |
|---|---|---|
| vLLM crashes mid-serving | `_await_exit` task observes `process.wait()` return | status flips to `failed`, GPUs released, last 50 lines into `last_error`; **no auto-restart** |
| Container OOM kill (k8s) | container restart | on boot, `model_runtime` is wiped, all models start as `pulled`; operator re-loads explicitly |
| HF token rotated mid-load | unrelated to runtime; only matters for pull | model keeps serving with already-loaded weights |

**Why no auto-restart:** today's wizard ships infinite respawn and that's how we ended up here. v2 surfaces failures explicitly so the operator can act on them. If a future need emerges, that's a v3 feature with a per-model "auto-restart on crash" toggle and a backoff-with-circuit-breaker policy. Not in scope.

### Proxy failures

| Failure | Behavior |
|---|---|
| Token missing/invalid | 401 from `verify_bearer` |
| Token revoked | 401 (revoked tokens don't match the WHERE clause) |
| Token not allow-listed for model | 403 |
| `model` field missing | 400 |
| Model name not loaded | 404 |
| Upstream vLLM unreachable mid-stream | 502; `_await_exit` has already flipped status to `failed` so `/v1/models` immediately stops listing it |
| Client disconnects mid-stream | upstream connection closed; bytes counted up to that point go to `error_count` (no usage delta) |

### Wizard recovery

| Failure | Behavior |
|---|---|
| User refreshes mid-wizard | `setup_state.draft` is the source of truth; UI re-renders from server state |
| HF token validation flaky | retry button on the same step |
| Container restarts mid-wizard | wizard reloads at the last saved `step`; draft answers persist |
| User wants to reset | container restart with `/data/secrets/` wiped by the operator |

### Container shutdown

```python
@app.on_event("shutdown")
async def shutdown():
    await asyncio.gather(*(supervisor.unload(mid)
                          for mid in list(supervisor._pool)),
                         return_exceptions=True)
```

K8s sends SIGTERM with default 30s grace; we issue our own 30s budget per child before SIGKILL. Worst case both budgets stack and k8s SIGKILLs the parent before we finish — children become orphans, but they're in their own session group (`start_new_session=True`) and get reaped by PID 1 (we run a tini-equivalent in the container).

### Observability

- **Per-model**: `/api/v1/models/{id}/logs?lines=500` returns the last 500 lines of vLLM's stdout/stderr. Available even after `failed`.
- **App-wide**: structured logs go to stdout in JSON (`model_id`, `event`, `level`). Container log is the source of truth.
- **Crash forensics**: when a model flips to `failed`, the timestamp + last 50 log lines are stored in `models.last_error` so the UI shows it without needing to fetch the full log.

## Testing strategy

Three layers, each with a clear seam.

### Layer 1 — Unit tests (no subprocess, no GPU)

- `tests/runtime/test_supervisor_logic.py` — supervisor with `asyncio.create_subprocess_exec` patched
  - GPU registry overlap detection
  - `unload()` releases GPUs even if the child has already exited
  - `_fail()` releases GPUs (regression test for "stranded GPU after failed load")
  - Health timeout fires `_fail` and updates DB to `failed`
  - Crash detection: `_await_exit` fires only if the model is still in the pool
  - Concurrent `load()` calls on the same model_id are serialised (lock test)
- `tests/runtime/test_env_construction.py` — the bug-fix:
  - `gpu_indices = [1, 2]` produces `CUDA_VISIBLE_DEVICES=1,2`, not `all`
  - `dtype='auto'` is omitted from argv (vLLM default)
  - `extra_args = ["--max-model-len=8192"]` lands at the end of argv
  - HF_TOKEN comes from `/data/secrets/hf_token`
- `tests/proxy/test_router.py` — proxy with `httpx.AsyncClient` mocked via `respx`
  - Routes by `model` field; 404 if not loaded; 403 if token allow-list excludes
  - Streaming SSE chunks pass through byte-for-byte
  - Token counter accumulates correctly for streaming + non-streaming
  - Client disconnect mid-stream → error counter +=1, no usage delta
- `tests/auth/test_tokens.py` — bcrypt session, sha256 token hashing, allow-list, revocation
- `tests/db/test_repos.py` — every repo function with `:memory:` SQLite
- `tests/setup/test_wizard_state.py` — wizard state machine
- `tests/stats/test_sampler.py` — minute bucketing, retention pruning at 7 days

### Layer 2 — Integration tests (real subprocess, fake vLLM)

`tests/fakes/fake_vllm.py` is a Python script that pretends to be vLLM:

```python
# usage: python fake_vllm.py --port N --behavior {ok,oom,hang,crash-after,500}
# - serves /health → 200 after 0.5s, /v1/* echoes back
# - --behavior=oom → exits rc=1 with "torch.OutOfMemoryError" on stderr after 1s
# - --behavior=hang → never serves /health
# - --behavior=crash-after → serves cleanly, exits rc=139 after 3s
```

Tests in `tests/integration/`:

- `test_supervisor_real_subprocess.py` — full lifecycle (spawn → ready → unload, spawn → OOM → fail, spawn → hang → timeout)
- `test_proxy_to_subprocess.py` — supervisor spawns fake_vllm; proxy makes a real HTTP request; assertions on byte-perfect passthrough
- `test_pull_disk_check.py` — pull task with `huggingface_hub` patched; verifies disk-space pre-flight rejects when `<1.5×` available

These run on CI without a GPU because the *fake* vLLM is just a Python HTTP server.

### Layer 3 — End-to-end (real vLLM, manual / staging only)

Not run on CI. Live against the bonus node (3 GPUs):

- `tests/e2e/test_smoke_qwen3.5-9b.sh`:
  1. Walk the wizard via the admin REST endpoints
  2. Add Qwen/Qwen3.5-9B@main on GPUs[1,2] TP=2
  3. Pull → wait for `pulled` → assert `disk_bytes > 0`
  4. Load → wait for `loaded` (timeout 10min) → assert `model_runtime` row exists
  5. Create token, call `/v1/chat/completions`, assert non-empty response
  6. Check `/api/v1/stats/now` shows non-zero counters
  7. Unload → assert GPUs released

This script becomes a runbook in the GitLab wiki and is the merge gate before tagging a release.

### Coverage targets

- Supervisor module: 100% line + branch (safety-critical)
- Proxy router: 100% (token-counting math + auth)
- Web routes: ≥80%
- Overall: ≥85%

### Test fixtures

- `tmp_path_factory` for `/data` overrides — every integration test gets its own data dir
- `db` fixture — fresh in-memory SQLite, migrations applied
- `supervisor` fixture — clean instance per test
- `fake_vllm_factory` — context manager that returns the spawn argv

### What we deliberately don't test

- The vLLM-serving GPU path itself — that's vLLM's test suite
- HF Hub's actual download mechanics — patched out
- Browser-rendering of htmx fragments — htmx contract tested by the project; we test the JSON+HTML our endpoints emit

### Tests that lock in the bug fixes

Explicit, named tests so the regression is named in the codebase:

```python
# tests/runtime/test_env_construction.py

def test_cuda_visible_devices_from_model_row_not_env():
    """Phase-1 bug: gpu_indices from the model row must override
    NVIDIA_VISIBLE_DEVICES=all so the vLLM process only sees the GPUs
    the user picked. Without this fix, vLLM saw the Quadro RTX 4000
    on GPU 0 and tried to load there, causing OOM/dtype mismatch."""
    model = ModelRow(id="m1", gpu_indices=json.dumps([1, 2]), ...)
    env = build_env(model)
    assert env["CUDA_VISIBLE_DEVICES"] == "1,2"

def test_failed_load_releases_gpus():
    """Regression: a load() that fails (OOM, timeout, exit) must
    release its GPU claim or subsequent loads on the same GPUs
    will be rejected forever."""
    sup = Supervisor(db, settings_dir=tmp_path)
    model = ModelRow(id="m1", gpu_indices=json.dumps([1, 2]), ...)
    with patch_subprocess_exec(behavior="oom"):
        await sup.load(model)              # spawns fake vLLM
    await wait_for_status(db, "m1", "failed")
    assert sup._gpu_owner == {}            # GPUs 1,2 released
    assert "m1" not in sup._pool
```

## Migration from Phase 1

The Phase 1 wizard codebase is replaced wholesale (greenfield rewrite). Operators with an existing v1 deploy migrate by:

1. Stop the v1 container.
2. Move `/data/wizard-draft.json` and `/data/wizard-state.json` aside (not used by v2; kept for reference).
3. Remove `/data/secrets/admin-password` if present (v1 may not have used this convention) — operator will re-run the wizard.
4. Start v2 image. Wizard launches.
5. (Optional) re-add the same model in v2's Models page; weights in `/data/cache/` are reused.

No data migration script. The catalog is small enough that re-adding 1-2 models manually is easier than writing converter code for what was effectively a single-model state.

## Out of scope (explicit non-goals)

- Multi-tenant: only one admin account; multi-user is a v3 concern.
- Model autoscaling / scheduling: GPUs are explicitly assigned per model.
- LoRA adapters / multi-LoRA serving.
- Persistent audit log of API requests.
- Prometheus `/metrics` endpoint.
- Cross-replica state (e.g., distributed lock service for the GPU registry).
- Auto-restart-on-crash for vLLM children.

These are reachable from this design without breaking architectural assumptions, but each is a separate decision and a separate implementation plan.
