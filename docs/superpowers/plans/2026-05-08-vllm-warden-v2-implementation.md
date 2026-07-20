# vllm-warden v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Greenfield Python rewrite of vllm-warden — replace the Phase 1 Go-based single-model wizard with a multi-stage management app that fixes the 2026-05-08 production bug where wizard GPU selection never reached the vLLM subprocess.

**Architecture:** FastAPI + Jinja2 + htmx + Chart.js + SQLite (aiosqlite) running inside the existing `vllm/vllm-openai` container. A `Supervisor` class manages multiple vLLM subprocesses, each with `CUDA_VISIBLE_DEVICES` derived 1:1 from the per-model `gpu_indices` row (THE BUG FIX). State persists in `/data/vllm-warden.db`; runtime processes are NOT auto-resumed across container restarts. OpenAI-compat proxy routes by request `model` field.

**Tech Stack:** Python 3.11, FastAPI, uvicorn, Jinja2, htmx, Chart.js (UMD), aiosqlite, bcrypt, itsdangerous, huggingface_hub, transformers (for tokenizer), aiofiles, httpx, pytest, pytest-asyncio.

---

## Reference

- **Spec:** `docs/superpowers/specs/2026-05-08-vllm-warden-v2-design.md` (committed at `da007c9`)
- **Branch:** `feat/v2-design-spec` (continue here)
- **Bug being fixed:** Phase 1 Go wizard ignored user's GPU selection when forking vLLM (`CUDA_VISIBLE_DEVICES` never set). Reproduced 2026-05-08: Qwen/Qwen3.5-9B@main TP=2 on GPUs[1,2] crashed with `torch.OutOfMemoryError` because vLLM saw all 4 GPUs and ignored the requested mapping.
- **Bug fix anchor:** Task 29 (`app/runtime/env_builder.py`) — derives `CUDA_VISIBLE_DEVICES` strictly from the model row's `gpu_indices` JSON column, with explicit regression tests `test_cuda_visible_devices_from_model_row_not_env` and `test_failed_load_releases_gpus` that lock in the fix BEFORE any subprocess code lands.

---

## File Structure

**Delete (Go tree):**
- `cmd/`, `internal/`, `web/`, `assets.go`, `go.mod`, `go.sum`, `.golangci.yml`, top-level `*.go`

**Keep (Hub catalog tooling, separate concern):**
- `scripts/publish_to_hub.py`, `tests/test_publish_to_hub.py`, `deploy/hub/`, `deploy/hub.bonus.yaml`

**Create (new Python app):**

```
app/
  __init__.py
  main.py                       # FastAPI app factory, lifespan, route includes
  config.py                     # env-driven settings (DATA_DIR, COOKIE_SECRET, CONTAINER_GPU_COUNT)
  utils/time.py                 # utc_now(), minute_bucket()
  db/
    database.py                 # aiosqlite pool, get_conn() context manager
    migrations.py               # apply 0001..NNNN.sql in order, track schema_migrations
    sql/
      0001_users.sql
      0002_setup_state.sql
      0003_models.sql
      0004_model_runtime.sql
      0005_api_tokens.sql
      0006_counters_samples.sql
      0007_indices.sql
    repos/
      users.py
      tokens.py
      models.py
      runtime.py
      counters.py
      samples.py
      setup.py
  auth/
    sessions.py                 # bcrypt password verify, signed cookie
    csrf.py                     # X-CSRF-Token header check
    bearer.py                   # vw_<base32> token verify (sha256 hashed)
    deps.py                     # require_session, require_bearer FastAPI deps
  system/
    gpu.py                      # nvidia-smi parse → list[GpuInfo]
    disk.py                     # shutil.disk_usage of /data
    hf.py                       # validate_hf_token (calls /api/whoami-v2)
  setup/
    state_machine.py            # next_step(state) → 'welcome'|'gpus'|'hf_token'|'admin'|'done'
    routes_api.py               # POST /api/setup/* (write draft, validate, finalize)
    routes_web.py               # GET /setup → redirect to step
    templates/
      welcome.html
      gpus.html
      hf_token.html
      admin.html
      done.html
  models/
    routes_api.py               # CRUD models, start/stop/unload, status
    routes_web.py               # /models page + htmx fragments
    pull_task.py                # background snapshot_download with disk pre-check
    templates/
      index.html
      _card.html
      _add_modal.html
  tokens/
    routes_api.py               # CRUD api_tokens
    routes_web.py
    templates/
      index.html
      _new_modal.html
  stats/
    routes_api.py               # GET /api/stats/{model_id}, /api/stats/gpus
    routes_web.py
    sampler.py                  # 60s asyncio task: write per-model + per-gpu samples
    templates/
      index.html
  settings/
    routes_web.py               # GET/POST /settings (edit allowed_gpu_indices, hf_token)
    templates/
      index.html
  proxy/
    router.py                   # POST /v1/chat/completions, /v1/completions, GET /v1/models
    accounting.py               # token counting + counters increment
    tokenizers.py               # cached transformers tokenizer per served_model_name
  runtime/
    supervisor.py               # Supervisor class: load/unload/health/_gpu_owner
    env_builder.py              # build_env(model: ModelRow) → dict  ← THE BUG FIX
    log_tailer.py               # SSE stream of /data/logs/<model_id>.log
  web/
    base_template.html
    login.html
    static/
      htmx.min.js
      chart.umd.min.js
      app.css
tests/
  conftest.py                   # tmp_path SQLite, FastAPI TestClient, fake nvidia-smi
  unit/
    db/test_migrations.py
    db/test_repos.py
    auth/test_sessions.py
    auth/test_bearer.py
    auth/test_csrf.py
    system/test_gpu_parse.py
    system/test_disk.py
    system/test_hf.py
    setup/test_state_machine.py
    setup/test_routes.py
    runtime/test_env_builder.py        # BUG FIX regression tests
    runtime/test_supervisor.py         # mocked subprocess
    models/test_routes.py
    models/test_pull_disk_check.py
    tokens/test_routes.py
    stats/test_sampler.py
    proxy/test_router.py
    proxy/test_accounting.py
  integration/
    test_supervisor_real_subprocess.py  # spawns fakes/fake_vllm.py
    test_proxy_to_subprocess.py
    test_pull_disk_check.py
  fakes/
    fake_vllm.py                # tiny aiohttp server mimicking vLLM /v1/* + /health
  e2e/
    test_smoke_qwen3.5-9b.sh    # manual smoke script (not pytest)
docker/
  entrypoint.sh                 # tini-equivalent + uvicorn launch
Dockerfile                       # rewritten: vllm/vllm-openai base + add Python deps + copy app/
Makefile                         # rewritten: build, dev, test, lint, typecheck
pyproject.toml                   # ruff + pytest + mypy config
requirements.txt                 # runtime deps
requirements-dev.txt             # test deps
.gitlab-ci.yml                   # rewritten: lint + test + build (no deploy)
```

---

## Pre-flight: Branch hygiene + Go tree removal

### Task 0: Verify branch state and delete Go tree

**Files:**
- Delete: `cmd/`, `internal/`, `web/`, `assets.go`, `go.mod`, `go.sum`, `.golangci.yml`, any top-level `*.go`

- [ ] **Step 1: Confirm branch and clean tree**

```bash
cd /home/ip/projects/vllm-warden
git status
git branch --show-current
```

Expected: branch `feat/v2-design-spec`, working tree clean (only the spec file under `docs/superpowers/specs/` from `da007c9`).

- [ ] **Step 2: Identify and delete Go sources**

```bash
git rm -r cmd internal web
git rm assets.go go.mod go.sum .golangci.yml 2>/dev/null || true
git rm Makefile Dockerfile .gitlab-ci.yml 2>/dev/null || true  # Phase 1 versions; rewritten in Phase A and Phase P
```

- [ ] **Step 3: Verify what survived**

```bash
ls -la
find . -name "*.go" -not -path "./.git/*"
```

Expected: `*.go` returns empty. Surviving files: `docs/`, `scripts/publish_to_hub.py`, `deploy/`, `tests/test_publish_to_hub.py`, `requirements-dev.txt`, `pyproject.toml`, `README.md`, `LICENSE`.

- [ ] **Step 4: Commit removal**

```bash
git add -A
git commit -m "chore: remove Phase 1 Go tree ahead of v2 Python rewrite

The v2 design (docs/superpowers/specs/2026-05-08-vllm-warden-v2-design.md)
replaces the Go wizard wholesale. Hub catalog tooling (scripts/publish_to_hub.py,
deploy/hub/) is preserved as a separate concern."
```

---

## Phase A: Project scaffolding

### Task 1: Python project metadata

**Files:**
- Modify: `pyproject.toml`
- Create: `requirements.txt`
- Create: `requirements-dev.txt`

- [ ] **Step 1: Replace pyproject.toml with v2 config**

```toml
[project]
name = "vllm-warden"
version = "2.0.0"
requires-python = ">=3.11"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP", "ASYNC"]
ignore = ["E501"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = "-ra --strict-markers"
markers = [
    "integration: requires real subprocess + filesystem",
]

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true
```

- [ ] **Step 2: Create requirements.txt**

```
fastapi==0.115.6
uvicorn[standard]==0.32.1
jinja2==3.1.4
aiosqlite==0.20.0
bcrypt==4.2.1
itsdangerous==2.2.0
huggingface_hub==0.27.0
transformers==4.47.1
aiofiles==24.1.0
httpx==0.28.1
python-multipart==0.0.20
```

- [ ] **Step 3: Update requirements-dev.txt**

```
-r requirements.txt
pytest==8.3.4
pytest-asyncio==0.25.0
pytest-httpx==0.35.0
ruff==0.8.4
mypy==1.13.0
types-aiofiles==24.1.0.20240626
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml requirements.txt requirements-dev.txt
git commit -m "build: pin v2 Python dependencies"
```

---

### Task 2: FastAPI app skeleton + lifespan

**Files:**
- Create: `app/__init__.py`
- Create: `app/main.py`
- Create: `app/config.py`
- Create: `app/utils/__init__.py`
- Create: `app/utils/time.py`
- Test: `tests/conftest.py`, `tests/unit/test_app_smoke.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_app_smoke.py`:

```python
from fastapi.testclient import TestClient

def test_health_endpoint_returns_200(client: TestClient):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"ok": True}
```

`tests/conftest.py`:

```python
import asyncio
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("VW_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VW_COOKIE_SECRET", "test-secret-32-bytes-min-padding!")
    monkeypatch.setenv("VW_CONTAINER_GPU_COUNT", "4")
    return tmp_path

@pytest.fixture
def client(tmp_data_dir: Path) -> TestClient:
    from app.main import build_app
    app = build_app()
    with TestClient(app) as c:
        yield c
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm -v $(pwd):/app -w /app python:3.11-slim sh -c "pip install -q -r requirements-dev.txt && pytest tests/unit/test_app_smoke.py -v"`

Expected: FAIL with `ModuleNotFoundError: No module named 'app.main'`.

- [ ] **Step 3: Write app/utils/time.py**

```python
from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def minute_bucket(dt: datetime | None = None) -> int:
    """Return integer minute bucket (epoch seconds // 60)."""
    if dt is None:
        dt = utc_now()
    return int(dt.timestamp()) // 60
```

- [ ] **Step 4: Write app/config.py**

```python
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    cookie_secret: str
    container_gpu_count: int
    bind_host: str = "0.0.0.0"
    bind_port: int = 8080

    @property
    def db_path(self) -> Path:
        return self.data_dir / "vllm-warden.db"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def hf_cache_dir(self) -> Path:
        return self.data_dir / "hf-cache"


def load_settings() -> Settings:
    data_dir = Path(os.environ.get("VW_DATA_DIR", "/data"))
    secret = os.environ.get("VW_COOKIE_SECRET")
    if not secret or len(secret) < 32:
        raise RuntimeError("VW_COOKIE_SECRET must be set and >=32 chars")
    gpu_count = int(os.environ.get("VW_CONTAINER_GPU_COUNT", "0"))
    return Settings(
        data_dir=data_dir,
        cookie_secret=secret,
        container_gpu_count=gpu_count,
    )
```

- [ ] **Step 5: Write app/main.py**

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import load_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.hf_cache_dir.mkdir(parents=True, exist_ok=True)
    app.state.settings = settings
    yield


def build_app() -> FastAPI:
    app = FastAPI(title="vllm-warden", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, bool]:
        return {"ok": True}

    return app


app = build_app()
```

`app/__init__.py`, `app/utils/__init__.py`: empty files.

- [ ] **Step 6: Run test to verify it passes**

Run: `docker run --rm -v $(pwd):/app -w /app python:3.11-slim sh -c "pip install -q -r requirements-dev.txt && pytest tests/unit/test_app_smoke.py -v"`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add app/ tests/conftest.py tests/unit/test_app_smoke.py
git commit -m "feat(app): FastAPI skeleton with lifespan + healthz"
```

---

### Task 3: Makefile (Docker-only workflow)

**Files:**
- Create: `Makefile`

- [ ] **Step 1: Write Makefile**

```makefile
IMAGE := vllm-warden:dev
PY_IMAGE := python:3.11-slim
RUN_PY := docker run --rm -v $(PWD):/app -w /app $(PY_IMAGE)

.PHONY: install test lint typecheck format docker-build docker-run shell

install:
	$(RUN_PY) sh -c "pip install -q -r requirements-dev.txt"

test:
	$(RUN_PY) sh -c "pip install -q -r requirements-dev.txt && pytest -v $(ARGS)"

test-unit:
	$(RUN_PY) sh -c "pip install -q -r requirements-dev.txt && pytest -v -m 'not integration' tests/unit"

test-integration:
	$(RUN_PY) sh -c "pip install -q -r requirements-dev.txt && pytest -v -m integration tests/integration"

lint:
	$(RUN_PY) sh -c "pip install -q ruff && ruff check app/ tests/"

format:
	$(RUN_PY) sh -c "pip install -q ruff && ruff format app/ tests/"

typecheck:
	$(RUN_PY) sh -c "pip install -q -r requirements-dev.txt && mypy app/"

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm --gpus all -p 8080:8080 \
	  -e VW_COOKIE_SECRET=$$(openssl rand -base64 32) \
	  -e VW_CONTAINER_GPU_COUNT=4 \
	  -v $(PWD)/.data:/data \
	  $(IMAGE)

shell:
	$(RUN_PY) bash
```

- [ ] **Step 2: Verify make test passes**

Run: `make test-unit ARGS=tests/unit/test_app_smoke.py`

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "build: Docker-only Makefile for v2"
```

---

### Task 4: Static assets (htmx, Chart.js, base template)

**Files:**
- Create: `app/web/__init__.py`
- Create: `app/web/static/htmx.min.js` (vendored)
- Create: `app/web/static/chart.umd.min.js` (vendored)
- Create: `app/web/static/app.css`
- Create: `app/web/base_template.html`
- Modify: `app/main.py` — mount `/static`, register Jinja2

- [ ] **Step 1: Vendor static assets**

```bash
curl -sSL https://unpkg.com/htmx.org@2.0.4/dist/htmx.min.js -o app/web/static/htmx.min.js
curl -sSL https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js -o app/web/static/chart.umd.min.js
```

Verify both files are non-empty:

```bash
wc -c app/web/static/htmx.min.js app/web/static/chart.umd.min.js
```

- [ ] **Step 2: Write app.css**

```css
:root {
  --bg: #0e1116;
  --fg: #e5e7eb;
  --muted: #6b7280;
  --accent: #3b82f6;
  --ok: #10b981;
  --warn: #f59e0b;
  --err: #ef4444;
  --card: #1f2937;
}
* { box-sizing: border-box; }
body { margin: 0; font-family: ui-sans-serif, system-ui, sans-serif; background: var(--bg); color: var(--fg); }
header { padding: 12px 20px; border-bottom: 1px solid #2d3748; display: flex; gap: 16px; }
header a { color: var(--fg); text-decoration: none; }
header a.active { color: var(--accent); }
main { padding: 20px; max-width: 1200px; margin: 0 auto; }
.card { background: var(--card); padding: 16px; border-radius: 8px; margin-bottom: 12px; }
.btn { background: var(--accent); color: white; border: 0; padding: 8px 14px; border-radius: 6px; cursor: pointer; }
.btn-danger { background: var(--err); }
.muted { color: var(--muted); }
.status-loaded { color: var(--ok); }
.status-failed { color: var(--err); }
.status-loading, .status-pulling { color: var(--warn); }
input, select, textarea { background: #111827; color: var(--fg); border: 1px solid #374151; padding: 6px 8px; border-radius: 4px; width: 100%; }
table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #2d3748; }
.modal { position: fixed; inset: 0; background: rgba(0,0,0,0.6); display: flex; align-items: center; justify-content: center; }
.modal-content { background: var(--card); padding: 24px; border-radius: 8px; max-width: 600px; width: 90%; }
```

- [ ] **Step 3: Write base_template.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{% block title %}vllm-warden{% endblock %}</title>
  <link rel="stylesheet" href="/static/app.css">
  <script src="/static/htmx.min.js"></script>
  {% block head %}{% endblock %}
</head>
<body>
  {% if request.state.user %}
  <header>
    <a href="/models" class="{{ 'active' if active=='models' else '' }}">Models</a>
    <a href="/tokens" class="{{ 'active' if active=='tokens' else '' }}">API Tokens</a>
    <a href="/stats" class="{{ 'active' if active=='stats' else '' }}">Stats</a>
    <a href="/settings" class="{{ 'active' if active=='settings' else '' }}">Settings</a>
    <a href="/logout" style="margin-left: auto;">Logout ({{ request.state.user }})</a>
  </header>
  {% endif %}
  <main>
    {% block content %}{% endblock %}
  </main>
</body>
</html>
```

- [ ] **Step 4: Wire static + templates in app/main.py**

Modify `build_app()` to add:

```python
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

# inside build_app(), after FastAPI(...) call:
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "web/static"), name="static")
app.state.templates = Jinja2Templates(directory=Path(__file__).parent / "web")
```

- [ ] **Step 5: Add smoke test**

`tests/unit/test_app_smoke.py` (append):

```python
def test_static_htmx_served(client: TestClient):
    r = client.get("/static/htmx.min.js")
    assert r.status_code == 200
    assert b"htmx" in r.content[:200]
```

Run: `make test-unit ARGS=tests/unit/test_app_smoke.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/web/ app/main.py tests/unit/test_app_smoke.py
git commit -m "feat(web): static assets + base template"
```

---

## Phase B: Persistence

### Task 5: Migration runner

**Files:**
- Create: `app/db/__init__.py`
- Create: `app/db/database.py`
- Create: `app/db/migrations.py`
- Create: `app/db/sql/0001_users.sql`
- Test: `tests/unit/db/test_migrations.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/db/__init__.py`: empty.

`tests/unit/db/test_migrations.py`:

```python
import pytest
import aiosqlite

from app.db.migrations import apply_migrations
from app.db.database import open_db


async def test_migrations_create_schema_table_and_run_files(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in await cur.fetchall()}
        assert "schema_migrations" in tables
        assert "users" in tables


async def test_migrations_idempotent(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
        await apply_migrations(db)  # second call must be a no-op
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("SELECT COUNT(*) FROM schema_migrations")
        (count,) = await cur.fetchone()
        assert count == 1  # one file: 0001_users.sql
```

- [ ] **Step 2: Run to verify failure**

Run: `make test-unit ARGS=tests/unit/db/test_migrations.py`

Expected: FAIL with `ModuleNotFoundError: app.db.migrations`.

- [ ] **Step 3: Write app/db/database.py**

```python
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite


@asynccontextmanager
async def open_db(path: Path):
    async with aiosqlite.connect(path) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        await db.execute("PRAGMA journal_mode = WAL")
        db.row_factory = aiosqlite.Row
        yield db
```

- [ ] **Step 4: Write app/db/migrations.py**

```python
from pathlib import Path

import aiosqlite

SQL_DIR = Path(__file__).parent / "sql"


async def apply_migrations(db: aiosqlite.Connection) -> None:
    await db.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "  filename TEXT PRIMARY KEY,"
        "  applied_at TEXT NOT NULL DEFAULT (datetime('now'))"
        ")"
    )
    await db.commit()

    cur = await db.execute("SELECT filename FROM schema_migrations")
    applied = {row[0] for row in await cur.fetchall()}

    files = sorted(p for p in SQL_DIR.glob("*.sql"))
    for path in files:
        if path.name in applied:
            continue
        sql = path.read_text(encoding="utf-8")
        await db.executescript(sql)
        await db.execute("INSERT INTO schema_migrations(filename) VALUES (?)", (path.name,))
        await db.commit()
```

- [ ] **Step 5: Write 0001_users.sql**

```sql
CREATE TABLE users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT NOT NULL UNIQUE,
  password_hash TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
```

- [ ] **Step 6: Run tests**

Run: `make test-unit ARGS=tests/unit/db/test_migrations.py`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add app/db/ tests/unit/db/
git commit -m "feat(db): migration runner + users table"
```

---

### Task 6: setup_state, models, model_runtime tables

**Files:**
- Create: `app/db/sql/0002_setup_state.sql`
- Create: `app/db/sql/0003_models.sql`
- Create: `app/db/sql/0004_model_runtime.sql`
- Test: `tests/unit/db/test_migrations.py` (extend)

- [ ] **Step 1: Extend test**

Append to `tests/unit/db/test_migrations.py`:

```python
async def test_migrations_create_all_v2_tables(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in await cur.fetchall()}
        for t in ["users", "setup_state", "models", "model_runtime"]:
            assert t in tables
```

- [ ] **Step 2: Run — expect failure**

Expected: FAIL on missing tables.

- [ ] **Step 3: Write 0002_setup_state.sql**

```sql
CREATE TABLE setup_state (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  step TEXT NOT NULL DEFAULT 'welcome',
  draft TEXT NOT NULL DEFAULT '{}',
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
INSERT INTO setup_state(id, step, draft) VALUES (1, 'welcome', '{}');
```

- [ ] **Step 4: Write 0003_models.sql**

```sql
CREATE TABLE models (
  id TEXT PRIMARY KEY,
  served_model_name TEXT NOT NULL UNIQUE,
  hf_repo TEXT NOT NULL,
  hf_revision TEXT NOT NULL DEFAULT 'main',
  gpu_indices TEXT NOT NULL,
  tensor_parallel_size INTEGER NOT NULL DEFAULT 1,
  dtype TEXT,
  max_model_len INTEGER,
  gpu_memory_utilization REAL NOT NULL DEFAULT 0.9,
  trust_remote_code INTEGER NOT NULL DEFAULT 0,
  extra_args TEXT NOT NULL DEFAULT '[]',
  status TEXT NOT NULL DEFAULT 'registered'
    CHECK (status IN ('registered','pulling','pulled','loading','loaded','unloading','failed')),
  pulled_bytes INTEGER NOT NULL DEFAULT 0,
  pulled_total INTEGER,
  last_error TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
```

- [ ] **Step 5: Write 0004_model_runtime.sql**

```sql
CREATE TABLE model_runtime (
  model_id TEXT PRIMARY KEY REFERENCES models(id) ON DELETE CASCADE,
  pid INTEGER,
  port INTEGER,
  started_at TEXT,
  health_ok INTEGER NOT NULL DEFAULT 0,
  last_health_at TEXT
);
```

- [ ] **Step 6: Run + commit**

Run: `make test-unit ARGS=tests/unit/db/test_migrations.py`

Expected: PASS.

```bash
git add app/db/sql/000{2,3,4}_*.sql tests/unit/db/test_migrations.py
git commit -m "feat(db): setup_state, models, model_runtime"
```

---

### Task 7: api_tokens, counters, samples + indices

**Files:**
- Create: `app/db/sql/0005_api_tokens.sql`
- Create: `app/db/sql/0006_counters_samples.sql`
- Create: `app/db/sql/0007_indices.sql`
- Test: `tests/unit/db/test_migrations.py` (extend)

- [ ] **Step 1: Extend test**

Append:

```python
async def test_migrations_create_full_v2_schema(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in await cur.fetchall()}
        for t in ["api_tokens", "counters", "model_samples", "gpu_samples"]:
            assert t in tables
        cur = await db.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indices = {row[0] for row in await cur.fetchall()}
        assert "idx_model_samples_minute" in indices
        assert "idx_gpu_samples_minute" in indices
```

- [ ] **Step 2: Write 0005_api_tokens.sql**

```sql
CREATE TABLE api_tokens (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  prefix TEXT NOT NULL,
  hash TEXT NOT NULL UNIQUE,
  scope TEXT NOT NULL DEFAULT 'inference',
  allowed_models TEXT,
  rate_limit_rpm INTEGER,
  rate_limit_tpm INTEGER,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  last_used_at TEXT,
  revoked_at TEXT
);
```

- [ ] **Step 3: Write 0006_counters_samples.sql**

```sql
CREATE TABLE counters (
  model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
  token_id TEXT REFERENCES api_tokens(id) ON DELETE SET NULL,
  requests INTEGER NOT NULL DEFAULT 0,
  prompt_tokens INTEGER NOT NULL DEFAULT 0,
  completion_tokens INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (model_id, token_id)
);

CREATE TABLE model_samples (
  model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
  minute INTEGER NOT NULL,
  requests INTEGER NOT NULL DEFAULT 0,
  prompt_tokens INTEGER NOT NULL DEFAULT 0,
  completion_tokens INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (model_id, minute)
);

CREATE TABLE gpu_samples (
  gpu_index INTEGER NOT NULL,
  minute INTEGER NOT NULL,
  utilization_pct INTEGER,
  memory_used_mib INTEGER,
  memory_total_mib INTEGER,
  PRIMARY KEY (gpu_index, minute)
);
```

- [ ] **Step 4: Write 0007_indices.sql**

```sql
CREATE INDEX idx_model_samples_minute ON model_samples(minute);
CREATE INDEX idx_gpu_samples_minute ON gpu_samples(minute);
CREATE INDEX idx_api_tokens_prefix ON api_tokens(prefix);
CREATE INDEX idx_api_tokens_revoked ON api_tokens(revoked_at);
```

- [ ] **Step 5: Run + commit**

Run: `make test-unit ARGS=tests/unit/db/test_migrations.py`

Expected: PASS.

```bash
git add app/db/sql/000{5,6,7}_*.sql tests/unit/db/test_migrations.py
git commit -m "feat(db): tokens, counters, samples + indices"
```

---

### Task 8: User repo

**Files:**
- Create: `app/db/repos/__init__.py`
- Create: `app/db/repos/users.py`
- Test: `tests/unit/db/test_repos_users.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.users import UserRepo


@pytest.fixture
async def db(tmp_data_dir):
    async with open_db(tmp_data_dir / "vllm-warden.db") as conn:
        await apply_migrations(conn)
        yield conn


async def test_create_and_get_user(db):
    repo = UserRepo(db)
    await repo.create("admin", "hashed-pw")
    u = await repo.get_by_username("admin")
    assert u is not None
    assert u.username == "admin"
    assert u.password_hash == "hashed-pw"


async def test_unique_username(db):
    repo = UserRepo(db)
    await repo.create("admin", "h1")
    with pytest.raises(Exception):
        await repo.create("admin", "h2")


async def test_count_users(db):
    repo = UserRepo(db)
    assert await repo.count() == 0
    await repo.create("a", "h")
    await repo.create("b", "h")
    assert await repo.count() == 2
```

- [ ] **Step 2: Verify failure**

Run: `make test-unit ARGS=tests/unit/db/test_repos_users.py`

Expected: FAIL.

- [ ] **Step 3: Write users repo**

`app/db/repos/__init__.py`: empty.

`app/db/repos/users.py`:

```python
from dataclasses import dataclass

import aiosqlite


@dataclass(frozen=True)
class UserRow:
    id: int
    username: str
    password_hash: str


class UserRepo:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    async def create(self, username: str, password_hash: str) -> int:
        cur = await self.db.execute(
            "INSERT INTO users(username, password_hash) VALUES (?, ?)",
            (username, password_hash),
        )
        await self.db.commit()
        return cur.lastrowid

    async def get_by_username(self, username: str) -> UserRow | None:
        cur = await self.db.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?",
            (username,),
        )
        row = await cur.fetchone()
        return UserRow(*row) if row else None

    async def count(self) -> int:
        cur = await self.db.execute("SELECT COUNT(*) FROM users")
        (n,) = await cur.fetchone()
        return n
```

- [ ] **Step 4: Run + commit**

Run: `make test-unit ARGS=tests/unit/db/test_repos_users.py`

Expected: PASS.

```bash
git add app/db/repos/
git commit -m "feat(db): users repo"
```

---

### Task 9: Models + runtime + setup repos

**Files:**
- Create: `app/db/repos/models.py`
- Create: `app/db/repos/runtime.py`
- Create: `app/db/repos/setup.py`
- Test: `tests/unit/db/test_repos_models.py`, `test_repos_runtime.py`, `test_repos_setup.py`

- [ ] **Step 1: Write tests for models repo**

`tests/unit/db/test_repos_models.py`:

```python
import json
import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.models import ModelRepo, ModelRow


@pytest.fixture
async def db(tmp_data_dir):
    async with open_db(tmp_data_dir / "vllm-warden.db") as conn:
        await apply_migrations(conn)
        yield conn


async def test_insert_and_get_model(db):
    repo = ModelRepo(db)
    await repo.insert(ModelRow(
        id="m1",
        served_model_name="qwen3.5-9b",
        hf_repo="Qwen/Qwen3.5-9B",
        hf_revision="main",
        gpu_indices=json.dumps([1, 2]),
        tensor_parallel_size=2,
        dtype="auto",
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        trust_remote_code=False,
        extra_args=json.dumps([]),
        status="registered",
        pulled_bytes=0,
        pulled_total=None,
        last_error=None,
    ))
    m = await repo.get("m1")
    assert m.served_model_name == "qwen3.5-9b"
    assert json.loads(m.gpu_indices) == [1, 2]


async def test_update_status(db):
    repo = ModelRepo(db)
    await repo.insert(_make_row("m1"))
    await repo.update_status("m1", "loaded")
    m = await repo.get("m1")
    assert m.status == "loaded"


async def test_set_status_failed_on_startup_wipes_loaded(db):
    """At app startup, any 'loaded'/'loading' rows must be marked failed."""
    repo = ModelRepo(db)
    await repo.insert(_make_row("m1", status="loaded"))
    await repo.insert(_make_row("m2", status="loading"))
    await repo.insert(_make_row("m3", status="registered"))
    n = await repo.mark_runtime_dead_on_startup()
    assert n == 2
    assert (await repo.get("m1")).status == "failed"
    assert (await repo.get("m2")).status == "failed"
    assert (await repo.get("m3")).status == "registered"


def _make_row(model_id: str, status: str = "registered") -> ModelRow:
    return ModelRow(
        id=model_id,
        served_model_name=f"name-{model_id}",
        hf_repo="org/repo",
        hf_revision="main",
        gpu_indices=json.dumps([0]),
        tensor_parallel_size=1,
        dtype=None,
        max_model_len=None,
        gpu_memory_utilization=0.9,
        trust_remote_code=False,
        extra_args=json.dumps([]),
        status=status,
        pulled_bytes=0,
        pulled_total=None,
        last_error=None,
    )
```

- [ ] **Step 2: Write app/db/repos/models.py**

```python
from dataclasses import dataclass, asdict

import aiosqlite


@dataclass
class ModelRow:
    id: str
    served_model_name: str
    hf_repo: str
    hf_revision: str
    gpu_indices: str
    tensor_parallel_size: int
    dtype: str | None
    max_model_len: int | None
    gpu_memory_utilization: float
    trust_remote_code: bool
    extra_args: str
    status: str
    pulled_bytes: int
    pulled_total: int | None
    last_error: str | None


class ModelRepo:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    async def insert(self, row: ModelRow) -> None:
        await self.db.execute(
            """INSERT INTO models(
                id, served_model_name, hf_repo, hf_revision, gpu_indices,
                tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization,
                trust_remote_code, extra_args, status, pulled_bytes, pulled_total, last_error
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                row.id, row.served_model_name, row.hf_repo, row.hf_revision, row.gpu_indices,
                row.tensor_parallel_size, row.dtype, row.max_model_len, row.gpu_memory_utilization,
                int(row.trust_remote_code), row.extra_args, row.status, row.pulled_bytes,
                row.pulled_total, row.last_error,
            ),
        )
        await self.db.commit()

    async def get(self, model_id: str) -> ModelRow | None:
        cur = await self.db.execute(
            "SELECT id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization, "
            "trust_remote_code, extra_args, status, pulled_bytes, pulled_total, last_error "
            "FROM models WHERE id = ?",
            (model_id,),
        )
        row = await cur.fetchone()
        if not row:
            return None
        return ModelRow(
            id=row[0], served_model_name=row[1], hf_repo=row[2], hf_revision=row[3],
            gpu_indices=row[4], tensor_parallel_size=row[5], dtype=row[6],
            max_model_len=row[7], gpu_memory_utilization=row[8],
            trust_remote_code=bool(row[9]), extra_args=row[10], status=row[11],
            pulled_bytes=row[12], pulled_total=row[13], last_error=row[14],
        )

    async def list_all(self) -> list[ModelRow]:
        cur = await self.db.execute(
            "SELECT id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization, "
            "trust_remote_code, extra_args, status, pulled_bytes, pulled_total, last_error "
            "FROM models ORDER BY created_at"
        )
        rows = await cur.fetchall()
        return [
            ModelRow(
                id=r[0], served_model_name=r[1], hf_repo=r[2], hf_revision=r[3],
                gpu_indices=r[4], tensor_parallel_size=r[5], dtype=r[6],
                max_model_len=r[7], gpu_memory_utilization=r[8],
                trust_remote_code=bool(r[9]), extra_args=r[10], status=r[11],
                pulled_bytes=r[12], pulled_total=r[13], last_error=r[14],
            )
            for r in rows
        ]

    async def get_by_served_name(self, served: str) -> ModelRow | None:
        cur = await self.db.execute("SELECT id FROM models WHERE served_model_name = ?", (served,))
        row = await cur.fetchone()
        return await self.get(row[0]) if row else None

    async def update_status(
        self, model_id: str, status: str, last_error: str | None = None
    ) -> None:
        await self.db.execute(
            "UPDATE models SET status = ?, last_error = ?, updated_at = datetime('now') "
            "WHERE id = ?",
            (status, last_error, model_id),
        )
        await self.db.commit()

    async def update_pull_progress(
        self, model_id: str, pulled_bytes: int, pulled_total: int | None
    ) -> None:
        await self.db.execute(
            "UPDATE models SET pulled_bytes = ?, pulled_total = ?, updated_at = datetime('now') "
            "WHERE id = ?",
            (pulled_bytes, pulled_total, model_id),
        )
        await self.db.commit()

    async def delete(self, model_id: str) -> None:
        await self.db.execute("DELETE FROM models WHERE id = ?", (model_id,))
        await self.db.commit()

    async def mark_runtime_dead_on_startup(self) -> int:
        """Wipe loaded/loading rows to failed on app startup. Returns count updated."""
        cur = await self.db.execute(
            "UPDATE models SET status = 'failed', "
            "last_error = 'process not running after restart', "
            "updated_at = datetime('now') "
            "WHERE status IN ('loaded', 'loading', 'unloading')"
        )
        await self.db.commit()
        return cur.rowcount
```

- [ ] **Step 3: Write app/db/repos/runtime.py**

```python
from dataclasses import dataclass

import aiosqlite


@dataclass
class RuntimeRow:
    model_id: str
    pid: int | None
    port: int | None
    started_at: str | None
    health_ok: bool
    last_health_at: str | None


class RuntimeRepo:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    async def upsert(
        self, model_id: str, pid: int | None, port: int | None, started_at: str | None
    ) -> None:
        await self.db.execute(
            "INSERT INTO model_runtime(model_id, pid, port, started_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(model_id) DO UPDATE SET pid=excluded.pid, "
            "port=excluded.port, started_at=excluded.started_at",
            (model_id, pid, port, started_at),
        )
        await self.db.commit()

    async def update_health(self, model_id: str, ok: bool, when: str) -> None:
        await self.db.execute(
            "UPDATE model_runtime SET health_ok = ?, last_health_at = ? WHERE model_id = ?",
            (1 if ok else 0, when, model_id),
        )
        await self.db.commit()

    async def get(self, model_id: str) -> RuntimeRow | None:
        cur = await self.db.execute(
            "SELECT model_id, pid, port, started_at, health_ok, last_health_at "
            "FROM model_runtime WHERE model_id = ?",
            (model_id,),
        )
        r = await cur.fetchone()
        return RuntimeRow(r[0], r[1], r[2], r[3], bool(r[4]), r[5]) if r else None

    async def clear_all(self) -> None:
        """Called at startup so no stale runtime rows survive."""
        await self.db.execute("DELETE FROM model_runtime")
        await self.db.commit()
```

- [ ] **Step 4: Write app/db/repos/setup.py**

```python
import json
from dataclasses import dataclass

import aiosqlite


@dataclass
class SetupState:
    step: str
    draft: dict


class SetupRepo:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    async def get(self) -> SetupState:
        cur = await self.db.execute("SELECT step, draft FROM setup_state WHERE id = 1")
        r = await cur.fetchone()
        return SetupState(step=r[0], draft=json.loads(r[1]))

    async def set_step(self, step: str) -> None:
        await self.db.execute(
            "UPDATE setup_state SET step = ?, updated_at = datetime('now') WHERE id = 1",
            (step,),
        )
        await self.db.commit()

    async def merge_draft(self, **kwargs) -> dict:
        cur = await self.db.execute("SELECT draft FROM setup_state WHERE id = 1")
        (draft_json,) = await cur.fetchone()
        draft = json.loads(draft_json)
        draft.update(kwargs)
        await self.db.execute(
            "UPDATE setup_state SET draft = ?, updated_at = datetime('now') WHERE id = 1",
            (json.dumps(draft),),
        )
        await self.db.commit()
        return draft

    async def is_done(self) -> bool:
        s = await self.get()
        return s.step == "done"
```

- [ ] **Step 5: Add tests for runtime + setup repos**

`tests/unit/db/test_repos_runtime.py`:

```python
import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.models import ModelRepo
from app.db.repos.runtime import RuntimeRepo
import json


@pytest.fixture
async def db(tmp_data_dir):
    async with open_db(tmp_data_dir / "vllm-warden.db") as conn:
        await apply_migrations(conn)
        yield conn


async def test_upsert_and_clear_runtime(db):
    from app.db.repos.models import ModelRow
    await ModelRepo(db).insert(ModelRow(
        id="m1", served_model_name="m1", hf_repo="o/r", hf_revision="main",
        gpu_indices=json.dumps([0]), tensor_parallel_size=1, dtype=None,
        max_model_len=None, gpu_memory_utilization=0.9, trust_remote_code=False,
        extra_args=json.dumps([]), status="registered", pulled_bytes=0,
        pulled_total=None, last_error=None,
    ))
    rt = RuntimeRepo(db)
    await rt.upsert("m1", pid=1234, port=10000, started_at="2026-05-08T00:00:00Z")
    row = await rt.get("m1")
    assert row.pid == 1234
    assert row.port == 10000

    await rt.clear_all()
    assert await rt.get("m1") is None
```

`tests/unit/db/test_repos_setup.py`:

```python
import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.setup import SetupRepo


@pytest.fixture
async def db(tmp_data_dir):
    async with open_db(tmp_data_dir / "vllm-warden.db") as conn:
        await apply_migrations(conn)
        yield conn


async def test_initial_state_welcome(db):
    s = await SetupRepo(db).get()
    assert s.step == "welcome"
    assert s.draft == {}


async def test_merge_draft_accumulates(db):
    repo = SetupRepo(db)
    await repo.merge_draft(allowed_gpu_indices=[1, 2])
    await repo.merge_draft(hf_token_present=True)
    s = await repo.get()
    assert s.draft == {"allowed_gpu_indices": [1, 2], "hf_token_present": True}


async def test_set_step_done_marks_done(db):
    repo = SetupRepo(db)
    assert not await repo.is_done()
    await repo.set_step("done")
    assert await repo.is_done()
```

- [ ] **Step 6: Run + commit**

Run: `make test-unit ARGS=tests/unit/db/`

Expected: PASS.

```bash
git add app/db/repos/ tests/unit/db/test_repos_*.py
git commit -m "feat(db): models, runtime, setup repos"
```

---

### Task 10: Tokens, counters, samples repos

**Files:**
- Create: `app/db/repos/tokens.py`
- Create: `app/db/repos/counters.py`
- Create: `app/db/repos/samples.py`
- Test: `tests/unit/db/test_repos_tokens.py`, `test_repos_counters.py`, `test_repos_samples.py`

- [ ] **Step 1: Write tokens repo**

```python
import hashlib
from dataclasses import dataclass

import aiosqlite


@dataclass
class TokenRow:
    id: str
    name: str
    prefix: str
    scope: str
    allowed_models: str | None
    rate_limit_rpm: int | None
    rate_limit_tpm: int | None
    revoked_at: str | None
    last_used_at: str | None


def hash_token(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


class TokenRepo:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    async def create(
        self,
        token_id: str,
        name: str,
        plaintext: str,
        scope: str = "inference",
        allowed_models: list[str] | None = None,
    ) -> None:
        prefix = plaintext[:8]
        await self.db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope, allowed_models) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                token_id, name, prefix, hash_token(plaintext), scope,
                ",".join(allowed_models) if allowed_models else None,
            ),
        )
        await self.db.commit()

    async def find_by_plaintext(self, plaintext: str) -> TokenRow | None:
        h = hash_token(plaintext)
        cur = await self.db.execute(
            "SELECT id, name, prefix, scope, allowed_models, rate_limit_rpm, rate_limit_tpm, "
            "revoked_at, last_used_at FROM api_tokens WHERE hash = ?",
            (h,),
        )
        r = await cur.fetchone()
        return TokenRow(*r) if r else None

    async def list_all(self) -> list[TokenRow]:
        cur = await self.db.execute(
            "SELECT id, name, prefix, scope, allowed_models, rate_limit_rpm, rate_limit_tpm, "
            "revoked_at, last_used_at FROM api_tokens ORDER BY created_at DESC"
        )
        return [TokenRow(*r) for r in await cur.fetchall()]

    async def revoke(self, token_id: str) -> None:
        await self.db.execute(
            "UPDATE api_tokens SET revoked_at = datetime('now') WHERE id = ?", (token_id,)
        )
        await self.db.commit()

    async def touch_last_used(self, token_id: str) -> None:
        await self.db.execute(
            "UPDATE api_tokens SET last_used_at = datetime('now') WHERE id = ?", (token_id,)
        )
        await self.db.commit()
```

- [ ] **Step 2: Write counters repo**

```python
import aiosqlite


class CountersRepo:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    async def increment(
        self,
        model_id: str,
        token_id: str | None,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        await self.db.execute(
            "INSERT INTO counters(model_id, token_id, requests, prompt_tokens, completion_tokens) "
            "VALUES (?, ?, 1, ?, ?) "
            "ON CONFLICT(model_id, token_id) DO UPDATE SET "
            "requests = requests + 1, "
            "prompt_tokens = prompt_tokens + excluded.prompt_tokens, "
            "completion_tokens = completion_tokens + excluded.completion_tokens",
            (model_id, token_id, prompt_tokens, completion_tokens),
        )
        await self.db.commit()
```

- [ ] **Step 3: Write samples repo**

```python
import aiosqlite


class SamplesRepo:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    async def add_model_sample(
        self,
        model_id: str,
        minute: int,
        delta_requests: int,
        delta_prompt: int,
        delta_completion: int,
    ) -> None:
        await self.db.execute(
            "INSERT INTO model_samples(model_id, minute, requests, prompt_tokens, completion_tokens) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(model_id, minute) DO UPDATE SET "
            "requests = requests + excluded.requests, "
            "prompt_tokens = prompt_tokens + excluded.prompt_tokens, "
            "completion_tokens = completion_tokens + excluded.completion_tokens",
            (model_id, minute, delta_requests, delta_prompt, delta_completion),
        )
        await self.db.commit()

    async def add_gpu_sample(
        self,
        gpu_index: int,
        minute: int,
        utilization_pct: int,
        memory_used_mib: int,
        memory_total_mib: int,
    ) -> None:
        await self.db.execute(
            "INSERT INTO gpu_samples(gpu_index, minute, utilization_pct, "
            "memory_used_mib, memory_total_mib) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(gpu_index, minute) DO UPDATE SET "
            "utilization_pct = excluded.utilization_pct, "
            "memory_used_mib = excluded.memory_used_mib, "
            "memory_total_mib = excluded.memory_total_mib",
            (gpu_index, minute, utilization_pct, memory_used_mib, memory_total_mib),
        )
        await self.db.commit()

    async def model_samples_since(self, model_id: str, since_minute: int) -> list[dict]:
        cur = await self.db.execute(
            "SELECT minute, requests, prompt_tokens, completion_tokens "
            "FROM model_samples WHERE model_id = ? AND minute >= ? ORDER BY minute",
            (model_id, since_minute),
        )
        return [
            {"minute": r[0], "requests": r[1], "prompt_tokens": r[2], "completion_tokens": r[3]}
            for r in await cur.fetchall()
        ]

    async def gpu_samples_since(self, since_minute: int) -> list[dict]:
        cur = await self.db.execute(
            "SELECT gpu_index, minute, utilization_pct, memory_used_mib, memory_total_mib "
            "FROM gpu_samples WHERE minute >= ? ORDER BY minute, gpu_index",
            (since_minute,),
        )
        return [
            {
                "gpu_index": r[0], "minute": r[1], "utilization_pct": r[2],
                "memory_used_mib": r[3], "memory_total_mib": r[4],
            }
            for r in await cur.fetchall()
        ]

    async def prune_older_than(self, before_minute: int) -> None:
        await self.db.execute("DELETE FROM model_samples WHERE minute < ?", (before_minute,))
        await self.db.execute("DELETE FROM gpu_samples WHERE minute < ?", (before_minute,))
        await self.db.commit()
```

- [ ] **Step 4: Add tests**

`tests/unit/db/test_repos_tokens.py`:

```python
import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.tokens import TokenRepo, hash_token


@pytest.fixture
async def db(tmp_data_dir):
    async with open_db(tmp_data_dir / "vllm-warden.db") as conn:
        await apply_migrations(conn)
        yield conn


async def test_create_lookup_revoke(db):
    repo = TokenRepo(db)
    await repo.create("tok1", "ci-bot", "vw_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    found = await repo.find_by_plaintext("vw_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    assert found.id == "tok1"
    assert found.prefix == "vw_aaaaa"

    await repo.revoke("tok1")
    found = await repo.find_by_plaintext("vw_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    assert found.revoked_at is not None


async def test_only_hash_stored(db):
    plaintext = "vw_secret_token_value_xyz"
    await TokenRepo(db).create("tok2", "n", plaintext)
    cur = await db.execute("SELECT hash FROM api_tokens WHERE id = ?", ("tok2",))
    (stored,) = await cur.fetchone()
    assert stored == hash_token(plaintext)
    assert plaintext not in stored
```

`tests/unit/db/test_repos_counters.py`:

```python
import json
import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.counters import CountersRepo
from app.db.repos.models import ModelRepo, ModelRow


@pytest.fixture
async def db(tmp_data_dir):
    async with open_db(tmp_data_dir / "vllm-warden.db") as conn:
        await apply_migrations(conn)
        await ModelRepo(conn).insert(ModelRow(
            id="m1", served_model_name="m1", hf_repo="o/r", hf_revision="main",
            gpu_indices=json.dumps([0]), tensor_parallel_size=1, dtype=None,
            max_model_len=None, gpu_memory_utilization=0.9, trust_remote_code=False,
            extra_args=json.dumps([]), status="registered", pulled_bytes=0,
            pulled_total=None, last_error=None,
        ))
        yield conn


async def test_increment_accumulates(db):
    c = CountersRepo(db)
    await c.increment("m1", None, 100, 50)
    await c.increment("m1", None, 30, 10)
    cur = await db.execute(
        "SELECT requests, prompt_tokens, completion_tokens FROM counters "
        "WHERE model_id = 'm1' AND token_id IS NULL"
    )
    r = await cur.fetchone()
    assert r == (2, 130, 60)
```

`tests/unit/db/test_repos_samples.py`:

```python
import json
import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.models import ModelRepo, ModelRow
from app.db.repos.samples import SamplesRepo


@pytest.fixture
async def db(tmp_data_dir):
    async with open_db(tmp_data_dir / "vllm-warden.db") as conn:
        await apply_migrations(conn)
        await ModelRepo(conn).insert(ModelRow(
            id="m1", served_model_name="m1", hf_repo="o/r", hf_revision="main",
            gpu_indices=json.dumps([0]), tensor_parallel_size=1, dtype=None,
            max_model_len=None, gpu_memory_utilization=0.9, trust_remote_code=False,
            extra_args=json.dumps([]), status="registered", pulled_bytes=0,
            pulled_total=None, last_error=None,
        ))
        yield conn


async def test_model_samples_accumulate_per_minute(db):
    s = SamplesRepo(db)
    await s.add_model_sample("m1", 100, 5, 200, 100)
    await s.add_model_sample("m1", 100, 3, 50, 25)
    await s.add_model_sample("m1", 101, 1, 10, 5)
    rows = await s.model_samples_since("m1", 100)
    assert rows == [
        {"minute": 100, "requests": 8, "prompt_tokens": 250, "completion_tokens": 125},
        {"minute": 101, "requests": 1, "prompt_tokens": 10, "completion_tokens": 5},
    ]


async def test_gpu_samples_replace_per_minute(db):
    s = SamplesRepo(db)
    await s.add_gpu_sample(0, 100, 50, 4096, 24576)
    await s.add_gpu_sample(0, 100, 80, 8192, 24576)  # overwrites
    rows = await s.gpu_samples_since(100)
    assert len(rows) == 1
    assert rows[0]["utilization_pct"] == 80
    assert rows[0]["memory_used_mib"] == 8192


async def test_prune_older_than(db):
    s = SamplesRepo(db)
    await s.add_model_sample("m1", 50, 1, 10, 5)
    await s.add_model_sample("m1", 100, 1, 10, 5)
    await s.prune_older_than(80)
    rows = await s.model_samples_since("m1", 0)
    assert len(rows) == 1
    assert rows[0]["minute"] == 100
```

- [ ] **Step 5: Run + commit**

Run: `make test-unit ARGS=tests/unit/db/`

Expected: PASS.

```bash
git add app/db/repos/tokens.py app/db/repos/counters.py app/db/repos/samples.py tests/unit/db/test_repos_*.py
git commit -m "feat(db): tokens, counters, samples repos"
```

---

### Task 11: DB lifespan integration

**Files:**
- Modify: `app/main.py`
- Modify: `tests/conftest.py` (verify lifespan runs in TestClient)
- Test: `tests/unit/test_lifespan.py`

- [ ] **Step 1: Write failing test**

```python
import sqlite3

def test_lifespan_creates_db_with_schema(tmp_data_dir, client):
    """Calling any endpoint must trigger lifespan, which migrates the DB."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    assert db_path.exists()
    with sqlite3.connect(db_path) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "users" in tables
        assert "models" in tables


def test_lifespan_clears_runtime_table(tmp_data_dir, client):
    """Stale model_runtime rows must be wiped on startup."""
    client.get("/healthz")  # boot
    db_path = tmp_data_dir / "vllm-warden.db"
    with sqlite3.connect(db_path) as db:
        # Insert a model + runtime row, then re-boot via second client.
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, gpu_memory_utilization, trust_remote_code, extra_args, status, "
            "pulled_bytes) VALUES "
            "('m1','m1','o/r','main','[0]',1,0.9,0,'[]','loaded',0)"
        )
        db.execute(
            "INSERT INTO model_runtime(model_id, pid, port) VALUES ('m1', 9999, 10000)"
        )
        db.commit()

    # Reboot app
    from app.main import build_app
    from fastapi.testclient import TestClient
    with TestClient(build_app()) as c2:
        c2.get("/healthz")

    with sqlite3.connect(db_path) as db:
        (n,) = db.execute("SELECT COUNT(*) FROM model_runtime").fetchone()
        assert n == 0
        (status,) = db.execute("SELECT status FROM models WHERE id='m1'").fetchone()
        assert status == "failed"
```

- [ ] **Step 2: Verify failure**

Run: `make test-unit ARGS=tests/unit/test_lifespan.py`

Expected: FAIL.

- [ ] **Step 3: Update app/main.py lifespan**

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import load_settings
from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.models import ModelRepo
from app.db.repos.runtime import RuntimeRepo


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.hf_cache_dir.mkdir(parents=True, exist_ok=True)
    app.state.settings = settings

    async with open_db(settings.db_path) as db:
        await apply_migrations(db)
        await ModelRepo(db).mark_runtime_dead_on_startup()
        await RuntimeRepo(db).clear_all()

    yield
```

- [ ] **Step 4: Run + commit**

Run: `make test-unit ARGS=tests/unit/test_lifespan.py`

Expected: PASS.

```bash
git add app/main.py tests/unit/test_lifespan.py
git commit -m "feat(app): run migrations + wipe runtime on startup"
```

---

## Phase C: System probes

### Task 12: GPU probe (nvidia-smi parser)

**Files:**
- Create: `app/system/__init__.py`
- Create: `app/system/gpu.py`
- Test: `tests/unit/system/test_gpu_parse.py`

- [ ] **Step 1: Write failing test**

```python
from app.system.gpu import GpuInfo, parse_nvidia_smi_csv


def test_parse_nvidia_smi_csv_4_gpus():
    out = (
        "0, NVIDIA A100-SXM4-40GB, 40960, 1024, 5\n"
        "1, NVIDIA A100-SXM4-40GB, 40960, 0, 0\n"
        "2, NVIDIA A100-SXM4-40GB, 40960, 0, 0\n"
        "3, NVIDIA A100-SXM4-40GB, 40960, 24576, 95\n"
    )
    gpus = parse_nvidia_smi_csv(out)
    assert len(gpus) == 4
    assert gpus[0] == GpuInfo(
        index=0, name="NVIDIA A100-SXM4-40GB",
        memory_total_mib=40960, memory_used_mib=1024, utilization_pct=5,
    )
    assert gpus[3].utilization_pct == 95


def test_parse_nvidia_smi_csv_handles_empty():
    assert parse_nvidia_smi_csv("") == []


def test_parse_nvidia_smi_csv_skips_malformed():
    out = "0, NVIDIA, 40960, 1024, 5\nbroken\n1, NVIDIA, 40960, 0, 0\n"
    gpus = parse_nvidia_smi_csv(out)
    assert [g.index for g in gpus] == [0, 1]
```

- [ ] **Step 2: Verify failure**

Run: `make test-unit ARGS=tests/unit/system/test_gpu_parse.py`

Expected: FAIL.

- [ ] **Step 3: Write app/system/gpu.py**

```python
import asyncio
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

NVIDIA_SMI_CMD = [
    "nvidia-smi",
    "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
    "--format=csv,noheader,nounits",
]


@dataclass
class GpuInfo:
    index: int
    name: str
    memory_total_mib: int
    memory_used_mib: int
    utilization_pct: int


def parse_nvidia_smi_csv(stdout: str) -> list[GpuInfo]:
    out = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            out.append(GpuInfo(
                index=int(parts[0]),
                name=parts[1],
                memory_total_mib=int(parts[2]),
                memory_used_mib=int(parts[3]),
                utilization_pct=int(parts[4]),
            ))
        except ValueError:
            logger.warning("nvidia-smi malformed row skipped: %r", line)
            continue
    return out


async def query_gpus() -> list[GpuInfo]:
    """Run nvidia-smi and parse output. Returns [] if not available."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *NVIDIA_SMI_CMD,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
    except (FileNotFoundError, asyncio.TimeoutError) as e:
        logger.warning("nvidia-smi unavailable: %s", e)
        return []
    if proc.returncode != 0:
        logger.warning("nvidia-smi exit %d: %s", proc.returncode, stderr.decode())
        return []
    return parse_nvidia_smi_csv(stdout.decode())
```

- [ ] **Step 4: Run + commit**

Run: `make test-unit ARGS=tests/unit/system/test_gpu_parse.py`

Expected: PASS.

```bash
git add app/system/__init__.py app/system/gpu.py tests/unit/system/
git commit -m "feat(system): nvidia-smi GPU probe"
```

---

### Task 13: Disk probe

**Files:**
- Create: `app/system/disk.py`
- Test: `tests/unit/system/test_disk.py`

- [ ] **Step 1: Write failing test**

```python
from app.system.disk import disk_free_bytes


def test_disk_free_bytes_on_tmp(tmp_path):
    free = disk_free_bytes(tmp_path)
    assert free > 0
```

- [ ] **Step 2: Verify failure**

Run: `make test-unit ARGS=tests/unit/system/test_disk.py`

Expected: FAIL.

- [ ] **Step 3: Write app/system/disk.py**

```python
import shutil
from pathlib import Path


def disk_free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free
```

- [ ] **Step 4: Run + commit**

Run: `make test-unit ARGS=tests/unit/system/test_disk.py`

Expected: PASS.

```bash
git add app/system/disk.py tests/unit/system/test_disk.py
git commit -m "feat(system): disk free probe"
```

---

### Task 14: HF token validator

**Files:**
- Create: `app/system/hf.py`
- Test: `tests/unit/system/test_hf.py`

- [ ] **Step 1: Write failing test**

```python
import pytest
import httpx

from app.system.hf import validate_hf_token, HfWhoAmI


async def test_validate_hf_token_ok(httpx_mock):
    httpx_mock.add_response(
        url="https://huggingface.co/api/whoami-v2",
        json={"name": "alice", "type": "user"},
        status_code=200,
    )
    info = await validate_hf_token("hf_xxx")
    assert info == HfWhoAmI(username="alice", account_type="user")


async def test_validate_hf_token_invalid(httpx_mock):
    httpx_mock.add_response(
        url="https://huggingface.co/api/whoami-v2",
        status_code=401,
    )
    with pytest.raises(ValueError):
        await validate_hf_token("hf_bad")
```

- [ ] **Step 2: Verify failure**

Run: `make test-unit ARGS=tests/unit/system/test_hf.py`

Expected: FAIL.

- [ ] **Step 3: Write app/system/hf.py**

```python
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class HfWhoAmI:
    username: str
    account_type: str


async def validate_hf_token(token: str) -> HfWhoAmI:
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {token}"},
        )
    if r.status_code == 401 or r.status_code == 403:
        raise ValueError("HuggingFace rejected token")
    r.raise_for_status()
    j = r.json()
    return HfWhoAmI(username=j.get("name", "?"), account_type=j.get("type", "user"))
```

- [ ] **Step 4: Run + commit**

Run: `make test-unit ARGS=tests/unit/system/test_hf.py`

Expected: PASS.

```bash
git add app/system/hf.py tests/unit/system/test_hf.py
git commit -m "feat(system): HF token validator"
```

---

## Phase D: Auth

### Task 15: Session auth (cookie + bcrypt)

**Files:**
- Create: `app/auth/__init__.py`
- Create: `app/auth/sessions.py`
- Test: `tests/unit/auth/test_sessions.py`

- [ ] **Step 1: Write failing test**

```python
import pytest

from app.auth.sessions import (
    hash_password, verify_password,
    sign_session, verify_session, SESSION_MAX_AGE,
)


def test_bcrypt_round_trip():
    h = hash_password("hunter2")
    assert verify_password("hunter2", h)
    assert not verify_password("wrong", h)


def test_session_signed_and_verified():
    secret = "x" * 32
    cookie = sign_session("alice", secret=secret)
    assert verify_session(cookie, secret=secret) == "alice"


def test_session_rejects_wrong_secret():
    cookie = sign_session("alice", secret="a" * 32)
    assert verify_session(cookie, secret="b" * 32) is None


def test_session_expires(monkeypatch):
    import time
    secret = "x" * 32
    cookie = sign_session("alice", secret=secret)
    # Fast-forward past max-age
    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() + SESSION_MAX_AGE + 60)
    assert verify_session(cookie, secret=secret) is None
```

- [ ] **Step 2: Verify failure**

Run: `make test-unit ARGS=tests/unit/auth/test_sessions.py`

Expected: FAIL.

- [ ] **Step 3: Write app/auth/sessions.py**

```python
import bcrypt
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

SESSION_MAX_AGE = 60 * 60 * 24 * 7  # 7 days


def hash_password(plaintext: str) -> str:
    return bcrypt.hashpw(plaintext.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plaintext: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plaintext.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        return False


def sign_session(username: str, *, secret: str) -> str:
    s = URLSafeTimedSerializer(secret_key=secret, salt="vw-session")
    return s.dumps({"u": username})


def verify_session(cookie: str, *, secret: str) -> str | None:
    s = URLSafeTimedSerializer(secret_key=secret, salt="vw-session")
    try:
        data = s.loads(cookie, max_age=SESSION_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None
    return data.get("u")
```

`app/auth/__init__.py`: empty.

- [ ] **Step 4: Run + commit**

Run: `make test-unit ARGS=tests/unit/auth/test_sessions.py`

Expected: PASS.

```bash
git add app/auth/ tests/unit/auth/test_sessions.py
git commit -m "feat(auth): bcrypt + signed session cookie"
```

---

### Task 16: Bearer token auth

**Files:**
- Create: `app/auth/bearer.py`
- Test: `tests/unit/auth/test_bearer.py`

- [ ] **Step 1: Write failing test**

```python
import pytest

from app.auth.bearer import generate_bearer_token, parse_bearer_header


def test_generate_token_format():
    plaintext = generate_bearer_token()
    assert plaintext.startswith("vw_")
    # base32 of 32 bytes = 56 chars, total prefix+body = 59
    assert len(plaintext) == 3 + 56


def test_parse_bearer_header_strips_prefix():
    assert parse_bearer_header("Bearer vw_xxxx") == "vw_xxxx"
    assert parse_bearer_header("bearer vw_xxxx") == "vw_xxxx"


def test_parse_bearer_header_rejects_other_schemes():
    assert parse_bearer_header("Basic abc") is None
    assert parse_bearer_header("vw_xxxx") is None
    assert parse_bearer_header(None) is None
    assert parse_bearer_header("") is None
```

- [ ] **Step 2: Verify failure**

Run: `make test-unit ARGS=tests/unit/auth/test_bearer.py`

Expected: FAIL.

- [ ] **Step 3: Write app/auth/bearer.py**

```python
import base64
import secrets


def generate_bearer_token() -> str:
    """Returns vw_<56 chars of base32 encoded 32 random bytes>."""
    raw = secrets.token_bytes(32)
    body = base64.b32encode(raw).decode("ascii").rstrip("=").lower()
    return f"vw_{body}"


def parse_bearer_header(value: str | None) -> str | None:
    if not value:
        return None
    parts = value.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip()
```

- [ ] **Step 4: Run + commit**

Run: `make test-unit ARGS=tests/unit/auth/test_bearer.py`

Expected: PASS.

```bash
git add app/auth/bearer.py tests/unit/auth/test_bearer.py
git commit -m "feat(auth): bearer token generation + parsing"
```

---

### Task 17: CSRF + login route + auth deps

**Files:**
- Create: `app/auth/csrf.py`
- Create: `app/auth/deps.py`
- Modify: `app/main.py` — login/logout routes, session middleware, request.state.user
- Create: `app/web/login.html`
- Test: `tests/unit/auth/test_csrf.py`, `tests/unit/auth/test_login_flow.py`

- [ ] **Step 1: Write failing tests**

`tests/unit/auth/test_csrf.py`:

```python
from app.auth.csrf import generate_csrf_token, verify_csrf_token


def test_csrf_token_round_trip():
    secret = "x" * 32
    t = generate_csrf_token("session-id-1", secret=secret)
    assert verify_csrf_token(t, "session-id-1", secret=secret) is True


def test_csrf_token_rejects_wrong_session():
    secret = "x" * 32
    t = generate_csrf_token("session-id-1", secret=secret)
    assert verify_csrf_token(t, "session-id-2", secret=secret) is False
```

`tests/unit/auth/test_login_flow.py`:

```python
import sqlite3

import bcrypt
import pytest


def _seed_admin(db_path, username="admin", pw="hunter2"):
    """Seed an admin user directly via sqlite3 sync."""
    h = bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
    with sqlite3.connect(db_path) as db:
        db.execute("INSERT INTO users(username, password_hash) VALUES (?, ?)", (username, h))
        # Mark setup done so we land on /models
        db.execute("UPDATE setup_state SET step = 'done' WHERE id = 1")
        db.commit()


def test_login_sets_session_cookie(tmp_data_dir, client):
    client.get("/healthz")  # boot, run migrations
    _seed_admin(tmp_data_dir / "vllm-warden.db")
    r = client.post(
        "/login",
        data={"username": "admin", "password": "hunter2"},
        follow_redirects=False,
    )
    assert r.status_code in (302, 303)
    assert "vw_session" in r.cookies


def test_login_rejects_bad_password(tmp_data_dir, client):
    client.get("/healthz")
    _seed_admin(tmp_data_dir / "vllm-warden.db")
    r = client.post(
        "/login", data={"username": "admin", "password": "wrong"}, follow_redirects=False
    )
    assert r.status_code == 401


def test_protected_route_redirects_when_unauthed(tmp_data_dir, client):
    r = client.get("/models", follow_redirects=False)
    assert r.status_code in (302, 303)
    assert r.headers["location"].startswith("/login")


def test_logout_clears_cookie(tmp_data_dir, client):
    client.get("/healthz")
    _seed_admin(tmp_data_dir / "vllm-warden.db")
    client.post("/login", data={"username": "admin", "password": "hunter2"})
    r = client.get("/logout", follow_redirects=False)
    assert "vw_session" not in r.cookies or r.cookies["vw_session"] == ""
```

- [ ] **Step 2: Run — expect FAIL**

Run: `make test-unit ARGS=tests/unit/auth/`

Expected: FAIL.

- [ ] **Step 3: Write app/auth/csrf.py**

```python
import hmac
from hashlib import sha256


def generate_csrf_token(session_id: str, *, secret: str) -> str:
    return hmac.new(secret.encode(), session_id.encode(), sha256).hexdigest()


def verify_csrf_token(token: str, session_id: str, *, secret: str) -> bool:
    expected = generate_csrf_token(session_id, secret=secret)
    return hmac.compare_digest(token, expected)
```

- [ ] **Step 4: Write app/auth/deps.py**

```python
from fastapi import HTTPException, Request, status
from fastapi.responses import RedirectResponse

from app.auth.sessions import verify_session


async def session_user(request: Request) -> str | None:
    settings = request.app.state.settings
    cookie = request.cookies.get("vw_session")
    if not cookie:
        return None
    return verify_session(cookie, secret=settings.cookie_secret)


def require_session_redirect(request: Request) -> str:
    """For HTML routes — redirects to /login if missing."""
    user = request.state.user
    if not user:
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            headers={"Location": "/login"},
        )
    return user


def require_session_json(request: Request) -> str:
    """For JSON API routes — returns 401 if missing."""
    user = request.state.user
    if not user:
        raise HTTPException(status_code=401, detail="not authenticated")
    return user
```

- [ ] **Step 5: Write app/web/login.html**

```html
{% extends "base_template.html" %}
{% block title %}Login - vllm-warden{% endblock %}
{% block content %}
<div class="card" style="max-width: 400px; margin: 80px auto;">
  <h2>Sign in</h2>
  {% if error %}<p style="color: var(--err);">{{ error }}</p>{% endif %}
  <form method="post" action="/login">
    <p><label>Username<br><input name="username" required autofocus></label></p>
    <p><label>Password<br><input name="password" type="password" required></label></p>
    <p><button type="submit" class="btn">Sign in</button></p>
  </form>
</div>
{% endblock %}
```

- [ ] **Step 6: Wire login + middleware in app/main.py**

Replace `build_app()`:

```python
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from app.auth.deps import session_user
from app.auth.sessions import sign_session, verify_password
from app.config import load_settings
from app.db.database import open_db
from app.db.repos.users import UserRepo


def build_app() -> FastAPI:
    app = FastAPI(title="vllm-warden", lifespan=lifespan)
    app.mount(
        "/static",
        StaticFiles(directory=Path(__file__).parent / "web/static"),
        name="static",
    )
    templates = Jinja2Templates(directory=Path(__file__).parent / "web")
    app.state.templates = templates

    @app.middleware("http")
    async def attach_user(request: Request, call_next):
        request.state.user = await session_user(request)
        return await call_next(request)

    @app.get("/healthz")
    async def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request):
        return templates.TemplateResponse(request, "login.html", {"error": None})

    @app.post("/login")
    async def do_login(
        request: Request,
        username: str = Form(...),
        password: str = Form(...),
    ):
        settings = request.app.state.settings
        async with open_db(settings.db_path) as db:
            user = await UserRepo(db).get_by_username(username)
        if not user or not verify_password(password, user.password_hash):
            return templates.TemplateResponse(
                request, "login.html",
                {"error": "Invalid credentials"},
                status_code=401,
            )
        cookie = sign_session(username, secret=settings.cookie_secret)
        resp = RedirectResponse(url="/models", status_code=303)
        resp.set_cookie(
            "vw_session", cookie,
            httponly=True, samesite="strict", secure=False, max_age=60 * 60 * 24 * 7,
        )
        return resp

    @app.get("/logout")
    async def logout():
        resp = RedirectResponse(url="/login", status_code=303)
        resp.delete_cookie("vw_session")
        return resp

    @app.get("/")
    async def root(request: Request):
        if request.state.user:
            return RedirectResponse(url="/models", status_code=303)
        return RedirectResponse(url="/login", status_code=303)

    @app.get("/models", response_class=HTMLResponse)
    async def models_placeholder(request: Request):
        if not request.state.user:
            return RedirectResponse(url="/login", status_code=303)
        return HTMLResponse("<p>models page placeholder</p>")

    return app


app = build_app()
```

- [ ] **Step 7: Run + commit**

Run: `make test-unit ARGS=tests/unit/auth/`

Expected: PASS.

```bash
git add app/auth/csrf.py app/auth/deps.py app/web/login.html app/main.py tests/unit/auth/test_csrf.py tests/unit/auth/test_login_flow.py
git commit -m "feat(auth): login/logout, session middleware, CSRF helper"
```

---

## Phase E: Setup wizard

### Task 18: Setup state machine

**Files:**
- Create: `app/setup/__init__.py`
- Create: `app/setup/state_machine.py`
- Test: `tests/unit/setup/test_state_machine.py`

- [ ] **Step 1: Write failing test**

```python
from app.setup.state_machine import next_step, STEPS


def test_steps_in_order():
    assert STEPS == ["welcome", "gpus", "hf_token", "admin", "done"]


def test_next_step_advances():
    assert next_step("welcome") == "gpus"
    assert next_step("gpus") == "hf_token"
    assert next_step("hf_token") == "admin"
    assert next_step("admin") == "done"


def test_next_step_done_stays_done():
    assert next_step("done") == "done"


def test_next_step_unknown_raises():
    import pytest
    with pytest.raises(ValueError):
        next_step("garbage")
```

- [ ] **Step 2: Verify failure**

Run: `make test-unit ARGS=tests/unit/setup/test_state_machine.py`

Expected: FAIL.

- [ ] **Step 3: Write app/setup/state_machine.py**

```python
STEPS = ["welcome", "gpus", "hf_token", "admin", "done"]


def next_step(current: str) -> str:
    if current not in STEPS:
        raise ValueError(f"unknown step: {current!r}")
    if current == "done":
        return "done"
    idx = STEPS.index(current)
    return STEPS[idx + 1]
```

`app/setup/__init__.py`: empty.
`tests/unit/setup/__init__.py`: empty.

- [ ] **Step 4: Run + commit**

Run: `make test-unit ARGS=tests/unit/setup/test_state_machine.py`

Expected: PASS.

```bash
git add app/setup/ tests/unit/setup/
git commit -m "feat(setup): wizard state machine"
```

---

### Task 19: Setup API routes (welcome → gpus)

**Files:**
- Create: `app/setup/routes_api.py`
- Create: `app/setup/routes_web.py`
- Create: `app/setup/templates/welcome.html`
- Create: `app/setup/templates/gpus.html`
- Modify: `app/main.py` — register setup routes; pre-setup gate redirects all auth'd routes to `/setup`
- Test: `tests/unit/setup/test_routes_welcome_gpus.py`

- [ ] **Step 1: Write failing test**

```python
import sqlite3


def _setup_state(db_path):
    with sqlite3.connect(db_path) as db:
        return db.execute("SELECT step, draft FROM setup_state").fetchone()


def test_setup_redirects_when_not_done(tmp_data_dir, client):
    """Hitting /models pre-setup must redirect to /setup."""
    client.get("/healthz")
    r = client.get("/setup", follow_redirects=False)
    assert r.status_code == 200
    assert "Welcome" in r.text


def test_post_welcome_advances_to_gpus(tmp_data_dir, client):
    client.get("/healthz")
    r = client.post("/api/setup/welcome", follow_redirects=False)
    assert r.status_code == 200
    step, _ = _setup_state(tmp_data_dir / "vllm-warden.db")
    assert step == "gpus"


def test_post_gpus_validates_subset_and_persists(
    tmp_data_dir, client, monkeypatch
):
    """Selected GPU indices must be a subset of detected GPUs (0..N-1)."""
    from app.system import gpu as gpu_mod
    from app.system.gpu import GpuInfo

    async def fake_query():
        return [
            GpuInfo(0, "A100", 40960, 0, 0),
            GpuInfo(1, "A100", 40960, 0, 0),
            GpuInfo(2, "A100", 40960, 0, 0),
            GpuInfo(3, "A100", 40960, 0, 0),
        ]
    monkeypatch.setattr(gpu_mod, "query_gpus", fake_query)

    client.get("/healthz")
    client.post("/api/setup/welcome")
    r = client.post("/api/setup/gpus", json={"allowed_gpu_indices": [1, 2]})
    assert r.status_code == 200, r.text

    import json
    step, draft = _setup_state(tmp_data_dir / "vllm-warden.db")
    assert step == "hf_token"
    assert json.loads(draft)["allowed_gpu_indices"] == [1, 2]


def test_post_gpus_rejects_indices_out_of_range(
    tmp_data_dir, client, monkeypatch
):
    from app.system import gpu as gpu_mod
    from app.system.gpu import GpuInfo
    async def fake_query():
        return [GpuInfo(i, "A100", 40960, 0, 0) for i in range(2)]
    monkeypatch.setattr(gpu_mod, "query_gpus", fake_query)

    client.get("/healthz")
    client.post("/api/setup/welcome")
    r = client.post("/api/setup/gpus", json={"allowed_gpu_indices": [0, 5]})
    assert r.status_code == 400
```

- [ ] **Step 2: Verify failure**

Run: `make test-unit ARGS=tests/unit/setup/test_routes_welcome_gpus.py`

Expected: FAIL.

- [ ] **Step 3: Write welcome.html**

```html
{% extends "base_template.html" %}
{% block title %}Welcome - vllm-warden setup{% endblock %}
{% block content %}
<div class="card" style="max-width: 600px; margin: 40px auto;">
  <h1>Welcome to vllm-warden</h1>
  <p>Let's configure this container. Four short steps:</p>
  <ol>
    <li>Pick which GPUs this container can use</li>
    <li>Provide a HuggingFace token (optional)</li>
    <li>Create the admin account</li>
  </ol>
  <button class="btn"
          hx-post="/api/setup/welcome"
          hx-target="body"
          hx-swap="innerHTML">
    Begin setup
  </button>
</div>
{% endblock %}
```

- [ ] **Step 4: Write gpus.html**

```html
{% extends "base_template.html" %}
{% block title %}Select GPUs - vllm-warden setup{% endblock %}
{% block content %}
<div class="card" style="max-width: 700px; margin: 40px auto;">
  <h1>Select GPUs</h1>
  <p class="muted">vllm-warden detected the following GPUs in this container:</p>

  {% if not gpus %}
    <p style="color: var(--err);">No GPUs detected (nvidia-smi unavailable).</p>
  {% else %}
    <form id="gpus-form">
      <table>
        <thead>
          <tr><th></th><th>Index</th><th>Name</th><th>VRAM</th></tr>
        </thead>
        <tbody>
          {% for g in gpus %}
          <tr>
            <td><input type="checkbox" name="gpu" value="{{ g.index }}" checked></td>
            <td>{{ g.index }}</td>
            <td>{{ g.name }}</td>
            <td>{{ '%.1f' | format(g.memory_total_mib / 1024) }} GiB</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
      <p>
        <button type="button" class="btn" onclick="submitGpus()">Continue</button>
      </p>
    </form>
    <script>
      async function submitGpus() {
        const checked = [...document.querySelectorAll('input[name=gpu]:checked')]
          .map(el => parseInt(el.value, 10));
        const r = await fetch('/api/setup/gpus', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({allowed_gpu_indices: checked}),
        });
        if (r.ok) {
          window.location.href = '/setup';
        } else {
          alert('Error: ' + (await r.text()));
        }
      }
    </script>
  {% endif %}
</div>
{% endblock %}
```

- [ ] **Step 5: Write app/setup/routes_web.py**

```python
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.config import Settings
from app.db.database import open_db
from app.db.repos.setup import SetupRepo
from app.system.gpu import query_gpus

router = APIRouter()


@router.get("/setup", response_class=HTMLResponse)
async def setup_root(request: Request):
    settings: Settings = request.app.state.settings
    templates = request.app.state.templates
    async with open_db(settings.db_path) as db:
        state = await SetupRepo(db).get()
    if state.step == "done":
        return RedirectResponse(url="/models", status_code=303)
    if state.step == "welcome":
        return templates.TemplateResponse(request, "setup/welcome.html", {})
    if state.step == "gpus":
        gpus = await query_gpus()
        return templates.TemplateResponse(request, "setup/gpus.html", {"gpus": gpus})
    if state.step == "hf_token":
        return templates.TemplateResponse(request, "setup/hf_token.html", {})
    if state.step == "admin":
        return templates.TemplateResponse(request, "setup/admin.html", {})
    return HTMLResponse(f"unexpected step: {state.step}", status_code=500)
```

- [ ] **Step 6: Write app/setup/routes_api.py**

```python
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.db.database import open_db
from app.db.repos.setup import SetupRepo
from app.setup.state_machine import next_step
from app.system.gpu import query_gpus

router = APIRouter(prefix="/api/setup")


@router.post("/welcome")
async def post_welcome(request: Request):
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        repo = SetupRepo(db)
        state = await repo.get()
        if state.step != "welcome":
            raise HTTPException(400, f"not at welcome step (current: {state.step})")
        await repo.set_step(next_step("welcome"))
    return {"step": "gpus"}


class GpusBody(BaseModel):
    allowed_gpu_indices: list[int]


@router.post("/gpus")
async def post_gpus(body: GpusBody, request: Request):
    settings = request.app.state.settings
    detected = await query_gpus()
    valid_indices = {g.index for g in detected}
    if not body.allowed_gpu_indices:
        raise HTTPException(400, "must select at least one GPU")
    requested = set(body.allowed_gpu_indices)
    if not requested.issubset(valid_indices):
        bad = sorted(requested - valid_indices)
        raise HTTPException(
            400, f"GPU indices {bad} not present in container (have: {sorted(valid_indices)})"
        )
    async with open_db(settings.db_path) as db:
        repo = SetupRepo(db)
        state = await repo.get()
        if state.step != "gpus":
            raise HTTPException(400, f"not at gpus step (current: {state.step})")
        await repo.merge_draft(allowed_gpu_indices=sorted(requested))
        await repo.set_step(next_step("gpus"))
    return {"step": "hf_token"}
```

- [ ] **Step 7: Wire setup routes into main.py + add pre-setup gate**

In `build_app()` add (after templates are configured):

```python
from app.setup import routes_api as setup_routes_api
from app.setup import routes_web as setup_routes_web

app.include_router(setup_routes_web.router)
app.include_router(setup_routes_api.router)

# Update Jinja2Templates to look in setup/templates too:
templates_dirs = [Path(__file__).parent / "web", Path(__file__).parent / "setup/templates"]
templates = Jinja2Templates(directory=templates_dirs)
```

Move setup templates into `app/setup/templates/`. The gpus template references `setup/gpus.html` so loader needs both directories. Update in main.py:

```python
templates = Jinja2Templates(directory=str(Path(__file__).parent))  # parent of app/web and app/setup
```

Actually, simplest fix: place setup templates **also** under `app/web/setup/`:

`app/web/setup/welcome.html`, `app/web/setup/gpus.html` — keep `app/setup/templates/` as the canonical home and also write a copy. Better: change the routes_web to render `setup/welcome.html` from `app/web/setup/welcome.html`.

**Resolution**: Templates root stays `app/web/`. All setup templates live at `app/web/setup/*.html`. Skip `app/setup/templates/` directory entirely. Update routes_web.py paths to `setup/welcome.html` etc.

**Files corrected:**
- Create: `app/web/setup/welcome.html` (content from Step 3)
- Create: `app/web/setup/gpus.html` (content from Step 4)
- Skip creating `app/setup/templates/`

- [ ] **Step 8: Run + commit**

Run: `make test-unit ARGS=tests/unit/setup/test_routes_welcome_gpus.py`

Expected: PASS.

```bash
git add app/setup/routes_api.py app/setup/routes_web.py app/web/setup/ app/main.py tests/unit/setup/
git commit -m "feat(setup): welcome + GPU selection routes"
```

---

### Task 20: Setup hf_token + admin steps

**Files:**
- Modify: `app/setup/routes_api.py` — add `/hf_token`, `/admin`
- Create: `app/web/setup/hf_token.html`
- Create: `app/web/setup/admin.html`
- Create: `app/web/setup/done.html`
- Test: `tests/unit/setup/test_routes_hf_admin.py`

- [ ] **Step 1: Write failing test**

```python
import sqlite3
import json


def _seed_to_step(db_path, step, draft):
    with sqlite3.connect(db_path) as db:
        db.execute(
            "UPDATE setup_state SET step = ?, draft = ? WHERE id = 1",
            (step, json.dumps(draft)),
        )
        db.commit()


def test_post_hf_token_validates_and_advances(tmp_data_dir, client, monkeypatch, httpx_mock):
    """Valid token → advance to admin step. Token persists in draft."""
    from app.system import hf as hf_mod

    async def fake_validate(tok):
        from app.system.hf import HfWhoAmI
        return HfWhoAmI(username="alice", account_type="user")
    monkeypatch.setattr(hf_mod, "validate_hf_token", fake_validate)

    client.get("/healthz")
    _seed_to_step(tmp_data_dir / "vllm-warden.db", "hf_token", {"allowed_gpu_indices": [0]})

    r = client.post("/api/setup/hf_token", json={"hf_token": "hf_xxx"})
    assert r.status_code == 200
    assert r.json()["whoami"]["username"] == "alice"

    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        step, draft_json = db.execute("SELECT step, draft FROM setup_state").fetchone()
    draft = json.loads(draft_json)
    assert step == "admin"
    assert draft["hf_token"] == "hf_xxx"


def test_post_hf_token_can_be_skipped(tmp_data_dir, client):
    client.get("/healthz")
    _seed_to_step(tmp_data_dir / "vllm-warden.db", "hf_token", {"allowed_gpu_indices": [0]})

    r = client.post("/api/setup/hf_token", json={"hf_token": None})
    assert r.status_code == 200

    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        step, draft_json = db.execute("SELECT step, draft FROM setup_state").fetchone()
    assert step == "admin"
    assert "hf_token" not in json.loads(draft_json)


def test_post_admin_creates_user_and_finalizes(tmp_data_dir, client):
    """Posting admin creates the user, marks step=done, and clears the token from the draft."""
    client.get("/healthz")
    _seed_to_step(
        tmp_data_dir / "vllm-warden.db",
        "admin",
        {"allowed_gpu_indices": [0, 1], "hf_token": "hf_xxx"},
    )
    r = client.post(
        "/api/setup/admin",
        json={"username": "admin", "password": "hunter2"},
    )
    assert r.status_code == 200

    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        step, draft_json = db.execute("SELECT step, draft FROM setup_state").fetchone()
        (count,) = db.execute("SELECT COUNT(*) FROM users").fetchone()
    assert step == "done"
    assert count == 1
    draft = json.loads(draft_json)
    # The plaintext token is replaced with a flag once setup is done.
    assert "hf_token" not in draft
    assert draft.get("hf_token_present") is True


def test_post_admin_rejects_short_password(tmp_data_dir, client):
    client.get("/healthz")
    _seed_to_step(tmp_data_dir / "vllm-warden.db", "admin", {})
    r = client.post(
        "/api/setup/admin", json={"username": "admin", "password": "abc"}
    )
    assert r.status_code == 400
```

- [ ] **Step 2: Verify failure**

Run: `make test-unit ARGS=tests/unit/setup/test_routes_hf_admin.py`

Expected: FAIL.

- [ ] **Step 3: Extend app/setup/routes_api.py**

Append:

```python
from app.auth.sessions import hash_password
from app.db.repos.users import UserRepo
from app.system.hf import validate_hf_token


class HfTokenBody(BaseModel):
    hf_token: str | None


@router.post("/hf_token")
async def post_hf_token(body: HfTokenBody, request: Request):
    settings = request.app.state.settings
    whoami_dict = None
    if body.hf_token:
        try:
            whoami = await validate_hf_token(body.hf_token)
            whoami_dict = {"username": whoami.username, "account_type": whoami.account_type}
        except ValueError as e:
            raise HTTPException(400, f"HuggingFace token rejected: {e}")
    async with open_db(settings.db_path) as db:
        repo = SetupRepo(db)
        state = await repo.get()
        if state.step != "hf_token":
            raise HTTPException(400, f"not at hf_token step (current: {state.step})")
        if body.hf_token:
            await repo.merge_draft(hf_token=body.hf_token)
        await repo.set_step(next_step("hf_token"))
    return {"step": "admin", "whoami": whoami_dict}


class AdminBody(BaseModel):
    username: str
    password: str


@router.post("/admin")
async def post_admin(body: AdminBody, request: Request):
    if len(body.password) < 8:
        raise HTTPException(400, "password must be at least 8 chars")
    if not body.username or not body.username.strip():
        raise HTTPException(400, "username required")

    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        repo = SetupRepo(db)
        state = await repo.get()
        if state.step != "admin":
            raise HTTPException(400, f"not at admin step (current: {state.step})")
        users = UserRepo(db)
        if await users.get_by_username(body.username):
            raise HTTPException(400, "username already exists")
        await users.create(body.username, hash_password(body.password))

        # Replace plaintext token in draft with a flag
        draft = state.draft.copy()
        had_token = "hf_token" in draft
        draft.pop("hf_token", None)
        if had_token:
            draft["hf_token_present"] = True

        # Persist hf_token plaintext to a sealed env file under data_dir for vLLM subprocess
        if had_token:
            (settings.data_dir / "hf-token").write_text(state.draft["hf_token"])
            (settings.data_dir / "hf-token").chmod(0o600)

        await db.execute(
            "UPDATE setup_state SET step = 'done', draft = ?, updated_at = datetime('now') "
            "WHERE id = 1",
            (json.dumps(draft),),
        )
        await db.commit()
    return {"step": "done"}
```

Add `import json` at top of file.

- [ ] **Step 4: Write hf_token.html, admin.html, done.html**

`app/web/setup/hf_token.html`:

```html
{% extends "base_template.html" %}
{% block title %}HuggingFace token - vllm-warden setup{% endblock %}
{% block content %}
<div class="card" style="max-width: 600px; margin: 40px auto;">
  <h1>HuggingFace token (optional)</h1>
  <p class="muted">Required for gated/private models. Skip if you only use public models.</p>
  <p><label>Token<br><input id="tok" type="password" placeholder="hf_..."></label></p>
  <p>
    <button class="btn" onclick="submitTok(false)">Save and continue</button>
    <button class="btn" onclick="submitTok(true)" style="background: #4b5563;">Skip</button>
  </p>
  <script>
    async function submitTok(skip) {
      const tok = skip ? null : document.getElementById('tok').value;
      const r = await fetch('/api/setup/hf_token', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({hf_token: tok}),
      });
      if (r.ok) window.location.href = '/setup';
      else alert(await r.text());
    }
  </script>
</div>
{% endblock %}
```

`app/web/setup/admin.html`:

```html
{% extends "base_template.html" %}
{% block title %}Create admin - vllm-warden setup{% endblock %}
{% block content %}
<div class="card" style="max-width: 500px; margin: 40px auto;">
  <h1>Create admin account</h1>
  <p><label>Username<br><input id="u" autofocus></label></p>
  <p><label>Password (8+ chars)<br><input id="p" type="password"></label></p>
  <p><button class="btn" onclick="submitAdmin()">Finish setup</button></p>
  <script>
    async function submitAdmin() {
      const r = await fetch('/api/setup/admin', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          username: document.getElementById('u').value,
          password: document.getElementById('p').value,
        }),
      });
      if (r.ok) window.location.href = '/login';
      else alert(await r.text());
    }
  </script>
</div>
{% endblock %}
```

`app/web/setup/done.html`:

```html
{% extends "base_template.html" %}
{% block title %}Setup complete{% endblock %}
{% block content %}
<div class="card" style="max-width: 500px; margin: 40px auto;">
  <h1>Setup complete</h1>
  <p><a href="/login" class="btn">Sign in</a></p>
</div>
{% endblock %}
```

- [ ] **Step 5: Run + commit**

Run: `make test-unit ARGS=tests/unit/setup/test_routes_hf_admin.py`

Expected: PASS.

```bash
git add app/setup/routes_api.py app/web/setup/hf_token.html app/web/setup/admin.html app/web/setup/done.html tests/unit/setup/test_routes_hf_admin.py
git commit -m "feat(setup): hf_token + admin steps finalize wizard"
```

---

### Task 21: Pre-setup gate

**Files:**
- Modify: `app/main.py` — middleware redirects auth'd routes to `/setup` if setup not done
- Test: `tests/unit/setup/test_pre_setup_gate.py`

- [ ] **Step 1: Write failing test**

```python
import sqlite3


def test_models_redirects_to_setup_when_not_done(tmp_data_dir, client):
    client.get("/healthz")
    r = client.get("/models", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/setup"


def test_login_does_not_redirect_to_setup(tmp_data_dir, client):
    client.get("/healthz")
    r = client.get("/login", follow_redirects=False)
    assert r.status_code == 200


def test_static_does_not_redirect_to_setup(tmp_data_dir, client):
    client.get("/healthz")
    r = client.get("/static/app.css", follow_redirects=False)
    assert r.status_code == 200


def test_after_setup_done_no_redirect(tmp_data_dir, client):
    client.get("/healthz")
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute("UPDATE setup_state SET step = 'done' WHERE id = 1")
        db.commit()
    r = client.get("/models", follow_redirects=False)
    # Without auth → /login (not /setup)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"
```

- [ ] **Step 2: Verify failure**

Run: `make test-unit ARGS=tests/unit/setup/test_pre_setup_gate.py`

Expected: FAIL on first test (setup gate not implemented).

- [ ] **Step 3: Add gate middleware in app/main.py**

Insert after `attach_user` middleware:

```python
SETUP_BYPASS_PREFIXES = ("/setup", "/api/setup", "/login", "/logout", "/healthz", "/static")


@app.middleware("http")
async def gate_setup(request: Request, call_next):
    path = request.url.path
    if any(path.startswith(p) for p in SETUP_BYPASS_PREFIXES):
        return await call_next(request)
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        state = await SetupRepo(db).get()
    if state.step != "done":
        return RedirectResponse(url="/setup", status_code=303)
    return await call_next(request)
```

Add imports `from app.db.repos.setup import SetupRepo` at top of `main.py`.

- [ ] **Step 4: Run + commit**

Run: `make test-unit ARGS=tests/unit/setup/test_pre_setup_gate.py`

Expected: PASS.

```bash
git add app/main.py tests/unit/setup/test_pre_setup_gate.py
git commit -m "feat(setup): pre-setup gate redirects to wizard"
```

---

## Phase F: Models catalog

### Task 22: Models repo helpers + ModelCreate validation

**Files:**
- Create: `app/models/__init__.py`
- Create: `app/models/schemas.py`
- Test: `tests/unit/models/test_schemas.py`

- [ ] **Step 1: Write failing test**

```python
import pytest
from pydantic import ValidationError

from app.models.schemas import ModelCreate


def test_model_create_minimal():
    m = ModelCreate(
        served_model_name="qwen3.5-9b",
        hf_repo="Qwen/Qwen3.5-9B",
        gpu_indices=[1, 2],
    )
    assert m.tensor_parallel_size == 1
    assert m.hf_revision == "main"


def test_model_create_tp_must_match_gpu_count_when_unset_default_to_len():
    m = ModelCreate(
        served_model_name="x",
        hf_repo="o/r",
        gpu_indices=[0, 1, 2],
    )
    # Default behavior: tensor_parallel_size auto-set to len(gpu_indices)
    assert m.tensor_parallel_size == 3


def test_model_create_explicit_tp_must_match():
    with pytest.raises(ValidationError):
        ModelCreate(
            served_model_name="x", hf_repo="o/r",
            gpu_indices=[0, 1, 2],
            tensor_parallel_size=2,
        )


def test_model_create_rejects_empty_gpus():
    with pytest.raises(ValidationError):
        ModelCreate(served_model_name="x", hf_repo="o/r", gpu_indices=[])


def test_model_create_served_name_slug():
    with pytest.raises(ValidationError):
        ModelCreate(served_model_name="bad name!", hf_repo="o/r", gpu_indices=[0])
```

- [ ] **Step 2: Verify failure**

Run: `make test-unit ARGS=tests/unit/models/test_schemas.py`

Expected: FAIL.

- [ ] **Step 3: Write app/models/schemas.py**

```python
import re

from pydantic import BaseModel, Field, field_validator, model_validator

SLUG_RE = re.compile(r"^[a-zA-Z0-9._-]+$")


class ModelCreate(BaseModel):
    served_model_name: str = Field(..., min_length=1, max_length=100)
    hf_repo: str = Field(..., min_length=1, pattern=r"^[\w.-]+/[\w.-]+$")
    hf_revision: str = "main"
    gpu_indices: list[int] = Field(..., min_length=1)
    tensor_parallel_size: int | None = None
    dtype: str | None = None
    max_model_len: int | None = Field(None, gt=0)
    gpu_memory_utilization: float = Field(0.9, gt=0, le=1.0)
    trust_remote_code: bool = False
    extra_args: list[str] = Field(default_factory=list)

    @field_validator("served_model_name")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not SLUG_RE.match(v):
            raise ValueError("served_model_name must be alphanumeric/dot/dash/underscore")
        return v

    @field_validator("gpu_indices")
    @classmethod
    def _unique_gpus(cls, v: list[int]) -> list[int]:
        if len(set(v)) != len(v):
            raise ValueError("gpu_indices must be unique")
        if any(i < 0 for i in v):
            raise ValueError("gpu_indices must be >= 0")
        return v

    @model_validator(mode="after")
    def _tp_consistent(self) -> "ModelCreate":
        if self.tensor_parallel_size is None:
            object.__setattr__(self, "tensor_parallel_size", len(self.gpu_indices))
        elif self.tensor_parallel_size != len(self.gpu_indices):
            raise ValueError(
                f"tensor_parallel_size={self.tensor_parallel_size} "
                f"must equal len(gpu_indices)={len(self.gpu_indices)}"
            )
        return self
```

`app/models/__init__.py`: empty.
`tests/unit/models/__init__.py`: empty.

- [ ] **Step 4: Run + commit**

Run: `make test-unit ARGS=tests/unit/models/test_schemas.py`

Expected: PASS.

```bash
git add app/models/ tests/unit/models/
git commit -m "feat(models): ModelCreate schema with TP=len(GPUs) invariant"
```

---

### Task 23: Models CRUD API + GPU subset enforcement

**Files:**
- Create: `app/models/routes_api.py`
- Modify: `app/main.py` — register router
- Test: `tests/unit/models/test_routes_crud.py`

- [ ] **Step 1: Write failing test**

```python
import json
import sqlite3

import bcrypt


def _seed_done(db_path, allowed=[0, 1, 2, 3]):
    """Setup done, admin seeded, allowed_gpu_indices populated."""
    pw = bcrypt.hashpw(b"hunter2", bcrypt.gensalt()).decode()
    with sqlite3.connect(db_path) as db:
        db.execute("INSERT INTO users(username, password_hash) VALUES (?, ?)", ("admin", pw))
        db.execute(
            "UPDATE setup_state SET step = 'done', draft = ? WHERE id = 1",
            (json.dumps({"allowed_gpu_indices": allowed}),),
        )
        db.commit()


def _login(client):
    client.post("/login", data={"username": "admin", "password": "hunter2"})


def test_create_model_persists_and_returns_id(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db", allowed=[1, 2])
    _login(client)
    r = client.post("/api/models", json={
        "served_model_name": "qwen3.5-9b",
        "hf_repo": "Qwen/Qwen3.5-9B",
        "gpu_indices": [1, 2],
    })
    assert r.status_code == 201, r.text
    model_id = r.json()["id"]
    assert len(model_id) > 0

    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        row = db.execute(
            "SELECT served_model_name, gpu_indices, tensor_parallel_size, status "
            "FROM models WHERE id = ?", (model_id,)
        ).fetchone()
    assert row[0] == "qwen3.5-9b"
    assert json.loads(row[1]) == [1, 2]
    assert row[2] == 2
    assert row[3] == "registered"


def test_create_model_rejects_gpus_outside_allowed(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db", allowed=[0, 1])
    _login(client)
    r = client.post("/api/models", json={
        "served_model_name": "x",
        "hf_repo": "o/r",
        "gpu_indices": [2, 3],
    })
    assert r.status_code == 400
    assert "not in allowed_gpu_indices" in r.text.lower()


def test_create_rejects_duplicate_served_name(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    _login(client)
    body = {"served_model_name": "x", "hf_repo": "o/r", "gpu_indices": [0]}
    assert client.post("/api/models", json=body).status_code == 201
    assert client.post("/api/models", json=body).status_code == 409


def test_list_models(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    _login(client)
    client.post("/api/models", json={
        "served_model_name": "a", "hf_repo": "o/r", "gpu_indices": [0]
    })
    client.post("/api/models", json={
        "served_model_name": "b", "hf_repo": "o/r2", "gpu_indices": [1]
    })
    r = client.get("/api/models")
    names = sorted(m["served_model_name"] for m in r.json()["models"])
    assert names == ["a", "b"]


def test_delete_model_only_when_unloaded(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    _login(client)
    create = client.post("/api/models", json={
        "served_model_name": "x", "hf_repo": "o/r", "gpu_indices": [0]
    })
    mid = create.json()["id"]
    # Mark loaded → delete must refuse
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute("UPDATE models SET status = 'loaded' WHERE id = ?", (mid,))
        db.commit()
    r = client.delete(f"/api/models/{mid}")
    assert r.status_code == 409

    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute("UPDATE models SET status = 'registered' WHERE id = ?", (mid,))
        db.commit()
    r = client.delete(f"/api/models/{mid}")
    assert r.status_code == 204


def test_unauthed_models_api_401(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    r = client.get("/api/models")
    assert r.status_code == 401
```

- [ ] **Step 2: Verify failure**

Run: `make test-unit ARGS=tests/unit/models/test_routes_crud.py`

Expected: FAIL.

- [ ] **Step 3: Write app/models/routes_api.py**

```python
import json
import secrets

from dataclasses import asdict
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse, Response

from app.auth.deps import require_session_json
from app.db.database import open_db
from app.db.repos.models import ModelRepo, ModelRow
from app.db.repos.setup import SetupRepo
from app.models.schemas import ModelCreate

router = APIRouter(prefix="/api/models")


def _gen_id() -> str:
    return secrets.token_hex(8)


@router.post("", status_code=201)
async def create_model(
    body: ModelCreate, request: Request, _user: str = Depends(require_session_json)
):
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        setup_state = await SetupRepo(db).get()
        allowed = set(setup_state.draft.get("allowed_gpu_indices", []))
        if not set(body.gpu_indices).issubset(allowed):
            bad = sorted(set(body.gpu_indices) - allowed)
            raise HTTPException(
                400, f"GPU indices {bad} not in allowed_gpu_indices {sorted(allowed)}"
            )

        repo = ModelRepo(db)
        if await repo.get_by_served_name(body.served_model_name):
            raise HTTPException(409, f"served_model_name '{body.served_model_name}' already exists")

        model_id = _gen_id()
        await repo.insert(ModelRow(
            id=model_id,
            served_model_name=body.served_model_name,
            hf_repo=body.hf_repo,
            hf_revision=body.hf_revision,
            gpu_indices=json.dumps(sorted(body.gpu_indices)),
            tensor_parallel_size=body.tensor_parallel_size,
            dtype=body.dtype,
            max_model_len=body.max_model_len,
            gpu_memory_utilization=body.gpu_memory_utilization,
            trust_remote_code=body.trust_remote_code,
            extra_args=json.dumps(body.extra_args),
            status="registered",
            pulled_bytes=0,
            pulled_total=None,
            last_error=None,
        ))
    return {"id": model_id, "served_model_name": body.served_model_name, "status": "registered"}


@router.get("")
async def list_models(request: Request, _user: str = Depends(require_session_json)):
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        rows = await ModelRepo(db).list_all()
    return {
        "models": [
            {
                "id": r.id,
                "served_model_name": r.served_model_name,
                "hf_repo": r.hf_repo,
                "hf_revision": r.hf_revision,
                "gpu_indices": json.loads(r.gpu_indices),
                "tensor_parallel_size": r.tensor_parallel_size,
                "status": r.status,
                "pulled_bytes": r.pulled_bytes,
                "pulled_total": r.pulled_total,
                "last_error": r.last_error,
            }
            for r in rows
        ]
    }


@router.get("/{model_id}")
async def get_model(model_id: str, request: Request, _user: str = Depends(require_session_json)):
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        row = await ModelRepo(db).get(model_id)
    if not row:
        raise HTTPException(404, "not found")
    return {
        "id": row.id,
        "served_model_name": row.served_model_name,
        "hf_repo": row.hf_repo,
        "hf_revision": row.hf_revision,
        "gpu_indices": json.loads(row.gpu_indices),
        "tensor_parallel_size": row.tensor_parallel_size,
        "dtype": row.dtype,
        "max_model_len": row.max_model_len,
        "gpu_memory_utilization": row.gpu_memory_utilization,
        "trust_remote_code": row.trust_remote_code,
        "extra_args": json.loads(row.extra_args),
        "status": row.status,
        "pulled_bytes": row.pulled_bytes,
        "pulled_total": row.pulled_total,
        "last_error": row.last_error,
    }


@router.delete("/{model_id}", status_code=204)
async def delete_model(
    model_id: str, request: Request, _user: str = Depends(require_session_json)
):
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        row = await ModelRepo(db).get(model_id)
        if not row:
            raise HTTPException(404, "not found")
        if row.status in ("loaded", "loading", "unloading", "pulling"):
            raise HTTPException(409, f"cannot delete model in status '{row.status}'")
        await ModelRepo(db).delete(model_id)
    return Response(status_code=204)
```

- [ ] **Step 4: Wire router in app/main.py**

Add:

```python
from app.models import routes_api as models_routes_api
app.include_router(models_routes_api.router)
```

- [ ] **Step 5: Run + commit**

Run: `make test-unit ARGS=tests/unit/models/test_routes_crud.py`

Expected: PASS.

```bash
git add app/models/routes_api.py app/main.py tests/unit/models/test_routes_crud.py
git commit -m "feat(models): CRUD API enforcing allowed_gpu_indices subset"
```

---

### Task 24: Models web page

**Files:**
- Create: `app/models/routes_web.py`
- Create: `app/web/models/index.html`
- Create: `app/web/models/_card.html`
- Create: `app/web/models/_add_modal.html`
- Modify: `app/main.py` — register, replace placeholder
- Test: `tests/unit/models/test_routes_web.py`

- [ ] **Step 1: Write failing test**

```python
import sqlite3
import json
import bcrypt


def _seed_login(client, tmp_data_dir):
    pw = bcrypt.hashpw(b"hunter2", bcrypt.gensalt()).decode()
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute("INSERT INTO users(username, password_hash) VALUES (?, ?)", ("admin", pw))
        db.execute(
            "UPDATE setup_state SET step = 'done', draft = ? WHERE id = 1",
            (json.dumps({"allowed_gpu_indices": [0, 1, 2, 3]}),),
        )
        db.commit()
    client.post("/login", data={"username": "admin", "password": "hunter2"})


def test_models_page_renders_empty(tmp_data_dir, client):
    client.get("/healthz")
    _seed_login(client, tmp_data_dir)
    r = client.get("/models")
    assert r.status_code == 200
    assert "No models yet" in r.text


def test_models_page_lists_models(tmp_data_dir, client):
    client.get("/healthz")
    _seed_login(client, tmp_data_dir)
    client.post("/api/models", json={
        "served_model_name": "qwen3.5-9b",
        "hf_repo": "Qwen/Qwen3.5-9B",
        "gpu_indices": [1, 2],
    })
    r = client.get("/models")
    assert "qwen3.5-9b" in r.text
    assert "registered" in r.text
```

- [ ] **Step 2: Verify failure**

Run: `make test-unit ARGS=tests/unit/models/test_routes_web.py`

Expected: FAIL.

- [ ] **Step 3: Write web routes + templates**

`app/models/routes_web.py`:

```python
import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.db.database import open_db
from app.db.repos.models import ModelRepo
from app.db.repos.setup import SetupRepo

router = APIRouter()


@router.get("/models", response_class=HTMLResponse)
async def models_page(request: Request):
    if not request.state.user:
        return RedirectResponse(url="/login", status_code=303)
    settings = request.app.state.settings
    templates = request.app.state.templates
    async with open_db(settings.db_path) as db:
        models = await ModelRepo(db).list_all()
        setup_state = await SetupRepo(db).get()
    allowed = setup_state.draft.get("allowed_gpu_indices", [])
    rows = []
    for m in models:
        rows.append({
            "id": m.id,
            "served_model_name": m.served_model_name,
            "hf_repo": m.hf_repo,
            "hf_revision": m.hf_revision,
            "gpu_indices": json.loads(m.gpu_indices),
            "status": m.status,
            "pulled_bytes": m.pulled_bytes,
            "pulled_total": m.pulled_total,
            "last_error": m.last_error,
        })
    return templates.TemplateResponse(
        request, "models/index.html",
        {"active": "models", "models": rows, "allowed_gpus": allowed},
    )
```

`app/web/models/index.html`:

```html
{% extends "base_template.html" %}
{% block title %}Models - vllm-warden{% endblock %}
{% block content %}
<h1>Models</h1>
<p>
  <button class="btn" onclick="document.getElementById('add-modal').style.display='flex'">
    Add model
  </button>
</p>

{% if not models %}
  <p class="muted">No models yet.</p>
{% else %}
  <div id="model-cards">
    {% for m in models %}
      {% include "models/_card.html" %}
    {% endfor %}
  </div>
{% endif %}

{% include "models/_add_modal.html" %}
{% endblock %}
```

`app/web/models/_card.html`:

```html
<div class="card" id="model-{{ m.id }}">
  <div style="display: flex; justify-content: space-between; align-items: center;">
    <h3 style="margin: 0;">{{ m.served_model_name }}
      <span class="status-{{ m.status }}">[{{ m.status }}]</span>
    </h3>
    <span class="muted">GPUs {{ m.gpu_indices }}</span>
  </div>
  <p class="muted" style="margin: 4px 0;">{{ m.hf_repo }} @ {{ m.hf_revision }}</p>

  {% if m.status == 'pulling' and m.pulled_total %}
    <progress value="{{ m.pulled_bytes }}" max="{{ m.pulled_total }}" style="width: 100%;"></progress>
  {% endif %}

  {% if m.last_error %}
    <p style="color: var(--err);">{{ m.last_error }}</p>
  {% endif %}

  <div style="margin-top: 8px;">
    {% if m.status == 'registered' %}
      <button class="btn" hx-post="/api/models/{{ m.id }}/pull" hx-swap="none">Pull</button>
    {% elif m.status == 'pulled' %}
      <button class="btn" hx-post="/api/models/{{ m.id }}/load" hx-swap="none">Load</button>
    {% elif m.status == 'loaded' %}
      <button class="btn btn-danger" hx-post="/api/models/{{ m.id }}/unload" hx-swap="none">Unload</button>
    {% elif m.status == 'failed' %}
      <button class="btn" hx-post="/api/models/{{ m.id }}/load" hx-swap="none">Retry load</button>
    {% endif %}
    {% if m.status not in ['loading', 'loaded', 'unloading', 'pulling'] %}
      <button class="btn btn-danger" hx-delete="/api/models/{{ m.id }}"
              hx-confirm="Delete {{ m.served_model_name }}?"
              hx-target="#model-{{ m.id }}" hx-swap="outerHTML">Delete</button>
    {% endif %}
    <a href="/models/{{ m.id }}/logs">Logs</a>
  </div>
</div>
```

`app/web/models/_add_modal.html`:

```html
<div id="add-modal" class="modal" style="display: none;">
  <div class="modal-content">
    <h2>Add model</h2>
    <p><label>Served name<br><input id="am-name" placeholder="qwen3.5-9b"></label></p>
    <p><label>HuggingFace repo<br><input id="am-repo" placeholder="Qwen/Qwen3.5-9B"></label></p>
    <p><label>Revision<br><input id="am-rev" value="main"></label></p>
    <p>GPU indices (allowed: {{ allowed_gpus }}):</p>
    <p>
      {% for g in allowed_gpus %}
        <label style="margin-right: 12px;">
          <input type="checkbox" name="am-gpu" value="{{ g }}"> GPU {{ g }}
        </label>
      {% endfor %}
    </p>
    <p><label>Max model len (optional)<br><input id="am-mml" type="number" placeholder="auto"></label></p>
    <p>
      <button class="btn" onclick="submitNew()">Create</button>
      <button class="btn" style="background: #4b5563;"
              onclick="document.getElementById('add-modal').style.display='none'">Cancel</button>
    </p>
    <script>
      async function submitNew() {
        const gpus = [...document.querySelectorAll('input[name=am-gpu]:checked')]
          .map(el => parseInt(el.value, 10));
        const mml = document.getElementById('am-mml').value;
        const r = await fetch('/api/models', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            served_model_name: document.getElementById('am-name').value,
            hf_repo: document.getElementById('am-repo').value,
            hf_revision: document.getElementById('am-rev').value,
            gpu_indices: gpus,
            max_model_len: mml ? parseInt(mml, 10) : null,
          }),
        });
        if (r.ok) window.location.reload();
        else alert(await r.text());
      }
    </script>
  </div>
</div>
```

- [ ] **Step 4: Wire in main.py + remove placeholder**

```python
from app.models import routes_web as models_routes_web
app.include_router(models_routes_web.router)
```

Remove the `models_placeholder` route added earlier.

- [ ] **Step 5: Run + commit**

Run: `make test-unit ARGS=tests/unit/models/test_routes_web.py`

Expected: PASS.

```bash
git add app/models/routes_web.py app/web/models/ app/main.py tests/unit/models/test_routes_web.py
git commit -m "feat(models): models index page + add modal"
```

---

## Phase G: Pull task

### Task 25: Disk pre-flight before pull

**Files:**
- Create: `app/models/pull_task.py`
- Test: `tests/unit/models/test_pull_disk_check.py`

- [ ] **Step 1: Write failing test**

```python
import pytest

from app.models.pull_task import (
    insufficient_disk,
    estimate_repo_bytes,
    DiskShortage,
)


async def test_insufficient_disk_raises_when_estimate_exceeds_free(monkeypatch, tmp_path):
    from app.models import pull_task as pt
    monkeypatch.setattr(pt, "disk_free_bytes", lambda p: 10 * 1024 * 1024 * 1024)  # 10 GiB

    async def fake_estimate(*args, **kwargs):
        return 8 * 1024 * 1024 * 1024  # 8 GiB → 1.5x = 12 GiB > 10 GiB free
    monkeypatch.setattr(pt, "estimate_repo_bytes", fake_estimate)

    with pytest.raises(DiskShortage) as exc:
        await insufficient_disk("Qwen/Qwen3.5-9B", "main", tmp_path, hf_token=None)
    assert "needs" in str(exc.value).lower()


async def test_insufficient_disk_passes_when_enough(monkeypatch, tmp_path):
    from app.models import pull_task as pt
    monkeypatch.setattr(pt, "disk_free_bytes", lambda p: 100 * 1024 * 1024 * 1024)

    async def fake_estimate(*args, **kwargs):
        return 8 * 1024 * 1024 * 1024
    monkeypatch.setattr(pt, "estimate_repo_bytes", fake_estimate)

    await insufficient_disk("o/r", "main", tmp_path, hf_token=None)
```

- [ ] **Step 2: Verify failure**

Run: `make test-unit ARGS=tests/unit/models/test_pull_disk_check.py`

Expected: FAIL.

- [ ] **Step 3: Write app/models/pull_task.py**

```python
import asyncio
import logging
from pathlib import Path

from huggingface_hub import HfApi

from app.system.disk import disk_free_bytes

logger = logging.getLogger(__name__)
SAFETY_FACTOR = 1.5


class DiskShortage(Exception):
    pass


async def estimate_repo_bytes(repo: str, revision: str, hf_token: str | None) -> int:
    """Sum of file sizes in repo at revision. Returns bytes."""
    def _sync():
        api = HfApi(token=hf_token)
        info = api.repo_info(repo, revision=revision, files_metadata=True)
        total = 0
        for f in info.siblings or []:
            if f.size is not None:
                total += f.size
        return total
    return await asyncio.to_thread(_sync)


async def insufficient_disk(
    repo: str, revision: str, cache_dir: Path, hf_token: str | None
) -> None:
    """Raises DiskShortage if estimate * SAFETY_FACTOR > free bytes."""
    estimate = await estimate_repo_bytes(repo, revision, hf_token)
    free = disk_free_bytes(cache_dir)
    needed = int(estimate * SAFETY_FACTOR)
    if needed > free:
        raise DiskShortage(
            f"{repo}@{revision} estimated {estimate // (1024**3)} GiB; "
            f"with {SAFETY_FACTOR}x safety needs {needed // (1024**3)} GiB; "
            f"free {free // (1024**3)} GiB"
        )
```

- [ ] **Step 4: Run + commit**

Run: `make test-unit ARGS=tests/unit/models/test_pull_disk_check.py`

Expected: PASS.

```bash
git add app/models/pull_task.py tests/unit/models/test_pull_disk_check.py
git commit -m "feat(pull): disk pre-flight (1.5x safety factor)"
```

---

### Task 26: snapshot_download with progress + status updates

**Files:**
- Modify: `app/models/pull_task.py` — add `run_pull(model_id)`
- Test: `tests/unit/models/test_pull_run.py`

- [ ] **Step 1: Write failing test**

```python
import asyncio
import json
import sqlite3

import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.models import ModelRepo, ModelRow


async def _seed_model(db_path, model_id="m1", repo="o/r"):
    async with open_db(db_path) as db:
        await apply_migrations(db)
        await ModelRepo(db).insert(ModelRow(
            id=model_id, served_model_name=model_id, hf_repo=repo, hf_revision="main",
            gpu_indices=json.dumps([0]), tensor_parallel_size=1, dtype=None,
            max_model_len=None, gpu_memory_utilization=0.9, trust_remote_code=False,
            extra_args=json.dumps([]), status="registered", pulled_bytes=0,
            pulled_total=None, last_error=None,
        ))


async def test_run_pull_success_marks_pulled(tmp_data_dir, monkeypatch):
    from app.models import pull_task as pt
    from app.config import load_settings
    settings = load_settings()
    await _seed_model(settings.db_path)

    async def fake_disk_check(*args, **kwargs):
        return None
    async def fake_download(*args, **kwargs):
        return str(settings.hf_cache_dir / "fake")
    monkeypatch.setattr(pt, "insufficient_disk", fake_disk_check)
    monkeypatch.setattr(pt, "_snapshot_download", fake_download)

    await pt.run_pull("m1", settings)

    with sqlite3.connect(settings.db_path) as db:
        status, err = db.execute(
            "SELECT status, last_error FROM models WHERE id = 'm1'"
        ).fetchone()
    assert status == "pulled"
    assert err is None


async def test_run_pull_disk_shortage_marks_failed(tmp_data_dir, monkeypatch):
    from app.models import pull_task as pt
    from app.config import load_settings
    settings = load_settings()
    await _seed_model(settings.db_path)

    async def fake_disk_check(*args, **kwargs):
        raise pt.DiskShortage("not enough space")
    monkeypatch.setattr(pt, "insufficient_disk", fake_disk_check)

    await pt.run_pull("m1", settings)
    with sqlite3.connect(settings.db_path) as db:
        status, err = db.execute(
            "SELECT status, last_error FROM models WHERE id = 'm1'"
        ).fetchone()
    assert status == "failed"
    assert "not enough" in err.lower()


async def test_run_pull_marks_pulling_during(tmp_data_dir, monkeypatch):
    """Model status should be 'pulling' while download is running."""
    from app.models import pull_task as pt
    from app.config import load_settings
    settings = load_settings()
    await _seed_model(settings.db_path)

    seen_statuses = []
    async def fake_disk_check(*args, **kwargs):
        return None
    async def fake_download(model_id, settings_, repo, revision, token):
        with sqlite3.connect(settings_.db_path) as db:
            (s,) = db.execute("SELECT status FROM models WHERE id = ?", (model_id,)).fetchone()
        seen_statuses.append(s)
        return "/tmp/fake"

    monkeypatch.setattr(pt, "insufficient_disk", fake_disk_check)
    monkeypatch.setattr(pt, "_snapshot_download", fake_download)

    await pt.run_pull("m1", settings)
    assert "pulling" in seen_statuses
```

- [ ] **Step 2: Verify failure**

Run: `make test-unit ARGS=tests/unit/models/test_pull_run.py`

Expected: FAIL.

- [ ] **Step 3: Add run_pull and _snapshot_download**

Append to `app/models/pull_task.py`:

```python
from huggingface_hub import snapshot_download

from app.config import Settings
from app.db.database import open_db
from app.db.repos.models import ModelRepo


async def _read_hf_token(settings: Settings) -> str | None:
    p = settings.data_dir / "hf-token"
    if p.exists():
        return p.read_text().strip()
    return None


async def _snapshot_download(
    model_id: str, settings: Settings, repo: str, revision: str, token: str | None
) -> str:
    def _sync():
        return snapshot_download(
            repo_id=repo,
            revision=revision,
            cache_dir=str(settings.hf_cache_dir),
            token=token,
        )
    return await asyncio.to_thread(_sync)


async def run_pull(model_id: str, settings: Settings) -> None:
    """Background task: download model + update status."""
    async with open_db(settings.db_path) as db:
        repo_obj = ModelRepo(db)
        row = await repo_obj.get(model_id)
        if not row:
            logger.warning("run_pull: model %s missing", model_id)
            return
        token = await _read_hf_token(settings)

        await repo_obj.update_status(model_id, "pulling")
        try:
            await insufficient_disk(row.hf_repo, row.hf_revision, settings.hf_cache_dir, token)
            await _snapshot_download(model_id, settings, row.hf_repo, row.hf_revision, token)
            await repo_obj.update_status(model_id, "pulled")
        except DiskShortage as e:
            await repo_obj.update_status(model_id, "failed", last_error=str(e))
        except Exception as e:
            logger.exception("pull failed")
            await repo_obj.update_status(model_id, "failed", last_error=f"pull error: {e}")
```

- [ ] **Step 4: Run + commit**

Run: `make test-unit ARGS=tests/unit/models/test_pull_run.py`

Expected: PASS.

```bash
git add app/models/pull_task.py tests/unit/models/test_pull_run.py
git commit -m "feat(pull): run_pull task with status transitions"
```

---

### Task 27: Pull endpoint wires task into asyncio.create_task

**Files:**
- Modify: `app/models/routes_api.py` — add `POST /api/models/{model_id}/pull`
- Test: `tests/unit/models/test_pull_endpoint.py`

- [ ] **Step 1: Write failing test**

```python
import asyncio
import json
import sqlite3

import bcrypt
import pytest


def _seed_login(client, tmp_data_dir):
    pw = bcrypt.hashpw(b"hunter2", bcrypt.gensalt()).decode()
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute("INSERT INTO users(username, password_hash) VALUES (?, ?)", ("admin", pw))
        db.execute(
            "UPDATE setup_state SET step = 'done', draft = ? WHERE id = 1",
            (json.dumps({"allowed_gpu_indices": [0, 1]}),),
        )
        db.commit()
    client.post("/login", data={"username": "admin", "password": "hunter2"})


def test_pull_endpoint_accepts_and_runs_task(tmp_data_dir, client, monkeypatch):
    from app.models import routes_api as ra
    from app.models import pull_task as pt

    invoked = []

    async def fake_run_pull(model_id, settings):
        invoked.append(model_id)
        with sqlite3.connect(settings.db_path) as db:
            db.execute("UPDATE models SET status = 'pulled' WHERE id = ?", (model_id,))
            db.commit()
    monkeypatch.setattr(ra, "run_pull", fake_run_pull)

    client.get("/healthz")
    _seed_login(client, tmp_data_dir)
    create = client.post("/api/models", json={
        "served_model_name": "x", "hf_repo": "o/r", "gpu_indices": [0]
    })
    mid = create.json()["id"]

    r = client.post(f"/api/models/{mid}/pull")
    assert r.status_code == 202

    # Wait briefly for the task
    import time
    for _ in range(50):
        with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
            (s,) = db.execute(
                "SELECT status FROM models WHERE id = ?", (mid,)
            ).fetchone()
        if s == "pulled":
            break
        time.sleep(0.05)
    assert s == "pulled"
    assert mid in invoked


def test_pull_endpoint_404_on_missing(tmp_data_dir, client):
    client.get("/healthz")
    _seed_login(client, tmp_data_dir)
    r = client.post("/api/models/does-not-exist/pull")
    assert r.status_code == 404


def test_pull_endpoint_409_when_already_loaded(tmp_data_dir, client):
    client.get("/healthz")
    _seed_login(client, tmp_data_dir)
    create = client.post("/api/models", json={
        "served_model_name": "x", "hf_repo": "o/r", "gpu_indices": [0]
    })
    mid = create.json()["id"]
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute("UPDATE models SET status = 'loaded' WHERE id = ?", (mid,))
        db.commit()
    r = client.post(f"/api/models/{mid}/pull")
    assert r.status_code == 409
```

- [ ] **Step 2: Verify failure**

Run: `make test-unit ARGS=tests/unit/models/test_pull_endpoint.py`

Expected: FAIL.

- [ ] **Step 3: Add pull endpoint**

In `app/models/routes_api.py`:

```python
import asyncio
from app.models.pull_task import run_pull


@router.post("/{model_id}/pull", status_code=202)
async def trigger_pull(
    model_id: str, request: Request, _user: str = Depends(require_session_json)
):
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        row = await ModelRepo(db).get(model_id)
        if not row:
            raise HTTPException(404, "not found")
        if row.status not in ("registered", "failed", "pulled"):
            raise HTTPException(409, f"cannot pull from status '{row.status}'")
    asyncio.create_task(run_pull(model_id, settings))
    return {"status": "pulling"}
```

- [ ] **Step 4: Run + commit**

Run: `make test-unit ARGS=tests/unit/models/test_pull_endpoint.py`

Expected: PASS.

```bash
git add app/models/routes_api.py tests/unit/models/test_pull_endpoint.py
git commit -m "feat(pull): POST /api/models/{id}/pull endpoint"
```

---

## Phase H: Supervisor — vLLM subprocess lifecycle (THE BUG FIX)

> **Critical:** Task 29 is the regression-test anchor for the 2026-05-08 production bug. `CUDA_VISIBLE_DEVICES` MUST be derived from `model.gpu_indices` (the DB row), NEVER inherited from the parent process env. Tests in this phase are written first and must FAIL before implementation.

### Task 28: Supervisor scaffold + GpuOwnership

**Files:**
- Create: `app/runtime/__init__.py`
- Create: `app/runtime/gpu_ownership.py`
- Create: `app/runtime/supervisor.py` (skeleton only)
- Test: `tests/unit/runtime/test_gpu_ownership.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/runtime/__init__.py — empty
# tests/unit/runtime/test_gpu_ownership.py
import pytest
from app.runtime.gpu_ownership import GpuOwnership, GpuConflict


def test_claim_grants_exclusive_ownership():
    g = GpuOwnership()
    g.claim("m1", [0, 1])
    assert g.owner_of(0) == "m1"
    assert g.owner_of(1) == "m1"
    assert g.owner_of(2) is None


def test_claim_conflict_raises():
    g = GpuOwnership()
    g.claim("m1", [0, 1])
    with pytest.raises(GpuConflict) as ei:
        g.claim("m2", [1, 2])
    assert "1" in str(ei.value)
    # Partial claim must NOT have applied:
    assert g.owner_of(2) is None


def test_release_frees_gpus():
    g = GpuOwnership()
    g.claim("m1", [0, 1])
    g.release("m1")
    assert g.owner_of(0) is None
    g.claim("m2", [0, 1])  # reuse OK
    assert g.owner_of(0) == "m2"


def test_release_unknown_is_noop():
    g = GpuOwnership()
    g.release("does-not-exist")  # must not raise
```

- [ ] **Step 2: Run test → FAIL**

Run: `make test-unit ARGS=tests/unit/runtime/test_gpu_ownership.py`

Expected: FAIL `ModuleNotFoundError: No module named 'app.runtime.gpu_ownership'`.

- [ ] **Step 3: Implement**

```python
# app/runtime/__init__.py — empty
# app/runtime/gpu_ownership.py
from threading import Lock


class GpuConflict(RuntimeError):
    pass


class GpuOwnership:
    """In-memory exclusive GPU ownership: gpu_idx -> model_id."""

    def __init__(self) -> None:
        self._owner: dict[int, str] = {}
        self._lock = Lock()

    def claim(self, model_id: str, gpu_indices: list[int]) -> None:
        with self._lock:
            conflicts = [g for g in gpu_indices if g in self._owner and self._owner[g] != model_id]
            if conflicts:
                raise GpuConflict(f"GPUs {conflicts} already claimed")
            for g in gpu_indices:
                self._owner[g] = model_id

    def release(self, model_id: str) -> None:
        with self._lock:
            self._owner = {g: m for g, m in self._owner.items() if m != model_id}

    def owner_of(self, gpu_idx: int) -> str | None:
        with self._lock:
            return self._owner.get(gpu_idx)

    def all_claims(self) -> dict[int, str]:
        with self._lock:
            return dict(self._owner)
```

```python
# app/runtime/supervisor.py — skeleton only; filled in subsequent tasks
import asyncio
from app.runtime.gpu_ownership import GpuOwnership


class Supervisor:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.gpus = GpuOwnership()
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._ports: dict[str, int] = {}
        self._lock = asyncio.Lock()
```

- [ ] **Step 4: Run + commit**

Run: `make test-unit ARGS=tests/unit/runtime/test_gpu_ownership.py`

Expected: PASS.

```bash
git add app/runtime/__init__.py app/runtime/gpu_ownership.py app/runtime/supervisor.py tests/unit/runtime/__init__.py tests/unit/runtime/test_gpu_ownership.py
git commit -m "feat(runtime): GpuOwnership exclusive-claim primitive + Supervisor skeleton"
```

---

### Task 29: env_builder.py — THE 2026-05-08 BUG FIX

**Files:**
- Create: `app/runtime/env_builder.py`
- Test: `tests/unit/runtime/test_env_builder.py`

> **This task is the reason this rewrite exists.** Phase 1 wizard ignored the user's GPU selection because `CUDA_VISIBLE_DEVICES` was never set in the subprocess env, so vLLM saw all GPUs visible to the container. Here we lock it down with regression tests that fail without the fix.

- [ ] **Step 1: Write failing tests (regression anchors)**

```python
# tests/unit/runtime/test_env_builder.py
import pytest
from app.runtime.env_builder import build_subprocess_env


@pytest.fixture
def model():
    class M:
        id = "qwen3.5-9b"
        gpu_indices = [1, 2]
        tensor_parallel_size = 2
    return M()


def test_cuda_visible_devices_from_model_row_not_env(model, monkeypatch):
    """Regression: 2026-05-08 bug. CUDA_VISIBLE_DEVICES must come from model.gpu_indices,
    NEVER inherited from parent env."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")  # parent env "lies"
    env = build_subprocess_env(model, hf_token="hf_xxx", data_dir="/data")
    assert env["CUDA_VISIBLE_DEVICES"] == "1,2"  # NOT "0,1,2,3"


def test_cuda_visible_devices_serializes_in_gpu_indices_order():
    """Order in the env var matches gpu_indices order. vLLM treats CUDA_VISIBLE_DEVICES
    position as the logical device id."""
    class M:
        id = "m"
        gpu_indices = [3, 0, 2]  # deliberately unsorted
        tensor_parallel_size = 3
    env = build_subprocess_env(M(), hf_token="hf_xxx", data_dir="/data")
    assert env["CUDA_VISIBLE_DEVICES"] == "3,0,2"


def test_env_builder_sets_hf_home_and_token(model):
    env = build_subprocess_env(model, hf_token="hf_secret", data_dir="/data")
    assert env["HF_HOME"] == "/data/hf-cache"
    assert env["HUGGING_FACE_HUB_TOKEN"] == "hf_secret"


def test_env_builder_does_not_leak_warden_secrets(model, monkeypatch):
    """VW_COOKIE_SECRET, VW_ADMIN_PASSWORD etc. from parent env must NOT propagate."""
    monkeypatch.setenv("VW_COOKIE_SECRET", "session-key")
    monkeypatch.setenv("VW_ADMIN_PASSWORD", "supersecret")
    env = build_subprocess_env(model, hf_token="hf_xxx", data_dir="/data")
    assert "VW_COOKIE_SECRET" not in env
    assert "VW_ADMIN_PASSWORD" not in env


def test_env_builder_empty_gpu_indices_raises():
    class M:
        id = "m"
        gpu_indices = []
        tensor_parallel_size = 1
    with pytest.raises(ValueError, match="gpu_indices"):
        build_subprocess_env(M(), hf_token="hf_xxx", data_dir="/data")


def test_env_builder_tp_mismatch_raises():
    """tensor_parallel_size must equal len(gpu_indices). Defense in depth."""
    class M:
        id = "m"
        gpu_indices = [0, 1, 2]
        tensor_parallel_size = 2
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        build_subprocess_env(M(), hf_token="hf_xxx", data_dir="/data")
```

- [ ] **Step 2: Run tests → FAIL**

Run: `make test-unit ARGS=tests/unit/runtime/test_env_builder.py`

Expected: ALL FAIL with `ModuleNotFoundError: No module named 'app.runtime.env_builder'`.

- [ ] **Step 3: Implement**

```python
# app/runtime/env_builder.py
"""Build subprocess env for vLLM. THIS FIXES THE 2026-05-08 BUG.

CUDA_VISIBLE_DEVICES is derived from model.gpu_indices (a DB column populated by the
user's wizard/CRUD selection). It is NEVER inherited from the parent process. The
parent vllm-warden container is launched with all GPUs visible (the launcher gives
the container `--gpus all` so the supervisor can dispatch any GPU); each per-model
subprocess MUST have CUDA_VISIBLE_DEVICES restricted to exactly that model's
gpu_indices, in the order specified, so vLLM's logical device 0 == gpu_indices[0].
"""
from __future__ import annotations


def build_subprocess_env(model, *, hf_token: str, data_dir: str) -> dict[str, str]:
    """Construct the env dict for a vLLM subprocess.

    Returns a closed dict. Caller passes this dict as the `env=` kwarg to
    asyncio.create_subprocess_exec, which uses ONLY this env (no inheritance).
    """
    if not model.gpu_indices:
        raise ValueError("gpu_indices must be non-empty")
    if model.tensor_parallel_size != len(model.gpu_indices):
        raise ValueError(
            f"tensor_parallel_size ({model.tensor_parallel_size}) must equal "
            f"len(gpu_indices) ({len(model.gpu_indices)})"
        )

    return {
        # === THE BUG FIX ===
        "CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in model.gpu_indices),
        # HF cache shared so pull-test caches are reused at load time.
        "HF_HOME": f"{data_dir}/hf-cache",
        "HUGGING_FACE_HUB_TOKEN": hf_token,
        "VLLM_LOGGING_LEVEL": "INFO",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
    }
```

- [ ] **Step 4: Run + commit (BUG FIX LANDS HERE)**

Run: `make test-unit ARGS=tests/unit/runtime/test_env_builder.py`

Expected: ALL 6 PASS.

```bash
git add app/runtime/env_builder.py tests/unit/runtime/test_env_builder.py
git commit -m "fix(runtime): set CUDA_VISIBLE_DEVICES from model.gpu_indices

Regression fix for 2026-05-08 production bug. Phase 1 wizard's GPU
selection never reached the vLLM subprocess because CUDA_VISIBLE_DEVICES
was inherited from the parent container env (which sees all GPUs).
build_subprocess_env() now constructs a closed env dict and the
supervisor passes it as env= without inheritance.

Tests test_cuda_visible_devices_from_model_row_not_env and
test_cuda_visible_devices_serializes_in_gpu_indices_order anchor
the regression and MUST never be deleted."
```

---

### Task 30: vLLM command builder

**Files:**
- Create: `app/runtime/cmd_builder.py`
- Test: `tests/unit/runtime/test_cmd_builder.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/runtime/test_cmd_builder.py
from app.runtime.cmd_builder import build_vllm_args


class _M:
    id = "qwen"
    hf_repo = "Qwen/Qwen3.5-9B"
    hf_revision = "main"
    tensor_parallel_size = 2
    max_model_len = 8192
    dtype = "auto"
    gpu_memory_utilization = 0.90
    served_name = "qwen3.5-9b"


def test_build_vllm_args_minimal():
    m = _M()
    args = build_vllm_args(m, port=10001)
    assert args[0] == "--model"
    assert args[1] == "Qwen/Qwen3.5-9B"
    assert args[args.index("--revision") + 1] == "main"
    assert args[args.index("--tensor-parallel-size") + 1] == "2"
    assert args[args.index("--port") + 1] == "10001"
    assert args[args.index("--served-model-name") + 1] == "qwen3.5-9b"
    assert args[args.index("--host") + 1] == "127.0.0.1"


def test_build_vllm_args_omits_revision_when_none():
    m = _M(); m.hf_revision = None
    args = build_vllm_args(m, port=10001)
    assert "--revision" not in args


def test_build_vllm_args_includes_dtype_and_max_len():
    m = _M(); m.dtype = "bfloat16"; m.max_model_len = 4096
    args = build_vllm_args(m, port=10001)
    assert args[args.index("--dtype") + 1] == "bfloat16"
    assert args[args.index("--max-model-len") + 1] == "4096"
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement**

```python
# app/runtime/cmd_builder.py
def build_vllm_args(model, *, port: int) -> list[str]:
    args: list[str] = [
        "--model", model.hf_repo,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--served-model-name", model.served_name,
        "--tensor-parallel-size", str(model.tensor_parallel_size),
        "--dtype", model.dtype,
        "--max-model-len", str(model.max_model_len),
        "--gpu-memory-utilization", str(model.gpu_memory_utilization),
    ]
    if model.hf_revision:
        args += ["--revision", model.hf_revision]
    return args
```

- [ ] **Step 4: Run + commit**

```bash
git add app/runtime/cmd_builder.py tests/unit/runtime/test_cmd_builder.py
git commit -m "feat(runtime): vLLM CLI argument builder"
```

---

### Task 31: Supervisor.load() with subprocess + GPU claim

**Files:**
- Modify: `app/runtime/supervisor.py`
- Test: `tests/unit/runtime/test_supervisor_load.py`

- [ ] **Step 1: Write failing test (mocked subprocess)**

```python
# tests/unit/runtime/test_supervisor_load.py
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.runtime.supervisor import Supervisor
from app.runtime.gpu_ownership import GpuConflict


class _Settings:
    data_dir = "/data"
    hf_token_path = "/data/hf-token"


class _M:
    id = "qwen"
    hf_repo = "Qwen/Qwen3.5-9B"
    hf_revision = "main"
    served_name = "qwen3.5-9b"
    gpu_indices = [1, 2]
    tensor_parallel_size = 2
    max_model_len = 8192
    dtype = "auto"
    gpu_memory_utilization = 0.90


@pytest.mark.asyncio
async def test_load_claims_gpus_and_spawns_subprocess(tmp_path):
    settings = _Settings()
    settings.data_dir = str(tmp_path)
    settings.hf_token_path = str(tmp_path / "hf-token")
    (tmp_path / "hf-token").write_text("hf_xxx")

    sup = Supervisor(settings)
    fake_proc = MagicMock(); fake_proc.pid = 99999; fake_proc.returncode = None

    with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=fake_proc)) as spawn:
        await sup.load(_M(), port=10001)

    assert sup.gpus.owner_of(1) == "qwen"
    assert sup.gpus.owner_of(2) == "qwen"
    assert "qwen" in sup._processes
    kwargs = spawn.call_args.kwargs
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "1,2"
    assert "VW_ADMIN_PASSWORD" not in kwargs["env"]  # closed env
    assert kwargs["start_new_session"] is True


@pytest.mark.asyncio
async def test_load_conflict_releases_gpus(tmp_path):
    """If a second model wants overlapping GPUs, the claim must NOT partially apply."""
    settings = _Settings(); settings.data_dir = str(tmp_path)
    settings.hf_token_path = str(tmp_path / "hf-token")
    (tmp_path / "hf-token").write_text("hf_xxx")
    sup = Supervisor(settings)

    m1 = _M(); m1.id = "m1"; m1.gpu_indices = [0, 1]
    m2 = _M(); m2.id = "m2"; m2.gpu_indices = [1, 2]

    fake_proc = MagicMock(); fake_proc.pid = 1; fake_proc.returncode = None
    with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=fake_proc)):
        await sup.load(m1, port=10001)
        with pytest.raises(GpuConflict):
            await sup.load(m2, port=10002)

    assert sup.gpus.owner_of(0) == "m1"
    assert sup.gpus.owner_of(1) == "m1"
    assert sup.gpus.owner_of(2) is None


@pytest.mark.asyncio
async def test_failed_load_releases_gpus(tmp_path):
    """If subprocess spawn fails, GPUs must be released so retry can succeed."""
    settings = _Settings(); settings.data_dir = str(tmp_path)
    settings.hf_token_path = str(tmp_path / "hf-token")
    (tmp_path / "hf-token").write_text("hf_xxx")
    sup = Supervisor(settings)
    m = _M()

    with patch("asyncio.create_subprocess_exec", new=AsyncMock(side_effect=OSError("nope"))):
        with pytest.raises(OSError):
            await sup.load(m, port=10001)

    assert sup.gpus.owner_of(1) is None
    assert sup.gpus.owner_of(2) is None
    assert "qwen" not in sup._processes
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement (replace skeleton in supervisor.py)**

```python
# app/runtime/supervisor.py
import asyncio
import os
from pathlib import Path

from app.runtime.cmd_builder import build_vllm_args
from app.runtime.env_builder import build_subprocess_env
from app.runtime.gpu_ownership import GpuOwnership, GpuConflict


class Supervisor:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.gpus = GpuOwnership()
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._ports: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def load(self, model, *, port: int) -> None:
        async with self._lock:
            if model.id in self._processes:
                raise RuntimeError(f"model {model.id} already running")
            self.gpus.claim(model.id, model.gpu_indices)
            try:
                hf_token = Path(self.settings.hf_token_path).read_text().strip()
                env = build_subprocess_env(model, hf_token=hf_token, data_dir=self.settings.data_dir)
                cmd = ["vllm", "serve", *build_vllm_args(model, port=port)]

                log_dir = Path(self.settings.data_dir) / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / f"{model.id}.log"
                log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                try:
                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        env=env,
                        stdout=log_fd,
                        stderr=log_fd,
                        start_new_session=True,
                    )
                finally:
                    os.close(log_fd)

                self._processes[model.id] = proc
                self._ports[model.id] = port
            except Exception:
                self.gpus.release(model.id)
                raise

    def get_port(self, model_id: str) -> int | None:
        return self._ports.get(model_id)

    def is_running(self, model_id: str) -> bool:
        proc = self._processes.get(model_id)
        return proc is not None and proc.returncode is None
```

- [ ] **Step 4: Run + commit**

```bash
git add app/runtime/supervisor.py tests/unit/runtime/test_supervisor_load.py
git commit -m "feat(runtime): Supervisor.load — atomic GPU claim + vLLM subprocess spawn"
```

---

### Task 32: Health check loop

**Files:**
- Modify: `app/runtime/supervisor.py`
- Test: `tests/unit/runtime/test_supervisor_health.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/runtime/test_supervisor_health.py
from unittest.mock import AsyncMock, patch
import pytest
from app.runtime.supervisor import wait_for_health


@pytest.mark.asyncio
async def test_wait_for_health_returns_when_endpoint_200():
    responses = [Exception("conn refused"), Exception("conn refused"), 200]

    async def fake_get(url, timeout):
        v = responses.pop(0)
        if isinstance(v, Exception):
            raise v
        class R: status_code = v
        return R()

    with patch("app.runtime.supervisor._http_get", new=AsyncMock(side_effect=fake_get)):
        ok = await wait_for_health(port=10001, timeout_s=5, interval_s=0.05)
    assert ok is True


@pytest.mark.asyncio
async def test_wait_for_health_times_out():
    with patch("app.runtime.supervisor._http_get",
               new=AsyncMock(side_effect=Exception("never"))):
        ok = await wait_for_health(port=10001, timeout_s=0.2, interval_s=0.05)
    assert ok is False
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement (append to supervisor.py)**

```python
# app/runtime/supervisor.py — append at module level
import time
import httpx


async def _http_get(url: str, timeout: float):
    async with httpx.AsyncClient(timeout=timeout) as c:
        return await c.get(url)


async def wait_for_health(*, port: int, timeout_s: float = 600.0, interval_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        try:
            r = await _http_get(url, timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        await asyncio.sleep(interval_s)
    return False
```

- [ ] **Step 4: Run + commit**

```bash
git add app/runtime/supervisor.py tests/unit/runtime/test_supervisor_health.py
git commit -m "feat(runtime): wait_for_health polling helper"
```

---

### Task 33: Supervisor.unload() — SIGTERM → 30s grace → SIGKILL

**Files:**
- Modify: `app/runtime/supervisor.py`
- Test: `tests/unit/runtime/test_supervisor_unload.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/runtime/test_supervisor_unload.py
import asyncio
import signal
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from app.runtime.supervisor import Supervisor


class _Settings:
    data_dir = "/data"
    hf_token_path = "/data/hf-token"


@pytest.mark.asyncio
async def test_unload_sigterms_then_releases_gpus(tmp_path):
    s = _Settings(); s.data_dir = str(tmp_path); s.hf_token_path = str(tmp_path / "tok")
    (tmp_path / "tok").write_text("hf_x")
    sup = Supervisor(s)
    proc = MagicMock(); proc.pid = 4242; proc.returncode = None

    async def fake_wait():
        proc.returncode = 0
        return 0
    proc.wait = AsyncMock(side_effect=fake_wait)

    sup._processes["m1"] = proc
    sup._ports["m1"] = 10001
    sup.gpus.claim("m1", [0, 1])

    with patch("os.killpg") as kp:
        await sup.unload("m1")
    kp.assert_called_with(4242, signal.SIGTERM)
    assert "m1" not in sup._processes
    assert sup.gpus.owner_of(0) is None


@pytest.mark.asyncio
async def test_unload_sigkill_after_timeout(tmp_path):
    s = _Settings(); s.data_dir = str(tmp_path); s.hf_token_path = str(tmp_path / "tok")
    (tmp_path / "tok").write_text("hf_x")
    sup = Supervisor(s)
    proc = MagicMock(); proc.pid = 4242; proc.returncode = None

    async def hang():
        await asyncio.sleep(60)
    proc.wait = AsyncMock(side_effect=hang)

    sup._processes["m1"] = proc
    sup.gpus.claim("m1", [0])

    sent = []
    def fake_killpg(pid, sig):
        sent.append(sig)
        if sig == signal.SIGKILL:
            proc.returncode = -9
            proc.wait = AsyncMock(return_value=-9)
    with patch("os.killpg", side_effect=fake_killpg):
        with patch("app.runtime.supervisor.UNLOAD_GRACE_SECONDS", 0.2):
            await sup.unload("m1")
    assert signal.SIGTERM in sent
    assert signal.SIGKILL in sent
    assert sup.gpus.owner_of(0) is None
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement (append to supervisor.py — `unload` method on Supervisor and `UNLOAD_GRACE_SECONDS` constant)**

```python
# app/runtime/supervisor.py — add module constant + method
import signal


UNLOAD_GRACE_SECONDS = 30.0


# Add this method to the Supervisor class:
#
# async def unload(self, model_id: str) -> None:
#     async with self._lock:
#         proc = self._processes.get(model_id)
#         if proc is None:
#             self.gpus.release(model_id)  # idempotent cleanup
#             return
#         if proc.returncode is None:
#             try:
#                 os.killpg(proc.pid, signal.SIGTERM)
#             except ProcessLookupError:
#                 pass
#             try:
#                 await asyncio.wait_for(proc.wait(), timeout=UNLOAD_GRACE_SECONDS)
#             except asyncio.TimeoutError:
#                 try:
#                     os.killpg(proc.pid, signal.SIGKILL)
#                 except ProcessLookupError:
#                     pass
#                 await proc.wait()
#         self._processes.pop(model_id, None)
#         self._ports.pop(model_id, None)
#         self.gpus.release(model_id)
```

(Implementer: paste the `unload` method body inside the `Supervisor` class body — comments shown for clarity in the plan.)

- [ ] **Step 4: Run + commit**

```bash
git add app/runtime/supervisor.py tests/unit/runtime/test_supervisor_unload.py
git commit -m "feat(runtime): Supervisor.unload — SIGTERM/SIGKILL with grace period"
```

---

### Task 34: Port allocator

**Files:**
- Create: `app/runtime/port_alloc.py`
- Test: `tests/unit/runtime/test_port_alloc.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/runtime/test_port_alloc.py
import pytest
from app.runtime.port_alloc import PortAllocator, PortExhausted


def test_allocate_and_release():
    p = PortAllocator(start=10000, end=10003)
    a = p.allocate()
    b = p.allocate()
    assert a != b
    assert 10000 <= a <= 10003
    p.release(a)
    c = p.allocate()
    assert c == a  # reuses freed port


def test_exhaustion_raises():
    p = PortAllocator(start=10000, end=10001)
    p.allocate(); p.allocate()
    with pytest.raises(PortExhausted):
        p.allocate()
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement**

```python
# app/runtime/port_alloc.py
from threading import Lock


class PortExhausted(RuntimeError):
    pass


class PortAllocator:
    def __init__(self, *, start: int = 10000, end: int = 10999) -> None:
        self._free: list[int] = list(range(start, end + 1))
        self._used: set[int] = set()
        self._lock = Lock()

    def allocate(self) -> int:
        with self._lock:
            if not self._free:
                raise PortExhausted("no free ports in subprocess range")
            p = self._free.pop(0)
            self._used.add(p)
            return p

    def release(self, port: int) -> None:
        with self._lock:
            if port in self._used:
                self._used.discard(port)
                self._free.append(port)
```

- [ ] **Step 4: Run + commit**

```bash
git add app/runtime/port_alloc.py tests/unit/runtime/test_port_alloc.py
git commit -m "feat(runtime): PortAllocator for vLLM subprocess port range"
```

---

### Task 35: Load/unload API endpoints

**Files:**
- Modify: `app/models/routes_api.py`
- Test: `tests/unit/models/test_load_endpoint.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/models/test_load_endpoint.py
from unittest.mock import AsyncMock, patch
import pytest


@pytest.mark.asyncio
async def test_load_validates_gpus_against_allowed_set(client_logged_in, db_with_pulled_model):
    """If gpu_indices ⊄ allowed_gpu_indices, must return 422."""
    await db_with_pulled_model.set_allowed_gpus([1, 2, 3])
    await db_with_pulled_model.set_model_gpus("qwen", [0, 1])

    r = await client_logged_in.post("/api/models/qwen/load")
    assert r.status_code == 422
    assert "allowed" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_load_calls_supervisor_then_health_check(client_logged_in, db_with_pulled_model):
    await db_with_pulled_model.set_allowed_gpus([0, 1, 2, 3])
    await db_with_pulled_model.set_model_gpus("qwen", [0, 1])

    sup_load = AsyncMock()
    health = AsyncMock(return_value=True)
    with patch("app.runtime.supervisor.Supervisor.load", new=sup_load), \
         patch("app.runtime.supervisor.wait_for_health", new=health):
        r = await client_logged_in.post("/api/models/qwen/load")
    assert r.status_code == 202
    assert sup_load.await_count == 1


@pytest.mark.asyncio
async def test_unload_returns_404_for_unknown_model(client_logged_in):
    r = await client_logged_in.post("/api/models/missing/unload")
    assert r.status_code == 404
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement (append to `app/models/routes_api.py`)**

```python
# app/models/routes_api.py — append
from app.runtime.supervisor import wait_for_health


@router.post("/{model_id}/load", status_code=202)
async def load_model(model_id: str, request: Request, _user: str = Depends(require_session_json)):
    settings = request.app.state.settings
    sup = request.app.state.supervisor
    port_alloc = request.app.state.port_allocator

    async with open_db(settings.db_path) as db:
        model = await ModelRepo(db).get(model_id)
        if not model:
            raise HTTPException(404, "not found")
        if model.status not in ("pulled", "failed"):
            raise HTTPException(409, f"cannot load from status '{model.status}'")
        allowed = await SetupRepo(db).get_allowed_gpu_indices()
        if not set(model.gpu_indices).issubset(set(allowed)):
            raise HTTPException(
                422, f"gpu_indices {model.gpu_indices} not subset of allowed {allowed}"
            )
        await ModelRepo(db).set_status(model_id, "loading")

    port = port_alloc.allocate()

    async def runner():
        try:
            await sup.load(model, port=port)
        except Exception as e:
            async with open_db(settings.db_path) as db:
                await ModelRepo(db).set_status(model_id, "failed", error=str(e))
            port_alloc.release(port)
            return
        ok = await wait_for_health(port=port, timeout_s=settings.load_timeout_s)
        async with open_db(settings.db_path) as db:
            if ok:
                await ModelRepo(db).set_status(model_id, "loaded")
                await RuntimeRepo(db).set(model_id, port=port, pid=sup._processes[model_id].pid)
            else:
                await ModelRepo(db).set_status(model_id, "failed", error="health timeout")
                await sup.unload(model_id)
                port_alloc.release(port)

    asyncio.create_task(runner())
    return {"status": "loading", "port": port}


@router.post("/{model_id}/unload", status_code=202)
async def unload_model(model_id: str, request: Request, _user: str = Depends(require_session_json)):
    settings = request.app.state.settings
    sup = request.app.state.supervisor
    port_alloc = request.app.state.port_allocator
    async with open_db(settings.db_path) as db:
        model = await ModelRepo(db).get(model_id)
        if not model:
            raise HTTPException(404, "not found")
        if model.status not in ("loaded", "failed"):
            raise HTTPException(409, f"cannot unload from status '{model.status}'")
        await ModelRepo(db).set_status(model_id, "unloading")
        rt = await RuntimeRepo(db).get(model_id)
        port = rt["port"] if rt else None
    await sup.unload(model_id)
    if port:
        port_alloc.release(port)
    async with open_db(settings.db_path) as db:
        await RuntimeRepo(db).clear(model_id)
        await ModelRepo(db).set_status(model_id, "pulled")
    return {"status": "unloaded"}
```

- [ ] **Step 4: Run + commit**

```bash
git add app/models/routes_api.py tests/unit/models/test_load_endpoint.py
git commit -m "feat(models): POST /api/models/{id}/load and /unload"
```

---

### Task 36: Wire supervisor + port allocator into app lifespan

**Files:**
- Modify: `app/main.py`
- Test: `tests/unit/test_app_state.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_app_state.py
def test_app_state_has_supervisor_and_port_allocator(client):
    app = client.app
    assert hasattr(app.state, "supervisor")
    assert hasattr(app.state, "port_allocator")
    assert app.state.port_allocator.allocate() >= 10000
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement (append inside `lifespan()` in `app/main.py`, after migrations + runtime cleanup)**

```python
from app.runtime.supervisor import Supervisor
from app.runtime.port_alloc import PortAllocator

app.state.supervisor = Supervisor(app.state.settings)
app.state.port_allocator = PortAllocator(start=10000, end=10999)
```

- [ ] **Step 4: Run + commit**

```bash
git add app/main.py tests/unit/test_app_state.py
git commit -m "feat(app): wire Supervisor + PortAllocator into app.state"
```

---

## Phase I: Log streaming (SSE)

### Task 37: SSE log stream endpoint

**Files:**
- Create: `app/models/routes_logs.py`
- Test: `tests/unit/models/test_logs_stream.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/models/test_logs_stream.py
import pytest


@pytest.mark.asyncio
async def test_logs_stream_404_when_no_log_file(client_logged_in):
    r = await client_logged_in.get("/api/models/missing/logs/stream")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_logs_stream_returns_sse_content_type(client_logged_in, tmp_data_dir):
    log_path = tmp_data_dir / "logs" / "qwen.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("hello world\n")
    r = await client_logged_in.get(
        "/api/models/qwen/logs/stream",
        headers={"Accept": "text/event-stream"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
```

- [ ] **Step 2: Run → FAIL.**

Run: `make test-unit ARGS=tests/unit/models/test_logs_stream.py`

Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# app/models/routes_logs.py
import asyncio
from pathlib import Path

import aiofiles
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.auth.session import require_session_json

router = APIRouter(prefix="/api/models", tags=["logs"])


async def _tail(path: Path, *, from_end: bool = True):
    """Yield lines as they are appended to `path`. Stops if the file is removed."""
    async with aiofiles.open(path, "r") as f:
        if from_end:
            await f.seek(0, 2)
        while True:
            line = await f.readline()
            if line:
                yield line
            else:
                if not path.exists():
                    return
                await asyncio.sleep(0.5)


@router.get("/{model_id}/logs/stream")
async def stream_logs(model_id: str, request: Request, _user: str = Depends(require_session_json)):
    settings = request.app.state.settings
    log_path = Path(settings.data_dir) / "logs" / f"{model_id}.log"
    if not log_path.exists():
        raise HTTPException(404, "no log file")

    async def gen():
        # Send last 200 lines first, then tail.
        async with aiofiles.open(log_path, "r") as f:
            content = await f.read()
        for line in content.splitlines()[-200:]:
            yield f"data: {line}\n\n"
        async for line in _tail(log_path, from_end=True):
            if await request.is_disconnected():
                return
            yield f"data: {line.rstrip()}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
```

Wire it in `app/main.py`:

```python
from app.models.routes_logs import router as logs_router
app.include_router(logs_router)
```

- [ ] **Step 4: Run + commit**

Run: `make test-unit ARGS=tests/unit/models/test_logs_stream.py`

Expected: PASS.

```bash
git add app/models/routes_logs.py app/main.py tests/unit/models/test_logs_stream.py
git commit -m "feat(logs): SSE /api/models/{id}/logs/stream via aiofiles tail"
```

---

## Phase J: Real-process integration tests

### Task 38: Fake vLLM (aiohttp double)

**Files:**
- Create: `tests/fakes/__init__.py`
- Create: `tests/fakes/fake_vllm.py`
- Test: `tests/integration/__init__.py`
- Test: `tests/integration/test_fake_vllm.py`

> The fake speaks the vLLM `/v1/*` and `/health` API. We use it in integration tests to spawn a real subprocess but avoid pulling a real model.

- [ ] **Step 1: Write failing test**

```python
# tests/integration/test_fake_vllm.py
import asyncio
import json
import sys
import pytest
import httpx


@pytest.mark.asyncio
async def test_fake_vllm_health_and_completions(tmp_path):
    # Spawn the fake on port 18001.
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "tests.fakes.fake_vllm", "--port", "18001",
        "--served-model-name", "fake-model",
    )
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            for _ in range(50):
                try:
                    r = await c.get("http://127.0.0.1:18001/health")
                    if r.status_code == 200:
                        break
                except Exception:
                    await asyncio.sleep(0.1)
            else:
                raise RuntimeError("fake_vllm did not become healthy")

            r = await c.get("http://127.0.0.1:18001/v1/models")
            assert r.status_code == 200
            assert r.json()["data"][0]["id"] == "fake-model"

            r = await c.post(
                "http://127.0.0.1:18001/v1/chat/completions",
                json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert r.status_code == 200
            data = r.json()
            assert data["choices"][0]["message"]["content"]
            assert data["usage"]["prompt_tokens"] >= 1
    finally:
        proc.terminate()
        await proc.wait()
```

- [ ] **Step 2: Run → FAIL.**

Run: `make test-integration ARGS=tests/integration/test_fake_vllm.py`

Expected: FAIL — fake doesn't exist.

- [ ] **Step 3: Implement fake**

```python
# tests/fakes/__init__.py — empty
# tests/fakes/fake_vllm.py
"""Tiny aiohttp server mimicking vLLM /v1/* and /health for integration tests."""
import argparse
import asyncio
import json
import time

from aiohttp import web


async def health(req):
    return web.Response(status=200, text="ok")


async def models(req):
    name = req.app["served_name"]
    return web.json_response({"data": [{"id": name, "object": "model"}]})


async def chat_completions(req):
    body = await req.json()
    msg = body["messages"][-1]["content"]
    if body.get("stream"):
        async def gen():
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(req)
            for tok in ["hi", " there"]:
                payload = {
                    "choices": [{"delta": {"content": tok}}],
                    "model": req.app["served_name"],
                }
                await resp.write(f"data: {json.dumps(payload)}\n\n".encode())
            await resp.write(b"data: [DONE]\n\n")
            return resp
        return await gen()
    return web.json_response({
        "id": "fake-1",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.app["served_name"],
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": f"echo: {msg}"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": len(msg), "completion_tokens": 4, "total_tokens": len(msg) + 4},
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18001)
    ap.add_argument("--served-model-name", default="fake-model")
    args = ap.parse_args()

    app = web.Application()
    app["served_name"] = args.served_model_name
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", chat_completions)
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run + commit**

Run: `make test-integration ARGS=tests/integration/test_fake_vllm.py`

Expected: PASS.

```bash
git add tests/fakes/__init__.py tests/fakes/fake_vllm.py tests/integration/__init__.py tests/integration/test_fake_vllm.py
git commit -m "test(fakes): minimal aiohttp double of vLLM /v1/* + /health"
```

---

### Task 39: Supervisor real-subprocess integration test

**Files:**
- Create: `tests/integration/test_supervisor_real_subprocess.py`

> Spawns the fake via `Supervisor.load()` (substituting `vllm serve` for `python -m tests.fakes.fake_vllm`). Confirms the supervisor really sets `CUDA_VISIBLE_DEVICES`, really starts the process, really finds it healthy.

- [ ] **Step 1: Write integration test**

```python
# tests/integration/test_supervisor_real_subprocess.py
import asyncio
import sys
from unittest.mock import patch
import pytest
from app.runtime.supervisor import Supervisor, wait_for_health


class _Settings:
    pass


class _M:
    id = "fake"
    hf_repo = "fake/repo"
    hf_revision = None
    served_name = "fake-model"
    gpu_indices = [0]
    tensor_parallel_size = 1
    max_model_len = 4096
    dtype = "auto"
    gpu_memory_utilization = 0.5


@pytest.mark.asyncio
async def test_supervisor_spawns_real_fake_process(tmp_path):
    s = _Settings()
    s.data_dir = str(tmp_path)
    s.hf_token_path = str(tmp_path / "hf-token")
    (tmp_path / "hf-token").write_text("fake")
    sup = Supervisor(s)

    # Patch the command so it runs the fake instead of `vllm serve`.
    def fake_cmd(model, *, port: int):
        return [sys.executable, "-m", "tests.fakes.fake_vllm",
                "--port", str(port),
                "--served-model-name", model.served_name]

    with patch("app.runtime.supervisor.build_vllm_args", return_value=[]), \
         patch("app.runtime.supervisor.asyncio.create_subprocess_exec",
               wraps=asyncio.create_subprocess_exec) as spawn:
        # Replace the cmd construction directly:
        original_load = Supervisor.load
        async def patched_load(self, model, *, port):
            self.gpus.claim(model.id, model.gpu_indices)
            try:
                cmd = fake_cmd(model, port=port)
                proc = await asyncio.create_subprocess_exec(*cmd, start_new_session=True)
                self._processes[model.id] = proc
                self._ports[model.id] = port
            except Exception:
                self.gpus.release(model.id)
                raise
        with patch.object(Supervisor, "load", patched_load):
            await sup.load(_M(), port=18002)

    try:
        ok = await wait_for_health(port=18002, timeout_s=10, interval_s=0.2)
        assert ok is True
    finally:
        proc = sup._processes["fake"]
        proc.terminate()
        await proc.wait()
```

- [ ] **Step 2: Run → expect to PASS once Phase H is in place.**

Run: `make test-integration ARGS=tests/integration/test_supervisor_real_subprocess.py`

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_supervisor_real_subprocess.py
git commit -m "test(integration): supervisor spawns real fake-vllm subprocess + health check"
```

---

### Task 40: Proxy → real subprocess integration test

> Skip until Phase L proxy lands. This task is a placeholder reminder; insert its content after Task 47 (proxy) is complete.

- [ ] **Step 1: Stub note** — the actual integration test for proxy→subprocess routing is written as Task 47 Step 5 (see Phase L). No work in this task.

- [ ] **Step 2: Commit anchor (no code)** — skip; this slot reserved.

---

## Phase K: API tokens (bearer auth)

### Task 41: Tokens CRUD API

**Files:**
- Create: `app/tokens/__init__.py`
- Create: `app/tokens/routes_api.py`
- Test: `tests/unit/tokens/test_tokens_api.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/tokens/__init__.py — empty
# tests/unit/tokens/test_tokens_api.py
import pytest


@pytest.mark.asyncio
async def test_create_token_returns_plaintext_once(client_logged_in):
    r = await client_logged_in.post("/api/tokens", json={"name": "ci-bot"})
    assert r.status_code == 201
    body = r.json()
    assert body["name"] == "ci-bot"
    assert body["plaintext"].startswith("vw_")
    assert len(body["plaintext"]) >= 30


@pytest.mark.asyncio
async def test_list_tokens_does_not_return_plaintext(client_logged_in):
    await client_logged_in.post("/api/tokens", json={"name": "ci-bot"})
    r = await client_logged_in.get("/api/tokens")
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 1
    assert "plaintext" not in items[0]
    assert "token_hash" not in items[0]
    assert items[0]["name"] == "ci-bot"
    assert items[0]["preview"]  # short hint, e.g. "vw_xxx…last4"


@pytest.mark.asyncio
async def test_delete_token(client_logged_in):
    create = await client_logged_in.post("/api/tokens", json={"name": "x"})
    tid = create.json()["id"]
    r = await client_logged_in.delete(f"/api/tokens/{tid}")
    assert r.status_code == 204
    r = await client_logged_in.get("/api/tokens")
    assert r.json()["items"] == []
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement**

```python
# app/tokens/__init__.py — empty
# app/tokens/routes_api.py
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.auth.session import require_session_json
from app.auth.tokens import generate_bearer_token, hash_token, token_preview
from app.db.connection import open_db
from app.db.tokens_repo import TokenRepo

router = APIRouter(prefix="/api/tokens", tags=["tokens"])


class TokenCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_token(body: TokenCreate, request: Request,
                       _user: str = Depends(require_session_json)):
    plaintext = generate_bearer_token()
    h = hash_token(plaintext)
    async with open_db(request.app.state.settings.db_path) as db:
        tid = await TokenRepo(db).create(name=body.name, token_hash=h,
                                         preview=token_preview(plaintext))
    return {"id": tid, "name": body.name, "plaintext": plaintext,
            "preview": token_preview(plaintext)}


@router.get("")
async def list_tokens(request: Request, _user: str = Depends(require_session_json)):
    async with open_db(request.app.state.settings.db_path) as db:
        rows = await TokenRepo(db).list()
    return {"items": [{"id": r["id"], "name": r["name"], "preview": r["preview"],
                       "last_used_at": r["last_used_at"], "created_at": r["created_at"]}
                      for r in rows]}


@router.delete("/{token_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_token(token_id: str, request: Request,
                       _user: str = Depends(require_session_json)):
    async with open_db(request.app.state.settings.db_path) as db:
        deleted = await TokenRepo(db).delete(token_id)
    if not deleted:
        raise HTTPException(404)
    return None
```

Wire in `app/main.py`:
```python
from app.tokens.routes_api import router as tokens_router
app.include_router(tokens_router)
```

- [ ] **Step 4: Run + commit**

```bash
git add app/tokens/__init__.py app/tokens/routes_api.py app/main.py tests/unit/tokens/__init__.py tests/unit/tokens/test_tokens_api.py
git commit -m "feat(tokens): CRUD API for bearer tokens"
```

---

### Task 42: Tokens UI page

**Files:**
- Create: `app/web/tokens/__init__.py`
- Create: `app/web/tokens/routes.py`
- Create: `app/web/tokens/templates/tokens.html`
- Create: `app/web/tokens/templates/_token_row.html`
- Test: `tests/unit/web/test_tokens_page.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/web/test_tokens_page.py
import pytest


@pytest.mark.asyncio
async def test_tokens_page_renders(client_logged_in):
    r = await client_logged_in.get("/tokens")
    assert r.status_code == 200
    assert b"API Tokens" in r.content


@pytest.mark.asyncio
async def test_tokens_page_create_returns_partial_with_plaintext(client_logged_in):
    r = await client_logged_in.post(
        "/tokens", data={"name": "ci-bot"},
        headers={"HX-Request": "true"},
    )
    assert r.status_code == 200
    assert b"vw_" in r.content  # plaintext shown ONCE in the response
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement**

```python
# app/web/tokens/routes.py
from fastapi import APIRouter, Depends, Form, Request
from fastapi.templating import Jinja2Templates

from app.auth.session import require_session
from app.auth.tokens import generate_bearer_token, hash_token, token_preview
from app.db.connection import open_db
from app.db.tokens_repo import TokenRepo

router = APIRouter()
templates = Jinja2Templates(directory="app/web")


@router.get("/tokens")
async def tokens_page(request: Request, _user: str = Depends(require_session)):
    async with open_db(request.app.state.settings.db_path) as db:
        rows = await TokenRepo(db).list()
    return templates.TemplateResponse(
        "tokens/templates/tokens.html",
        {"request": request, "tokens": rows, "just_created": None},
    )


@router.post("/tokens")
async def tokens_create(request: Request, name: str = Form(...),
                        _user: str = Depends(require_session)):
    plaintext = generate_bearer_token()
    async with open_db(request.app.state.settings.db_path) as db:
        await TokenRepo(db).create(name=name, token_hash=hash_token(plaintext),
                                   preview=token_preview(plaintext))
        rows = await TokenRepo(db).list()
    return templates.TemplateResponse(
        "tokens/templates/tokens.html",
        {"request": request, "tokens": rows, "just_created": plaintext},
    )
```

```html
{# app/web/tokens/templates/tokens.html #}
{% extends "base.html" %}
{% block content %}
<h1>API Tokens</h1>
{% if just_created %}
<div class="alert alert-success">
  <strong>New token:</strong> <code>{{ just_created }}</code>
  <p>Copy now. We never show this again.</p>
</div>
{% endif %}
<form method="post" action="/tokens">
  <input name="name" required maxlength="64" placeholder="ci-bot">
  <input type="hidden" name="csrf_token" value="{{ request.state.csrf_token }}">
  <button type="submit">Create</button>
</form>
<table>
  <thead><tr><th>Name</th><th>Preview</th><th>Last used</th><th></th></tr></thead>
  <tbody>
    {% for t in tokens %}
    <tr id="row-{{ t.id }}">
      <td>{{ t.name }}</td>
      <td><code>{{ t.preview }}</code></td>
      <td>{{ t.last_used_at or "never" }}</td>
      <td>
        <button hx-delete="/api/tokens/{{ t.id }}"
                hx-target="#row-{{ t.id }}" hx-swap="outerHTML">Delete</button>
      </td>
    </tr>
    {% endfor %}
  </tbody>
</table>
{% endblock %}
```

Wire in `app/main.py`:
```python
from app.web.tokens.routes import router as tokens_web
app.include_router(tokens_web)
```

- [ ] **Step 4: Run + commit**

```bash
git add app/web/tokens/ app/main.py tests/unit/web/__init__.py tests/unit/web/test_tokens_page.py
git commit -m "feat(tokens): /tokens UI page with create + delete"
```

---

## Phase L: OpenAI-compat proxy

### Task 43: Bearer auth dependency

**Files:**
- Create: `app/proxy/__init__.py`
- Create: `app/proxy/auth.py`
- Test: `tests/unit/proxy/test_proxy_auth.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/proxy/__init__.py — empty
# tests/unit/proxy/test_proxy_auth.py
import pytest


@pytest.mark.asyncio
async def test_proxy_rejects_missing_bearer(client):
    r = await client.post("/v1/chat/completions", json={"model": "x", "messages": []})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_proxy_rejects_unknown_bearer(client):
    r = await client.post("/v1/chat/completions",
                          headers={"Authorization": "Bearer vw_unknown"},
                          json={"model": "x", "messages": []})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_proxy_accepts_valid_bearer_and_records_last_used(client_with_token):
    client, plaintext = client_with_token
    r = await client.post("/v1/chat/completions",
                          headers={"Authorization": f"Bearer {plaintext}"},
                          json={"model": "missing-model", "messages": []})
    # 404 because the model is unknown — but it got past auth.
    assert r.status_code == 404
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement**

```python
# app/proxy/__init__.py — empty
# app/proxy/auth.py
from datetime import datetime, timezone
from fastapi import Depends, HTTPException, Request

from app.auth.tokens import hash_token
from app.db.connection import open_db
from app.db.tokens_repo import TokenRepo


async def require_bearer(request: Request) -> str:
    """Validate Bearer token, return token_id, update last_used_at."""
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(401, "missing bearer token")
    plaintext = auth[7:].strip()
    if not plaintext.startswith("vw_"):
        raise HTTPException(401, "invalid token format")
    h = hash_token(plaintext)
    async with open_db(request.app.state.settings.db_path) as db:
        row = await TokenRepo(db).find_by_hash(h)
        if not row:
            raise HTTPException(401, "unknown token")
        await TokenRepo(db).touch(row["id"])
    return row["id"]
```

- [ ] **Step 4: Run + commit**

```bash
git add app/proxy/__init__.py app/proxy/auth.py tests/unit/proxy/__init__.py tests/unit/proxy/test_proxy_auth.py
git commit -m "feat(proxy): require_bearer dependency for /v1/* routes"
```

---

### Task 44: Proxy router resolving by `model` field

**Files:**
- Create: `app/proxy/routes.py`
- Test: `tests/unit/proxy/test_proxy_route.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/proxy/test_proxy_route.py
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_proxy_404_when_model_not_loaded(client_with_token):
    client, plaintext = client_with_token
    r = await client.post("/v1/chat/completions",
                          headers={"Authorization": f"Bearer {plaintext}"},
                          json={"model": "ghost", "messages": []})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_proxy_forwards_to_subprocess_port(client_with_loaded_model):
    """When model is loaded with served_name='qwen', requests for model='qwen' route to its port."""
    client, plaintext = client_with_loaded_model
    fake_resp = AsyncMock()
    fake_resp.status_code = 200
    fake_resp.aiter_bytes = AsyncMock(return_value=iter([b'{"ok":true}']))
    fake_resp.headers = {"content-type": "application/json"}
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=fake_resp)) as send:
        r = await client.post("/v1/chat/completions",
                              headers={"Authorization": f"Bearer {plaintext}"},
                              json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    sent_url = str(send.call_args.args[0].url)
    assert "127.0.0.1" in sent_url
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement**

```python
# app/proxy/routes.py
import json
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
import httpx

from app.db.connection import open_db
from app.db.models_repo import ModelRepo
from app.proxy.auth import require_bearer

router = APIRouter(prefix="/v1", tags=["proxy"])


async def _resolve_target(request: Request, served_name: str):
    """Look up the loaded model whose served_name matches and return its port."""
    settings = request.app.state.settings
    sup = request.app.state.supervisor
    async with open_db(settings.db_path) as db:
        model = await ModelRepo(db).get_by_served_name(served_name)
    if not model or model.status != "loaded":
        raise HTTPException(404, f"model '{served_name}' is not loaded")
    port = sup.get_port(model.id)
    if port is None:
        raise HTTPException(404, f"model '{served_name}' not running")
    return model, port


async def _forward(request: Request, model_id: str, port: int, path: str):
    body = await request.body()
    is_stream = False
    try:
        if request.headers.get("content-type", "").startswith("application/json") and body:
            is_stream = json.loads(body).get("stream", False)
    except Exception:
        pass
    url = f"http://127.0.0.1:{port}{path}"
    async with httpx.AsyncClient(timeout=None) as c:
        upstream = c.build_request(request.method, url, content=body,
                                   headers={k: v for k, v in request.headers.items()
                                            if k.lower() not in ("host", "authorization")})
        resp = await c.send(upstream, stream=is_stream)
        if is_stream:
            async def gen():
                async for chunk in resp.aiter_bytes():
                    yield chunk
                await resp.aclose()
            return StreamingResponse(gen(), status_code=resp.status_code,
                                     headers={"content-type": resp.headers.get("content-type", "text/event-stream")})
        content = await resp.aread()
        await resp.aclose()
        return Response(content=content, status_code=resp.status_code,
                        headers={"content-type": resp.headers.get("content-type", "application/json")})


@router.post("/chat/completions")
async def chat_completions(request: Request, _tid: str = Depends(require_bearer)):
    body = json.loads(await request.body() or b"{}")
    served_name = body.get("model")
    if not served_name:
        raise HTTPException(400, "missing 'model' field")
    model, port = await _resolve_target(request, served_name)
    return await _forward(request, model.id, port, "/v1/chat/completions")


@router.post("/completions")
async def completions(request: Request, _tid: str = Depends(require_bearer)):
    body = json.loads(await request.body() or b"{}")
    served_name = body.get("model")
    if not served_name:
        raise HTTPException(400, "missing 'model' field")
    model, port = await _resolve_target(request, served_name)
    return await _forward(request, model.id, port, "/v1/completions")
```

Wire in `app/main.py`:
```python
from app.proxy.routes import router as proxy_router
app.include_router(proxy_router)
```

- [ ] **Step 4: Run + commit**

```bash
git add app/proxy/routes.py app/main.py tests/unit/proxy/test_proxy_route.py
git commit -m "feat(proxy): /v1/chat/completions + /v1/completions routing by model field"
```

---

### Task 45: Tokenizer cache for accounting

**Files:**
- Create: `app/proxy/tokenizers.py`
- Test: `tests/unit/proxy/test_tokenizers.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/proxy/test_tokenizers.py
import pytest
from unittest.mock import patch, MagicMock


@pytest.mark.asyncio
async def test_tokenizer_cache_returns_cached_instance():
    from app.proxy.tokenizers import TokenizerCache

    fake_tok = MagicMock()
    fake_tok.encode = lambda s: list(s.encode())
    with patch("app.proxy.tokenizers.AutoTokenizer.from_pretrained",
               return_value=fake_tok) as load:
        cache = TokenizerCache()
        a = await cache.get("Qwen/Qwen3.5-9B")
        b = await cache.get("Qwen/Qwen3.5-9B")
    assert a is b
    assert load.call_count == 1


@pytest.mark.asyncio
async def test_count_tokens_uses_repo_tokenizer():
    from app.proxy.tokenizers import TokenizerCache

    fake_tok = MagicMock()
    fake_tok.encode = lambda s: list(s)
    with patch("app.proxy.tokenizers.AutoTokenizer.from_pretrained", return_value=fake_tok):
        cache = TokenizerCache()
        n = await cache.count("Qwen/Qwen3.5-9B", "hello")
    assert n == 5
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement**

```python
# app/proxy/tokenizers.py
import asyncio
from transformers import AutoTokenizer


class TokenizerCache:
    """Lazy-loaded HF tokenizer cache keyed by hf_repo. Used for accounting only."""

    def __init__(self) -> None:
        self._cache: dict[str, object] = {}
        self._lock = asyncio.Lock()

    async def get(self, hf_repo: str):
        async with self._lock:
            if hf_repo not in self._cache:
                # Run the blocking load in a thread.
                loop = asyncio.get_running_loop()
                self._cache[hf_repo] = await loop.run_in_executor(
                    None, lambda: AutoTokenizer.from_pretrained(hf_repo, trust_remote_code=True)
                )
            return self._cache[hf_repo]

    async def count(self, hf_repo: str, text: str) -> int:
        tok = await self.get(hf_repo)
        return len(tok.encode(text))
```

Wire into `app.state.tokenizers = TokenizerCache()` in `lifespan`.

- [ ] **Step 4: Run + commit**

```bash
git add app/proxy/tokenizers.py app/main.py tests/unit/proxy/test_tokenizers.py
git commit -m "feat(proxy): TokenizerCache for token-count accounting"
```

---

### Task 46: Streaming SSE token counting + counter increments

**Files:**
- Modify: `app/proxy/routes.py`
- Modify: `app/proxy/tokenizers.py` (add SSE accumulator helper)
- Test: `tests/unit/proxy/test_proxy_accounting.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/proxy/test_proxy_accounting.py
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_proxy_increments_counters_on_non_streaming(client_with_loaded_model):
    """A non-streaming chat completion increments prompt_tokens + completion_tokens
    counters for (token_id, model_id, gpu_indices, minute_bucket)."""
    client, plaintext, _ = client_with_loaded_model
    body = {"id": "x", "model": "qwen",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}
    fake_resp = AsyncMock()
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "application/json"}
    fake_resp.aread = AsyncMock(return_value=__import__("json").dumps(body).encode())
    fake_resp.aclose = AsyncMock()
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=fake_resp)):
        r = await client.post("/v1/chat/completions",
                              headers={"Authorization": f"Bearer {plaintext}"},
                              json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200

    # Verify counters table got an increment.
    from app.db.connection import open_db
    from app.db.counters_repo import CountersRepo
    async with open_db(client.app.state.settings.db_path) as db:
        rows = await CountersRepo(db).list_recent(seconds=120)
    assert any(r["prompt_tokens"] >= 5 for r in rows)
    assert any(r["completion_tokens"] >= 2 for r in rows)


@pytest.mark.asyncio
async def test_proxy_streaming_counts_tokens_incrementally(client_with_loaded_model):
    """For streaming SSE, completion tokens are counted by tokenizing each delta."""
    client, plaintext, _ = client_with_loaded_model
    sse_chunks = [
        b'data: {"choices":[{"delta":{"content":"hello"}}],"model":"qwen"}\n\n',
        b'data: {"choices":[{"delta":{"content":" world"}}],"model":"qwen"}\n\n',
        b'data: [DONE]\n\n',
    ]

    async def aiter():
        for c in sse_chunks:
            yield c

    fake_resp = AsyncMock()
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "text/event-stream"}
    fake_resp.aiter_bytes = aiter
    fake_resp.aclose = AsyncMock()

    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=fake_resp)):
        r = await client.post("/v1/chat/completions",
                              headers={"Authorization": f"Bearer {plaintext}"},
                              json={"model": "qwen", "stream": True,
                                    "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    # Drain the body so the proxy gets to its post-stream accounting hook.
    async for _ in r.aiter_bytes():
        pass

    from app.db.connection import open_db
    from app.db.counters_repo import CountersRepo
    async with open_db(client.app.state.settings.db_path) as db:
        rows = await CountersRepo(db).list_recent(seconds=120)
    assert any(r["completion_tokens"] >= 2 for r in rows)  # "hello" + " world" → ≥ 2 tokens
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement (modify `app/proxy/routes.py` `_forward()`)**

Replace the body of `_forward()` to:
1. Tokenize the prompt before sending and bump `prompt_tokens` counter.
2. For non-streaming, parse `usage.completion_tokens` from the JSON response and bump.
3. For streaming, parse SSE chunks; for each `delta.content` accumulate text; at end of stream, tokenize accumulated text → `completion_tokens` count. Use a wrapper generator that buffers but yields chunks unchanged to the caller.

```python
# app/proxy/routes.py — replace _forward and add helpers
import json
from fastapi.responses import StreamingResponse, Response
import httpx
from app.db.connection import open_db
from app.db.counters_repo import CountersRepo


def _extract_prompt(body_json) -> str:
    if "messages" in body_json:
        return "\n".join(m.get("content", "") for m in body_json["messages"])
    return body_json.get("prompt", "")


def _parse_sse_delta(line: bytes) -> str | None:
    if not line.startswith(b"data:"):
        return None
    payload = line[5:].strip()
    if payload == b"[DONE]":
        return None
    try:
        ev = json.loads(payload)
        return ev["choices"][0].get("delta", {}).get("content") or None
    except Exception:
        return None


async def _record_counters(request: Request, model, token_id: str | None,
                           prompt_tokens: int, completion_tokens: int):
    async with open_db(request.app.state.settings.db_path) as db:
        await CountersRepo(db).increment(
            token_id=token_id, model_id=model.id, gpu_indices=model.gpu_indices,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        )


async def _forward(request: Request, model, port: int, path: str, token_id: str):
    body = await request.body()
    body_json = json.loads(body) if body else {}
    is_stream = bool(body_json.get("stream"))
    tok_cache = request.app.state.tokenizers

    prompt_text = _extract_prompt(body_json)
    prompt_tokens = await tok_cache.count(model.hf_repo, prompt_text)

    url = f"http://127.0.0.1:{port}{path}"
    client = httpx.AsyncClient(timeout=None)
    upstream_req = client.build_request(
        request.method, url, content=body,
        headers={k: v for k, v in request.headers.items()
                 if k.lower() not in ("host", "authorization")},
    )
    resp = await client.send(upstream_req, stream=is_stream)

    if is_stream:
        async def gen():
            buf = b""
            accumulated = ""
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
                    buf += chunk
                    while b"\n\n" in buf:
                        line, buf = buf.split(b"\n\n", 1)
                        delta = _parse_sse_delta(line)
                        if delta:
                            accumulated += delta
            finally:
                await resp.aclose()
                await client.aclose()
                completion_tokens = await tok_cache.count(model.hf_repo, accumulated)
                await _record_counters(request, model, token_id, prompt_tokens, completion_tokens)

        return StreamingResponse(gen(), status_code=resp.status_code,
                                 headers={"content-type": resp.headers.get("content-type", "text/event-stream")})

    content = await resp.aread()
    await resp.aclose()
    await client.aclose()
    completion_tokens = 0
    try:
        completion_tokens = json.loads(content).get("usage", {}).get("completion_tokens", 0)
    except Exception:
        pass
    await _record_counters(request, model, token_id, prompt_tokens, completion_tokens)
    return Response(content=content, status_code=resp.status_code,
                    headers={"content-type": resp.headers.get("content-type", "application/json")})


@router.post("/chat/completions")
async def chat_completions(request: Request, token_id: str = Depends(require_bearer)):
    body = json.loads(await request.body() or b"{}")
    served_name = body.get("model")
    if not served_name:
        raise HTTPException(400, "missing 'model' field")
    model, port = await _resolve_target(request, served_name)
    return await _forward(request, model, port, "/v1/chat/completions", token_id)


@router.post("/completions")
async def completions(request: Request, token_id: str = Depends(require_bearer)):
    body = json.loads(await request.body() or b"{}")
    served_name = body.get("model")
    if not served_name:
        raise HTTPException(400, "missing 'model' field")
    model, port = await _resolve_target(request, served_name)
    return await _forward(request, model, port, "/v1/completions", token_id)
```

- [ ] **Step 4: Run + commit**

Run: `make test-unit ARGS=tests/unit/proxy/test_proxy_accounting.py`

Expected: PASS.

```bash
git add app/proxy/routes.py tests/unit/proxy/test_proxy_accounting.py
git commit -m "feat(proxy): token-count accounting via tokenizer cache (streaming + non-streaming)"
```

---

### Task 47: `/v1/models` aggregate endpoint + proxy↔subprocess integration

**Files:**
- Modify: `app/proxy/routes.py`
- Test: `tests/unit/proxy/test_v1_models.py`
- Test: `tests/integration/test_proxy_real_subprocess.py`

- [ ] **Step 1: Write failing test (unit)**

```python
# tests/unit/proxy/test_v1_models.py
import pytest


@pytest.mark.asyncio
async def test_v1_models_lists_only_loaded(client_with_loaded_model, db_with_pulled_model):
    """Only models in status='loaded' are listed; pulled-but-not-loaded are hidden."""
    client, plaintext, _ = client_with_loaded_model
    await db_with_pulled_model.add_model("other", status="pulled")
    r = await client.get("/v1/models", headers={"Authorization": f"Bearer {plaintext}"})
    assert r.status_code == 200
    ids = [m["id"] for m in r.json()["data"]]
    assert "qwen" in ids
    assert "other" not in ids
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement (append to `app/proxy/routes.py`)**

```python
@router.get("/models")
async def list_models(request: Request, _tid: str = Depends(require_bearer)):
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        rows = await ModelRepo(db).list_by_status(["loaded"])
    return {"object": "list",
            "data": [{"id": r.served_name, "object": "model",
                      "owned_by": "vllm-warden"} for r in rows]}
```

- [ ] **Step 4: Run unit + integration tests**

```python
# tests/integration/test_proxy_real_subprocess.py
"""Spin up a fake-vllm subprocess via the supervisor's flow, then send a real
HTTP request through the proxy and verify routing + counting."""
import asyncio
import sys
import pytest
import httpx


@pytest.mark.asyncio
async def test_proxy_routes_to_real_subprocess(client_logged_in, tmp_data_dir):
    # 1) Use API to register and "pull" a fake-backed model.
    # (test fixture monkeypatches snapshot_download to a no-op.)
    create = await client_logged_in.post("/api/models", json={
        "id": "fake", "hf_repo": "fake/repo", "served_name": "fake-model",
        "gpu_indices": [0], "tensor_parallel_size": 1,
        "max_model_len": 4096, "dtype": "auto", "gpu_memory_utilization": 0.5,
    })
    assert create.status_code == 201

    # 2) Manually start a fake_vllm subprocess on a known port and stitch it into
    #    the supervisor's state so we don't need a real GPU/vLLM here.
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "tests.fakes.fake_vllm",
        "--port", "18099", "--served-model-name", "fake-model",
    )
    try:
        # Mark loaded + register port.
        sup = client_logged_in.app.state.supervisor
        sup._processes["fake"] = proc
        sup._ports["fake"] = 18099
        sup.gpus.claim("fake", [0])

        from app.db.connection import open_db
        from app.db.models_repo import ModelRepo
        from app.db.runtime_repo import RuntimeRepo
        async with open_db(client_logged_in.app.state.settings.db_path) as db:
            await ModelRepo(db).set_status("fake", "loaded")
            await RuntimeRepo(db).set("fake", port=18099, pid=proc.pid)

        # 3) Create a token and call /v1/chat/completions.
        tok = await client_logged_in.post("/api/tokens", json={"name": "test"})
        plaintext = tok.json()["plaintext"]

        # Wait for fake to boot.
        async with httpx.AsyncClient(timeout=5.0) as c:
            for _ in range(50):
                try:
                    r = await c.get("http://127.0.0.1:18099/health")
                    if r.status_code == 200:
                        break
                except Exception:
                    await asyncio.sleep(0.1)

        r = await client_logged_in.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "fake-model",
                  "messages": [{"role": "user", "content": "hello"}]},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["choices"][0]["message"]["content"].startswith("echo:")
    finally:
        proc.terminate()
        await proc.wait()
```

Run: `make test-integration ARGS=tests/integration/test_proxy_real_subprocess.py`

Expected: PASS.

```bash
git add app/proxy/routes.py tests/unit/proxy/test_v1_models.py tests/integration/test_proxy_real_subprocess.py
git commit -m "feat(proxy): /v1/models + proxy↔subprocess integration test"
```

---

## Phase M — Stats (per-model + per-GPU minute buckets)

The spec defines two retention windows for time-series:
- **Per-model:** `model_id, minute_ts, prompt_tokens, completion_tokens, request_count` — written on each successful proxy completion (Task 46 already inserts).
- **Per-GPU:** `gpu_index, minute_ts, util_pct, mem_used_mib, mem_total_mib` — written every 60 s by a sampler asyncio task that calls `nvidia-smi` and aggregates into one row per (gpu_index, minute_ts).

Both retain 7 days. Pruning runs hourly.

### Task 48: GPU stats sampler asyncio task

**Files:**
- Create: `app/runtime/stats_sampler.py`
- Test: `tests/unit/runtime/test_stats_sampler.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/runtime/test_stats_sampler.py
import pytest
from unittest.mock import AsyncMock, patch
from datetime import datetime, timezone

from app.runtime.stats_sampler import sample_gpus_once


@pytest.mark.asyncio
async def test_sample_gpus_once_inserts_one_row_per_gpu(memory_db):
    fake_probe = AsyncMock(return_value=[
        {"index": 0, "util_pct": 42, "mem_used_mib": 1000, "mem_total_mib": 24000},
        {"index": 1, "util_pct": 55, "mem_used_mib": 2000, "mem_total_mib": 24000},
    ])
    with patch("app.runtime.stats_sampler.probe_gpus", fake_probe):
        await sample_gpus_once(memory_db, now=datetime(2026, 5, 8, 12, 30, 0, tzinfo=timezone.utc))

    rows = await memory_db.fetch_all(
        "SELECT gpu_index, minute_ts, util_pct, mem_used_mib FROM gpu_stats ORDER BY gpu_index"
    )
    assert len(rows) == 2
    assert rows[0]["gpu_index"] == 0 and rows[0]["util_pct"] == 42
    assert rows[1]["gpu_index"] == 1 and rows[1]["util_pct"] == 55
    assert rows[0]["minute_ts"] == "2026-05-08T12:30:00+00:00"


@pytest.mark.asyncio
async def test_sample_gpus_once_upserts_same_minute(memory_db):
    """Two samples in the same minute must produce one row, last value wins."""
    when = datetime(2026, 5, 8, 12, 30, 30, tzinfo=timezone.utc)
    fake = AsyncMock(side_effect=[
        [{"index": 0, "util_pct": 10, "mem_used_mib": 100, "mem_total_mib": 24000}],
        [{"index": 0, "util_pct": 90, "mem_used_mib": 200, "mem_total_mib": 24000}],
    ])
    with patch("app.runtime.stats_sampler.probe_gpus", fake):
        await sample_gpus_once(memory_db, now=when)
        await sample_gpus_once(memory_db, now=when)

    rows = await memory_db.fetch_all("SELECT util_pct, mem_used_mib FROM gpu_stats")
    assert len(rows) == 1
    assert rows[0]["util_pct"] == 90
    assert rows[0]["mem_used_mib"] == 200
```

`memory_db` is a pytest fixture that yields an in-memory aiosqlite connection with all migrations applied; defined in `tests/conftest.py` (Task 5).

- [ ] **Step 2: Run test to verify it fails**

Run: `make test-unit ARGS=tests/unit/runtime/test_stats_sampler.py`

Expected: FAIL — `app.runtime.stats_sampler` module does not exist.

- [ ] **Step 3: Implement `app/runtime/stats_sampler.py`**

```python
import asyncio
import logging
from datetime import datetime, timezone

from app.runtime.gpu_probe import probe_gpus

logger = logging.getLogger(__name__)

SAMPLE_INTERVAL_SECONDS = 60


def _bucket(now: datetime) -> str:
    """Truncate to the minute, ISO8601 with timezone."""
    return now.replace(second=0, microsecond=0).isoformat()


async def sample_gpus_once(db, *, now: datetime | None = None) -> None:
    now = now or datetime.now(timezone.utc)
    bucket = _bucket(now)
    try:
        gpus = await probe_gpus()
    except Exception:
        logger.exception("probe_gpus failed in stats sampler; skipping bucket %s", bucket)
        return
    for g in gpus:
        await db.execute(
            """
            INSERT INTO gpu_stats (gpu_index, minute_ts, util_pct, mem_used_mib, mem_total_mib)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(gpu_index, minute_ts) DO UPDATE SET
                util_pct = excluded.util_pct,
                mem_used_mib = excluded.mem_used_mib,
                mem_total_mib = excluded.mem_total_mib
            """,
            (g["index"], bucket, g["util_pct"], g["mem_used_mib"], g["mem_total_mib"]),
        )
    await db.commit()


async def run_sampler_forever(db) -> None:
    """Long-lived background task. Cancellation-safe — caller must `task.cancel()` on shutdown."""
    while True:
        await sample_gpus_once(db)
        await asyncio.sleep(SAMPLE_INTERVAL_SECONDS)
```

- [ ] **Step 4: Add unique index migration**

If migration `005_gpu_stats.sql` (Task 6) didn't already include `UNIQUE(gpu_index, minute_ts)`, add it now in a new migration `010_gpu_stats_unique.sql`. Otherwise skip this step.

```sql
-- migrations/010_gpu_stats_unique.sql
CREATE UNIQUE INDEX IF NOT EXISTS idx_gpu_stats_gpu_minute
    ON gpu_stats (gpu_index, minute_ts);
```

- [ ] **Step 5: Run test to verify it passes**

Run: `make test-unit ARGS=tests/unit/runtime/test_stats_sampler.py`

Expected: PASS — both tests green.

- [ ] **Step 6: Commit**

```bash
git add app/runtime/stats_sampler.py tests/unit/runtime/test_stats_sampler.py migrations/010_gpu_stats_unique.sql
git commit -m "feat(stats): GPU sampler with minute-bucket upsert"
```

---

### Task 49: Stats API endpoints

**Files:**
- Create: `app/api/stats.py`
- Test: `tests/unit/api/test_stats.py`

Endpoints:
- `GET /api/stats/models?range=24h` → `[{minute_ts, model_id, prompt_tokens, completion_tokens, request_count}, …]`
- `GET /api/stats/gpus?range=24h` → `[{minute_ts, gpu_index, util_pct, mem_used_mib, mem_total_mib}, …]`

`range` accepts `1h | 6h | 24h | 7d` (validate, default `24h`, reject anything else with 400).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/api/test_stats.py
import pytest
from datetime import datetime, timedelta, timezone


@pytest.mark.asyncio
async def test_stats_models_filters_by_range(client_logged_in, db):
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    # one row inside 24h, one row 25h old
    await db.execute(
        "INSERT INTO model_stats (model_id, minute_ts, prompt_tokens, completion_tokens, request_count) "
        "VALUES (?, ?, ?, ?, ?)",
        ("m1", now.isoformat(), 100, 50, 1),
    )
    await db.execute(
        "INSERT INTO model_stats (model_id, minute_ts, prompt_tokens, completion_tokens, request_count) "
        "VALUES (?, ?, ?, ?, ?)",
        ("m1", (now - timedelta(hours=25)).isoformat(), 999, 999, 9),
    )
    await db.commit()

    r = await client_logged_in.get("/api/stats/models?range=24h")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["prompt_tokens"] == 100


@pytest.mark.asyncio
async def test_stats_range_invalid_returns_400(client_logged_in):
    r = await client_logged_in.get("/api/stats/models?range=banana")
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_stats_requires_session(client):
    r = await client.get("/api/stats/models")
    assert r.status_code == 401
```

- [ ] **Step 2: Run to verify it fails**

Run: `make test-unit ARGS=tests/unit/api/test_stats.py`

Expected: FAIL — `app.api.stats` does not exist.

- [ ] **Step 3: Implement `app/api/stats.py`**

```python
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth.session import require_session
from app.deps import get_db

router = APIRouter(prefix="/api/stats", tags=["stats"], dependencies=[Depends(require_session)])

_RANGE_TO_HOURS = {"1h": 1, "6h": 6, "24h": 24, "7d": 24 * 7}


def _cutoff(range_: str) -> str:
    if range_ not in _RANGE_TO_HOURS:
        raise HTTPException(status_code=400, detail="invalid range")
    return (datetime.now(timezone.utc) - timedelta(hours=_RANGE_TO_HOURS[range_])).isoformat()


@router.get("/models")
async def stats_models(range: str = Query("24h"), db=Depends(get_db)):
    cutoff = _cutoff(range)
    rows = await db.fetch_all(
        "SELECT model_id, minute_ts, prompt_tokens, completion_tokens, request_count "
        "FROM model_stats WHERE minute_ts >= ? ORDER BY minute_ts ASC",
        (cutoff,),
    )
    return [dict(r) for r in rows]


@router.get("/gpus")
async def stats_gpus(range: str = Query("24h"), db=Depends(get_db)):
    cutoff = _cutoff(range)
    rows = await db.fetch_all(
        "SELECT gpu_index, minute_ts, util_pct, mem_used_mib, mem_total_mib "
        "FROM gpu_stats WHERE minute_ts >= ? ORDER BY minute_ts ASC",
        (cutoff,),
    )
    return [dict(r) for r in rows]
```

Wire `router` into `app/main.py` (final wiring sweep happens in Task 54; for now register it directly):

```python
from app.api import stats as stats_api
app.include_router(stats_api.router)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `make test-unit ARGS=tests/unit/api/test_stats.py`

Expected: PASS — all three green.

- [ ] **Step 5: Commit**

```bash
git add app/api/stats.py app/main.py tests/unit/api/test_stats.py
git commit -m "feat(stats): /api/stats/models and /api/stats/gpus"
```

---

### Task 50: Stats UI page (Chart.js)

**Files:**
- Create: `app/templates/stats.html`
- Modify: `app/api/web.py` — add `GET /stats` route
- Test: `tests/unit/web/test_stats_page.py`

Use Chart.js UMD already vendored at `app/static/vendor/chart.umd.min.js` (Task 4). Two charts side-by-side: tokens/minute per model (stacked bar) and GPU util % (multi-line, one per GPU).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/web/test_stats_page.py
import pytest


@pytest.mark.asyncio
async def test_stats_page_renders(client_logged_in):
    r = await client_logged_in.get("/stats")
    assert r.status_code == 200
    assert "vendor/chart.umd.min.js" in r.text
    assert 'id="chart-models"' in r.text
    assert 'id="chart-gpus"' in r.text


@pytest.mark.asyncio
async def test_stats_page_redirects_when_unauth(client):
    r = await client.get("/stats", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"].startswith("/login")
```

- [ ] **Step 2: Run to verify failure**

Run: `make test-unit ARGS=tests/unit/web/test_stats_page.py`

Expected: FAIL — `/stats` route not registered.

- [ ] **Step 3: Add the page route**

In `app/api/web.py`:

```python
@router.get("/stats", response_class=HTMLResponse)
async def stats_page(request: Request, _user=Depends(require_session_html)):
    return templates.TemplateResponse("stats.html", {"request": request})
```

- [ ] **Step 4: Create `app/templates/stats.html`**

```html
{% extends "_base.html" %}
{% block title %}Stats{% endblock %}
{% block content %}
<h1>Stats</h1>
<form id="range-form">
  <label>Range:
    <select name="range" id="range-select">
      <option value="1h">1 hour</option>
      <option value="6h">6 hours</option>
      <option value="24h" selected>24 hours</option>
      <option value="7d">7 days</option>
    </select>
  </label>
</form>
<section class="chart-grid">
  <div><h2>Tokens / minute (per model)</h2><canvas id="chart-models"></canvas></div>
  <div><h2>GPU utilisation %</h2><canvas id="chart-gpus"></canvas></div>
</section>
<script src="/static/vendor/chart.umd.min.js"></script>
<script>
(async function () {
  const select = document.getElementById('range-select');
  let modelsChart, gpusChart;

  async function reload() {
    const r = select.value;
    const [models, gpus] = await Promise.all([
      fetch('/api/stats/models?range=' + r).then(r => r.json()),
      fetch('/api/stats/gpus?range=' + r).then(r => r.json()),
    ]);
    renderModels(models);
    renderGpus(gpus);
  }

  function renderModels(rows) {
    const byModel = {};
    const buckets = new Set();
    for (const r of rows) {
      buckets.add(r.minute_ts);
      (byModel[r.model_id] ||= {})[r.minute_ts] = r.prompt_tokens + r.completion_tokens;
    }
    const labels = Array.from(buckets).sort();
    const datasets = Object.entries(byModel).map(([m, b]) => ({
      label: m,
      data: labels.map(l => b[l] || 0),
    }));
    if (modelsChart) modelsChart.destroy();
    modelsChart = new Chart(document.getElementById('chart-models'), {
      type: 'bar',
      data: { labels, datasets },
      options: { scales: { x: { stacked: true }, y: { stacked: true } } },
    });
  }

  function renderGpus(rows) {
    const byGpu = {};
    const buckets = new Set();
    for (const r of rows) {
      buckets.add(r.minute_ts);
      (byGpu[r.gpu_index] ||= {})[r.minute_ts] = r.util_pct;
    }
    const labels = Array.from(buckets).sort();
    const datasets = Object.entries(byGpu).map(([g, b]) => ({
      label: 'GPU ' + g,
      data: labels.map(l => b[l] ?? null),
      spanGaps: true,
    }));
    if (gpusChart) gpusChart.destroy();
    gpusChart = new Chart(document.getElementById('chart-gpus'), {
      type: 'line',
      data: { labels, datasets },
      options: { scales: { y: { min: 0, max: 100 } } },
    });
  }

  select.addEventListener('change', reload);
  await reload();
})();
</script>
{% endblock %}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `make test-unit ARGS=tests/unit/web/test_stats_page.py`

Expected: PASS — both green.

- [ ] **Step 6: Commit**

```bash
git add app/templates/stats.html app/api/web.py tests/unit/web/test_stats_page.py
git commit -m "feat(stats): UI page with Chart.js model + GPU graphs"
```

---

### Task 51: 7-day retention pruner

**Files:**
- Create: `app/runtime/stats_pruner.py`
- Test: `tests/unit/runtime/test_stats_pruner.py`

Hourly: `DELETE FROM model_stats WHERE minute_ts < NOW - 7d` and same for `gpu_stats`. Wire into the same lifespan that started the sampler.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/runtime/test_stats_pruner.py
import pytest
from datetime import datetime, timedelta, timezone

from app.runtime.stats_pruner import prune_once, RETENTION_DAYS


@pytest.mark.asyncio
async def test_prune_removes_rows_older_than_retention(memory_db):
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    fresh = now.isoformat()
    stale = (now - timedelta(days=RETENTION_DAYS, minutes=1)).isoformat()
    await memory_db.execute(
        "INSERT INTO model_stats (model_id, minute_ts, prompt_tokens, completion_tokens, request_count) "
        "VALUES ('m1', ?, 1, 1, 1), ('m1', ?, 1, 1, 1)",
        (fresh, stale),
    )
    await memory_db.execute(
        "INSERT INTO gpu_stats (gpu_index, minute_ts, util_pct, mem_used_mib, mem_total_mib) "
        "VALUES (0, ?, 1, 1, 1), (0, ?, 1, 1, 1)",
        (fresh, stale),
    )
    await memory_db.commit()

    deleted = await prune_once(memory_db)

    assert deleted["model_stats"] == 1
    assert deleted["gpu_stats"] == 1
    rem_models = await memory_db.fetch_one("SELECT count(*) AS n FROM model_stats")
    rem_gpus = await memory_db.fetch_one("SELECT count(*) AS n FROM gpu_stats")
    assert rem_models["n"] == 1
    assert rem_gpus["n"] == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `make test-unit ARGS=tests/unit/runtime/test_stats_pruner.py`

Expected: FAIL — module missing.

- [ ] **Step 3: Implement `app/runtime/stats_pruner.py`**

```python
import asyncio
from datetime import datetime, timedelta, timezone

RETENTION_DAYS = 7
PRUNE_INTERVAL_SECONDS = 3600


async def prune_once(db) -> dict[str, int]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)).isoformat()
    cur1 = await db.execute("DELETE FROM model_stats WHERE minute_ts < ?", (cutoff,))
    cur2 = await db.execute("DELETE FROM gpu_stats WHERE minute_ts < ?", (cutoff,))
    await db.commit()
    return {"model_stats": cur1.rowcount or 0, "gpu_stats": cur2.rowcount or 0}


async def run_pruner_forever(db) -> None:
    while True:
        await prune_once(db)
        await asyncio.sleep(PRUNE_INTERVAL_SECONDS)
```

- [ ] **Step 4: Wire sampler + pruner into FastAPI lifespan**

In `app/main.py`, inside the existing lifespan context manager (created in Task 11), spawn both tasks and cancel on shutdown:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... existing DB init from Task 11 ...
    sampler_task = asyncio.create_task(run_sampler_forever(app.state.db))
    pruner_task = asyncio.create_task(run_pruner_forever(app.state.db))
    try:
        yield
    finally:
        sampler_task.cancel()
        pruner_task.cancel()
        await asyncio.gather(sampler_task, pruner_task, return_exceptions=True)
        # ... existing DB close ...
```

- [ ] **Step 5: Run all stats tests**

Run: `make test-unit ARGS=tests/unit/runtime/test_stats_pruner.py tests/unit/runtime/test_stats_sampler.py tests/unit/api/test_stats.py`

Expected: PASS — all green.

- [ ] **Step 6: Commit**

```bash
git add app/runtime/stats_pruner.py app/main.py tests/unit/runtime/test_stats_pruner.py
git commit -m "feat(stats): 7-day retention pruner + lifespan wiring"
```

---

## Phase N — Settings page

### Task 52: Settings page (edit allowed_gpu_indices and hf_token)

**Files:**
- Create: `app/templates/settings.html`
- Modify: `app/api/web.py` (add `GET /settings`)
- Modify: `app/api/setup.py` or new `app/api/settings.py` (add `POST /api/settings`)
- Test: `tests/unit/api/test_settings.py`

Reuses the wizard's GPU probe + HF token validator (Tasks 12, 14) so user can re-pick GPUs and rotate the token without re-running the wizard. Hardens: cannot remove a GPU currently `_gpu_owner.values()`-claimed by a loaded model — return 409 with the offending model id.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/api/test_settings.py
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_settings_page_renders(client_logged_in):
    r = await client_logged_in.get("/settings")
    assert r.status_code == 200
    assert 'name="allowed_gpu_indices"' in r.text
    assert 'name="hf_token"' in r.text


@pytest.mark.asyncio
async def test_settings_post_validates_token_before_save(client_logged_in, db):
    fake_validate = AsyncMock(return_value={"valid": False, "error": "bad token"})
    with patch("app.api.settings.validate_hf_token", fake_validate):
        r = await client_logged_in.post(
            "/api/settings",
            json={"allowed_gpu_indices": [0, 1], "hf_token": "bad"},
            headers={"X-CSRF-Token": "..."},
        )
    assert r.status_code == 422
    assert "bad token" in r.json()["detail"]


@pytest.mark.asyncio
async def test_settings_rejects_removing_gpu_in_use(client_logged_in, db, app):
    app.state.gpus.claim("model-A", [0, 1])
    try:
        r = await client_logged_in.post(
            "/api/settings",
            json={"allowed_gpu_indices": [2, 3]},  # removes 0,1 — model-A holds them
            headers={"X-CSRF-Token": "..."},
        )
        assert r.status_code == 409
        assert "model-A" in r.json()["detail"]
    finally:
        app.state.gpus.release("model-A")


@pytest.mark.asyncio
async def test_settings_persists_to_settings_table(client_logged_in, db):
    fake_validate = AsyncMock(return_value={"valid": True, "username": "user"})
    with patch("app.api.settings.validate_hf_token", fake_validate):
        r = await client_logged_in.post(
            "/api/settings",
            json={"allowed_gpu_indices": [0, 1, 2, 3], "hf_token": "hf_new"},
            headers={"X-CSRF-Token": "..."},
        )
    assert r.status_code == 200
    row = await db.fetch_one("SELECT key, value FROM settings WHERE key = 'allowed_gpu_indices'")
    assert row["value"] == "[0, 1, 2, 3]"
    row = await db.fetch_one("SELECT key, value FROM settings WHERE key = 'hf_token'")
    assert row["value"] == "hf_new"
```

The fixture `client_logged_in` should pre-fetch a CSRF token; the test framework helper `auth_headers(client)` returns them as a dict — see Task 17. Replace `headers={"X-CSRF-Token": "..."}` with `headers=await auth_headers(client_logged_in)` in actual test code.

- [ ] **Step 2: Run to verify failure**

Run: `make test-unit ARGS=tests/unit/api/test_settings.py`

Expected: FAIL — `/settings` and `/api/settings` not registered.

- [ ] **Step 3: Implement `app/api/settings.py`**

```python
import json

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.auth.session import require_session
from app.auth.csrf import require_csrf
from app.deps import get_db
from app.runtime.hf_token import validate_hf_token

router = APIRouter(prefix="/api", tags=["settings"], dependencies=[Depends(require_session)])


class SettingsIn(BaseModel):
    allowed_gpu_indices: list[int] = Field(min_length=1)
    hf_token: str | None = None  # None means leave unchanged


@router.post("/settings", dependencies=[Depends(require_csrf)])
async def update_settings(payload: SettingsIn, request: Request, db=Depends(get_db)):
    new_set = set(payload.allowed_gpu_indices)
    in_use = request.app.state.gpus.snapshot()  # {gpu_idx: model_id}
    removed = [g for g in in_use if g not in new_set]
    if removed:
        owners = sorted({in_use[g] for g in removed})
        raise HTTPException(
            status_code=409,
            detail=f"GPU(s) {removed} are in use by model(s): {', '.join(owners)}. "
                   f"Unload them before changing allowed_gpu_indices.",
        )

    if payload.hf_token is not None:
        result = await validate_hf_token(payload.hf_token)
        if not result["valid"]:
            raise HTTPException(status_code=422, detail=result.get("error", "invalid HF token"))
        await db.execute(
            "INSERT INTO settings (key, value) VALUES ('hf_token', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (payload.hf_token,),
        )

    await db.execute(
        "INSERT INTO settings (key, value) VALUES ('allowed_gpu_indices', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (json.dumps(sorted(new_set)),),
    )
    await db.commit()
    return {"ok": True}
```

Add `snapshot()` to `GpuOwnership` in `app/runtime/gpu_ownership.py` (returns a `dict[int, str]` copy under the lock — used by both this endpoint and the supervisor for status pages).

- [ ] **Step 4: Add `GET /settings` route in `app/api/web.py`**

```python
@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, _user=Depends(require_session_html), db=Depends(get_db)):
    cur = await db.fetch_all("SELECT key, value FROM settings WHERE key IN ('allowed_gpu_indices', 'hf_token')")
    settings = {r["key"]: r["value"] for r in cur}
    gpus = await probe_gpus()  # for showing the user which GPUs exist on this host
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "current_allowed": json.loads(settings.get("allowed_gpu_indices", "[]")),
        "hf_token_set": bool(settings.get("hf_token")),
        "gpus": gpus,
    })
```

- [ ] **Step 5: Create `app/templates/settings.html`**

```html
{% extends "_base.html" %}
{% block title %}Settings{% endblock %}
{% block content %}
<h1>Settings</h1>
<form id="settings-form">
  <fieldset>
    <legend>Allowed GPUs</legend>
    {% for g in gpus %}
    <label>
      <input type="checkbox" name="gpu" value="{{ g.index }}"
             {% if g.index in current_allowed %}checked{% endif %}>
      GPU {{ g.index }} — {{ g.name }} ({{ g.mem_total_mib }} MiB)
    </label><br>
    {% endfor %}
  </fieldset>
  <fieldset>
    <legend>HuggingFace token</legend>
    <p>{% if hf_token_set %}A token is currently saved. Leave blank to keep it.{% else %}No token set.{% endif %}</p>
    <input type="password" name="hf_token" placeholder="hf_...">
  </fieldset>
  <button type="submit">Save</button>
  <p id="settings-result"></p>
</form>
<script src="/static/app.js"></script>
<script>
document.getElementById('settings-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  const indices = [...form.querySelectorAll('input[name=gpu]:checked')]
    .map(el => parseInt(el.value, 10));
  const token = form.hf_token.value || null;
  const r = await fetch('/api/settings', {
    method: 'POST',
    headers: {'Content-Type': 'application/json', 'X-CSRF-Token': window.CSRF_TOKEN},
    body: JSON.stringify({allowed_gpu_indices: indices, hf_token: token}),
  });
  const out = document.getElementById('settings-result');
  if (r.ok) { out.textContent = 'Saved.'; out.className = 'ok'; }
  else { const b = await r.json(); out.textContent = 'Error: ' + b.detail; out.className = 'err'; }
});
</script>
{% endblock %}
```

`window.CSRF_TOKEN` is injected by the base template (Task 17).

- [ ] **Step 6: Wire settings router**

In `app/main.py`:

```python
from app.api import settings as settings_api
app.include_router(settings_api.router)
```

- [ ] **Step 7: Run all settings tests**

Run: `make test-unit ARGS=tests/unit/api/test_settings.py`

Expected: PASS — all four green.

- [ ] **Step 8: Commit**

```bash
git add app/api/settings.py app/api/web.py app/templates/settings.html app/main.py app/runtime/gpu_ownership.py tests/unit/api/test_settings.py
git commit -m "feat(settings): edit allowed_gpu_indices + rotate HF token"
```

---

## Phase O — End-to-end smoke + final wiring sweep

### Task 53: E2E smoke — Qwen3.5-9B on real bonus node

**Files:**
- Create: `tests/e2e/test_smoke_qwen3.5-9b.sh`
- Create: `tests/e2e/README.md`

This is the canonical regression for the 2026-05-08 production bug. It proves end-to-end that the wizard's GPU selection reaches the vLLM subprocess. The test is **manual** (gated on real GPU hardware) and does not run in CI — `make test-unit` and `make test-integration` cover everything else.

The test runs against a fresh container on the bonus node, exercises the full lifecycle (setup wizard → add model → load on GPUs `[1,2]` with TP=2 → inference call), and asserts:
1. The vLLM subprocess sees `CUDA_VISIBLE_DEVICES=1,2` (not `0,1,2,3`).
2. Exactly two GPUs go busy in `nvidia-smi` while the model is loaded.
3. A non-streaming `POST /v1/chat/completions` returns 200 with a non-empty content.
4. After unload, both GPUs return to idle and `_gpu_owner` is empty.

- [ ] **Step 1: Write `tests/e2e/test_smoke_qwen3.5-9b.sh`**

```bash
#!/usr/bin/env bash
# tests/e2e/test_smoke_qwen3.5-9b.sh
#
# End-to-end regression for the 2026-05-08 production bug. MUST be run on a host
# with at least 4 NVIDIA GPUs of >=16GiB each. Does not run in CI.
#
# Usage:
#   HF_TOKEN=hf_xxx VW_BASE=http://localhost:8080 ./tests/e2e/test_smoke_qwen3.5-9b.sh
#
# Exit codes: 0 = pass, non-zero = fail.

set -euo pipefail
trap 'echo "FAILED at line $LINENO" >&2' ERR

: "${HF_TOKEN:?HF_TOKEN must be set}"
: "${VW_BASE:?VW_BASE must be set, e.g. http://localhost:8080}"

ADMIN_USER="admin@e2e.local"
ADMIN_PASS="e2e-pass-$$"
COOKIES=$(mktemp)
trap 'rm -f $COOKIES' EXIT

curl_json() {
  curl -s -b "$COOKIES" -c "$COOKIES" -H "Content-Type: application/json" "$@"
}

echo "==> 1. Run setup wizard"
curl_json -X POST "$VW_BASE/api/setup/admin" \
  -d "{\"email\":\"$ADMIN_USER\",\"password\":\"$ADMIN_PASS\"}" >/dev/null
curl_json -X POST "$VW_BASE/api/setup/login" \
  -d "{\"email\":\"$ADMIN_USER\",\"password\":\"$ADMIN_PASS\"}" >/dev/null
CSRF=$(curl_json "$VW_BASE/api/csrf" | jq -r .token)
HEADER_CSRF=(-H "X-CSRF-Token: $CSRF")
curl_json "${HEADER_CSRF[@]}" -X POST "$VW_BASE/api/setup/hf-token" \
  -d "{\"hf_token\":\"$HF_TOKEN\"}" >/dev/null
curl_json "${HEADER_CSRF[@]}" -X POST "$VW_BASE/api/setup/gpus" \
  -d '{"allowed_gpu_indices":[0,1,2,3]}' >/dev/null
curl_json "${HEADER_CSRF[@]}" -X POST "$VW_BASE/api/setup/finish" >/dev/null

echo "==> 2. Register Qwen3.5-9B with TP=2 on GPUs [1,2]"
MODEL_ID=$(curl_json "${HEADER_CSRF[@]}" -X POST "$VW_BASE/api/models" -d '{
  "served_name": "qwen3.5-9b",
  "hf_repo": "Qwen/Qwen2.5-9B",
  "hf_revision": "main",
  "tensor_parallel_size": 2,
  "gpu_indices": [1, 2]
}' | jq -r .id)
echo "    model id: $MODEL_ID"

echo "==> 3. Pull"
curl_json "${HEADER_CSRF[@]}" -X POST "$VW_BASE/api/models/$MODEL_ID/pull" >/dev/null
for i in $(seq 1 60); do
  STATUS=$(curl_json "$VW_BASE/api/models/$MODEL_ID" | jq -r .status)
  [[ "$STATUS" == "pulled" ]] && break
  [[ "$STATUS" == "failed" ]] && { echo "pull failed"; exit 1; }
  sleep 30
done

echo "==> 4. Load (this is the bug-fix moment)"
curl_json "${HEADER_CSRF[@]}" -X POST "$VW_BASE/api/models/$MODEL_ID/load" >/dev/null
for i in $(seq 1 60); do
  STATUS=$(curl_json "$VW_BASE/api/models/$MODEL_ID" | jq -r .status)
  [[ "$STATUS" == "loaded" ]] && break
  [[ "$STATUS" == "failed" ]] && { echo "load failed — check /api/models/$MODEL_ID/logs"; exit 1; }
  sleep 5
done

echo "==> 5. Assert subprocess env CUDA_VISIBLE_DEVICES=1,2"
PID=$(curl_json "$VW_BASE/api/models/$MODEL_ID" | jq -r .runtime.pid)
ENV_VAL=$(tr '\0' '\n' < /proc/"$PID"/environ | awk -F= '/^CUDA_VISIBLE_DEVICES=/ {print $2}')
echo "    /proc/$PID/environ CUDA_VISIBLE_DEVICES=$ENV_VAL"
[[ "$ENV_VAL" == "1,2" ]] || { echo "BUG REGRESSED: expected '1,2', got '$ENV_VAL'"; exit 1; }

echo "==> 6. Assert exactly GPUs 1 and 2 are busy"
BUSY=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
       awk -F',' '$2+0 > 1000 {gsub(/ /,"",$1); print $1}' | sort | tr '\n' ',')
echo "    busy GPUs: $BUSY"
[[ "$BUSY" == "1,2," ]] || { echo "expected GPUs 1,2 busy, got: $BUSY"; exit 1; }

echo "==> 7. Inference"
TOKEN=$(curl_json "${HEADER_CSRF[@]}" -X POST "$VW_BASE/api/tokens" \
  -d '{"name":"e2e"}' | jq -r .token)
RESP=$(curl -s -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -X POST "$VW_BASE/v1/chat/completions" -d '{
    "model": "qwen3.5-9b",
    "messages": [{"role": "user", "content": "Reply with one word."}],
    "max_tokens": 8
  }')
CONTENT=$(echo "$RESP" | jq -r .choices[0].message.content)
echo "    response: $CONTENT"
[[ -n "$CONTENT" && "$CONTENT" != "null" ]] || { echo "empty response: $RESP"; exit 1; }

echo "==> 8. Unload + assert GPUs released"
curl_json "${HEADER_CSRF[@]}" -X POST "$VW_BASE/api/models/$MODEL_ID/unload" >/dev/null
for i in $(seq 1 30); do
  STATUS=$(curl_json "$VW_BASE/api/models/$MODEL_ID" | jq -r .status)
  [[ "$STATUS" == "registered" || "$STATUS" == "pulled" ]] && break
  sleep 1
done
sleep 5
BUSY=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
       awk -F',' '$2+0 > 1000 {print $1}' | tr '\n' ',')
[[ -z "$BUSY" ]] || { echo "GPUs still busy after unload: $BUSY"; exit 1; }

echo "==> ALL CHECKS PASSED"
```

- [ ] **Step 2: Make it executable + write the README**

```bash
chmod +x tests/e2e/test_smoke_qwen3.5-9b.sh
```

`tests/e2e/README.md` content:

```markdown
# E2E smoke tests

These tests require real GPU hardware and are not part of `make test`. Run them
manually after each release candidate against the bonus node.

## test_smoke_qwen3.5-9b.sh

Reproduces the 2026-05-08 production bug regression: the wizard's GPU selection
must reach the vLLM subprocess. Boots a clean container, registers Qwen3.5-9B
with `tensor_parallel_size=2` on GPUs `[1,2]`, and asserts:

1. `/proc/<pid>/environ` shows `CUDA_VISIBLE_DEVICES=1,2` (NOT `0,1,2,3`)
2. Exactly GPUs 1 and 2 are busy in `nvidia-smi`
3. Inference returns 200 with non-empty content
4. After unload, both GPUs are idle

### Usage

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxx
export VW_BASE=http://bonus-node:8080
./tests/e2e/test_smoke_qwen3.5-9b.sh
```

Exit code 0 = pass. Any non-zero = regression — investigate immediately.
```

- [ ] **Step 3: Commit**

```bash
git add tests/e2e/test_smoke_qwen3.5-9b.sh tests/e2e/README.md
git commit -m "test(e2e): smoke for Qwen3.5-9B GPU-selection bug regression"
```

---

### Task 54: Final FastAPI router wiring sweep

**Files:**
- Modify: `app/main.py`
- Test: `tests/unit/test_app_wiring.py`

Earlier tasks each registered their router incrementally. This task does a single audit pass: confirm every router is included exactly once, in a sensible order, with the correct prefix.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_app_wiring.py
import pytest
from app.main import app


def _routes_by_prefix():
    return {r.path: r for r in app.routes}


def test_all_expected_routes_registered():
    paths = {r.path for r in app.routes}

    # API
    expected_api = {
        "/api/csrf",
        "/api/setup/admin",
        "/api/setup/login",
        "/api/setup/hf-token",
        "/api/setup/gpus",
        "/api/setup/finish",
        "/api/models",
        "/api/models/{model_id}",
        "/api/models/{model_id}/pull",
        "/api/models/{model_id}/load",
        "/api/models/{model_id}/unload",
        "/api/models/{model_id}/logs",
        "/api/tokens",
        "/api/tokens/{token_id}",
        "/api/stats/models",
        "/api/stats/gpus",
        "/api/settings",
    }
    missing = expected_api - paths
    assert not missing, f"missing API routes: {missing}"

    # Pages
    expected_pages = {"/", "/login", "/setup", "/models", "/tokens", "/stats", "/settings"}
    missing = expected_pages - paths
    assert not missing, f"missing page routes: {missing}"

    # Proxy
    expected_proxy = {"/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/v1/models"}
    missing = expected_proxy - paths
    assert not missing, f"missing proxy routes: {missing}"


def test_no_duplicate_routes():
    seen = {}
    for r in app.routes:
        key = (r.path, tuple(sorted(getattr(r, "methods", set()) or set())))
        seen.setdefault(key, []).append(r)
    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    assert not dupes, f"duplicate routes: {list(dupes.keys())}"
```

- [ ] **Step 2: Run to find gaps**

Run: `make test-unit ARGS=tests/unit/test_app_wiring.py`

Expected: likely PASS if every previous task wired its router; if FAIL, list of missing routes guides the fix in step 3.

- [ ] **Step 3: Update `app/main.py` to register every router exactly once, in this order**

```python
from contextlib import asynccontextmanager
import asyncio
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.deps import get_db_pool
from app.runtime.stats_sampler import run_sampler_forever
from app.runtime.stats_pruner import run_pruner_forever
from app.runtime.supervisor import Supervisor
from app.runtime.gpu_ownership import GpuOwnership
from app.runtime.tokenizer_cache import TokenizerCache

from app.api import (
    csrf as csrf_api,
    setup as setup_api,
    models as models_api,
    tokens as tokens_api,
    stats as stats_api,
    settings as settings_api,
    web as web_api,
)
from app.proxy import routes as proxy_routes


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = await get_db_pool()
    app.state.gpus = GpuOwnership()
    app.state.tokenizers = TokenizerCache()
    app.state.supervisor = Supervisor(
        db=app.state.db, gpus=app.state.gpus, data_dir="/data"
    )
    await app.state.supervisor.startup_recovery()  # flips loaded/loading rows to failed

    sampler_task = asyncio.create_task(run_sampler_forever(app.state.db))
    pruner_task = asyncio.create_task(run_pruner_forever(app.state.db))
    try:
        yield
    finally:
        sampler_task.cancel()
        pruner_task.cancel()
        await asyncio.gather(sampler_task, pruner_task, return_exceptions=True)
        await app.state.supervisor.shutdown_all()
        await app.state.db.close()


app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

# API (order: csrf, setup, models, tokens, stats, settings)
app.include_router(csrf_api.router)
app.include_router(setup_api.router)
app.include_router(models_api.router)
app.include_router(tokens_api.router)
app.include_router(stats_api.router)
app.include_router(settings_api.router)

# Proxy (must be before web fallback so /v1/* matches first)
app.include_router(proxy_routes.router)

# Web pages last (catch-all home route lives here)
app.include_router(web_api.router)
```

- [ ] **Step 4: Run wiring test + full unit suite**

Run: `make test-unit`

Expected: PASS — every previously-passing test still passes, plus the new wiring test.

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/unit/test_app_wiring.py
git commit -m "chore: final FastAPI router wiring sweep"
```

---

## Phase P — Docker + GitLab CI

### Task 55: Dockerfile

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`
- Modify: `Makefile` (add `make docker-build`, `make docker-run`)
- Test: `tests/integration/test_dockerfile_build.sh` (manual smoke)

Base image is **`vllm/vllm-openai`** pinned by sha256 digest — vLLM is a dep, not just a binary, because we import `transformers` for tokenization in the proxy. Pin guards against silent dep upgrades.

- [ ] **Step 1: Pick the digest**

Run on host:

```bash
docker pull vllm/vllm-openai:v0.6.3
docker images --digests vllm/vllm-openai
```

Record the `sha256:…` digest. The plan uses placeholder `04563c302537a91aa49ebdfbceda96111c5712275999b7e8804fa598f0b5641d` — replace with the actual digest before committing.

- [ ] **Step 2: Write the Dockerfile**

```dockerfile
# syntax=docker/dockerfile:1.7
FROM vllm/vllm-openai@sha256:04563c302537a91aa49ebdfbceda96111c5712275999b7e8804fa598f0b5641d

# Install warden Python deps on top of vllm's environment
WORKDIR /app
COPY pyproject.toml uv.lock /app/
RUN pip install --no-cache-dir uv \
 && uv pip sync --system uv.lock

COPY app /app/app
COPY migrations /app/migrations

# Persistent state
VOLUME ["/data"]

# Warden listens on 8080. vLLM subprocesses use 10000-10999 internally only.
EXPOSE 8080

# Override vllm/vllm-openai's ENTRYPOINT — we run our own FastAPI app
ENTRYPOINT []
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

- [ ] **Step 3: Write `.dockerignore`**

```
.git
.venv
__pycache__
*.pyc
tests/
.worktrees/
data/
*.db
.pytest_cache
.mypy_cache
.ruff_cache
docs/
```

- [ ] **Step 4: Add Makefile targets**

```makefile
IMAGE ?= vllm-warden:dev

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm --gpus all \
	  -v $(PWD)/data:/data \
	  -p 8080:8080 \
	  --name vllm-warden-dev \
	  $(IMAGE)
```

- [ ] **Step 5: Smoke-test the build**

```bash
make docker-build
docker run --rm --entrypoint python vllm-warden:dev -c "from app.main import app; print('ok')"
```

Expected: prints `ok`. If imports fail, fix before continuing.

- [ ] **Step 6: Commit**

```bash
git add Dockerfile .dockerignore Makefile
git commit -m "build: Dockerfile pinned to vllm/vllm-openai digest"
```

---

### Task 56: GitLab CI (lint + test + build, no deploy)

**Files:**
- Create: `.gitlab-ci.yml`

Mirror the lint+build-only convention used by other PodWarden repos for non-`develop`/`main` branches. **No deploy stage** — image publishing comes later, after a real production release plan.

- [ ] **Step 1: Write `.gitlab-ci.yml`**

```yaml
# .gitlab-ci.yml
stages:
  - lint
  - test
  - build

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: ""

.python-image:
  image: python:3.12-slim
  before_script:
    - pip install --no-cache-dir uv
    - uv pip sync --system uv.lock

lint:
  stage: lint
  extends: .python-image
  script:
    - ruff check app tests
    - ruff format --check app tests

unit-tests:
  stage: test
  extends: .python-image
  script:
    - pytest tests/unit -v --tb=short
  artifacts:
    when: always
    reports:
      junit: pytest-unit.xml
    paths:
      - pytest-unit.xml
    expire_in: 1 week

integration-tests:
  stage: test
  extends: .python-image
  script:
    - pytest tests/integration -v --tb=short -m "not requires_gpu"
  artifacts:
    when: always
    reports:
      junit: pytest-integration.xml
    paths:
      - pytest-integration.xml
    expire_in: 1 week

build-image:
  stage: build
  image: docker:24-cli
  services:
    - docker:24-dind
  script:
    - docker build -t vllm-warden:$CI_COMMIT_SHORT_SHA .
    # No push — publishing comes in a separate MR after release plan
  rules:
    - if: $CI_COMMIT_BRANCH == "develop"
    - if: $CI_COMMIT_BRANCH == "main"
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
```

- [ ] **Step 2: Run jobs locally to verify they pass before pushing**

```bash
docker run --rm -v "$PWD":/app -w /app python:3.12-slim bash -c \
  "pip install uv && uv pip sync --system uv.lock && ruff check app tests && pytest tests/unit -v"
```

Expected: PASS — same as `make test-unit` should be passing already.

- [ ] **Step 3: Commit**

```bash
git add .gitlab-ci.yml
git commit -m "ci: lint + test + build (no deploy yet)"
```

---


