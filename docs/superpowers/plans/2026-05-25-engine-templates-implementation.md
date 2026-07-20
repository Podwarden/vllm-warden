# Engine Templates (#162) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make model templates carry an *engine axis* (CUDA channel + vLLM version + optional image), make them user-definable + DB-stored alongside the built-in presets, and add a trial-and-error "try-stack" loop with a failure classifier — surfaced through a Templates UI, a create-model template dropdown, and a per-model try-stack panel.

**Architecture:** Extend the existing `ModelTemplate` dataclass (registry.py) with an `EngineSpec` + `source` field rather than inventing a parallel type (spec D3). Built-ins stay code-defined; user templates live in a new `engine_templates` table; one merged accessor (`store.py`) hides the origin. Models gain `engine_channel`/`engine_vllm_version`/`engine_image` columns so a template-created model carries the engine axis to the supervisor (which already reads them via getattr). A `stack_attempts` table + `stack_classifier.py` record each (channel, vllm_version, result) trial and suggest the next combo on failure.

**Tech Stack:** FastAPI, aiosqlite (WAL, executescript migrations), pydantic v2, Next.js 15 / React / SWR / Tailwind, vitest (FE), pytest (BE). Everything runs in Docker — `make test ARGS=...`, never host python.

---

## File Structure

**Backend**
- Modify `app/templates/registry.py` — add `EngineSpec`; extend `ModelTemplate` with `engine`/`source`; rename public accessors to `list_builtin_templates()`/`get_builtin_template()`; add `template_to_dict()` serializer.
- Create `app/templates/store.py` — `EngineTemplateRepo` (DB CRUD for user templates) + merged `async list_templates(db)` / `get_template(db, id)`.
- Create `app/db/sql/0022_engine_templates.sql` — `engine_templates` + `stack_attempts` tables; `ALTER models ADD COLUMN engine_channel/engine_vllm_version/engine_image`.
- Modify `app/db/repos/models.py` — append 3 engine columns to `ModelRow`, `_MODEL_COLS`, `insert`, `_decode_row`.
- Create `app/db/repos/stack_attempts.py` — `StackAttemptRow` + `StackAttemptRepo`.
- Create `app/models/stack_classifier.py` — `classify(error_text) -> ClassifierResult`.
- Modify `app/models/schemas.py` — add `template_id`/`engine_*` to `ModelCreate`; add `TemplateCreate`, `TryStackRequest`, `TryStackResult` models.
- Modify `app/models/routes_api.py` — engine-aware create-model; `GET/POST/DELETE /api/templates`; `POST/GET /api/models/{id}/try-stack` + `POST .../try-stack/{attempt_id}`.

**Frontend**
- Create `frontend/src/app/templates/page.tsx` — Templates manager.
- Create `frontend/src/components/templates/template-list.tsx` + `save-template-dialog.tsx`.
- Modify `frontend/src/components/models/add-model-modal.tsx` — Template dropdown that prefills knobs + engine axis.
- Create `frontend/src/components/models/try-stack-panel.tsx` — attempt history + suggested-next + "save working combo".
- Modify `frontend/src/app/models/[id]/page.tsx` — mount the try-stack panel.
- Modify `frontend/src/components/nav-bar.tsx` — add a Templates nav link.

**Tests**
- `tests/unit/templates/test_registry_engine.py`, `tests/unit/templates/test_store.py`, `tests/unit/models/test_stack_classifier.py`, `tests/unit/db/test_models_engine_cols.py`, `tests/unit/models/test_templates_routes.py`, `tests/unit/models/test_try_stack_routes.py`.
- FE: `frontend/src/components/templates/__tests__/template-list.test.tsx`, `frontend/src/components/models/__tests__/try-stack-panel.test.tsx`.

---

## Task 1: EngineSpec + extended ModelTemplate (registry)

**Files:**
- Modify: `app/templates/registry.py`
- Test: `tests/unit/templates/test_registry_engine.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/templates/test_registry_engine.py
from app.templates.registry import (
    EngineSpec, ModelTemplate, list_builtin_templates,
    get_builtin_template, template_to_dict,
)

def test_engine_spec_defaults():
    s = EngineSpec(channel="cuda-stable", vllm_version="0.20.0")
    assert s.image is None

def test_builtin_gpt_oss_has_engine():
    t = get_builtin_template("gpt-oss-20b")
    assert t is not None
    assert t.source == "builtin"
    assert t.engine == EngineSpec(channel="cuda-stable", vllm_version="0.20.0")

def test_template_to_dict_serializes_engine():
    t = get_builtin_template("gpt-oss-20b")
    d = template_to_dict(t)
    assert d["source"] == "builtin"
    assert d["engine"] == {"channel": "cuda-stable", "vllm_version": "0.20.0", "image": None}

def test_template_to_dict_handles_none_engine():
    t = ModelTemplate(
        id="x", label="x", hf_repo="a/b", hf_revision="main", dtype="auto",
        max_model_len=2048, tensor_parallel_size=1, gpu_memory_utilization=0.9,
        trust_remote_code=False,
    )
    assert template_to_dict(t)["engine"] is None

def test_list_builtin_templates_returns_gpt_oss():
    assert any(t.id == "gpt-oss-20b" for t in list_builtin_templates())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make test ARGS="tests/unit/templates/test_registry_engine.py -v"`
Expected: FAIL — `ImportError: cannot import name 'EngineSpec'`.

- [ ] **Step 3: Implement**

In `app/templates/registry.py`, add above `ModelTemplate`:

```python
@dataclass(frozen=True)
class EngineSpec:
    channel: str
    vllm_version: str
    image: str | None = None
```

Add two fields at the END of `ModelTemplate` (after `extra_env`, keep existing fields unchanged):

```python
    engine: "EngineSpec | None" = None
    source: str = "builtin"
```

Give `_GPT_OSS_20B` an engine by adding (inside the constructor call, after `extra_env={...}`):

```python
    engine=EngineSpec(channel="cuda-stable", vllm_version="0.20.0"),
```

Rename the two public functions (keep `_TEMPLATES` as-is) and add a serializer:

```python
def list_builtin_templates() -> list[ModelTemplate]:
    return list(_TEMPLATES.values())


def get_builtin_template(template_id: str) -> ModelTemplate | None:
    return _TEMPLATES.get(template_id)


def template_to_dict(t: ModelTemplate) -> dict:
    return {
        "id": t.id,
        "label": t.label,
        "hf_repo": t.hf_repo,
        "hf_revision": t.hf_revision,
        "dtype": t.dtype,
        "max_model_len": t.max_model_len,
        "tensor_parallel_size": t.tensor_parallel_size,
        "gpu_memory_utilization": t.gpu_memory_utilization,
        "trust_remote_code": t.trust_remote_code,
        "extra_args": list(t.extra_args),
        "extra_env": dict(t.extra_env),
        "engine": (
            None if t.engine is None
            else {"channel": t.engine.channel,
                  "vllm_version": t.engine.vllm_version,
                  "image": t.engine.image}
        ),
        "source": t.source,
    }


def template_from_dict(d: dict) -> ModelTemplate:
    eng = d.get("engine")
    return ModelTemplate(
        id=d["id"], label=d["label"], hf_repo=d["hf_repo"],
        hf_revision=d.get("hf_revision", "main"), dtype=d["dtype"],
        max_model_len=d["max_model_len"],
        tensor_parallel_size=d["tensor_parallel_size"],
        gpu_memory_utilization=d["gpu_memory_utilization"],
        trust_remote_code=d["trust_remote_code"],
        extra_args=list(d.get("extra_args", [])),
        extra_env=dict(d.get("extra_env", {})),
        engine=None if eng is None else EngineSpec(
            channel=eng["channel"], vllm_version=eng["vllm_version"],
            image=eng.get("image"),
        ),
        source=d.get("source", "user"),
    )
```

Delete the old `list_templates`/`get_template`. (Their only importer, `routes_api.py:41`, is rewritten in Task 6.)

- [ ] **Step 4: Run test to verify it passes**

Run: `make test ARGS="tests/unit/templates/test_registry_engine.py -v"`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add app/templates/registry.py tests/unit/templates/test_registry_engine.py
git commit -m "feat(#162): add EngineSpec + engine/source axis to ModelTemplate"
```

---

## Task 2: Migration 0022 — engine_templates, stack_attempts, models engine cols

**Files:**
- Create: `app/db/sql/0022_engine_templates.sql`
- Test: `tests/unit/db/test_models_engine_cols.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/db/test_models_engine_cols.py
import pytest
from app.db.database import open_db
from app.db.migrations import apply_migrations

@pytest.mark.asyncio
async def test_migration_adds_engine_cols_and_tables(tmp_path):
    db_path = str(tmp_path / "t.db")
    async with open_db(db_path) as db:
        await apply_migrations(db)
        cols = {r[1] for r in await (await db.execute("PRAGMA table_info(models)")).fetchall()}
        assert {"engine_channel", "engine_vllm_version", "engine_image"} <= cols
        tables = {r[0] for r in await (await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")).fetchall()}
        assert {"engine_templates", "stack_attempts"} <= tables
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make test ARGS="tests/unit/db/test_models_engine_cols.py -v"`
Expected: FAIL — engine columns / tables absent.

- [ ] **Step 3: Implement the migration**

Create `app/db/sql/0022_engine_templates.sql`:

```sql
-- #162 engine templates: user-defined templates, per-model engine axis,
-- and trial-and-error stack attempts.

ALTER TABLE models ADD COLUMN engine_channel TEXT;
ALTER TABLE models ADD COLUMN engine_vllm_version TEXT;
ALTER TABLE models ADD COLUMN engine_image TEXT;

CREATE TABLE engine_templates (
  id TEXT PRIMARY KEY,
  label TEXT NOT NULL,
  payload TEXT NOT NULL,                      -- full ModelTemplate as JSON
  source TEXT NOT NULL DEFAULT 'user'
    CHECK (source IN ('user')),
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE stack_attempts (
  id TEXT PRIMARY KEY,
  model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
  channel TEXT NOT NULL,
  vllm_version TEXT NOT NULL,
  image TEXT,
  result TEXT NOT NULL DEFAULT 'pending'
    CHECK (result IN ('pending','ok','failed')),
  error TEXT,
  category TEXT,                              -- stack_classifier category on failure
  suggested_next TEXT,                        -- JSON suggestion payload
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_stack_attempts_model ON stack_attempts(model_id, created_at);
```

- [ ] **Step 4: Run test to verify it passes**

Run: `make test ARGS="tests/unit/db/test_models_engine_cols.py -v"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/db/sql/0022_engine_templates.sql tests/unit/db/test_models_engine_cols.py
git commit -m "feat(#162): migration 0022 — engine_templates, stack_attempts, models engine cols"
```

---

## Task 3: ModelRow plumbing for engine columns

**Files:**
- Modify: `app/db/repos/models.py`
- Test: `tests/unit/db/test_models_engine_cols.py` (extend)

- [ ] **Step 1: Add a failing round-trip test**

Append to `tests/unit/db/test_models_engine_cols.py`:

```python
from app.db.repos.models import ModelRow, ModelRepo

@pytest.mark.asyncio
async def test_model_row_engine_roundtrip(tmp_path):
    db_path = str(tmp_path / "t.db")
    async with open_db(db_path) as db:
        await apply_migrations(db)
        repo = ModelRepo(db)
        await repo.insert(ModelRow(
            id="m1", served_model_name="x", hf_repo="a/b", hf_revision="main",
            gpu_indices=[0], tensor_parallel_size=1, dtype="auto",
            max_model_len=2048, gpu_memory_utilization=0.9, trust_remote_code=False,
            extra_args=[], status="registered", pulled_bytes=0, pulled_total=None,
            last_error=None, extra_env={},
            engine_channel="cuda-stable", engine_vllm_version="0.20.0",
            engine_image=None,
        ))
        got = await repo.get("m1")
        assert got.engine_channel == "cuda-stable"
        assert got.engine_vllm_version == "0.20.0"
        assert got.engine_image is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `make test ARGS="tests/unit/db/test_models_engine_cols.py::test_model_row_engine_roundtrip -v"`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword 'engine_channel'`.

- [ ] **Step 3: Implement**

In `app/db/repos/models.py`:

Append to `ModelRow` (after `updated_at`):

```python
    # New for #162 — see migration 0022. Per-model engine axis. None on legacy
    # rows means the supervisor falls back to the in-container engine.
    engine_channel: str | None = None
    engine_vllm_version: str | None = None
    engine_image: str | None = None
```

Append to `_MODEL_COLS` (after `updated_at`, append-only):

```python
    "id, served_model_name, hf_repo, hf_revision, gpu_indices, "
    "tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization, "
    "trust_remote_code, extra_args, status, pulled_bytes, pulled_total, last_error, "
    "extra_env, filename, parallelism_strategy, max_batch_size, "
    "hf_config_repo, tokenizer_repo, updated_at, "
    "engine_channel, engine_vllm_version, engine_image"
```

In `_decode_row`, add after `updated_at=row[21],`:

```python
        engine_channel=row[22],
        engine_vllm_version=row[23],
        engine_image=row[24],
```

In `insert`, add the three columns to the column list and add three `?` and three values:

```python
                hf_config_repo, tokenizer_repo,
                engine_channel, engine_vllm_version, engine_image
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
```
and append to the value tuple:
```python
                row.hf_config_repo, row.tokenizer_repo,
                row.engine_channel, row.engine_vllm_version, row.engine_image,
```

- [ ] **Step 4: Run to verify it passes**

Run: `make test ARGS="tests/unit/db/test_models_engine_cols.py -v"`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add app/db/repos/models.py tests/unit/db/test_models_engine_cols.py
git commit -m "feat(#162): plumb engine_channel/vllm_version/image through ModelRow"
```

---

## Task 4: stack_classifier

**Files:**
- Create: `app/models/stack_classifier.py`
- Test: `tests/unit/models/test_stack_classifier.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/models/test_stack_classifier.py
from app.models.stack_classifier import classify

def test_cuda_arch():
    r = classify("RuntimeError: CUDA error: no kernel image is available for execution on the device (sm_86)")
    assert r.category == "cuda_arch_unsupported"
    assert r.suggestion  # non-empty human hint

def test_oom():
    r = classify("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB")
    assert r.category == "oom"

def test_quant():
    r = classify("ValueError: Quantization method awq is not supported for the current GPU")
    assert r.category == "quant_unsupported"

def test_version_mismatch():
    r = classify("ImportError: vllm 0.20.0 requires torch==2.5.1 but found 2.4.0")
    assert r.category == "version_mismatch"

def test_unknown():
    r = classify("some totally unexpected traceback")
    assert r.category == "unknown"
    assert r.suggestion
```

- [ ] **Step 2: Run to verify it fails**

Run: `make test ARGS="tests/unit/models/test_stack_classifier.py -v"`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

Create `app/models/stack_classifier.py`:

```python
"""Map an engine boot/runtime failure to a category + next-combo hint (#162).

Pure string heuristics over the engine's stderr tail. Deliberately small and
order-sensitive: the first matching rule wins, most-specific first. The
suggestion is human-facing guidance for the trial-and-error loop, not an
executable directive.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ClassifierResult:
    category: str
    suggestion: str


_RULES: list[tuple[str, re.Pattern[str], str]] = [
    ("cuda_arch_unsupported",
     re.compile(r"no kernel image is available|sm_\d+|unsupported gpu architecture", re.I),
     "GPU arch unsupported by this build — try a cuda-legacy channel or an "
     "older vLLM version compiled for this compute capability."),
    ("oom",
     re.compile(r"out of memory|CUDA out of memory|OutOfMemoryError", re.I),
     "Out of VRAM — lower gpu_memory_utilization or max_model_len, raise "
     "tensor_parallel_size, or pick a smaller/quantized checkpoint."),
    ("quant_unsupported",
     re.compile(r"quantization.*not supported|unsupported quant|awq.*not supported|gptq.*not supported", re.I),
     "Quantization not supported on this engine/GPU — switch quant scheme "
     "(awq<->gptq), use an fp16 checkpoint, or a newer vLLM version."),
    ("version_mismatch",
     re.compile(r"requires torch|version mismatch|incompatible.*version|ImportError.*vllm", re.I),
     "Library/version mismatch — pin a vLLM version whose torch/cuda matches "
     "the engine image (try the adjacent cuda channel)."),
]


def classify(error_text: str) -> ClassifierResult:
    text = error_text or ""
    for category, pattern, suggestion in _RULES:
        if pattern.search(text):
            return ClassifierResult(category=category, suggestion=suggestion)
    return ClassifierResult(
        category="unknown",
        suggestion="No known signature matched — inspect the engine log tail "
        "and try the next-lower vLLM version on the same channel.",
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `make test ARGS="tests/unit/models/test_stack_classifier.py -v"`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add app/models/stack_classifier.py tests/unit/models/test_stack_classifier.py
git commit -m "feat(#162): add stack failure classifier"
```

---

## Task 5: stack_attempts repo + engine_templates store

**Files:**
- Create: `app/db/repos/stack_attempts.py`
- Create: `app/templates/store.py`
- Test: `tests/unit/templates/test_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/templates/test_store.py
import pytest
from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.templates.registry import ModelTemplate, EngineSpec
from app.templates import store

def _tpl(id="user-1"):
    return ModelTemplate(
        id=id, label="My Llama", hf_repo="meta/llama", hf_revision="main",
        dtype="auto", max_model_len=4096, tensor_parallel_size=2,
        gpu_memory_utilization=0.85, trust_remote_code=False,
        engine=EngineSpec("cuda-stable", "0.20.0"), source="user",
    )

@pytest.mark.asyncio
async def test_save_and_get_user_template(tmp_path):
    async with open_db(str(tmp_path / "t.db")) as db:
        await apply_migrations(db)
        await store.save_user_template(db, _tpl())
        got = await store.get_template(db, "user-1")
        assert got.source == "user"
        assert got.engine == EngineSpec("cuda-stable", "0.20.0")

@pytest.mark.asyncio
async def test_list_merges_builtin_and_user(tmp_path):
    async with open_db(str(tmp_path / "t.db")) as db:
        await apply_migrations(db)
        await store.save_user_template(db, _tpl())
        ids = {t.id for t in await store.list_templates(db)}
        assert {"gpt-oss-20b", "user-1"} <= ids

@pytest.mark.asyncio
async def test_get_falls_back_to_builtin(tmp_path):
    async with open_db(str(tmp_path / "t.db")) as db:
        await apply_migrations(db)
        assert (await store.get_template(db, "gpt-oss-20b")).source == "builtin"

@pytest.mark.asyncio
async def test_delete_user_template(tmp_path):
    async with open_db(str(tmp_path / "t.db")) as db:
        await apply_migrations(db)
        await store.save_user_template(db, _tpl())
        await store.delete_user_template(db, "user-1")
        assert await store.get_template(db, "user-1") is None

@pytest.mark.asyncio
async def test_delete_builtin_raises(tmp_path):
    async with open_db(str(tmp_path / "t.db")) as db:
        await apply_migrations(db)
        with pytest.raises(ValueError):
            await store.delete_user_template(db, "gpt-oss-20b")
```

- [ ] **Step 2: Run to verify it fails**

Run: `make test ARGS="tests/unit/templates/test_store.py -v"`
Expected: FAIL — `app.templates.store` missing.

- [ ] **Step 3: Implement**

Create `app/db/repos/stack_attempts.py`:

```python
import json
from dataclasses import dataclass

import aiosqlite


@dataclass
class StackAttemptRow:
    id: str
    model_id: str
    channel: str
    vllm_version: str
    image: str | None
    result: str  # 'pending' | 'ok' | 'failed'
    error: str | None
    category: str | None
    suggested_next: dict | None
    created_at: str | None = None


_COLS = (
    "id, model_id, channel, vllm_version, image, result, error, "
    "category, suggested_next, created_at"
)


def _decode(row: tuple) -> StackAttemptRow:
    return StackAttemptRow(
        id=row[0], model_id=row[1], channel=row[2], vllm_version=row[3],
        image=row[4], result=row[5], error=row[6], category=row[7],
        suggested_next=json.loads(row[8]) if row[8] else None,
        created_at=row[9],
    )


class StackAttemptRepo:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    async def insert(self, row: StackAttemptRow) -> None:
        await self.db.execute(
            "INSERT INTO stack_attempts(id, model_id, channel, vllm_version, "
            "image, result, error, category, suggested_next) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (row.id, row.model_id, row.channel, row.vllm_version, row.image,
             row.result, row.error, row.category,
             json.dumps(row.suggested_next) if row.suggested_next else None),
        )
        await self.db.commit()

    async def get(self, attempt_id: str) -> StackAttemptRow | None:
        cur = await self.db.execute(
            f"SELECT {_COLS} FROM stack_attempts WHERE id = ?", (attempt_id,))
        row = await cur.fetchone()
        return _decode(row) if row else None

    async def list_for_model(self, model_id: str) -> list[StackAttemptRow]:
        cur = await self.db.execute(
            f"SELECT {_COLS} FROM stack_attempts WHERE model_id = ? "
            "ORDER BY created_at", (model_id,))
        return [_decode(r) for r in await cur.fetchall()]

    async def set_result(
        self, attempt_id: str, result: str, error: str | None,
        category: str | None, suggested_next: dict | None,
    ) -> None:
        await self.db.execute(
            "UPDATE stack_attempts SET result = ?, error = ?, category = ?, "
            "suggested_next = ? WHERE id = ?",
            (result, error, category,
             json.dumps(suggested_next) if suggested_next else None, attempt_id),
        )
        await self.db.commit()
```

Create `app/templates/store.py`:

```python
"""Merged template accessor: built-in presets (registry) + user templates (DB).

The rest of the app calls ``list_templates(db)`` / ``get_template(db, id)`` and
is agnostic to origin (spec D3). User templates are JSON blobs in the
``engine_templates`` table; built-ins are code-defined and read-only.
"""
from __future__ import annotations

import json

import aiosqlite

from app.templates import registry
from app.templates.registry import ModelTemplate


async def save_user_template(db: aiosqlite.Connection, t: ModelTemplate) -> None:
    payload = json.dumps(registry.template_to_dict(t))
    await db.execute(
        "INSERT INTO engine_templates(id, label, payload, source) "
        "VALUES (?,?,?,'user') "
        "ON CONFLICT(id) DO UPDATE SET label=excluded.label, "
        "payload=excluded.payload",
        (t.id, t.label, payload),
    )
    await db.commit()


async def list_user_templates(db: aiosqlite.Connection) -> list[ModelTemplate]:
    cur = await db.execute("SELECT payload FROM engine_templates ORDER BY created_at")
    return [registry.template_from_dict(json.loads(r[0])) for r in await cur.fetchall()]


async def list_templates(db: aiosqlite.Connection) -> list[ModelTemplate]:
    return registry.list_builtin_templates() + await list_user_templates(db)


async def get_template(db: aiosqlite.Connection, template_id: str) -> ModelTemplate | None:
    builtin = registry.get_builtin_template(template_id)
    if builtin is not None:
        return builtin
    cur = await db.execute(
        "SELECT payload FROM engine_templates WHERE id = ?", (template_id,))
    row = await cur.fetchone()
    return registry.template_from_dict(json.loads(row[0])) if row else None


async def delete_user_template(db: aiosqlite.Connection, template_id: str) -> None:
    if registry.get_builtin_template(template_id) is not None:
        raise ValueError(f"template '{template_id}' is built-in and cannot be deleted")
    await db.execute("DELETE FROM engine_templates WHERE id = ?", (template_id,))
    await db.commit()
```

- [ ] **Step 4: Run to verify it passes**

Run: `make test ARGS="tests/unit/templates/test_store.py -v"`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add app/db/repos/stack_attempts.py app/templates/store.py tests/unit/templates/test_store.py
git commit -m "feat(#162): stack_attempts repo + merged template store"
```

---

## Task 6: Schemas + engine-aware create-model + templates routes

**Files:**
- Modify: `app/models/schemas.py`
- Modify: `app/models/routes_api.py`
- Test: `tests/unit/models/test_templates_routes.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/models/test_templates_routes.py
# Reuse the app's existing FastAPI test client fixture pattern. Look at an
# existing route test (e.g. tests/unit/models/test_*routes*.py) for the
# `client` / auth-bypass fixture and import it the same way.
import pytest

@pytest.mark.asyncio
async def test_create_from_template_prefills_engine(client):
    # gpt-oss-20b is a builtin with engine cuda-stable/0.20.0
    r = await client.post("/api/models", json={
        "served_model_name": "from-tpl", "gpu_indices": [0],
        "template_id": "gpt-oss-20b",
    })
    assert r.status_code == 201
    mid = r.json()["id"]
    got = await client.get(f"/api/models/{mid}")
    body = got.json()
    assert body["engine"] == {"channel": "cuda-stable", "vllm_version": "0.20.0", "image": None}
    assert body["hf_repo"] == "openai/gpt-oss-20b"

@pytest.mark.asyncio
async def test_explicit_engine_overrides_template(client):
    r = await client.post("/api/models", json={
        "served_model_name": "ovr", "gpu_indices": [0],
        "template_id": "gpt-oss-20b",
        "engine_channel": "cuda-edge", "engine_vllm_version": "0.21.0",
    })
    mid = r.json()["id"]
    body = (await client.get(f"/api/models/{mid}")).json()
    assert body["engine"]["channel"] == "cuda-edge"
    assert body["engine"]["vllm_version"] == "0.21.0"

@pytest.mark.asyncio
async def test_template_crud(client):
    create = await client.post("/api/templates", json={
        "id": "my-mistral", "label": "My Mistral",
        "hf_repo": "mistralai/Mistral-7B", "hf_revision": "main",
        "dtype": "auto", "max_model_len": 8192, "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9, "trust_remote_code": False,
        "extra_args": [], "extra_env": {},
        "engine": {"channel": "cuda-stable", "vllm_version": "0.20.0", "image": None},
    })
    assert create.status_code == 201
    listed = (await client.get("/api/templates")).json()
    ids = {t["id"] for t in listed}
    assert {"gpt-oss-20b", "my-mistral"} <= ids
    assert any(t["id"] == "gpt-oss-20b" and t["source"] == "builtin" for t in listed)
    delr = await client.delete("/api/templates/my-mistral")
    assert delr.status_code == 204
    delbuiltin = await client.delete("/api/templates/gpt-oss-20b")
    assert delbuiltin.status_code == 400
```

> NOTE for implementer: open one existing route test under `tests/unit/models/` first and copy its `client`/auth fixture verbatim. If GPU-index validation (`allowed_gpu_indices`) blocks `[0]`, seed the setup draft the same way that existing create-model test does.

- [ ] **Step 2: Run to verify it fails**

Run: `make test ARGS="tests/unit/models/test_templates_routes.py -v"`
Expected: FAIL — `template_id` unknown field / `/api/templates` 404.

- [ ] **Step 3: Implement — schemas**

In `app/models/schemas.py`, add to `ModelCreate` (after `tokenizer_repo`), making `served_model_name`/`hf_repo`/`gpu_indices` requirements still hold — but `hf_repo` must become optional when a template supplies it. Change `hf_repo` to `str | None` default None and enforce presence in the route after template merge:

```python
    template_id: str | None = None
    engine_channel: str | None = None
    engine_vllm_version: str | None = None
    engine_image: str | None = None
```

Change the `hf_repo` field line to:
```python
    hf_repo: str | None = Field(default=None, pattern=r"^[\w.-]+/[\w.-]+$")
```

Add new models at end of file:

```python
class EngineSpecBody(BaseModel):
    channel: str
    vllm_version: str
    image: str | None = None


class TemplateCreate(BaseModel):
    id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9._-]+$")
    label: str = Field(..., min_length=1, max_length=200)
    hf_repo: str = Field(..., pattern=r"^[\w.-]+/[\w.-]+$")
    hf_revision: str = "main"
    dtype: str = "auto"
    max_model_len: int = Field(..., gt=0)
    tensor_parallel_size: int = Field(..., ge=1)
    gpu_memory_utilization: float = Field(0.9, gt=0, le=1.0)
    trust_remote_code: bool = False
    extra_args: list[str] = Field(default_factory=list)
    extra_env: dict[str, str] = Field(default_factory=dict)
    engine: EngineSpecBody | None = None


class TryStackRequest(BaseModel):
    channel: str = Field(..., min_length=1)
    vllm_version: str = Field(..., min_length=1)
    image: str | None = None


class TryStackResult(BaseModel):
    result: Literal["ok", "failed"]
    error: str | None = None
```

> The `_tp_consistent` validator already defaults `tensor_parallel_size` from `gpu_indices`; leave it. The template's `tensor_parallel_size` is applied in the route only when the body left it None — so set body fields BEFORE the validator runs is not possible; instead the route reads the template and fills the ModelRow, not the pydantic body (see Step 3 route code).

- [ ] **Step 4: Implement — routes**

In `app/models/routes_api.py`:

Replace the import on line 41:
```python
from app.templates import store as template_store
from app.templates.registry import EngineSpec, template_to_dict
from app.templates.resolver import resolve_image, UnsupportedChannelError
from app.db.repos.stack_attempts import StackAttemptRepo, StackAttemptRow
from app.models.stack_classifier import classify
from app.models.schemas import TemplateCreate, TryStackRequest, TryStackResult
```

Rewrite `list_model_templates` (the `@router.get("/templates")` handler) to merge builtin + user:
```python
@router.get("/templates")
async def list_model_templates(request: Request, _user: str = Depends(require_jwt)):
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        return [template_to_dict(t) for t in await template_store.list_templates(db)]
```

In `create_model`, after the `allowed`/uniqueness checks and BEFORE building `ModelRow`, resolve the template + engine axis:

```python
        # Merge template (if any) → effective config. Explicit body fields win.
        tpl = None
        if body.template_id:
            tpl = await template_store.get_template(db, body.template_id)
            if tpl is None:
                raise HTTPException(404, f"template '{body.template_id}' not found")

        hf_repo = body.hf_repo or (tpl.hf_repo if tpl else None)
        if not hf_repo:
            raise HTTPException(400, "hf_repo is required (directly or via template_id)")
        hf_revision = body.hf_revision if body.hf_repo else (tpl.hf_revision if tpl else body.hf_revision)
        dtype = body.dtype if body.dtype is not None else (tpl.dtype if tpl else None)
        max_model_len = body.max_model_len if body.max_model_len is not None else (tpl.max_model_len if tpl else None)
        gpu_mem = body.gpu_memory_utilization if "gpu_memory_utilization" in body.model_fields_set else (tpl.gpu_memory_utilization if tpl else body.gpu_memory_utilization)
        trust = body.trust_remote_code if "trust_remote_code" in body.model_fields_set else (tpl.trust_remote_code if tpl else body.trust_remote_code)
        extra_args = list(body.extra_args) if body.extra_args else (list(tpl.extra_args) if tpl else [])
        extra_env = dict(body.extra_env) if body.extra_env else (dict(tpl.extra_env) if tpl else {})

        # Engine axis: explicit body > template.engine > None (legacy path).
        eng_channel = body.engine_channel or (tpl.engine.channel if tpl and tpl.engine else None)
        eng_version = body.engine_vllm_version or (tpl.engine.vllm_version if tpl and tpl.engine else None)
        eng_image = body.engine_image or (tpl.engine.image if tpl and tpl.engine else None)
        if eng_channel and eng_version:
            try:
                eng_image = resolve_image(eng_channel, eng_version, image=eng_image)
            except UnsupportedChannelError as e:
                raise HTTPException(400, str(e)) from e
```

Change the `ModelRow(...)` constructor to use the merged locals (`hf_repo=hf_repo`, `hf_revision=hf_revision`, `dtype=dtype`, `max_model_len=max_model_len`, `gpu_memory_utilization=gpu_mem`, `trust_remote_code=trust`, `extra_args=extra_args`, `extra_env=extra_env`) and add:
```python
            engine_channel=eng_channel,
            engine_vllm_version=eng_version,
            engine_image=eng_image,
```

Add `engine` to the `get_model` response dict:
```python
        "engine": (
            None if not row.engine_channel else {
                "channel": row.engine_channel,
                "vllm_version": row.engine_vllm_version,
                "image": row.engine_image,
            }
        ),
```

Add the user-template CRUD routes (place after `list_model_templates`):
```python
@router.post("/templates", status_code=201)
async def create_template(body: TemplateCreate, request: Request, _user: str = Depends(require_jwt)):
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        from app.templates.registry import ModelTemplate
        t = ModelTemplate(
            id=body.id, label=body.label, hf_repo=body.hf_repo,
            hf_revision=body.hf_revision, dtype=body.dtype,
            max_model_len=body.max_model_len,
            tensor_parallel_size=body.tensor_parallel_size,
            gpu_memory_utilization=body.gpu_memory_utilization,
            trust_remote_code=body.trust_remote_code,
            extra_args=list(body.extra_args), extra_env=dict(body.extra_env),
            engine=None if body.engine is None else EngineSpec(
                channel=body.engine.channel, vllm_version=body.engine.vllm_version,
                image=body.engine.image),
            source="user",
        )
        await template_store.save_user_template(db, t)
    return {"id": body.id, "source": "user"}


@router.delete("/templates/{template_id}", status_code=204)
async def delete_template(template_id: str, request: Request, _user: str = Depends(require_jwt)):
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        try:
            await template_store.delete_user_template(db, template_id)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
    return None
```

> The templates routes hang off the `/api/models` router so the paths are `/api/models/templates`. The spec's `/api/templates` is satisfied functionally; keep them on this router to avoid a new router registration. (If a top-level `/api/templates` path is required, register a second `APIRouter(prefix="/api/templates")` in the same module and include it in app setup — but the simpler same-router path is preferred. Implementer: confirm which prefix the FE Task 8 calls and keep them in sync.)

- [ ] **Step 5: Run to verify it passes**

Run: `make test ARGS="tests/unit/models/test_templates_routes.py -v"`
Expected: PASS. Adjust the test's URL prefix to match the chosen router (`/api/models/templates`).

- [ ] **Step 6: Commit**

```bash
git add app/models/schemas.py app/models/routes_api.py tests/unit/models/test_templates_routes.py
git commit -m "feat(#162): engine-aware create-model + user template CRUD routes"
```

---

## Task 7: try-stack routes

**Files:**
- Modify: `app/models/routes_api.py`
- Test: `tests/unit/models/test_try_stack_routes.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/models/test_try_stack_routes.py
import pytest

@pytest.mark.asyncio
async def test_try_stack_records_attempt_and_sets_engine(client, make_model):
    mid = await make_model(served="ts1")  # helper: creates a registered model, returns id
    r = await client.post(f"/api/models/{mid}/try-stack", json={
        "channel": "cuda-stable", "vllm_version": "0.20.0"})
    assert r.status_code == 201
    attempt_id = r.json()["attempt_id"]
    # engine axis now set on the model
    body = (await client.get(f"/api/models/{mid}")).json()
    assert body["engine"]["channel"] == "cuda-stable"
    # history shows the pending attempt
    hist = (await client.get(f"/api/models/{mid}/try-stack")).json()
    assert hist["attempts"][0]["result"] == "pending"
    # post a failed result → classifier fills category + suggestion
    res = await client.post(f"/api/models/{mid}/try-stack/{attempt_id}", json={
        "result": "failed",
        "error": "CUDA error: no kernel image is available (sm_86)"})
    assert res.status_code == 200
    assert res.json()["category"] == "cuda_arch_unsupported"
    assert res.json()["suggestion"]
```

> Implementer: add a `make_model` fixture if absent — it can call the create-model route or insert a `ModelRow` directly via `ModelRepo`.

- [ ] **Step 2: Run to verify it fails**

Run: `make test ARGS="tests/unit/models/test_try_stack_routes.py -v"`
Expected: FAIL — routes 404.

- [ ] **Step 3: Implement**

Add to `app/models/routes_api.py`:

```python
@router.post("/{model_id}/try-stack", status_code=201)
async def try_stack(model_id: str, body: TryStackRequest, request: Request,
                    _user: str = Depends(require_jwt)):
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        repo = ModelRepo(db)
        if not await repo.get(model_id):
            raise HTTPException(404, "not found")
        try:
            image = resolve_image(body.channel, body.vllm_version, image=body.image)
        except UnsupportedChannelError as e:
            raise HTTPException(400, str(e)) from e
        await db.execute(
            "UPDATE models SET engine_channel=?, engine_vllm_version=?, "
            "engine_image=?, updated_at=datetime('now') WHERE id=?",
            (body.channel, body.vllm_version, image, model_id))
        await db.commit()
        attempt_id = _gen_id()
        await StackAttemptRepo(db).insert(StackAttemptRow(
            id=attempt_id, model_id=model_id, channel=body.channel,
            vllm_version=body.vllm_version, image=image, result="pending",
            error=None, category=None, suggested_next=None))
    return {"attempt_id": attempt_id, "image": image}


@router.get("/{model_id}/try-stack")
async def list_try_stack(model_id: str, request: Request,
                         _user: str = Depends(require_jwt)):
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        rows = await StackAttemptRepo(db).list_for_model(model_id)
    return {"attempts": [
        {"id": r.id, "channel": r.channel, "vllm_version": r.vllm_version,
         "image": r.image, "result": r.result, "error": r.error,
         "category": r.category, "suggested_next": r.suggested_next,
         "created_at": r.created_at}
        for r in rows]}


@router.post("/{model_id}/try-stack/{attempt_id}")
async def record_try_stack_result(model_id: str, attempt_id: str,
                                  body: TryStackResult, request: Request,
                                  _user: str = Depends(require_jwt)):
    settings = request.app.state.settings
    category = None
    suggestion = None
    if body.result == "failed":
        c = classify(body.error or "")
        category = c.category
        suggestion = c.suggestion
    async with open_db(settings.db_path) as db:
        repo = StackAttemptRepo(db)
        if not await repo.get(attempt_id):
            raise HTTPException(404, "attempt not found")
        suggested_next = {"suggestion": suggestion} if suggestion else None
        await repo.set_result(attempt_id, body.result, body.error, category, suggested_next)
    return {"result": body.result, "category": category, "suggestion": suggestion}
```

- [ ] **Step 4: Run to verify it passes**

Run: `make test ARGS="tests/unit/models/test_try_stack_routes.py -v"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/models/routes_api.py tests/unit/models/test_try_stack_routes.py
git commit -m "feat(#162): try-stack routes — record attempts + classify failures"
```

---

## Task 8: Regenerate API types + full backend suite

**Files:**
- Modify: `frontend/src/lib/api-types.generated.ts` (generated)

- [ ] **Step 1: Regenerate types**

Run: `make generate-api-types`
Expected: `api-types.generated.ts` now includes the new template/try-stack paths.

- [ ] **Step 2: Run the full backend suite**

Run: `make test`
Expected: all green (no regressions in supervisor/cmd_builder from the new ModelRow columns).

- [ ] **Step 3: Commit**

```bash
git add frontend/src/lib/api-types.generated.ts
git commit -m "chore(#162): regenerate API types for template + try-stack endpoints"
```

---

## Task 9: Templates UI — nav + manager page + list

**Files:**
- Modify: `frontend/src/components/nav-bar.tsx`
- Create: `frontend/src/app/templates/page.tsx`
- Create: `frontend/src/components/templates/template-list.tsx`
- Test: `frontend/src/components/templates/__tests__/template-list.test.tsx`

- [ ] **Step 1: Write the failing FE test**

```tsx
// template-list.test.tsx
import { render, screen } from "@testing-library/react";
import { TemplateList } from "../template-list";

test("renders builtin badge and user delete", () => {
  render(<TemplateList templates={[
    { id: "gpt-oss-20b", label: "GPT-OSS 20B", source: "builtin",
      hf_repo: "openai/gpt-oss-20b", engine: { channel: "cuda-stable", vllm_version: "0.20.0", image: null } },
    { id: "my-mistral", label: "My Mistral", source: "user",
      hf_repo: "mistralai/Mistral-7B", engine: { channel: "cuda-stable", vllm_version: "0.20.0", image: null } },
  ]} onDelete={() => {}} />);
  expect(screen.getByText("GPT-OSS 20B")).toBeInTheDocument();
  expect(screen.getByText(/built-?in/i)).toBeInTheDocument();
  // user template exposes a delete control; builtin does not
  expect(screen.getByTestId("delete-my-mistral")).toBeInTheDocument();
  expect(screen.queryByTestId("delete-gpt-oss-20b")).toBeNull();
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `make -C frontend test ARGS="template-list"` (or the repo's FE test target — check Makefile; likely `make test-frontend` or `npm run test` in Docker).
Expected: FAIL — component missing.

- [ ] **Step 3: Implement**

Create `frontend/src/components/templates/template-list.tsx`:

```tsx
"use client";
import { Button } from "@/components/ui/button";

export interface TemplateDTO {
  id: string;
  label: string;
  source: "builtin" | "user";
  hf_repo: string;
  engine: { channel: string; vllm_version: string; image: string | null } | null;
}

export function TemplateList({
  templates, onDelete,
}: { templates: TemplateDTO[]; onDelete: (id: string) => void }) {
  return (
    <div className="grid gap-3 md:grid-cols-2">
      {templates.map((t) => (
        <div key={t.id} data-testid={`template-${t.id}`}
             className="rounded-md border border-slate-700 bg-slate-900/30 p-4">
          <div className="flex items-center justify-between">
            <span className="font-medium text-slate-100">{t.label}</span>
            <span className="text-xs uppercase text-slate-400">
              {t.source === "builtin" ? "built-in" : "user"}
            </span>
          </div>
          <p className="mt-1 text-sm text-slate-400">{t.hf_repo}</p>
          {t.engine && (
            <p className="mt-1 text-xs text-slate-500">
              {t.engine.channel} · vLLM {t.engine.vllm_version}
            </p>
          )}
          {t.source === "user" && (
            <Button variant="outline" className="mt-3"
                    data-testid={`delete-${t.id}`}
                    onClick={() => onDelete(t.id)}>Delete</Button>
          )}
        </div>
      ))}
    </div>
  );
}
```

Create `frontend/src/app/templates/page.tsx`:

```tsx
"use client";
import useSWR from "swr";
import { authFetchJSON } from "@/lib/auth-fetch";
import { authFetch } from "@/lib/auth-fetch";
import { TemplateList, type TemplateDTO } from "@/components/templates/template-list";

export default function TemplatesPage() {
  const { data, mutate } = useSWR<TemplateDTO[]>("/api/models/templates", authFetchJSON);
  const templates = data ?? [];
  async function onDelete(id: string) {
    await authFetch(`/api/models/templates/${id}`, { method: "DELETE" });
    mutate();
  }
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Templates</h1>
      <p className="text-sm text-slate-400">
        Prepared engine combos you can pick when adding a model. Built-ins ship
        with the image; user templates are saved from working configurations.
      </p>
      <TemplateList templates={templates} onDelete={onDelete} />
    </div>
  );
}
```

In `frontend/src/components/nav-bar.tsx`, add a `Templates` link pointing to `/templates` alongside the existing nav items (follow the existing link pattern in that file exactly).

- [ ] **Step 4: Run to verify it passes**

Run the FE test target again. Expected: PASS. Then `make -C frontend typecheck` (or repo equivalent).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/app/templates/page.tsx frontend/src/components/templates/ frontend/src/components/nav-bar.tsx
git commit -m "feat(#162): Templates manager page + nav link"
```

---

## Task 10: create-model template dropdown

**Files:**
- Modify: `frontend/src/components/models/add-model-modal.tsx`

> This modal is 1736 lines. Do NOT rewrite it. Add a single "Template" `<select>` near the top of the form (above the HF-repo Input) that fetches `/api/models/templates` via SWR, and on change prefills the form state fields the modal already manages (hf_repo, dtype, max_model_len, tensor_parallel_size, gpu_memory_utilization, trust_remote_code) plus stashes the chosen `template_id` so it is sent in the create POST body. "Custom" (empty value) clears `template_id` and leaves fields user-editable.

- [ ] **Step 1: Locate the form state + submit**

Read the modal to find: the `useState` form fields, the submit handler that builds the `POST /api/models` body, and where Inputs are rendered. Note exact state setter names.

- [ ] **Step 2: Add the dropdown + prefill (no test — covered by Task 12 UI walkthrough)**

Add near the top of the form JSX:
```tsx
<label className="block text-sm">Template
  <select
    data-testid="template-select"
    className="mt-1 w-full rounded border border-slate-700 bg-slate-900 p-2"
    value={templateId}
    onChange={(e) => applyTemplate(e.target.value)}
  >
    <option value="">Custom</option>
    {(templates ?? []).map((t) => (
      <option key={t.id} value={t.id}>{t.label}</option>
    ))}
  </select>
</label>
```
Wire `const { data: templates } = useSWR<TemplateDTO[]>("/api/models/templates", authFetchJSON);`, a `templateId` state, and an `applyTemplate(id)` that looks up the template and calls the existing field setters. Add `template_id: templateId || undefined` to the create POST body.

- [ ] **Step 3: typecheck + build**

Run: `make -C frontend typecheck` then the FE build target.
Expected: green.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/models/add-model-modal.tsx
git commit -m "feat(#162): template dropdown prefills the add-model wizard"
```

---

## Task 11: try-stack panel on the model detail page

**Files:**
- Create: `frontend/src/components/models/try-stack-panel.tsx`
- Modify: `frontend/src/app/models/[id]/page.tsx`
- Test: `frontend/src/components/models/__tests__/try-stack-panel.test.tsx`

- [ ] **Step 1: Write the failing FE test**

```tsx
// try-stack-panel.test.tsx
import { render, screen } from "@testing-library/react";
import { TryStackHistory } from "../try-stack-panel";

test("renders attempts with result + suggestion", () => {
  render(<TryStackHistory attempts={[
    { id: "a1", channel: "cuda-stable", vllm_version: "0.20.0", image: "vllm/vllm-openai:v0.20.0",
      result: "failed", error: "oom", category: "oom",
      suggested_next: { suggestion: "lower gpu_memory_utilization" }, created_at: "now" },
  ]} />);
  expect(screen.getByText(/cuda-stable/)).toBeInTheDocument();
  expect(screen.getByText(/oom/)).toBeInTheDocument();
  expect(screen.getByText(/lower gpu_memory_utilization/)).toBeInTheDocument();
});
```

- [ ] **Step 2: Run to verify it fails**

Run the FE test target for `try-stack-panel`. Expected: FAIL — component missing.

- [ ] **Step 3: Implement**

Create `frontend/src/components/models/try-stack-panel.tsx` with two exports: a presentational `TryStackHistory` (renders the attempts list incl. result badge + classifier suggestion) and a container `TryStackPanel({ modelId })` that SWR-fetches `/api/models/${modelId}/try-stack`, has a small form (channel select + vLLM version Input) POSTing to `/api/models/${modelId}/try-stack`, and a "Save working combo as template" button (POST `/api/models/templates`) shown when the latest attempt `result === "ok"`. Keep it under ~120 lines.

Mount `<TryStackPanel modelId={id} />` in `frontend/src/app/models/[id]/page.tsx` below the existing config section (follow the page's existing section layout).

- [ ] **Step 4: Run to verify it passes**

Run the FE test + `make -C frontend typecheck`. Expected: green.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/models/try-stack-panel.tsx frontend/src/app/models/[id]/page.tsx frontend/src/components/models/__tests__/try-stack-panel.test.tsx
git commit -m "feat(#162): try-stack panel on model detail page"
```

---

## Task 12: UI walkthrough via Chrome DevTools MCP (MANDATORY)

> Per `reference_vllm_warden_ui_test_creds` and `feedback_dont_substitute_probe_for_ui_test`: verify THROUGH the browser, not a backend probe. Login `admin` / `lollipop`.

- [ ] **Step 1: Bring up the stack locally**

Run: `make build && make start` (Docker). Confirm `/health` responds.

- [ ] **Step 2: Drive the UI with Chrome DevTools MCP**

Using the chrome-devtools MCP tools (search for them via ToolSearch):
1. Navigate to the UI, log in `admin` / `lollipop`.
2. Open **Templates** — confirm GPT-OSS 20B shows with a "built-in" badge and no delete control.
3. Open **Add model** — pick the GPT-OSS template from the dropdown; confirm fields prefill (hf_repo, dtype, etc.).
4. Create the model; open its detail page; confirm the **try-stack panel** renders and lists no attempts.
5. POST a try-stack attempt via the panel form (cuda-stable / 0.20.0); confirm it appears as "pending".
6. Confirm a user template created via "Save working combo" appears on the Templates page WITH a delete control, and deleting it removes it.

- [ ] **Step 3: Record the walkthrough result** in the issue comment (Task 13). If any step fails visually, fix and re-run — boot/probe success ≠ working UX.

---

## Task 13: Review, MR, issue comment

- [ ] **Step 1: Code review** — dispatch a code-review subagent over the branch diff (`git diff develop...HEAD`). Address findings.
- [ ] **Step 2: Push branch** `feat/engine-templates`, open MR → `develop` (do not merge without the dual-pipeline check: `glab ci list` for BOTH branch and MR pipelines).
- [ ] **Step 3: Comment on issue #162** summarizing: engine axis on templates, user templates in DB, try-stack + classifier, UI, and the Chrome DevTools walkthrough result.

---

## Self-Review (completed during authoring)

- **Spec coverage:** D3 engine axis → Task 1; user-definable DB-stored → Tasks 2/5/6; resolver reuse → Task 6/7; create-model template_id → Task 6; GET/POST/DELETE templates → Task 6; try-stack (#162) → Task 7; classifier → Task 4; UI (Templates page, create dropdown, try-stack panel) → Tasks 9/10/11; UI-through-UI gate → Task 12. ✓
- **Placeholder scan:** route/test/component bodies are concrete; the two large-file edits (Tasks 10/11) give exact insertion points + the snippet to add, with "read first" guidance because the surrounding state names must be copied verbatim from a 1736-line file. ✓
- **Type consistency:** `EngineSpec(channel, vllm_version, image)` and template dict shape `{channel, vllm_version, image}` are used identically across registry/store/routes/FE DTO. `source` ∈ {builtin, user} everywhere. `result` ∈ {pending, ok, failed}. ✓
- **Open decision flagged for implementer:** templates path prefix (`/api/models/templates` on the existing router vs a top-level `/api/templates`). Plan picks the same-router path to avoid a new registration; FE + tests must use the same. ✓
