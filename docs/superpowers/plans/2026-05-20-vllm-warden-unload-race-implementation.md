# vllm-warden Unload-Race Elimination — Implementation Plan

> **Historical note (epic/overhaul S1, 2026-05-22):** Task 5 ("Bench supervisor — defensive `force=True`") and every other reference in this plan to `app/bench/supervisor.py`, the bench CLI, `bench_health_wait_s`, or `bench_*` tests is **obsolete**. The entire Bench v2 subsystem was excised as part of the overhaul; those files no longer exist. The remaining tasks (Supervisor state machine, `?force=true` HTTP gate, warmup probe) still apply and were shipped on `develop` before this branch was cut. Future readers: treat the bench-related steps as superseded; they document a pre-overhaul code shape.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate every code path that can SIGTERM a vLLM subprocess while it is still in startup or multimodal warmup, by moving lifecycle state into the supervisor and gating `unload()` on a `READY` state.

**Architecture:** `Supervisor` gains a per-model `ModelState` (`LOADING → WARMING → READY → UNLOADING`) and an `UnloadRefused` exception. The HTTP load runner stops auto-calling `sup.unload()` on failures and adds a warmup verification probe (`POST /v1/completions max_tokens=1`) that must succeed before status flips to `loaded`. The HTTP unload route gains a `?force=true` query param that is the only path to bypass the state gate. The bench supervisor passes `force=True` defensively when it coordinates lifecycle internally.

**Tech Stack:** Python 3.11, FastAPI, httpx, asyncio, pytest, pytest-asyncio. Tests run in Docker via `make test-unit` / `make test-integration`.

**Spec:** `docs/superpowers/specs/2026-05-20-vllm-warden-unload-race-design.md`

---

## File Structure

**Modified:**
- `app/runtime/supervisor.py` — add `ModelState` enum, `UnloadRefused`, `_state` dict, `mark_warming()`, `mark_ready()`, `force=` param on `unload()`
- `app/models/routes_api.py` — restructure load runner; remove auto-`sup.unload()` on health timeout; add warmup probe call; add `?force=true` query param; catch `UnloadRefused` → 409
- `app/bench/supervisor.py` — pass `force=True` on internal `sup.unload()` calls
- `app/config.py` — add `warmup_probe_timeout_s: float = 60.0`
- `changelog.md` — Unreleased entry

**Created:**
- `app/runtime/warmup_probe.py` — single-responsibility module: `async def warmup_probe(*, port, served_model_name, timeout_s) -> ProbeResult`
- `tests/unit/runtime/test_supervisor_state.py` — state machine + UnloadRefused tests
- `tests/unit/runtime/test_warmup_probe.py` — probe success/failure/timeout tests
- `tests/integration/test_load_lifecycle.py` — mock-subprocess end-to-end test

**Not changing:** DB schema, `ACTIVE_STATUSES` tuple, `mark_runtime_dead_on_startup`, `UNLOAD_GRACE_SECONDS`.

---

## Task 1: Add `warmup_probe_timeout_s` setting

**Files:**
- Modify: `app/config.py:14`

- [ ] **Step 1: Add the field to Settings**

In `app/config.py`, after `load_timeout_s: float = 600.0`, add:

```python
    # Max time the warmup verification probe waits for a successful
    # POST /v1/completions before marking the load failed. Closes the
    # window between /health 200 and actual serving readiness (e.g.
    # Qwen3-VL's _warmup_mm_processor). Configurable via env override.
    warmup_probe_timeout_s: float = 60.0
```

- [ ] **Step 2: Add env override in `load_settings()`**

In `app/config.py`, inside `load_settings()`, add before the `return Settings(...)`:

```python
    warmup_probe_timeout_s = float(
        os.environ.get("VW_WARMUP_PROBE_TIMEOUT_S", "60.0")
    )
```

And add the field to the `Settings(...)` constructor call:

```python
    return Settings(
        data_dir=data_dir,
        hf_cache_dir=hf_cache_dir,
        cookie_secret=secret,
        container_gpu_count=gpu_count,
        frontend_origin=frontend_origin,
        bench_health_wait_s=bench_health_wait_s,
        warmup_probe_timeout_s=warmup_probe_timeout_s,
    )
```

- [ ] **Step 3: Verify it loads**

Run: `make test-unit ARGS="tests/unit -k config"` (or `pytest tests/unit -k 'config or settings'` if a config test exists)
Expected: PASS (or no tests collected if none exist; ensure imports still succeed).

Sanity-check via Python: `docker run --rm -v $(pwd):/app -w /app python:3.11-slim sh -c "pip install -q -r requirements-dev.txt && python -c 'from app.config import Settings; s = Settings(data_dir=__import__(\"pathlib\").Path(\"/tmp\"), hf_cache_dir=__import__(\"pathlib\").Path(\"/tmp\"), cookie_secret=\"x\"*32, container_gpu_count=0); print(s.warmup_probe_timeout_s)'"` → expect `60.0`.

- [ ] **Step 4: Hold commit**

Don't commit yet — Task 2 commits this together with the state machine.

---

## Task 2: Supervisor state machine + `UnloadRefused`

**Files:**
- Modify: `app/runtime/supervisor.py:1-31` (imports + class init)
- Modify: `app/runtime/supervisor.py:56-101` (load)
- Modify: `app/runtime/supervisor.py:33-54` (_watch_exit)
- Modify: `app/runtime/supervisor.py:144-169` (unload)
- Modify: `app/runtime/supervisor.py:103-130` (new mark_warming/mark_ready, get_state)
- Create: `tests/unit/runtime/test_supervisor_state.py`

- [ ] **Step 1: Write failing tests for the state machine**

Create `tests/unit/runtime/test_supervisor_state.py`:

```python
import asyncio
import signal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.runtime.supervisor import ModelState, Supervisor, UnloadRefused


class _Settings:
    pass


def _make_sup(tmp_path):
    s = _Settings()
    s.data_dir = str(tmp_path)
    s.hf_token_path = str(tmp_path / "tok")
    (tmp_path / "tok").write_text("hf_x")
    return Supervisor(s)


def _register_proc(sup, model_id="m1", state=ModelState.LOADING):
    proc = MagicMock()
    proc.pid = 4242
    proc.returncode = None

    async def fake_wait():
        proc.returncode = 0
        return 0
    proc.wait = AsyncMock(side_effect=fake_wait)

    sup._processes[model_id] = proc
    sup._ports[model_id] = 10001
    sup._state[model_id] = state
    sup.gpus.claim(model_id, [0])
    return proc


@pytest.mark.asyncio
async def test_load_sets_state_loading(tmp_path):
    sup = _make_sup(tmp_path)
    proc = MagicMock()
    proc.pid = 1
    proc.returncode = None
    with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)):
        model = MagicMock()
        model.id = "m1"
        model.gpu_indices = [0]
        model.tensor_parallel_size = 1
        with patch("app.runtime.supervisor.build_subprocess_env", return_value={}):
            with patch("app.runtime.supervisor.build_vllm_args", return_value=[]):
                await sup.load(model, port=10001)
    assert sup.get_state("m1") == ModelState.LOADING


@pytest.mark.asyncio
async def test_mark_warming_then_ready_transitions(tmp_path):
    sup = _make_sup(tmp_path)
    _register_proc(sup, state=ModelState.LOADING)
    await sup.mark_warming("m1")
    assert sup.get_state("m1") == ModelState.WARMING
    await sup.mark_ready("m1")
    assert sup.get_state("m1") == ModelState.READY


@pytest.mark.asyncio
async def test_unload_refused_from_loading(tmp_path):
    sup = _make_sup(tmp_path)
    _register_proc(sup, state=ModelState.LOADING)
    with pytest.raises(UnloadRefused) as exc:
        await sup.unload("m1")
    assert "LOADING" in str(exc.value)
    # Process must still be registered — no SIGTERM sent
    assert "m1" in sup._processes


@pytest.mark.asyncio
async def test_unload_refused_from_warming(tmp_path):
    sup = _make_sup(tmp_path)
    _register_proc(sup, state=ModelState.WARMING)
    with pytest.raises(UnloadRefused):
        await sup.unload("m1")
    assert "m1" in sup._processes


@pytest.mark.asyncio
async def test_unload_force_bypasses_state_gate_from_loading(tmp_path):
    sup = _make_sup(tmp_path)
    _register_proc(sup, state=ModelState.LOADING)
    with patch("os.killpg") as kp:
        await sup.unload("m1", force=True)
    kp.assert_called_with(4242, signal.SIGTERM)
    assert "m1" not in sup._processes


@pytest.mark.asyncio
async def test_unload_from_ready_works_without_force(tmp_path):
    sup = _make_sup(tmp_path)
    _register_proc(sup, state=ModelState.READY)
    with patch("os.killpg") as kp:
        await sup.unload("m1")
    kp.assert_called_with(4242, signal.SIGTERM)
    assert "m1" not in sup._processes


@pytest.mark.asyncio
async def test_watch_exit_clears_state(tmp_path):
    sup = _make_sup(tmp_path)
    proc = _register_proc(sup, state=ModelState.WARMING)
    # Trigger natural exit
    await sup._watch_exit("m1", None)
    assert sup.get_state("m1") is None
    assert "m1" not in sup._processes


@pytest.mark.asyncio
async def test_get_state_returns_none_for_unknown(tmp_path):
    sup = _make_sup(tmp_path)
    assert sup.get_state("does-not-exist") is None


@pytest.mark.asyncio
async def test_mark_warming_raises_if_not_loading(tmp_path):
    sup = _make_sup(tmp_path)
    _register_proc(sup, state=ModelState.READY)
    with pytest.raises(RuntimeError):
        await sup.mark_warming("m1")
```

- [ ] **Step 2: Run the tests — verify they fail**

Run: `make test-unit ARGS="tests/unit/runtime/test_supervisor_state.py -v"`
Expected: ImportError on `ModelState`, `UnloadRefused` → all tests fail to collect.

- [ ] **Step 3: Implement the state machine in `supervisor.py`**

Edit `app/runtime/supervisor.py`. At the top, after the existing imports, add:

```python
from enum import Enum


class ModelState(str, Enum):
    LOADING = "loading"
    WARMING = "warming"
    READY = "ready"
    UNLOADING = "unloading"


class UnloadRefused(Exception):
    """Raised when unload() is called on a model whose supervisor state
    is not READY and the caller did not pass ``force=True``.

    The exception message names the current state so the HTTP layer can
    translate to a 409 with a useful body.
    """

    def __init__(self, model_id: str, state: ModelState) -> None:
        self.model_id = model_id
        self.state = state
        super().__init__(
            f"refusing to unload model {model_id!r}: state is {state.name}, "
            f"not READY (pass force=True to override)"
        )
```

In `Supervisor.__init__`, add `_state` alongside `_processes`:

```python
        self._state: dict[str, ModelState] = {}
```

In `Supervisor.load()`, after `self._processes[model.id] = proc`:

```python
                self._state[model.id] = ModelState.LOADING
```

In `Supervisor._watch_exit()`, inside the `if model_id not in self._processes:` block's else branch (where it currently pops `_processes`, `_ports`, `_overrides`, `_watchers`), also pop `_state`:

```python
                self._state.pop(model_id, None)
```

After the existing `get_pid` method, add:

```python
    def get_state(self, model_id: str) -> ModelState | None:
        """Current supervisor lifecycle state for ``model_id``.

        ``None`` if no process is registered. Public read for callers
        that need to display lifecycle (UI, bench).
        """
        return self._state.get(model_id)

    async def mark_warming(self, model_id: str) -> None:
        """Transition ``model_id`` from LOADING to WARMING.

        Called by the load runner after ``wait_for_health`` succeeds and
        before the warmup verification probe runs.
        """
        async with self._lock:
            cur = self._state.get(model_id)
            if cur is not ModelState.LOADING:
                raise RuntimeError(
                    f"cannot mark warming from state {cur}: expected LOADING"
                )
            self._state[model_id] = ModelState.WARMING

    async def mark_ready(self, model_id: str) -> None:
        """Transition ``model_id`` from WARMING to READY.

        Called by the load runner after the warmup probe succeeds.
        After this transition, ``unload()`` is permitted without force.
        """
        async with self._lock:
            cur = self._state.get(model_id)
            if cur is not ModelState.WARMING:
                raise RuntimeError(
                    f"cannot mark ready from state {cur}: expected WARMING"
                )
            self._state[model_id] = ModelState.READY
```

Replace the `unload()` method:

```python
    async def unload(self, model_id: str, *, force: bool = False) -> None:
        async with self._lock:
            cur = self._state.get(model_id)
            if cur is not None and cur is not ModelState.READY and not force:
                raise UnloadRefused(model_id, cur)
            watcher = self._watchers.pop(model_id, None)
            if watcher is not None and not watcher.done():
                watcher.cancel()
            proc = self._processes.get(model_id)
            if proc is None:
                self.gpus.release(model_id)
                self._state.pop(model_id, None)
                return
            self._state[model_id] = ModelState.UNLOADING
            if proc.returncode is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=UNLOAD_GRACE_SECONDS)
                except TimeoutError:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    await proc.wait()
            self._processes.pop(model_id, None)
            self._ports.pop(model_id, None)
            self._overrides.pop(model_id, None)
            self._state.pop(model_id, None)
            self.gpus.release(model_id)
```

- [ ] **Step 4: Re-run the state tests — verify they pass**

Run: `make test-unit ARGS="tests/unit/runtime/test_supervisor_state.py -v"`
Expected: all 9 tests PASS.

- [ ] **Step 5: Re-run the existing unload tests — verify no regression**

Run: `make test-unit ARGS="tests/unit/runtime/test_supervisor_unload.py -v"`
Expected: PASS. Note: the existing tests do `sup._processes["m1"] = proc` directly without setting `_state` — `unload()` will see `cur is None` (no state entry) and proceed without raising, preserving backward compat for tests that bypass `load()`.

- [ ] **Step 6: Re-run all runtime/supervisor unit tests**

Run: `make test-unit ARGS="tests/unit/runtime/ -v"`
Expected: PASS.

- [ ] **Step 7: Commit Task 1 + Task 2 together**

```bash
git add app/config.py app/runtime/supervisor.py tests/unit/runtime/test_supervisor_state.py
git commit -m "$(cat <<'EOF'
feat(supervisor): add LOADING/WARMING/READY state machine + UnloadRefused

Move per-model lifecycle state into the supervisor itself. Callers
must transition LOADING -> WARMING -> READY before unload() will
SIGTERM; otherwise UnloadRefused is raised. force=True bypasses for
the operator force-unload path.

State entries are only created by load() and cleared by _watch_exit()
or unload(), so direct _processes mutation in tests (no state entry)
still passes the gate — backward-compatible with the existing
test_supervisor_unload.py fixtures.

Also adds warmup_probe_timeout_s setting (default 60s, env override
VW_WARMUP_PROBE_TIMEOUT_S), wired in Task 4.

Spec: docs/superpowers/specs/2026-05-20-vllm-warden-unload-race-design.md
EOF
)"
```

---

## Task 3: Warmup probe module

**Files:**
- Create: `app/runtime/warmup_probe.py`
- Create: `tests/unit/runtime/test_warmup_probe.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/runtime/test_warmup_probe.py`:

```python
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.runtime.warmup_probe import ProbeResult, warmup_probe


class _FakeResponse:
    def __init__(self, status_code: int, json_body: dict):
        self.status_code = status_code
        self._json = json_body
        self.text = str(json_body)

    def json(self):
        return self._json


@pytest.mark.asyncio
async def test_probe_succeeds_on_200_with_choices():
    fake = _FakeResponse(200, {"choices": [{"text": " "}]})
    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=fake)):
        result = await warmup_probe(
            port=10001, served_model_name="m1", timeout_s=5.0
        )
    assert result.ok is True
    assert result.detail is None


@pytest.mark.asyncio
async def test_probe_fails_on_5xx():
    fake = _FakeResponse(503, {"error": "engine warming"})
    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=fake)):
        result = await warmup_probe(
            port=10001, served_model_name="m1", timeout_s=5.0
        )
    assert result.ok is False
    assert "503" in result.detail


@pytest.mark.asyncio
async def test_probe_fails_on_200_without_choices():
    fake = _FakeResponse(200, {"unexpected": "shape"})
    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=fake)):
        result = await warmup_probe(
            port=10001, served_model_name="m1", timeout_s=5.0
        )
    assert result.ok is False
    assert "choices" in result.detail


@pytest.mark.asyncio
async def test_probe_fails_on_timeout():
    with patch(
        "httpx.AsyncClient.post",
        new=AsyncMock(side_effect=httpx.ReadTimeout("timed out")),
    ):
        result = await warmup_probe(
            port=10001, served_model_name="m1", timeout_s=0.1
        )
    assert result.ok is False
    assert "timeout" in result.detail.lower()


@pytest.mark.asyncio
async def test_probe_fails_on_connection_error():
    with patch(
        "httpx.AsyncClient.post",
        new=AsyncMock(side_effect=httpx.ConnectError("refused")),
    ):
        result = await warmup_probe(
            port=10001, served_model_name="m1", timeout_s=5.0
        )
    assert result.ok is False
    assert "connect" in result.detail.lower()


@pytest.mark.asyncio
async def test_probe_sends_max_tokens_1():
    fake = _FakeResponse(200, {"choices": [{"text": " "}]})
    captured = {}

    async def fake_post(self, url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return fake

    with patch("httpx.AsyncClient.post", new=fake_post):
        await warmup_probe(port=10001, served_model_name="m1", timeout_s=5.0)
    assert captured["json"]["max_tokens"] == 1
    assert captured["json"]["model"] == "m1"
    assert captured["json"]["stream"] is False
    assert captured["url"].endswith("/v1/completions")
    assert "127.0.0.1:10001" in captured["url"]
```

- [ ] **Step 2: Run the tests — verify they fail**

Run: `make test-unit ARGS="tests/unit/runtime/test_warmup_probe.py -v"`
Expected: ImportError on `app.runtime.warmup_probe` → all tests fail to collect.

- [ ] **Step 3: Implement the module**

Create `app/runtime/warmup_probe.py`:

```python
"""Verify that a vLLM subprocess is actually serving before flipping
DB status to ``loaded``.

vLLM's ``/health`` returns 200 once the engine reports up, which can
happen BEFORE multimodal warmup (``_warmup_mm_processor``) completes.
A unload arriving in that window SIGTERMs an actively-warming
subprocess and aborts the load. Sending a cheap completion request
forces the engine to actually serve, closing the race window.

Spec: docs/superpowers/specs/2026-05-20-vllm-warden-unload-race-design.md
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    detail: str | None


async def warmup_probe(
    *, port: int, served_model_name: str, timeout_s: float
) -> ProbeResult:
    """Send one POST /v1/completions to localhost:port and report success.

    Success = HTTP 200 with a non-empty ``choices`` array in the response
    body. Any other outcome (non-2xx, timeout, malformed body, network
    error) returns ``ok=False`` with a short ``detail`` string suitable
    for ``models.last_error``.

    Does NOT kill the subprocess on failure — the caller decides cleanup.
    """
    url = f"http://127.0.0.1:{port}/v1/completions"
    payload = {
        "model": served_model_name,
        "prompt": " ",
        "max_tokens": 1,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(url, json=payload, timeout=timeout_s)
    except httpx.ReadTimeout:
        return ProbeResult(False, "warmup probe timeout")
    except httpx.ConnectError as e:
        return ProbeResult(False, f"warmup probe connect error: {e}")
    except Exception as e:  # noqa: BLE001 — best-effort classification
        return ProbeResult(False, f"warmup probe error: {e!r}")

    if r.status_code != 200:
        return ProbeResult(
            False,
            f"warmup probe HTTP {r.status_code}: {r.text[:200]}",
        )
    try:
        body = r.json()
    except Exception as e:  # noqa: BLE001
        return ProbeResult(False, f"warmup probe non-JSON body: {e!r}")
    choices = body.get("choices") if isinstance(body, dict) else None
    if not choices:
        return ProbeResult(
            False, "warmup probe response missing 'choices' array"
        )
    return ProbeResult(True, None)
```

- [ ] **Step 4: Re-run probe tests — verify they pass**

Run: `make test-unit ARGS="tests/unit/runtime/test_warmup_probe.py -v"`
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/runtime/warmup_probe.py tests/unit/runtime/test_warmup_probe.py
git commit -m "$(cat <<'EOF'
feat(supervisor): add warmup verification probe

POST /v1/completions max_tokens=1 against the loaded vLLM port,
returning ProbeResult(ok, detail). Closes the window between
/health 200 and actual serving readiness. Used by Task 4 in the
load runner before flipping DB status to 'loaded'.

Spec: docs/superpowers/specs/2026-05-20-vllm-warden-unload-race-design.md
EOF
)"
```

---

## Task 4: Load route restructure + force-unload param

**Files:**
- Modify: `app/models/routes_api.py:627-707` (load + unload routes)
- Modify: `tests/unit/models/test_load_endpoint.py` (augment existing tests)

- [ ] **Step 1: Write failing tests for the new route behavior**

Append to `tests/unit/models/test_load_endpoint.py`:

```python
def test_load_runs_warmup_probe_before_flipping_to_loaded(tmp_data_dir, client):
    """Status must remain 'loading' until the warmup probe succeeds, not
    just when /health returns 200. Regression for 2026-05-20 Qwen3-VL
    crash loop."""
    client.get("/healthz")
    _seed_done_with_pulled_model(
        tmp_data_dir / "vllm-warden.db", allowed=[0, 1, 2, 3], gpus=[0, 1],
        model_id="probe-model",
    )
    auth = _jwt_login(client)

    from app.runtime.warmup_probe import ProbeResult
    sup_load = AsyncMock()
    sup_mark_warming = AsyncMock()
    sup_mark_ready = AsyncMock()
    health = AsyncMock(return_value=True)
    probe = AsyncMock(return_value=ProbeResult(ok=True, detail=None))

    with patch("app.runtime.supervisor.Supervisor.load", new=sup_load), \
         patch("app.runtime.supervisor.Supervisor.mark_warming", new=sup_mark_warming), \
         patch("app.runtime.supervisor.Supervisor.mark_ready", new=sup_mark_ready), \
         patch("app.models.routes_api.wait_for_health", new=health), \
         patch("app.models.routes_api.warmup_probe", new=probe):
        r = client.post(
            "/api/models/probe-model/load",
            headers={**auth, **csrf_header(client)},
        )
    assert r.status_code == 202

    # Wait for the background runner to complete
    for _ in range(50):
        if probe.await_count >= 1:
            break
        time.sleep(0.05)
    assert probe.await_count == 1
    sup_mark_warming.assert_awaited_once()
    sup_mark_ready.assert_awaited_once()
    # Order matters: mark_warming before probe before mark_ready
    # (We can't assert call order across separate mocks easily, but probe
    # success is what gates mark_ready, so checking both were called is
    # sufficient.)


def test_load_probe_failure_marks_failed_without_unloading(tmp_data_dir, client):
    """When the warmup probe fails, the model row goes to 'failed' but
    the subprocess is left running (no SIGTERM). Operator must
    force-unload to release GPUs."""
    client.get("/healthz")
    _seed_done_with_pulled_model(
        tmp_data_dir / "vllm-warden.db", allowed=[0, 1, 2, 3], gpus=[0, 1],
        model_id="probe-fail-model",
    )
    auth = _jwt_login(client)

    from app.runtime.warmup_probe import ProbeResult
    sup_load = AsyncMock()
    sup_unload = AsyncMock()
    sup_mark_warming = AsyncMock()
    health = AsyncMock(return_value=True)
    probe = AsyncMock(return_value=ProbeResult(ok=False, detail="HTTP 503"))

    with patch("app.runtime.supervisor.Supervisor.load", new=sup_load), \
         patch("app.runtime.supervisor.Supervisor.unload", new=sup_unload), \
         patch("app.runtime.supervisor.Supervisor.mark_warming", new=sup_mark_warming), \
         patch("app.models.routes_api.wait_for_health", new=health), \
         patch("app.models.routes_api.warmup_probe", new=probe):
        r = client.post(
            "/api/models/probe-fail-model/load",
            headers={**auth, **csrf_header(client)},
        )
    assert r.status_code == 202
    for _ in range(50):
        if probe.await_count >= 1:
            break
        time.sleep(0.05)

    # Status should be 'failed', subprocess NOT unloaded
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        row = db.execute(
            "SELECT status, last_error FROM models WHERE id = 'probe-fail-model'"
        ).fetchone()
    assert row[0] == "failed"
    assert "503" in row[1]
    sup_unload.assert_not_awaited()


def test_load_health_timeout_marks_failed_without_unloading(tmp_data_dir, client):
    """When wait_for_health times out, the model row goes to 'failed'
    but the subprocess is left running. Removes the legacy auto-SIGTERM
    that was the original race trigger."""
    client.get("/healthz")
    _seed_done_with_pulled_model(
        tmp_data_dir / "vllm-warden.db", allowed=[0, 1, 2, 3], gpus=[0, 1],
        model_id="health-timeout-model",
    )
    auth = _jwt_login(client)

    sup_load = AsyncMock()
    sup_unload = AsyncMock()
    health = AsyncMock(return_value=False)  # health never goes green

    with patch("app.runtime.supervisor.Supervisor.load", new=sup_load), \
         patch("app.runtime.supervisor.Supervisor.unload", new=sup_unload), \
         patch("app.models.routes_api.wait_for_health", new=health):
        r = client.post(
            "/api/models/health-timeout-model/load",
            headers={**auth, **csrf_header(client)},
        )
    assert r.status_code == 202
    for _ in range(50):
        if health.await_count >= 1:
            break
        time.sleep(0.05)

    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        row = db.execute(
            "SELECT status, last_error FROM models WHERE id = 'health-timeout-model'"
        ).fetchone()
    assert row[0] == "failed"
    assert "health timeout" in row[1]
    sup_unload.assert_not_awaited()


def test_unload_returns_409_when_supervisor_refuses(tmp_data_dir, client):
    """When the supervisor raises UnloadRefused, the route returns 409
    with the current state in the body."""
    from app.runtime.supervisor import ModelState, UnloadRefused

    client.get("/healthz")
    _seed_done_with_pulled_model(
        tmp_data_dir / "vllm-warden.db", allowed=[0, 1], gpus=[0],
        model_id="refused-model",
    )
    # Seed DB status as 'loaded' so the route gets past its own status check
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute("UPDATE models SET status='loaded' WHERE id='refused-model'")
        db.commit()
    auth = _jwt_login(client)

    sup_unload = AsyncMock(
        side_effect=UnloadRefused("refused-model", ModelState.WARMING)
    )
    with patch("app.runtime.supervisor.Supervisor.unload", new=sup_unload):
        r = client.post(
            "/api/models/refused-model/unload",
            headers={**auth, **csrf_header(client)},
        )
    assert r.status_code == 409
    assert "WARMING" in r.json()["detail"]


def test_unload_force_query_param_bypasses_refusal(tmp_data_dir, client):
    """?force=true must call sup.unload with force=True and succeed even
    when the supervisor would otherwise refuse."""
    client.get("/healthz")
    _seed_done_with_pulled_model(
        tmp_data_dir / "vllm-warden.db", allowed=[0, 1], gpus=[0],
        model_id="force-model",
    )
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute("UPDATE models SET status='loaded' WHERE id='force-model'")
        db.commit()
    auth = _jwt_login(client)

    captured = {}

    async def fake_unload(self, model_id, *, force=False):
        captured["force"] = force

    with patch("app.runtime.supervisor.Supervisor.unload", new=fake_unload):
        r = client.post(
            "/api/models/force-model/unload?force=true",
            headers={**auth, **csrf_header(client)},
        )
    assert r.status_code == 202
    assert captured["force"] is True
```

- [ ] **Step 2: Run the tests — verify they fail**

Run: `make test-unit ARGS="tests/unit/models/test_load_endpoint.py -v -k 'probe or refused or force or health_timeout'"`
Expected: tests fail (imports may pass, but routes don't call probe / don't accept force query param yet).

- [ ] **Step 3: Implement the load route restructure**

In `app/models/routes_api.py`, add to imports at the top of the file:

```python
from app.runtime.supervisor import UnloadRefused
from app.runtime.warmup_probe import warmup_probe
```

Replace the `runner()` body inside the `load_model` route (`app/models/routes_api.py:660-681`) with:

```python
    async def runner():
        try:
            await sup.load(model, port=port, on_exit=on_exit)
        except Exception as e:
            async with open_db(settings.db_path) as db:
                await ModelRepo(db).update_status(model_id, "failed", last_error=str(e))
            port_alloc.release(port)
            return
        ok = await wait_for_health(port=port, timeout_s=settings.load_timeout_s)
        if not ok:
            # Subprocess is left running; operator must force-unload to
            # release GPUs. This is intentional (spec §"Load-route changes")
            # — auto-SIGTERM here was the original race trigger.
            async with open_db(settings.db_path) as db:
                await ModelRepo(db).update_status(
                    model_id,
                    "failed",
                    last_error=(
                        "health timeout; subprocess still holding GPUs — "
                        "force-unload to release"
                    ),
                )
            return
        await sup.mark_warming(model_id)
        probe_result = await warmup_probe(
            port=port,
            served_model_name=model.served_model_name,
            timeout_s=settings.warmup_probe_timeout_s,
        )
        if not probe_result.ok:
            async with open_db(settings.db_path) as db:
                await ModelRepo(db).update_status(
                    model_id,
                    "failed",
                    last_error=(
                        f"{probe_result.detail}; subprocess still holding "
                        f"GPUs — force-unload to release"
                    ),
                )
            return
        await sup.mark_ready(model_id)
        async with open_db(settings.db_path) as db:
            await ModelRepo(db).update_status(model_id, "loaded")
            await RuntimeRepo(db).upsert(
                model_id,
                pid=sup._processes[model_id].pid,
                port=port,
                started_at=datetime.now(UTC).isoformat(),
            )
```

Note: the two `await sup.unload(model_id)` calls at the old `routes_api.py:680` and inside the `else` branch are removed. The `port_alloc.release(port)` on failure paths is also removed because the subprocess is left running and still owns its port; release happens when the operator force-unloads.

- [ ] **Step 4: Implement the force-unload query param + UnloadRefused → 409**

Replace the `unload_model` route signature and body (`app/models/routes_api.py:687-707`):

```python
@router.post("/{model_id}/unload", status_code=202)
async def unload_model(
    model_id: str,
    request: Request,
    force: bool = False,
    _user: str = Depends(require_jwt),
):
    settings = request.app.state.settings
    sup = request.app.state.supervisor
    port_alloc = request.app.state.port_allocator
    async with open_db(settings.db_path) as db:
        model = await ModelRepo(db).get(model_id)
        if not model:
            raise HTTPException(404, "not found")
        if model.status not in ("loaded", "failed"):
            raise HTTPException(409, f"cannot unload from status '{model.status}'")
        await ModelRepo(db).update_status(model_id, "unloading")
        rt = await RuntimeRepo(db).get(model_id)
        port = rt.port if rt else None
    try:
        await sup.unload(model_id, force=force)
    except UnloadRefused as e:
        # Roll back the status flip — caller can retry with ?force=true
        async with open_db(settings.db_path) as db:
            await ModelRepo(db).update_status(model_id, model.status)
        raise HTTPException(
            409,
            (
                f"refused: supervisor state is {e.state.name}; "
                f"use ?force=true to override"
            ),
        )
    if port:
        port_alloc.release(port)
    async with open_db(settings.db_path) as db:
        await RuntimeRepo(db).clear(model_id)
        await ModelRepo(db).update_status(model_id, "pulled")
    return {"status": "unloaded"}
```

- [ ] **Step 5: Re-run the new route tests — verify they pass**

Run: `make test-unit ARGS="tests/unit/models/test_load_endpoint.py -v -k 'probe or refused or force or health_timeout'"`
Expected: all 5 new tests PASS.

- [ ] **Step 6: Re-run all load_endpoint tests — verify no regression**

Run: `make test-unit ARGS="tests/unit/models/test_load_endpoint.py -v"`
Expected: PASS (including the original `test_load_calls_supervisor_then_health_check` and `test_on_exit_callback_flips_status_to_failed_and_releases_port`).

Note: the original `test_load_calls_supervisor_then_health_check` may need a minor update — it patches `wait_for_health` to True but doesn't patch `warmup_probe`. The default mock behavior may cause it to hang trying a real httpx call. If it fails, add the patches for `mark_warming`, `warmup_probe`, and `mark_ready`:

```python
from app.runtime.warmup_probe import ProbeResult
# ...
probe = AsyncMock(return_value=ProbeResult(ok=True, detail=None))
with patch("app.runtime.supervisor.Supervisor.load", new=sup_load), \
     patch("app.runtime.supervisor.Supervisor.mark_warming", new=AsyncMock()), \
     patch("app.runtime.supervisor.Supervisor.mark_ready", new=AsyncMock()), \
     patch("app.models.routes_api.wait_for_health", new=health), \
     patch("app.models.routes_api.warmup_probe", new=probe):
```

Update the test only if it fails. Otherwise leave it.

- [ ] **Step 7: Commit**

```bash
git add app/models/routes_api.py tests/unit/models/test_load_endpoint.py
git commit -m "$(cat <<'EOF'
feat(routes): warmup probe + force-unload query param

Restructure the load runner to:
  1. wait_for_health -> mark_warming -> warmup_probe -> mark_ready
  2. on any failure (health timeout, probe fail), mark DB failed but
     DO NOT call sup.unload — the subprocess is left running and the
     operator must explicitly force-unload to release GPUs. This
     removes the legacy auto-SIGTERM path that was the original race
     trigger.

Unload route now:
  - accepts ?force=true query param, passed through to sup.unload
  - catches UnloadRefused and returns 409 with the current state in
    the body, rolling back the 'unloading' status flip

Spec: docs/superpowers/specs/2026-05-20-vllm-warden-unload-race-design.md
EOF
)"
```

---

## Task 5: Bench supervisor — defensive `force=True`

**Files:**
- Modify: `app/bench/supervisor.py` (lines 1014, 1050, 1064, 1160, 1193)

- [ ] **Step 1: Read the bench supervisor's unload call sites**

Open `app/bench/supervisor.py` and locate every `await sup.unload(model_id)` call. There are five (at line offsets 1014, 1050, 1064, 1160, 1193 in the current file — verify with `grep -n "sup.unload\|self.runtime_supervisor.unload" app/bench/supervisor.py`).

Each of these is a bench-supervisor-coordinated unload where the bench has its own state machine and knows it is safe to terminate. They should bypass the new state gate because the bench may legitimately need to terminate a model that hasn't yet reached READY (e.g. when a load-config attempt times out partway through).

- [ ] **Step 2: Replace each call site with `force=True`**

Run: `grep -n "await sup.unload(model_id)" /home/ip/projects/vllm-warden/app/bench/supervisor.py`

For each result, change `await sup.unload(model_id)` to `await sup.unload(model_id, force=True)`. There should be five replacement sites. Use sed if you prefer:

```bash
sed -i 's|await sup\.unload(model_id)|await sup.unload(model_id, force=True)|g' app/bench/supervisor.py
```

Verify exactly five lines changed:

```bash
git diff --stat app/bench/supervisor.py
# Expect: 5 insertions, 5 deletions on app/bench/supervisor.py
grep -c "force=True" app/bench/supervisor.py
# Expect: 5
```

- [ ] **Step 3: Run bench supervisor tests — verify no regression**

Run: `make test-unit ARGS="tests/unit/bench/ -v"`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add app/bench/supervisor.py
git commit -m "$(cat <<'EOF'
fix(bench): pass force=True on internal sup.unload calls

The new supervisor state gate refuses unload from LOADING/WARMING
without force=True. Bench-supervisor-coordinated unloads happen
during apply_load_config and restore_after_run where bench owns
the lifecycle — explicitly force=True documents that the bench
caller knows what it's doing and bypasses the UI-protection gate.

Spec: docs/superpowers/specs/2026-05-20-vllm-warden-unload-race-design.md
EOF
)"
```

---

## Task 6: Integration test — mock vLLM subprocess

**Files:**
- Create: `tests/integration/test_load_lifecycle.py`

This test runs a tiny FastAPI app that impersonates vLLM's `/health` and `/v1/completions` endpoints with a programmable warmup delay. It verifies the end-to-end flow: load → health green → probe → ready, and a parallel unload attempt during the probe window is refused.

- [ ] **Step 1: Read existing integration test conventions**

Run: `cat tests/integration/conftest.py | head -60`
Expected: see how integration tests are bootstrapped (Settings fixture, client, etc.). Mirror that conventions in the new file.

- [ ] **Step 2: Write the test**

Create `tests/integration/test_load_lifecycle.py`:

```python
"""End-to-end test: load runner + warmup probe + supervisor state gate.

Spawns a real subprocess (a tiny FastAPI app) that impersonates vLLM's
``/health`` and ``/v1/completions``. The fake's completions endpoint
returns 503 for the first N seconds and 200 after — simulating the
multimodal-warmup window where /health is already green. We verify:

1. DB status stays 'loading' during the 503 window.
2. An unload attempt during that window returns 409 (UnloadRefused).
3. After the probe succeeds, DB flips to 'loaded' and unload works.
"""
import asyncio
import json
import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from unittest.mock import patch

import bcrypt
import pytest


pytestmark = pytest.mark.integration


# A standalone uvicorn app that fakes vLLM's two relevant endpoints.
# Lives in a tempfile so the test owns its lifecycle.
FAKE_VLLM_SOURCE = textwrap.dedent("""
    import os
    import time
    from fastapi import FastAPI, Response
    import uvicorn

    START = time.monotonic()
    WARMUP_DELAY = float(os.environ.get("FAKE_VLLM_WARMUP_S", "5.0"))
    PORT = int(os.environ["FAKE_VLLM_PORT"])

    app = FastAPI()

    @app.get("/health")
    async def health():
        return Response(status_code=200)

    @app.post("/v1/completions")
    async def completions(body: dict):
        elapsed = time.monotonic() - START
        if elapsed < WARMUP_DELAY:
            return Response(status_code=503, content='{"error":"warming"}',
                            media_type="application/json")
        return {"choices": [{"text": " "}], "model": body.get("model")}

    if __name__ == "__main__":
        uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="error")
""")


@pytest.fixture
def fake_vllm(tmp_path):
    """Spawn the fake vLLM on a free port; tear down after the test."""
    # Pick a free port
    import socket
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    src = tmp_path / "fake_vllm.py"
    src.write_text(FAKE_VLLM_SOURCE)

    env = {
        **__import__("os").environ,
        "FAKE_VLLM_PORT": str(port),
        "FAKE_VLLM_WARMUP_S": "3.0",
    }
    proc = subprocess.Popen(
        [sys.executable, str(src)], env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # Wait until it's listening
    import socket as _s
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            with _s.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.05)
    else:
        proc.kill()
        raise RuntimeError("fake vllm never came up")
    yield port
    proc.terminate()
    proc.wait(timeout=5)


@pytest.mark.asyncio
async def test_status_stays_loading_during_probe_window(fake_vllm):
    """The race window between /health 200 and probe success must keep
    DB status at 'loading', and an unload attempt in that window must
    return 409."""
    from app.config import Settings
    from app.runtime.supervisor import ModelState, Supervisor, UnloadRefused
    from app.runtime.warmup_probe import warmup_probe

    port = fake_vllm
    # Health should be 200 immediately
    import httpx
    async with httpx.AsyncClient() as c:
        r = await c.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
    assert r.status_code == 200

    # Probe attempted now should fail (warmup still pending)
    early = await warmup_probe(
        port=port, served_model_name="fake", timeout_s=0.5
    )
    assert early.ok is False

    # Wait for warmup window to end, then probe again
    await asyncio.sleep(3.5)
    late = await warmup_probe(
        port=port, served_model_name="fake", timeout_s=2.0
    )
    assert late.ok is True


@pytest.mark.asyncio
async def test_supervisor_unload_refused_during_warming(tmp_path):
    """Direct unit test of the state gate at the supervisor level —
    integration-marked because it exercises the full Supervisor class
    without mocking _state."""
    from app.runtime.supervisor import ModelState, Supervisor, UnloadRefused

    class _Settings:
        pass

    s = _Settings()
    s.data_dir = str(tmp_path)
    s.hf_token_path = str(tmp_path / "tok")
    (tmp_path / "tok").write_text("hf_x")
    sup = Supervisor(s)

    from unittest.mock import AsyncMock, MagicMock
    proc = MagicMock()
    proc.pid = 9999
    proc.returncode = None
    proc.wait = AsyncMock(return_value=0)
    sup._processes["m1"] = proc
    sup._state["m1"] = ModelState.WARMING
    sup.gpus.claim("m1", [0])

    with pytest.raises(UnloadRefused) as exc:
        await sup.unload("m1")
    assert exc.value.state == ModelState.WARMING

    # Process still registered, no SIGTERM
    assert "m1" in sup._processes
    proc.wait.assert_not_called()

    # Force bypass works
    from unittest.mock import patch
    with patch("os.killpg"):
        await sup.unload("m1", force=True)
    assert "m1" not in sup._processes
```

- [ ] **Step 3: Run the integration test**

Run: `make test-integration ARGS="tests/integration/test_load_lifecycle.py -v"`
Expected: both tests PASS.

If the fake-vllm spawn fails because `uvicorn` isn't installed in the test image, add it to `requirements-dev.txt` if not already present:

```bash
grep -q '^uvicorn' requirements-dev.txt || echo 'uvicorn' >> requirements-dev.txt
```

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_load_lifecycle.py
# Only add requirements-dev.txt if it was modified
git status --short requirements-dev.txt | grep -q M && git add requirements-dev.txt
git commit -m "$(cat <<'EOF'
test(integration): end-to-end load-lifecycle with mock vLLM

Spawns a real subprocess that fakes vLLM's /health (always 200)
and /v1/completions (503 for first 3s, then 200). Verifies the
warmup probe correctly distinguishes the two phases, and that
sup.unload() raises UnloadRefused during WARMING but succeeds
with force=True.

This is the regression test for the 2026-05-20 Qwen3-VL crash
loop: before the fix, an unload click during the 8-second window
between /health 200 and multimodal-warmup completion SIGTERMed
the subprocess. With the fix, that click 409s.

Spec: docs/superpowers/specs/2026-05-20-vllm-warden-unload-race-design.md
EOF
)"
```

---

## Task 7: Changelog + UI affordance note

**Files:**
- Modify: `changelog.md`

The actual UI affordance (a "Force unload" button in the model row when `last_error` contains `subprocess still holding`) is a follow-up ticket. This task only documents the API behavior. Operators can hit `POST /api/models/{id}/unload?force=true` directly until the UI catches up.

- [ ] **Step 1: Add the changelog entry**

Edit `changelog.md`. Under `## [Unreleased]`, before the existing `### Fixed` section added in commit `909b866`, add a new `### Changed` section:

```markdown
### Changed
- **Supervisor refuses unload during model load/warmup.** The vLLM
  subprocess lifecycle now goes `LOADING → WARMING → READY → UNLOADING`
  inside the supervisor. Calling `POST /api/models/{id}/unload` while
  the model is not yet `READY` returns 409 with the current state in
  the body. To force-terminate a stuck or still-warming model, use
  `POST /api/models/{id}/unload?force=true`. A warmup verification
  probe (`POST /v1/completions max_tokens=1`) runs after vLLM's
  `/health` returns 200 — DB status only flips to `loaded` once that
  probe succeeds, closing the race window where Qwen3-VL's
  `_warmup_mm_processor` was still running but the row looked
  serviceable. When the probe or health-wait fails, the row is marked
  `failed` and the subprocess is **left running** holding its GPUs
  until an explicit `?force=true` unload — operators must check
  `last_error` (contains `subprocess still holding`) and decide
  whether to retry or release. The `warmup_probe_timeout_s` setting
  (default 60s, env `VW_WARMUP_PROBE_TIMEOUT_S`) controls the probe
  budget. See spec
  `docs/superpowers/specs/2026-05-20-vllm-warden-unload-race-design.md`.
```

- [ ] **Step 2: Commit**

```bash
git add changelog.md
git commit -m "$(cat <<'EOF'
docs(changelog): supervisor state machine + force-unload behavior

Records the user-visible API change: unload during LOADING/WARMING
now returns 409, force=true bypass, warmup probe gating, and the
new 'failed-with-subprocess-still-running' state operators must
handle by force-unload.
EOF
)"
```

---

## Task 8: Manual smoke test on warden

**Files:** none (manual test against deployed image)

This is the acceptance test before merging. It must run against the actual warden host and exercise the real Qwen3-VL load path that triggered the original incident.

- [ ] **Step 1: Build the new image**

On the warden host (10.10.0.187):

```bash
ssh ip@10.10.0.187 'bash -c "pushd /home/ip/projects/vllm-warden >/dev/null && sudo docker compose build api"'
```

Sync the new source files to the warden (or pull the feature branch directly there once pushed).

- [ ] **Step 2: Restart and wait for healthy**

```bash
ssh ip@10.10.0.187 'sudo docker compose restart api && sleep 5 && sudo docker compose ps'
```

Expected: `vllm-warden-api-1` shows `Up`.

- [ ] **Step 3: Load Qwen3-VL and race the unload click**

```bash
# Start load via API (use existing model row 0607dafd536baa30)
curl -X POST https://warden.h:8080/api/models/0607dafd536baa30/load \
  -H "Authorization: Bearer $VW_JWT"

# Watch logs; once you see "/health" return 200, immediately call unload
ssh ip@10.10.0.187 'sudo docker exec vllm-warden-api-1 tail -f /data/logs/0607dafd536baa30.log' &
TAIL_PID=$!

# Wait for /health (in another shell, or after seeing the log line)
# Then:
curl -X POST https://warden.h:8080/api/models/0607dafd536baa30/unload \
  -H "Authorization: Bearer $VW_JWT" \
  -w "\nHTTP %{http_code}\n"
```

Expected output of the unload curl: `HTTP 409` with body mentioning `state is WARMING`.
Expected in vLLM log: NO `KeyboardInterrupt("terminated")`. The warmup proceeds.

- [ ] **Step 4: Verify load completes**

After ~10–60 seconds, the warmup probe should succeed and DB status flips to `loaded`. Verify:

```bash
ssh ip@10.10.0.187 'sudo sqlite3 /data/vllm-warden.db \
  "SELECT id, status, last_error FROM models WHERE id = '\''0607dafd536baa30'\''"'
```

Expected: `status=loaded`, `last_error=NULL`.

- [ ] **Step 5: Verify ready-state unload works**

```bash
curl -X POST https://warden.h:8080/api/models/0607dafd536baa30/unload \
  -H "Authorization: Bearer $VW_JWT" \
  -w "\nHTTP %{http_code}\n"
```

Expected: `HTTP 202`, GPUs released, `nvidia-smi` shows `0 MiB` on the four target GPUs within 30s.

- [ ] **Step 6: Verify force-unload from a failed/stuck load**

Force a slow probe failure (set `VW_WARMUP_PROBE_TIMEOUT_S=1` in compose env for this run only) and reload. Verify:
- DB status flips to `failed` with `last_error` containing `subprocess still holding`
- `nvidia-smi` still shows GPU memory in use (subprocess alive)
- `POST /unload?force=true` returns 202 and releases the GPUs

Restore the env var to default after the test.

- [ ] **Step 7: Mark the task complete**

If all six smoke steps pass, mark the PR as ready for review. Open the MR against `develop`:

```bash
git push -u origin feature/supervisor-unload-state-machine
glab mr create --title "fix: eliminate unload-during-warmup race" \
  --description-file docs/superpowers/specs/2026-05-20-vllm-warden-unload-race-design.md \
  --target-branch develop
```
