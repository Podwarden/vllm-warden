# P1: Engine-Driver Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce an `EngineDriver` abstraction so the vLLM engine can run either as today's in-container subprocess OR as a sibling container via the host Docker socket (DooD), with the in-container path remaining the default and `Supervisor`'s public API unchanged.

**Architecture:** `Supervisor` keeps ownership of locking, the LOADING→WARMING→READY→UNLOADING state machine, GPU ownership, and exit-watchers. It delegates the three engine primitives — *spawn*, *observe exit*, *terminate* — to an injected `EngineDriver`. A driver returns an `EngineHandle` (exposes `pid`, `returncode`, `wait()`); `Supervisor` stores handles instead of raw `asyncio.subprocess.Process`. Two drivers ship: `LocalSubprocessDriver` (current behaviour, default) and `DockerSocketDriver` (sibling container). Implements issue #160.

**Tech Stack:** Python 3.10, asyncio, FastAPI, pytest. Docker engine via the `docker` Python SDK (already a transitive dep through the engine image tooling — confirm in Task 4) or `docker` CLI subprocess fallback.

---

## File Structure

- `app/runtime/engine/__init__.py` (Create) — `EngineDriver` Protocol + `EngineHandle` Protocol + `EngineSpec` dataclass. The contract; no logic.
- `app/runtime/engine/local_subprocess.py` (Create) — `LocalSubprocessDriver` + `LocalHandle`. Extracts the subprocess spawn/killpg logic currently inline in `supervisor.py`.
- `app/runtime/engine/docker_socket.py` (Create) — `DockerSocketDriver` + `DockerHandle`. Runs the engine as a sibling container over `/var/run/docker.sock`.
- `app/runtime/supervisor.py` (Modify) — delegate spawn/observe/terminate to `self._driver`; store `EngineHandle` in `_processes`→renamed `_handles`.
- `app/settings.py` (Modify) — add `engine_driver: str = "local"`.
- `app/main.py:44` (Modify) — pick driver from settings when constructing `Supervisor`.
- `tests/fakes/fake_engine.py` (Create) — `FakeDriver`/`FakeHandle` for Supervisor unit tests.
- `tests/unit/runtime/engine/test_local_subprocess.py` (Create)
- `tests/unit/runtime/engine/test_docker_socket.py` (Create)
- `tests/unit/runtime/test_supervisor_driver.py` (Create)

---

## Task 1: Engine driver contract

**Files:**
- Create: `app/runtime/engine/__init__.py`
- Test: `tests/unit/runtime/engine/test_contract.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/runtime/engine/test_contract.py
from app.runtime.engine import EngineSpec, EngineDriver, EngineHandle


def test_engine_spec_defaults():
    spec = EngineSpec(model_arg="openai/gpt-oss-20b", args=["--port", "8001"],
                      env={"VLLM_USE_V1": "1"}, port=8001, model_id="m1")
    assert spec.image is None          # local driver ignores image
    assert spec.gpu_indices == []      # default no pinning


def test_protocols_are_runtime_checkable():
    # Protocols must be importable and usable in isinstance checks.
    assert hasattr(EngineDriver, "spawn")
    assert hasattr(EngineHandle, "wait")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm -v $(pwd):/app -w /app vllm-warden-test pytest tests/unit/runtime/engine/test_contract.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.runtime.engine'`

- [ ] **Step 3: Write minimal implementation**

```python
# app/runtime/engine/__init__.py
"""Engine driver contract: how the vLLM engine is spawned/observed/killed.

Supervisor owns lifecycle state, locking, GPU ownership and exit-watchers.
A driver owns only the three engine primitives below. This lets the engine
run as an in-container subprocess (LocalSubprocessDriver, default) or as a
sibling container via the host Docker socket (DockerSocketDriver) without
the Supervisor caring which. Implements #160.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class EngineSpec:
    """Everything a driver needs to start one engine. Built by Supervisor
    from a model row + the resolved argv; drivers stay model-agnostic."""
    model_id: str
    model_arg: str            # the --model value (repo or repo:quant)
    args: list[str]           # full `vllm serve` argv tail (post-binary)
    env: dict[str, str]       # process/container env
    port: int                 # host port the engine listens on
    image: str | None = None  # engine container image (docker driver only)
    gpu_indices: list[int] = field(default_factory=list)


@runtime_checkable
class EngineHandle(Protocol):
    """Live reference to one running engine."""

    @property
    def pid(self) -> int | None:
        """Host PID of the engine's main process for GPU attribution.
        None if not yet known / not applicable."""

    @property
    def returncode(self) -> int | None:
        """Exit code once exited, else None."""

    async def wait(self) -> int:
        """Block until the engine exits, return its exit code."""


@runtime_checkable
class EngineDriver(Protocol):
    async def spawn(self, spec: EngineSpec) -> EngineHandle:
        """Start the engine described by spec and return its handle."""

    async def terminate(self, handle: EngineHandle, *, grace_s: float) -> None:
        """Graceful stop (SIGTERM/`docker stop`); SIGKILL after grace_s."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker run --rm -v $(pwd):/app -w /app vllm-warden-test pytest tests/unit/runtime/engine/test_contract.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add app/runtime/engine/__init__.py tests/unit/runtime/engine/test_contract.py
git commit -m "feat(#160): engine driver contract (EngineSpec/EngineDriver/EngineHandle)"
```

---

## Task 2: LocalSubprocessDriver (extract current behaviour)

**Files:**
- Create: `app/runtime/engine/local_subprocess.py`
- Test: `tests/unit/runtime/engine/test_local_subprocess.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/runtime/engine/test_local_subprocess.py
import asyncio
import pytest
from app.runtime.engine import EngineSpec
from app.runtime.engine.local_subprocess import LocalSubprocessDriver


@pytest.mark.asyncio
async def test_spawn_runs_command_and_handle_waits(tmp_path):
    # Use /bin/sh as a stand-in "engine" that exits 0 immediately.
    spec = EngineSpec(model_id="m1", model_arg="x",
                      args=["-c", "exit 0"], env={}, port=8001)
    driver = LocalSubprocessDriver(binary="/bin/sh", log_dir=str(tmp_path))
    handle = await driver.spawn(spec)
    assert handle.pid is not None
    rc = await handle.wait()
    assert rc == 0
    assert handle.returncode == 0


@pytest.mark.asyncio
async def test_terminate_kills_long_running(tmp_path):
    spec = EngineSpec(model_id="m2", model_arg="x",
                      args=["-c", "sleep 60"], env={}, port=8002)
    driver = LocalSubprocessDriver(binary="/bin/sh", log_dir=str(tmp_path))
    handle = await driver.spawn(spec)
    await driver.terminate(handle, grace_s=0.5)
    assert handle.returncode is not None  # exited after term/kill
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm -v $(pwd):/app -w /app vllm-warden-test pytest tests/unit/runtime/engine/test_local_subprocess.py -v`
Expected: FAIL with `ModuleNotFoundError: app.runtime.engine.local_subprocess`

- [ ] **Step 3: Write minimal implementation**

```python
# app/runtime/engine/local_subprocess.py
"""In-container subprocess engine driver — the default, identical in
behaviour to the pre-#160 inline Supervisor logic. Spawns `vllm serve`
(or an injected binary, for tests) in a new session, logs to a per-model
file, and stops it via process-group SIGTERM→SIGKILL."""
from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path

from app.runtime.engine import EngineHandle, EngineSpec


class LocalHandle:
    def __init__(self, proc: asyncio.subprocess.Process) -> None:
        self._proc = proc

    @property
    def pid(self) -> int | None:
        return self._proc.pid

    @property
    def returncode(self) -> int | None:
        return self._proc.returncode

    async def wait(self) -> int:
        return await self._proc.wait()


class LocalSubprocessDriver:
    def __init__(self, *, binary: str = "vllm", log_dir: str) -> None:
        self._binary = binary
        self._log_dir = Path(log_dir)

    async def spawn(self, spec: EngineSpec) -> LocalHandle:
        self._log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._log_dir / f"{spec.model_id}.log"
        log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            # `vllm serve` for the real binary; tests inject /bin/sh and pass
            # the subcommand inside args, so only prepend "serve" for vllm.
            head = [self._binary] + (["serve"] if self._binary == "vllm" else [])
            proc = await asyncio.create_subprocess_exec(
                *head, *spec.args,
                env={**os.environ, **spec.env},
                stdout=log_fd, stderr=log_fd,
                start_new_session=True,
            )
        finally:
            os.close(log_fd)
        return LocalHandle(proc)

    async def terminate(self, handle: LocalHandle, *, grace_s: float) -> None:
        pid = handle.pid
        if pid is None or handle.returncode is not None:
            return
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(handle.wait(), timeout=grace_s)
        except (TimeoutError, asyncio.TimeoutError):
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await handle.wait()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker run --rm -v $(pwd):/app -w /app vllm-warden-test pytest tests/unit/runtime/engine/test_local_subprocess.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add app/runtime/engine/local_subprocess.py tests/unit/runtime/engine/test_local_subprocess.py
git commit -m "feat(#160): LocalSubprocessDriver extracting in-container spawn/kill"
```

---

## Task 3: Refactor Supervisor to delegate to a driver

**Files:**
- Modify: `app/runtime/supervisor.py`
- Create: `tests/fakes/fake_engine.py`
- Create: `tests/unit/runtime/test_supervisor_driver.py`

- [ ] **Step 1: Write the fake driver**

```python
# tests/fakes/fake_engine.py
import asyncio
from app.runtime.engine import EngineSpec


class FakeHandle:
    def __init__(self):
        self._evt = asyncio.Event()
        self._rc = None
        self.pid = 4242

    @property
    def returncode(self):
        return self._rc

    async def wait(self):
        await self._evt.wait()
        return self._rc

    def _finish(self, rc: int):
        self._rc = rc
        self._evt.set()


class FakeDriver:
    def __init__(self):
        self.spawned: list[EngineSpec] = []
        self.handles: dict[str, FakeHandle] = {}

    async def spawn(self, spec: EngineSpec) -> FakeHandle:
        self.spawned.append(spec)
        h = FakeHandle()
        self.handles[spec.model_id] = h
        return h

    async def terminate(self, handle: FakeHandle, *, grace_s: float) -> None:
        handle._finish(0)
```

- [ ] **Step 2: Write the failing Supervisor test**

```python
# tests/unit/runtime/test_supervisor_driver.py
import pytest
from app.runtime.supervisor import Supervisor, ModelState
from tests.fakes.fake_engine import FakeDriver


class _Settings:
    def __init__(self, tmp_path):
        self.data_dir = str(tmp_path)
        self.hf_token_path = str(tmp_path / "tok")


class _M:
    id = "qwen"; hf_repo = "Qwen/Q"; hf_revision = "main"
    served_model_name = "q"; gpu_indices = [0]; tensor_parallel_size = 1
    max_model_len = 4096; dtype = "auto"; gpu_memory_utilization = 0.9
    extra_env = {}; extra_args = []


@pytest.mark.asyncio
async def test_load_uses_injected_driver(tmp_path):
    drv = FakeDriver()
    sup = Supervisor(_Settings(tmp_path), driver=drv)
    await sup.load(_M(), port=8001)
    assert len(drv.spawned) == 1
    assert drv.spawned[0].port == 8001
    assert sup.get_state("qwen") is ModelState.LOADING
    assert sup.get_pid("qwen") == 4242


@pytest.mark.asyncio
async def test_unload_terminates_via_driver(tmp_path):
    drv = FakeDriver()
    sup = Supervisor(_Settings(tmp_path), driver=drv)
    await sup.load(_M(), port=8001)
    await sup.unload("qwen", force=True)
    assert sup.is_running("qwen") is False
```

- [ ] **Step 3: Run to verify it fails**

Run: `docker run --rm -v $(pwd):/app -w /app vllm-warden-test pytest tests/unit/runtime/test_supervisor_driver.py -v`
Expected: FAIL — `Supervisor.__init__() got an unexpected keyword argument 'driver'`

- [ ] **Step 4: Refactor Supervisor**

In `app/runtime/supervisor.py`:

Replace the imports/top of `Supervisor.__init__`:

```python
from app.runtime.cmd_builder import build_vllm_args
from app.runtime.engine import EngineSpec
from app.runtime.engine.local_subprocess import LocalSubprocessDriver
from app.runtime.env_builder import build_subprocess_env
from app.runtime.gpu_ownership import GpuOwnership
```

```python
    def __init__(self, settings, *, driver=None) -> None:
        self.settings = settings
        self.gpus = GpuOwnership()
        self._driver = driver or LocalSubprocessDriver(
            log_dir=str(Path(settings.data_dir) / "logs")
        )
        self._handles: dict[str, object] = {}   # model_id -> EngineHandle
        self._ports: dict[str, int] = {}
        self._overrides: dict[str, dict | None] = {}
        self._watchers: dict[str, asyncio.Task] = {}
        self._state: dict[str, ModelState] = {}
        self._lock = asyncio.Lock()
```

Rewrite `load()`'s body inside the lock (replacing the subprocess block) to build a spec and call the driver:

```python
            self.gpus.claim(model.id, model.gpu_indices)
            try:
                hf_token_path = Path(self.settings.hf_token_path)

                def _read_hf_token() -> str:
                    return hf_token_path.read_text().strip() if hf_token_path.exists() else ""

                hf_token = await asyncio.to_thread(_read_hf_token)
                env = build_subprocess_env(model, hf_token=hf_token,
                                           data_dir=str(self.settings.data_dir))
                args = build_vllm_args(model, port=port, overrides=overrides)
                spec = EngineSpec(
                    model_id=model.id,
                    model_arg=model.hf_repo,
                    args=args,
                    env=env,
                    port=port,
                    gpu_indices=list(model.gpu_indices),
                )
                handle = await self._driver.spawn(spec)
                self._handles[model.id] = handle
                self._ports[model.id] = port
                self._overrides[model.id] = overrides
                self._state[model.id] = ModelState.LOADING
                self._watchers[model.id] = asyncio.create_task(
                    self._watch_exit(model.id, on_exit)
                )
            except Exception:
                self.gpus.release(model.id)
                raise
```

Update `_watch_exit`, `get_pid`, `is_running`, `parent_pid_to_model`, and `unload` to use `_handles` + the driver:

```python
    async def _watch_exit(self, model_id, on_exit):
        handle = self._handles.get(model_id)
        if handle is None:
            return
        try:
            rc = await handle.wait()
        except asyncio.CancelledError:
            return
        async with self._lock:
            if model_id not in self._handles:
                return
            self._handles.pop(model_id, None)
            self._ports.pop(model_id, None)
            self._overrides.pop(model_id, None)
            self._watchers.pop(model_id, None)
            self._state.pop(model_id, None)
            self.gpus.release(model_id)
        if on_exit is not None:
            await on_exit(rc)

    def get_pid(self, model_id):
        h = self._handles.get(model_id)
        return h.pid if h is not None else None

    def is_running(self, model_id):
        h = self._handles.get(model_id)
        return h is not None and h.returncode is None

    def parent_pid_to_model(self):
        out = {}
        for model_id, h in self._handles.items():
            if h.returncode is None and h.pid is not None:
                out[h.pid] = model_id
        return out
```

`unload()` — replace the `os.killpg(...)` grace/kill block with the driver:

```python
            self._state[model_id] = ModelState.UNLOADING
            handle = self._handles.get(model_id)
            if handle is not None and handle.returncode is None:
                await self._driver.terminate(handle, grace_s=UNLOAD_GRACE_SECONDS)
            self._handles.pop(model_id, None)
            self._ports.pop(model_id, None)
            self._overrides.pop(model_id, None)
            self._state.pop(model_id, None)
            self.gpus.release(model_id)
```

(and the early `if proc is None:` guard becomes `if handle is None:` reading from `_handles`). Remove the now-unused `import os` / `import signal` only if nothing else needs them — `os` is still used elsewhere? It is not after this change; verify with grep before deleting.

- [ ] **Step 5: Run the new + existing supervisor tests**

Run: `docker run --rm -v $(pwd):/app -w /app vllm-warden-test pytest tests/unit/runtime/ -v`
Expected: PASS. Tests that previously poked `sup._processes` directly must be updated to `sup._handles` and to use `FakeDriver`/`FakeHandle` instead of MagicMock procs. Update each failing test minimally; do NOT delete coverage.

- [ ] **Step 6: Fix routes_api.py private-state read**

`app/models/routes_api.py:844` reads `sup._processes[model_id].pid`. Replace with the public accessor:

```python
                pid=sup.get_pid(model_id),
```

- [ ] **Step 7: Run full unit suite**

Run: `docker run --rm -v $(pwd):/app -w /app vllm-warden-test pytest tests/unit -q`
Expected: PASS (no regressions).

- [ ] **Step 8: Commit**

```bash
git add app/runtime/supervisor.py app/models/routes_api.py tests/fakes/fake_engine.py tests/unit/runtime/
git commit -m "refactor(#160): Supervisor delegates spawn/observe/terminate to EngineDriver"
```

---

## Task 4: DockerSocketDriver (sibling container via host docker socket)

**Files:**
- Create: `app/runtime/engine/docker_socket.py`
- Test: `tests/unit/runtime/engine/test_docker_socket.py`

**Design:** The driver talks to the host Docker daemon through the mounted `/var/run/docker.sock`. It runs the engine image with `--gpus` for the pinned indices, named volumes `vllm-warden-hfcache` and `vllm-warden-data` (bind paths would resolve on the host, not in our container — see architecture spec), publishes the engine port, and injects env. The handle wraps the container id; `pid` comes from `docker inspect .State.Pid` (the host PID, which is exactly what GPU attribution needs). `terminate` = `docker stop -t <grace>` then `docker rm -f`. The Docker client is abstracted behind a tiny `_DockerClient` Protocol so tests inject a fake — no daemon needed in unit tests.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/runtime/engine/test_docker_socket.py
import pytest
from app.runtime.engine import EngineSpec
from app.runtime.engine.docker_socket import DockerSocketDriver


class _FakeContainers:
    def __init__(self):
        self.run_kwargs = None
        self.removed = []

    def run(self, image, **kwargs):
        self.run_kwargs = {"image": image, **kwargs}
        return _FakeContainer("cid-1")


class _FakeContainer:
    def __init__(self, cid):
        self.id = cid
        self._stopped = False
        self.attrs = {"State": {"Pid": 9999, "Running": True, "ExitCode": 0}}

    def reload(self):
        if self._stopped:
            self.attrs["State"] = {"Pid": 0, "Running": False, "ExitCode": 0}

    def stop(self, timeout=None):
        self._stopped = True

    def remove(self, force=False):
        pass

    def wait(self):
        return {"StatusCode": 0}


class _FakeClient:
    def __init__(self):
        self.containers = _FakeContainers()


@pytest.mark.asyncio
async def test_spawn_runs_engine_image_with_gpus():
    client = _FakeClient()
    drv = DockerSocketDriver(client=client, image="vllm/vllm-openai:v0.20.0")
    spec = EngineSpec(model_id="m1", model_arg="openai/gpt-oss-20b",
                      args=["--model", "openai/gpt-oss-20b", "--port", "8001"],
                      env={"VLLM_USE_V1": "1"}, port=8001, gpu_indices=[0, 1])
    handle = await drv.spawn(spec)
    kw = client.containers.run_kwargs
    assert kw["image"] == "vllm/vllm-openai:v0.20.0"
    assert kw["detach"] is True
    assert "0,1" in str(kw["device_requests"]) or kw.get("gpus") == "0,1"
    assert kw["ports"] == {"8001/tcp": 8001}
    assert handle.pid == 9999


@pytest.mark.asyncio
async def test_terminate_stops_and_removes():
    client = _FakeClient()
    drv = DockerSocketDriver(client=client, image="img:tag")
    spec = EngineSpec(model_id="m1", model_arg="x", args=[], env={}, port=8001)
    handle = await drv.spawn(spec)
    await drv.terminate(handle, grace_s=1.0)
    assert handle.returncode == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `docker run --rm -v $(pwd):/app -w /app vllm-warden-test pytest tests/unit/runtime/engine/test_docker_socket.py -v`
Expected: FAIL — `ModuleNotFoundError: app.runtime.engine.docker_socket`

- [ ] **Step 3: Implement the driver**

```python
# app/runtime/engine/docker_socket.py
"""Sibling-container engine driver (Docker-out-of-Docker). Asks the HOST
docker daemon (via mounted /var/run/docker.sock) to run the engine image
as a sibling container. GPUs are mapped by the host NVIDIA Container
Toolkit; volumes MUST be named (bind paths resolve on the host). The
handle's pid is the engine container's HOST pid (docker inspect .State.Pid)
which is exactly what nvidia-smi attribution keys on. Implements #160."""
from __future__ import annotations

import asyncio

HFCACHE_VOLUME = "vllm-warden-hfcache"
DATA_VOLUME = "vllm-warden-data"


class DockerHandle:
    def __init__(self, container) -> None:
        self._c = container

    @property
    def pid(self) -> int | None:
        try:
            self._c.reload()
            pid = self._c.attrs.get("State", {}).get("Pid", 0)
            return pid or None
        except Exception:
            return None

    @property
    def returncode(self) -> int | None:
        try:
            self._c.reload()
            state = self._c.attrs.get("State", {})
            if state.get("Running"):
                return None
            return int(state.get("ExitCode", 0))
        except Exception:
            return None

    async def wait(self) -> int:
        res = await asyncio.to_thread(self._c.wait)
        return int(res.get("StatusCode", 0))


class DockerSocketDriver:
    def __init__(self, *, client, image: str) -> None:
        self._client = client
        self._image = image

    async def spawn(self, spec) -> DockerHandle:
        gpus = ",".join(str(i) for i in spec.gpu_indices) if spec.gpu_indices else "all"
        kwargs = dict(
            command=spec.args,
            detach=True,
            environment=dict(spec.env),
            ports={f"{spec.port}/tcp": spec.port},
            volumes={
                HFCACHE_VOLUME: {"bind": "/root/.cache/huggingface", "mode": "rw"},
                DATA_VOLUME: {"bind": "/data", "mode": "rw"},
            },
            gpus=gpus,                       # docker SDK >=7 accepts gpus=
            name=f"vllm-warden-engine-{spec.model_id}",
            labels={"vllm-warden.model_id": spec.model_id},
        )
        container = await asyncio.to_thread(
            self._client.containers.run, self._image, **kwargs
        )
        return DockerHandle(container)

    async def terminate(self, handle: DockerHandle, *, grace_s: float) -> None:
        def _stop_rm():
            try:
                handle._c.stop(timeout=int(grace_s))
            except Exception:
                pass
            try:
                handle._c.remove(force=True)
            except Exception:
                pass
        await asyncio.to_thread(_stop_rm)
```

> **Worker note:** the fake test asserts `kw.get("gpus") == "0,1"`; real `docker` SDK ≥7 supports `gpus=`. If the installed SDK is older (no `gpus=`), translate to `device_requests=[docker.types.DeviceRequest(device_ids=["0","1"], capabilities=[["gpu"]])]` and update the test's assertion branch accordingly — confirm the SDK version in the image first with `docker run --rm vllm-warden-test python -c "import docker; print(docker.__version__)"`.

- [ ] **Step 4: Run to verify it passes**

Run: `docker run --rm -v $(pwd):/app -w /app vllm-warden-test pytest tests/unit/runtime/engine/test_docker_socket.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add app/runtime/engine/docker_socket.py tests/unit/runtime/engine/test_docker_socket.py
git commit -m "feat(#160): DockerSocketDriver runs engine as sibling container (DooD)"
```

---

## Task 5: Driver selection from settings

**Files:**
- Modify: `app/settings.py`
- Modify: `app/main.py:44`
- Test: `tests/unit/test_app_state.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/unit/test_app_state.py
def test_engine_driver_setting_defaults_local():
    from app.settings import Settings
    s = Settings()  # adapt to the project's Settings constructor/fixture
    assert s.engine_driver == "local"
```

- [ ] **Step 2: Run to verify it fails**

Run: `docker run --rm -v $(pwd):/app -w /app vllm-warden-test pytest tests/unit/test_app_state.py::test_engine_driver_setting_defaults_local -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'engine_driver'`

- [ ] **Step 3: Add the setting**

In `app/settings.py`, add to the Settings model (match the existing field style — pydantic `Field` or attribute):

```python
    engine_driver: str = "local"   # "local" | "docker"
```

- [ ] **Step 4: Wire it in main.py**

Replace `app/main.py:44`:

```python
    app.state.supervisor = Supervisor(
        app.state.settings,
        driver=_build_engine_driver(app.state.settings),
    )
```

Add a module-level helper in `app/main.py`:

```python
def _build_engine_driver(settings):
    from pathlib import Path
    if settings.engine_driver == "docker":
        import docker
        from app.runtime.engine.docker_socket import DockerSocketDriver
        # Image resolution lands in P2 (#161); until then require an explicit
        # VLLM_ENGINE_IMAGE so the docker path is opt-in and never guesses.
        import os
        image = os.environ["VLLM_ENGINE_IMAGE"]
        return DockerSocketDriver(client=docker.from_env(), image=image)
    from app.runtime.engine.local_subprocess import LocalSubprocessDriver
    return LocalSubprocessDriver(log_dir=str(Path(settings.data_dir) / "logs"))
```

- [ ] **Step 5: Run to verify it passes + full suite**

Run: `docker run --rm -v $(pwd):/app -w /app vllm-warden-test pytest tests/unit -q`
Expected: PASS.

- [ ] **Step 6: Container-boot smoke (per feedback_container_boot_smoke_required)**

Run: `cd /home/ip/projects/vllm-warden && docker compose build && docker compose up -d && sleep 8 && docker compose logs --tail=40 app`
Expected: the app startup banner, no import/traceback. Then `docker compose down`.

- [ ] **Step 7: Commit**

```bash
git add app/settings.py app/main.py tests/unit/test_app_state.py
git commit -m "feat(#160): select engine driver from settings (default local)"
```

---

## Self-Review Notes (completed)

- **Spec coverage:** P1 row of the delivery plan (engine-sidecar split + EngineDriver + DockerSocketDriver) is covered by Tasks 1–5. Channel→image resolution is explicitly deferred to P2/#161 (Task 5 requires an explicit `VLLM_ENGINE_IMAGE` so the docker path never guesses an image before P2 lands).
- **Type consistency:** `EngineSpec`, `EngineHandle`, `EngineDriver` names and the `spawn`/`terminate`/`wait`/`pid`/`returncode` members are identical across Tasks 1–4. `Supervisor._handles` (renamed from `_processes`) is used consistently in Task 3.
- **Placeholder scan:** none — every code step has full code; the one judgement call (docker SDK `gpus=` vs `device_requests`) has an explicit decision rule and verification command.
- **Risk:** existing tests poke `sup._processes`; Task 3 Step 5 mandates updating them to `_handles`+`FakeDriver` without dropping coverage. The container-boot smoke (Task 5 Step 6) guards against import/lockfile regressions per memory.
