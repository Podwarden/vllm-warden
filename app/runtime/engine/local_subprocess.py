"""In-container subprocess engine driver — the default, identical in
behaviour to the pre-#160 inline Supervisor logic. Spawns `vllm serve`
(or an injected binary, for tests) in a new session, logs to a per-model
file, and stops it via process-group SIGTERM->SIGKILL.

Behaviour note (deviation from plan): ``spec.env`` is passed to the child
*verbatim*, NOT merged over ``os.environ``. This preserves the pre-#160
contract — Supervisor builds a curated env via ``build_subprocess_env``
(which deliberately omits secrets like ``VW_ADMIN_PASSWORD``); merging
``os.environ`` back in would leak those into the engine and break the
existing ``test_supervisor_load`` assertions.
"""
from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path

from app.runtime.engine import EngineSpec


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
    # The engine runs as a subprocess of the warden container, so its vLLM
    # version is whatever is baked into the warden image — ``EngineSpec.image``
    # is meaningless here and silently ignored. Surfaced so Supervisor.load can
    # refuse an engine-version pin instead of launching the wrong version.
    supports_engine_image = False

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
                env=dict(spec.env),
                stdout=log_fd, stderr=log_fd,
                start_new_session=True,
            )
        finally:
            os.close(log_fd)
        return LocalHandle(proc)

    def engine_host(self, model_id: str) -> str:
        # In-container subprocess shares the control-plane's network
        # namespace, so the engine is reachable on loopback.
        return "127.0.0.1"

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
        except TimeoutError:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await handle.wait()
