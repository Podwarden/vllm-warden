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
