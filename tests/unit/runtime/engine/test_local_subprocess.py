import pytest

from app.runtime.engine import EngineSpec
from app.runtime.engine.local_subprocess import LocalSubprocessDriver
from app.runtime.engine.run_marker import is_run_sentinel


@pytest.mark.asyncio
async def test_spawn_runs_command_and_handle_waits(tmp_path):
    # Use /bin/sh as a stand-in "engine" that exits 0 immediately.
    spec = EngineSpec(model_id="m1", model_arg="x",
                      argv=["/bin/sh", "-c", "exit 0"], env={}, port=8001)
    driver = LocalSubprocessDriver(log_dir=str(tmp_path))
    handle = await driver.spawn(spec)
    assert handle.pid is not None
    rc = await handle.wait()
    assert rc == 0
    assert handle.returncode == 0


@pytest.mark.asyncio
async def test_terminate_kills_long_running(tmp_path):
    spec = EngineSpec(model_id="m2", model_arg="x",
                      argv=["/bin/sh", "-c", "sleep 60"], env={}, port=8002)
    driver = LocalSubprocessDriver(log_dir=str(tmp_path))
    handle = await driver.spawn(spec)
    await driver.terminate(handle, grace_s=0.5)
    assert handle.returncode is not None  # exited after term/kill


@pytest.mark.asyncio
async def test_spawn_execs_argv_zero_verbatim(tmp_path):
    """The driver no longer knows what program it is starting.

    Before sub-project B, spawn() prepended ['vllm', 'serve'] to spec.args --
    the single line that made a PLACEMENT concern (where does a process run)
    also decide WHICH program runs. argv[0] now comes from the backend's
    LaunchPlan, so the driver is backend-agnostic.
    """
    spec = EngineSpec(model_id="m3", model_arg="x",
                      argv=["/bin/sh", "-c", "exit 7"], env={}, port=8003)
    driver = LocalSubprocessDriver(log_dir=str(tmp_path))
    handle = await driver.spawn(spec)
    assert await handle.wait() == 7


@pytest.mark.asyncio
async def test_spawn_delimits_each_run_and_keeps_history(tmp_path):
    """#234: the log is append-only, so each spawn must stamp a run boundary
    that the diagnosis reader can start from. History stays -- the operator
    comparing attempt N-1 with attempt N is why we delimit instead of truncate.
    """
    driver = LocalSubprocessDriver(log_dir=str(tmp_path))
    for i, marker in enumerate(("run-one-output", "run-two-output")):
        spec = EngineSpec(model_id="m4", model_arg="x",
                          argv=["/bin/sh", "-c", f"echo {marker}"],
                          env={}, port=8004 + i)
        await (await driver.spawn(spec)).wait()

    lines = (tmp_path / "m4.log").read_text().splitlines()
    sentinels = [i for i, ln in enumerate(lines) if is_run_sentinel(ln)]
    assert len(sentinels) == 2
    # Previous run preserved for the operator...
    assert "run-one-output" in lines[sentinels[0]:sentinels[1]]
    # ...but strictly before the boundary the reader starts from.
    assert "run-two-output" in lines[sentinels[1]:]
    assert "run-one-output" not in lines[sentinels[1]:]


@pytest.mark.asyncio
async def test_first_load_rotation_is_silent(tmp_path, caplog):
    """A model's FIRST load has no log file yet -- rotating must say nothing.

    ``_rotate`` stat()s the log before every spawn. On a first load that
    raised FileNotFoundError, which the blanket ``except OSError`` logged as
    ``could not rotate ...`` with a full traceback. Nothing was lost (spawn
    creates the file immediately afterwards and the engine log was captured
    correctly) but it is the very first thing a new operator sees in the logs
    and it reads like a fault.
    """
    caplog.set_level("WARNING")
    driver = LocalSubprocessDriver(log_dir=str(tmp_path), log_max_bytes=1024)
    spec = EngineSpec(model_id="fresh", model_arg="x",
                      argv=["/bin/sh", "-c", "exit 0"], env={}, port=8010)
    await (await driver.spawn(spec)).wait()

    assert (tmp_path / "fresh.log").exists()
    assert caplog.records == [], [r.getMessage() for r in caplog.records]


@pytest.mark.asyncio
async def test_rotation_still_happens_once_the_log_is_big(tmp_path):
    """The silence above must not have cost us the rotation itself.

    The log shares /data with the SQLite DB; filling that volume takes the
    database down with it, which is the whole reason rotation exists.
    """
    driver = LocalSubprocessDriver(log_dir=str(tmp_path), log_max_bytes=64)
    log_path = tmp_path / "big.log"
    log_path.write_text("x" * 200)

    spec = EngineSpec(model_id="big", model_arg="x",
                      argv=["/bin/sh", "-c", "exit 0"], env={}, port=8011)
    await (await driver.spawn(spec)).wait()

    assert (tmp_path / "big.log.1").read_text() == "x" * 200
    assert "x" * 200 not in log_path.read_text()


def test_driver_has_no_binary_kwarg():
    """The 'binary' escape hatch existed only so tests could inject /bin/sh
    past the hard-coded head. With argv[0] on the spec it is dead weight, and
    leaving it would be a second, competing way to decide argv[0]."""
    import inspect
    sig = inspect.signature(LocalSubprocessDriver.__init__)
    assert "binary" not in sig.parameters
