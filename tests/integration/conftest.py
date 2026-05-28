# CI integration-tests wedge (#43): why this file daemonizes aiosqlite
# ------------------------------------------------------------------
# Symptom: pytest reports `47 passed in ~25s`, then the container hangs
# 25+ minutes until the runner kills it (`canceling` past the 10-minute
# cancel ack window). Reproduced locally in `python:3.11-slim` via
# `docker run --rm` — container never exits.
#
# Empirical proof (faulthandler.dump_traceback_later(30) in pytest_unconfigure):
#
#   [wedge-probe] pytest_unconfigure: active threads at teardown=2
#   [wedge-probe]   thread name='MainThread' daemon=False ident=...
#   [wedge-probe]   thread name='Thread-124' daemon=False ident=...
#   ...
#   ============================= 47 passed in 41.06s ==============================
#   Timeout (0:00:30)!
#   Thread 0x...4a6006c0 (most recent call first):
#     File ".../aiosqlite/core.py", line 107 in run
#     File ".../threading.py", line 1045 in _bootstrap_inner
#     File ".../threading.py", line 1002 in _bootstrap
#   Thread 0x...90e60300 (most recent call first):
#     File ".../threading.py", line 1590 in _shutdown
#
# Diagnosis: aiosqlite.Connection extends threading.Thread and is created
# with the default daemon=False. Its run() loop parks on self._tx.get()
# until __aexit__ enqueues a stop sentinel via Connection.close(). When
# pytest-asyncio tears down the per-test event loop, occasional close
# calls fail to complete the sentinel handshake (the worker thread's
# loop.call_soon_threadsafe raises "Event loop is closed"). The thread
# stays parked, non-daemon, and blocks threading._shutdown — the
# interpreter never returns from main, container PID 1 never exits,
# `docker run --rm` never returns, the GitLab Runner waits forever
# (and can't even ack cancellation because it's itself blocked on the
# same docker process).
#
# Fix: in the test process only, mark aiosqlite worker threads daemon.
# Daemon threads do not block interpreter shutdown — Python reaps them
# when main exits. We do NOT change app code; production keeps the
# same (correct, well-shaped) async-with semantics. This only affects
# how the test interpreter terminates after pytest finishes.
#
# The earlier CI workaround (`pkill -9 -f 'python|sqlite|uvicorn|vllm'`
# in .gitlab-ci.yml) becomes redundant once this fix lands; PM owns
# removing it in a follow-up.
import faulthandler
import sys
import threading

import aiosqlite.core
import pytest

# Monkey-patch aiosqlite.Connection.__init__ so its worker thread is a
# daemon. The wrapped __init__ still runs unchanged; we only flip the
# daemon flag after super().__init__() has finalized.
_orig_connection_init = aiosqlite.core.Connection.__init__


def _patched_connection_init(self, *args, **kwargs):
    _orig_connection_init(self, *args, **kwargs)
    self.daemon = True


aiosqlite.core.Connection.__init__ = _patched_connection_init


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "tests/integration" in str(item.fspath) or "tests\\integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


def pytest_unconfigure(config):
    # Diagnostic: dump the live-thread census at pytest teardown, and
    # arm a 30-second faulthandler trace dump so any future re-emergence
    # of the wedge shows exactly which thread is parked where. On the
    # happy path (process exits within 30s) the dump never fires.
    print(
        f"[wedge-probe] pytest_unconfigure: active threads at teardown="
        f"{threading.active_count()}",
        file=sys.stderr,
        flush=True,
    )
    for t in threading.enumerate():
        print(
            f"[wedge-probe]   thread name={t.name!r} daemon={t.daemon} ident={t.ident}",
            file=sys.stderr,
            flush=True,
        )

    faulthandler.enable()
    faulthandler.dump_traceback_later(30, repeat=False, file=sys.stderr)
