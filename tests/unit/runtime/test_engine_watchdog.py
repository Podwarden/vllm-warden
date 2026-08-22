"""#218 — the orphan GPU reaper must kill only processes this warden fathered.

Before this, ``reap_orphan_gpu_holders`` decided ownership with
``"vllm" in cmdline.lower()``. That matched a second vllm-warden on the same
box, another team's vLLM container and a researcher running
``python -m vllm.entrypoints.openai.api_server`` — and SIGKILLed all of them the
moment this warden recovered a crashed model. The ownership signal is now
session membership: ``LocalSubprocessDriver.spawn`` uses
``start_new_session=True``, so a wrapper's session id is its own pid and every
worker it forks inherits that id and keeps it after the wrapper dies.

These tests pin both halves of the trade: a stranger is never killed even when
its cmdline is indistinguishable from ours, and a genuine orphan of ours still
is.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from app.runtime import watchdog as wd
from app.runtime.watchdog import EngineOwnership

# A real foreign command line, of the shape the old substring test could not
# tell from our own workers.
FOREIGN_CMD = (
    "/home/researcher/.venv/bin/python -m vllm.entrypoints.openai.api_server "
    "--model mistralai/Mistral-7B-v0.3 --port 8000"
)


def _patch_procs(monkeypatch, *, apps, cmds, sids, parents=None):
    """Fake out every /proc + nvidia-smi read the reaper makes.

    ``sids`` is the piece that matters: it is the process table's answer to
    "which session does this pid belong to", which is what ownership now keys on.
    """
    parents = parents or {}
    monkeypatch.setattr(wd, "_gpu_compute_apps", lambda: apps)
    monkeypatch.setattr(wd, "_cmdline", lambda pid: cmds.get(pid, ""))
    monkeypatch.setattr(wd, "_sid", lambda pid: sids.get(pid))
    monkeypatch.setattr(wd, "_ppid", lambda pid: parents.get(pid))
    killed: list[int] = []
    monkeypatch.setattr(wd.os, "kill", lambda pid, sig: killed.append(pid))
    return killed


# --- the stranger ------------------------------------------------------------

def test_never_kills_a_foreign_vllm_process(monkeypatch):
    """The exact process #218 is about: another tenant's vLLM, same cmdline."""
    killed = _patch_procs(
        monkeypatch,
        apps=[(31337, 10550)],
        cmds={31337: FOREIGN_CMD},
        sids={31337: 31000},   # a session this warden never started
    )
    owned = EngineOwnership(sessions=frozenset({800137}), live_pids=frozenset())

    assert wd.reap_orphan_gpu_holders(owned) == []
    assert killed == [], "a cmdline substring is not an ownership signal"


def test_never_kills_a_second_wardens_orphaned_workers(monkeypatch):
    """Two vllm-wardens on one box. Their orphans look exactly like ours —
    same VLLM::Worker_TP0 comm, same dead-wrapper shape — and are still theirs."""
    killed = _patch_procs(
        monkeypatch,
        apps=[(700001, 10550), (700002, 10550)],
        cmds={700001: "VLLM::Worker_TP0", 700002: "VLLM::Worker_TP1"},
        sids={700001: 690000, 700002: 690000},   # the other warden's wrapper
        parents={700001: 1, 700002: 1},
    )
    owned = EngineOwnership(sessions=frozenset({800137}), live_pids=frozenset())

    assert wd.reap_orphan_gpu_holders(owned) == []
    assert killed == []


def test_kills_nothing_when_it_can_attribute_nothing(monkeypatch):
    """Empty ownership — e.g. straight after a warden restart — must be inert.

    This is the fail-safe: an un-reaped orphan costs one failed reload, a
    wrongly reaped stranger costs somebody their job.
    """
    killed = _patch_procs(
        monkeypatch,
        apps=[(801467, 10550), (31337, 4000)],
        cmds={801467: "VLLM::Worker_TP0", 31337: FOREIGN_CMD},
        sids={801467: 800137, 31337: 31000},
    )

    assert wd.reap_orphan_gpu_holders(EngineOwnership()) == []
    assert killed == []


def test_declined_kills_are_logged_with_a_reason(monkeypatch, caplog):
    """An operator looking at "Free memory on device cuda:0" after a failed
    reload has to be able to see that something we refused to touch is sitting
    on the VRAM. Silence here turns a diagnosable failure into a mystery.

    Both decline reasons are exercised, because both have to carry the MiB:
    31337 is a stranger's session, 555 is in a session of ours but fails the
    pid-wraparound cmdline check. Whichever one the operator hits, the number
    they need is the one in the CUDA error they are staring at.
    """
    _patch_procs(
        monkeypatch,
        apps=[(31337, 10550), (555, 8000)],
        cmds={31337: FOREIGN_CMD, 555: "/usr/bin/python3 train_resnet.py"},
        sids={31337: 31000, 555: 800137},
        parents={555: 1},
    )
    owned = EngineOwnership(sessions=frozenset({800137}), live_pids=frozenset())
    with caplog.at_level("WARNING", logger="app.runtime.watchdog"):
        assert wd.reap_orphan_gpu_holders(owned) == []

    text = caplog.text
    assert "31337" in text, "the pid we declined to kill must be named"
    assert "10550" in text, "how much VRAM it holds is the operator's whole question"
    assert "555" in text
    assert "8000 MiB" in text, "the second decline reason must carry the VRAM too"
    assert "#218" in text


def test_unreadable_session_is_not_attributed(monkeypatch):
    """_sid returns None for a process in another PID namespace, or one that
    exited between the nvidia-smi read and the /proc read. Unknown is not ours."""
    killed = _patch_procs(
        monkeypatch,
        apps=[(801467, 10550)],
        cmds={801467: "VLLM::Worker_TP0"},
        sids={801467: None},
    )
    owned = EngineOwnership(sessions=frozenset({800137}), live_pids=frozenset())

    assert wd.reap_orphan_gpu_holders(owned) == []
    assert killed == []


def test_pid_reuse_in_an_owned_session_still_needs_to_look_like_vllm(monkeypatch):
    """Defence in depth: session ids are pids and pid numbers wrap. If a session
    we recorded is re-created by an unrelated leader, the second test saves us."""
    killed = _patch_procs(
        monkeypatch,
        apps=[(555, 8000)],
        cmds={555: "/usr/bin/python3 train_resnet.py"},
        sids={555: 800137},   # collides with a wrapper pid we once spawned
        parents={555: 1},
    )
    owned = EngineOwnership(sessions=frozenset({800137}), live_pids=frozenset())

    assert wd.reap_orphan_gpu_holders(owned) == []
    assert killed == []


# --- our own orphans ---------------------------------------------------------

def test_reaps_our_own_orphaned_workers(monkeypatch):
    """2026-08-17: four VLLM::Worker_TP* held 10550 MiB each after EngineCore
    and the wrapper were both gone, and every reload then failed on free VRAM."""
    killed = _patch_procs(
        monkeypatch,
        apps=[(801467, 10550), (801590, 10550)],
        cmds={801467: "VLLM::Worker_TP0", 801590: "VLLM::Worker_TP1"},
        sids={801467: 800137, 801590: 800137},   # 800137 = our dead wrapper
        # Reparented to the container shim, NOT pid 1 — the host PID namespace
        # is shared, so an "orphan == ppid 1" test would miss these entirely.
        parents={801467: 794190, 801590: 794190, 794190: 0},
    )
    owned = EngineOwnership(sessions=frozenset({800137}), live_pids=frozenset())

    assert sorted(wd.reap_orphan_gpu_holders(owned)) == [801467, 801590]
    assert sorted(killed) == [801467, 801590]


def test_never_kills_workers_of_a_live_engine(monkeypatch):
    """A second, healthy model must survive the first one's recovery — even
    though its wrapper's session IS one of ours."""
    killed = _patch_procs(
        monkeypatch,
        apps=[(900001, 10550)],
        cmds={900001: "VLLM::Worker_TP0"},
        sids={900001: 888000},
        parents={900001: 888000, 888000: 794190},
    )
    owned = EngineOwnership(
        sessions=frozenset({800137, 888000}), live_pids=frozenset({888000})
    )

    assert wd.reap_orphan_gpu_holders(owned) == []
    assert killed == []


def test_live_engine_survives_a_broken_parent_chain(monkeypatch):
    """The ancestor walk fails when an intermediate process is gone or /proc
    is unreadable. Session membership of a LIVE wrapper still protects the
    workers — which is the difference between a spurious kill and a no-op."""
    killed = _patch_procs(
        monkeypatch,
        apps=[(900001, 10550)],
        cmds={900001: "VLLM::Worker_TP0"},
        sids={900001: 888000},
        parents={},   # no ancestry available at all
    )
    owned = EngineOwnership(
        sessions=frozenset({888000}), live_pids=frozenset({888000})
    )

    assert wd.reap_orphan_gpu_holders(owned) == []
    assert killed == []


# --- building the ownership set ---------------------------------------------

class _Handle:
    def __init__(self, pid, returncode=None):
        self.pid = pid
        self.returncode = returncode


class _Sup:
    def __init__(self, handles):
        self._handles = dict(handles)

    def get_pid(self, model_id):
        h = self._handles.get(model_id)
        return h.pid if h else None


def test_ownership_separates_live_wrappers_from_exited_ones():
    """An exited wrapper is still an owned SESSION (its workers may be holding
    VRAM) but must not be in live_pids, or its orphans are protected forever."""
    sup = _Sup({"a": _Handle(800137, returncode=1), "b": _Handle(888000)})

    owned = wd.engine_ownership(sup)

    assert owned.sessions == frozenset({800137, 888000})
    assert owned.live_pids == frozenset({888000})


def test_ownership_carries_the_pid_the_unload_is_about_to_erase():
    """Supervisor.unload pops the handle in a finally, so _restart has to hand
    the doomed pid in explicitly or the reaper is a no-op on its only case."""
    owned = wd.engine_ownership(_Sup({}), also_sessions=(800137, None))

    assert owned.sessions == frozenset({800137})
    assert owned.live_pids == frozenset()


# --- the wrapper-pid ledger --------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_ledger():
    wd._WRAPPER_PIDS.clear()
    yield
    wd._WRAPPER_PIDS.clear()


def test_ledger_survives_the_supervisor_dropping_the_handle():
    """_watch_exit pops the handle the instant the wrapper exits, which is
    BEFORE the sweep gets round to restarting the model. Without the ledger the
    exit path — the commonest crash shape — could never attribute its orphans."""
    sup = _Sup({"m1": _Handle(800137)})
    wd.remember_wrapper_pids(sup)

    sup._handles.clear()   # what _watch_exit does on exit

    assert sup.get_pid("m1") is None
    assert wd.last_wrapper_pid("m1") == 800137


def test_ledger_entry_expires(monkeypatch):
    """A model that failed and was never restarted must not leave a pid lying
    around until the kernel reissues that number to a stranger."""
    wd.remember_wrapper_pids(_Sup({"m1": _Handle(800137)}))
    now = time.monotonic()
    monkeypatch.setattr(wd.time, "monotonic", lambda: now + wd._WRAPPER_PID_TTL_S + 1)

    assert wd.last_wrapper_pid("m1") is None
    assert "m1" not in wd._WRAPPER_PIDS, "an expired entry must be dropped, not re-read"


def test_ledger_forgets_deleted_models():
    wd.remember_wrapper_pids(_Sup({"m1": _Handle(800137), "m2": _Handle(888000)}))

    wd.forget_wrapper_pids({"m1"})

    assert wd.last_wrapper_pid("m1") == 800137
    assert wd.last_wrapper_pid("m2") is None


# --- waiting for the VRAM to come back ---------------------------------------

@pytest.mark.asyncio
async def test_wait_for_gpu_release_returns_when_our_orphans_are_gone(monkeypatch):
    _patch_procs(monkeypatch, apps=[], cmds={}, sids={})
    owned = EngineOwnership(sessions=frozenset({800137}))

    assert await wd.wait_for_gpu_release(owned, timeout_s=1, interval_s=0.01)


@pytest.mark.asyncio
async def test_wait_for_gpu_release_times_out_while_ours_still_holds(monkeypatch):
    """Freeing is not instant after SIGKILL; reloading too early reproduces the
    very failure being recovered from."""
    _patch_procs(
        monkeypatch,
        apps=[(801467, 10550)],
        cmds={801467: "VLLM::Worker_TP0"},
        sids={801467: 800137},
        parents={801467: 794190, 794190: 0},
    )
    owned = EngineOwnership(sessions=frozenset({800137}))

    assert not await wd.wait_for_gpu_release(owned, timeout_s=0.2, interval_s=0.05)


@pytest.mark.asyncio
async def test_wait_does_not_block_on_a_foreign_holder(monkeypatch):
    """We are never going to free a stranger's VRAM, so waiting on it just adds
    30s to a reload that will fail anyway. The declined-kill log is the signal."""
    _patch_procs(
        monkeypatch,
        apps=[(31337, 10550)],
        cmds={31337: FOREIGN_CMD},
        sids={31337: 31000},
    )
    owned = EngineOwnership(sessions=frozenset({800137}))

    assert await wd.wait_for_gpu_release(owned, timeout_s=1, interval_s=0.01)


# --- the restart path wires it together --------------------------------------

@pytest.mark.asyncio
async def test_restart_samples_the_wrapper_pid_before_unloading(monkeypatch, tmp_path):
    """The whole fix hinges on reading get_pid() BEFORE unload() pops the
    handle. Sampling it afterwards yields None and reaps nothing."""
    seen: dict = {}

    class Sup:
        def __init__(self):
            self._handles = {"m1": _Handle(800137)}

        def get_pid(self, model_id):
            h = self._handles.get(model_id)
            return h.pid if h else None

        def get_overrides(self, model_id):
            return None

        async def unload(self, model_id, *, force=False):
            self._handles.pop(model_id, None)   # what Supervisor.unload does

    class _NullDB:
        async def __aenter__(self):
            return None

        async def __aexit__(self, *a):
            return False

    class FakeRepo:
        def __init__(self, db):
            pass

        async def get(self, mid):
            return SimpleNamespace(id=mid, prior_status="loaded", status="loaded")

        async def update_status(self, mid, status, last_error=None):
            pass

        async def set_prior_status(self, mid, prior):
            pass

    async def fake_start_engine(*a, **k):
        return None

    def spy_reap(owned):
        seen["sessions"] = owned.sessions
        return []

    monkeypatch.setattr(wd, "open_db", lambda *a, **k: _NullDB())
    monkeypatch.setattr(wd, "ModelRepo", FakeRepo)
    monkeypatch.setattr(wd, "reap_orphan_gpu_holders", spy_reap)
    monkeypatch.setattr("app.models.routes_api.start_engine", fake_start_engine)

    app_state = SimpleNamespace(
        supervisor=Sup(),
        port_allocator=SimpleNamespace(allocate=lambda: 10000, release=lambda p: None),
    )
    settings = SimpleNamespace(db_path=tmp_path / "x.db")

    await wd._restart(settings, app_state, "m1", None)

    assert 800137 in seen["sessions"], (
        "the doomed wrapper's pid is the session id of the orphans it left "
        "behind; losing it makes the reaper a no-op"
    )
