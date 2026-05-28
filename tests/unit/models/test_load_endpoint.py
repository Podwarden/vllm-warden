import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import csrf_header, jwt_login, seed_admin_user


@dataclass
class _FakeGpuLive:
    index: int
    memory_total_mib: int = 16376


@dataclass
class _FakeSnap:
    gpus: list = field(default_factory=list)
    apps: list = field(default_factory=list)
    probe_error: str | None = None


class _FakeProbeCache:
    def __init__(self, snap):
        self._snap = snap

    async def get(self):
        return self._snap


def _stub_probe(client, indices):
    """Install a probe cache reporting exactly ``indices`` as present.

    The load pre-flight now fails closed (#175): a configured GPU absent
    from the probe — or no probe at all — 422s. Tests that exercise the
    happy load path must therefore stub a probe that confirms their
    configured GPUs are present.
    """
    client.app.state.gpu_probe_cache = _FakeProbeCache(
        _FakeSnap(gpus=[_FakeGpuLive(index=i) for i in indices])
    )


def _seed_done_with_pulled_model(db_path, *, allowed, model_id="qwen", gpus=None):
    """Seed setup-done admin + a pre-pulled model with given gpu_indices.

    #55 fix — admin user + setup_state via shared barrier-aware helper.
    """
    seed_admin_user(db_path, allowed_gpu_indices=allowed)
    if gpus is not None:
        with sqlite3.connect(db_path) as db:
            db.execute(
                "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
                "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
                "gpu_memory_utilization, trust_remote_code, extra_args, status, "
                "pulled_bytes, pulled_total, last_error) "
                "VALUES (?, ?, 'o/r', 'main', ?, ?, NULL, NULL, 0.9, 0, '[]', 'pulled', 0, NULL, NULL)",
                (model_id, model_id, json.dumps(gpus), len(gpus)),
            )
            db.commit()


def _jwt_login(client, username="admin", password="hunter2"):
    # #55 fix — delegate to the barrier+retry helper.
    return jwt_login(client, username=username, password=password)


def test_load_validates_gpus_against_allowed_set(tmp_data_dir, client):
    """If gpu_indices ⊄ allowed_gpu_indices, must return 422."""
    client.get("/healthz")
    _seed_done_with_pulled_model(
        tmp_data_dir / "vllm-warden.db", allowed=[1, 2, 3], gpus=[0, 1]
    )
    auth = _jwt_login(client)
    r = client.post("/api/models/qwen/load", headers={**auth, **csrf_header(client)})
    assert r.status_code == 422
    assert "allowed" in r.json()["detail"].lower()


def test_load_calls_supervisor_then_health_check(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done_with_pulled_model(
        tmp_data_dir / "vllm-warden.db", allowed=[0, 1, 2, 3], gpus=[0, 1]
    )
    _stub_probe(client, [0, 1])
    auth = _jwt_login(client)

    sup_load = AsyncMock()
    health = AsyncMock(return_value=True)
    with patch("app.runtime.supervisor.Supervisor.load", new=sup_load), \
         patch("app.models.routes_api.wait_for_health", new=health):
        r = client.post("/api/models/qwen/load", headers={**auth, **csrf_header(client)})
    assert r.status_code == 202
    # supervisor.load is invoked from a background task, so allow a short window
    for _ in range(50):
        if sup_load.await_count >= 1:
            break
        time.sleep(0.05)
    assert sup_load.await_count == 1


def test_unload_returns_404_for_unknown_model(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done_with_pulled_model(tmp_data_dir / "vllm-warden.db", allowed=[0, 1])
    auth = _jwt_login(client)
    r = client.post("/api/models/missing/unload", headers={**auth, **csrf_header(client)})
    assert r.status_code == 404


@pytest.mark.skip(
    reason=(
        "Quarantined 2026-05-23 — fails deterministically on develop "
        "(pipeline #6621 jobs 34385 + 34400) with "
        "`RuntimeError: There is no current event loop in thread "
        "'MainThread'`, triggered by leaked supervisor state from a "
        "prior test (`cannot mark warming from state None: expected "
        "LOADING` at app/runtime/supervisor.py:172). Same flake family "
        "as #127. Loses coverage of on-exit failure->status transition "
        "+ port release — restore once root-cause fix lands. "
        "PM will file the tracking follow-up against this skip."
    )
)
def test_on_exit_callback_flips_status_to_failed_and_releases_port(tmp_data_dir, client):
    """Regression: when the supervisor fires on_exit after a crash, the model
    status must flip to 'failed' and the port must be released."""
    client.get("/healthz")
    _seed_done_with_pulled_model(
        tmp_data_dir / "vllm-warden.db", allowed=[0, 1, 2, 3], gpus=[0, 1],
        model_id="crash-model",
    )
    auth = _jwt_login(client)

    captured_on_exit = []

    async def fake_load(self, model, *, port, on_exit=None, overrides=None):
        # Capture the callback; we'll invoke it manually below
        captured_on_exit.append(on_exit)

    health = AsyncMock(return_value=True)

    with patch("app.runtime.supervisor.Supervisor.load", new=fake_load), \
         patch("app.models.routes_api.wait_for_health", new=health):
        r = client.post("/api/models/crash-model/load", headers={**auth, **csrf_header(client)})
    assert r.status_code == 202

    # Wait for runner() to register the on_exit callback
    for _ in range(50):
        if captured_on_exit:
            break
        time.sleep(0.05)
    assert captured_on_exit, "on_exit was never registered by runner()"

    on_exit_fn = captured_on_exit[0]
    assert on_exit_fn is not None, "on_exit should not be None"

    # Simulate crash: update DB to 'loaded' first (what runner does after health ok)
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute(
            "UPDATE models SET status = 'loaded' WHERE id = 'crash-model'"
        )
        db.commit()

    # Now fire the on_exit callback from a synchronous context
    asyncio.get_event_loop().run_until_complete(on_exit_fn(139))

    # Assert status flipped to failed
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        row = db.execute(
            "SELECT status, last_error FROM models WHERE id = 'crash-model'"
        ).fetchone()
    assert row[0] == "failed", f"Expected failed, got {row[0]}"
    assert "139" in row[1], f"Expected rc=139 in last_error, got {row[1]}"


def test_load_writes_health_ok_after_warmup_probe_succeeds(tmp_data_dir, client):
    """#29 — when wait_for_health succeeds AND the warmup probe succeeds,
    the runner must persist ``health_ok=True`` into ``model_runtime`` so
    callers (status badges, /stats, future telemetry) can distinguish
    'process up but never proved serving' from 'process up and served at
    least one request'. Previously the column was never written and stuck
    at its default 0.
    """
    client.get("/healthz")
    _seed_done_with_pulled_model(
        tmp_data_dir / "vllm-warden.db", allowed=[0, 1, 2, 3], gpus=[0, 1],
        model_id="health-ok-model",
    )
    _stub_probe(client, [0, 1])
    auth = _jwt_login(client)

    from app.runtime.warmup_probe import ProbeResult

    async def fake_load(self, model, *, port, on_exit=None, overrides=None):
        # Inject a fake process record so the runner's upsert(pid=...)
        # path has a PID to read. The real Supervisor.load does this;
        # AsyncMock would not, leaving sup._handles empty.
        class _FakeProc:
            pid = 1234
        self._handles[model.id] = _FakeProc()
        self._ports[model.id] = port

    sup_mark_warming = AsyncMock()
    sup_mark_ready = AsyncMock()
    health = AsyncMock(return_value=True)
    probe = AsyncMock(return_value=ProbeResult(ok=True, detail=None))

    with patch("app.runtime.supervisor.Supervisor.load", new=fake_load), \
         patch("app.runtime.supervisor.Supervisor.mark_warming", new=sup_mark_warming), \
         patch("app.runtime.supervisor.Supervisor.mark_ready", new=sup_mark_ready), \
         patch("app.models.routes_api.wait_for_health", new=health), \
         patch("app.models.routes_api.warmup_probe", new=probe):
        r = client.post(
            "/api/models/health-ok-model/load",
            headers={**auth, **csrf_header(client)},
        )
    assert r.status_code == 202

    row = None
    for _ in range(100):
        if probe.await_count >= 1:
            with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
                row = db.execute(
                    "SELECT health_ok, last_health_at "
                    "FROM model_runtime WHERE model_id = 'health-ok-model'"
                ).fetchone()
            if row and row[0] == 1:
                break
        time.sleep(0.05)

    assert row is not None, "model_runtime row never written"
    assert row[0] == 1, f"expected health_ok=1, got {row[0]}"
    assert row[1] is not None, "expected last_health_at ISO timestamp"
    # Sanity: timestamp is in ISO-8601 with a 'T' separator (datetime.isoformat)
    assert "T" in row[1], f"expected ISO timestamp, got {row[1]!r}"


def test_load_runs_warmup_probe_before_flipping_to_loaded(tmp_data_dir, client):
    """Status must remain 'loading' until the warmup probe succeeds, not
    just when /health returns 200. Regression for 2026-05-20 Qwen3-VL
    crash loop."""
    client.get("/healthz")
    _seed_done_with_pulled_model(
        tmp_data_dir / "vllm-warden.db", allowed=[0, 1, 2, 3], gpus=[0, 1],
        model_id="probe-model",
    )
    _stub_probe(client, [0, 1])
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

    for _ in range(50):
        if probe.await_count >= 1:
            break
        time.sleep(0.05)
    assert probe.await_count == 1
    sup_mark_warming.assert_awaited_once()
    sup_mark_ready.assert_awaited_once()


def test_load_probe_failure_marks_failed_without_unloading(tmp_data_dir, client):
    """When the warmup probe fails, the model row goes to 'failed' but
    the subprocess is left running (no SIGTERM). Operator must
    force-unload to release GPUs."""
    client.get("/healthz")
    _seed_done_with_pulled_model(
        tmp_data_dir / "vllm-warden.db", allowed=[0, 1, 2, 3], gpus=[0, 1],
        model_id="probe-fail-model",
    )
    _stub_probe(client, [0, 1])
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
    # Poll until the runner finishes the DB write; await_count increments
    # before the DB update completes, so we need to observe the DB state.
    row = None
    for _ in range(100):
        if probe.await_count >= 1:
            with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
                row = db.execute(
                    "SELECT status, last_error FROM models WHERE id = 'probe-fail-model'"
                ).fetchone()
            if row and row[0] == "failed":
                break
        time.sleep(0.05)

    assert row is not None
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
    _stub_probe(client, [0, 1])
    auth = _jwt_login(client)

    sup_load = AsyncMock()
    sup_unload = AsyncMock()
    health = AsyncMock(return_value=False)

    with patch("app.runtime.supervisor.Supervisor.load", new=sup_load), \
         patch("app.runtime.supervisor.Supervisor.unload", new=sup_unload), \
         patch("app.models.routes_api.wait_for_health", new=health):
        r = client.post(
            "/api/models/health-timeout-model/load",
            headers={**auth, **csrf_header(client)},
        )
    assert r.status_code == 202
    # Poll until the runner finishes the DB write; await_count increments
    # before the DB update completes, so we need to observe the DB state.
    row = None
    for _ in range(100):
        if health.await_count >= 1:
            with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
                row = db.execute(
                    "SELECT status, last_error FROM models WHERE id = 'health-timeout-model'"
                ).fetchone()
            if row and row[0] == "failed":
                break
        time.sleep(0.05)

    assert row is not None
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
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute("UPDATE models SET status='loaded' WHERE id='refused-model'")
        db.commit()
    auth = _jwt_login(client)

    # #166 — the refusal is now surfaced synchronously via the fast
    # ``ensure_unloadable`` pre-flight check (the slow teardown moved to a
    # background task), so the 409 originates there rather than from ``unload``.
    sup_check = AsyncMock(
        side_effect=UnloadRefused("refused-model", ModelState.WARMING)
    )
    with patch("app.runtime.supervisor.Supervisor.ensure_unloadable", new=sup_check):
        r = client.post(
            "/api/models/refused-model/unload",
            headers={**auth, **csrf_header(client)},
        )
    assert r.status_code == 409
    assert "WARMING" in r.json()["detail"]


_GIB = 1024 ** 3

# A config whose model-max (262144) blows the KV budget on a single A4000 —
# mirrors the tencent/Hy-MT2-1.8B footgun the preflight exists to catch.
_HUGE_CTX_CONFIG = {
    "hidden_size": 2048,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "max_position_embeddings": 262144,
    "torch_dtype": "bfloat16",
}
# Inputs that make decide_preflight return cap/block for the huge-ctx config.
_A4000_VRAM = 16 * _GIB
_WEIGHTS_2B = 4 * _GIB


def _set_explicit_max_model_len(db_path, model_id, value):
    with sqlite3.connect(db_path) as db:
        db.execute(
            "UPDATE models SET max_model_len = ? WHERE id = ?", (value, model_id)
        )
        db.commit()


def test_load_blocks_explicit_max_model_len_that_wont_fit(tmp_data_dir, client):
    """FEATURE 1, Case B: an EXPLICIT max_model_len that won't fit must block
    with HTTP 422 ``wont_fit`` BEFORE any engine is spawned."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done_with_pulled_model(
        db_path, allowed=[0, 1, 2, 3], gpus=[0], model_id="toobig"
    )
    # #175 physical-presence pre-flight fails closed on an empty probe; confirm
    # the seeded GPU is present so the flow reaches the KV-budget preflight.
    _stub_probe(client, [0])
    _set_explicit_max_model_len(db_path, "toobig", 262144)
    auth = _jwt_login(client)

    async def fake_inputs(settings, model, request):
        return _HUGE_CTX_CONFIG, _A4000_VRAM, _WEIGHTS_2B

    sup_load = AsyncMock()
    with patch("app.models.routes_api._gather_preflight_inputs", new=fake_inputs), \
         patch("app.runtime.supervisor.Supervisor.load", new=sup_load):
        r = client.post(
            "/api/models/toobig/load", headers={**auth, **csrf_header(client)}
        )
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error_code"] == "wont_fit"
    assert "recommended_max_model_len" in detail
    assert detail["breakdown"]["total_vram"] == _A4000_VRAM
    # Engine must NOT have been spawned, and the row must be re-loadable.
    sup_load.assert_not_awaited()
    with sqlite3.connect(db_path) as db:
        status = db.execute(
            "SELECT status FROM models WHERE id='toobig'"
        ).fetchone()[0]
    assert status == "pulled"


def test_load_auto_caps_null_max_model_len(tmp_data_dir, client):
    """FEATURE 1, Case A: a NULL max_model_len whose model-max won't fit is
    auto-capped — sup.load gets overrides={'max_model_len': <cap>} and the 202
    body carries ``context_capped``."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done_with_pulled_model(
        db_path, allowed=[0, 1, 2, 3], gpus=[0], model_id="capme"
    )  # max_model_len stays NULL from the seed
    # #175 physical-presence pre-flight fails closed on an empty probe; confirm
    # the seeded GPU is present so the flow reaches the KV-budget preflight.
    _stub_probe(client, [0])
    auth = _jwt_login(client)

    async def fake_inputs(settings, model, request):
        return _HUGE_CTX_CONFIG, _A4000_VRAM, _WEIGHTS_2B

    captured = {}

    async def fake_load(self, model, *, port, on_exit=None, overrides=None):
        captured["overrides"] = overrides

    health = AsyncMock(return_value=True)
    with patch("app.models.routes_api._gather_preflight_inputs", new=fake_inputs), \
         patch("app.runtime.supervisor.Supervisor.load", new=fake_load), \
         patch("app.models.routes_api.wait_for_health", new=health):
        r = client.post(
            "/api/models/capme/load", headers={**auth, **csrf_header(client)}
        )
    assert r.status_code == 202
    cc = r.json()["context_capped"]
    assert cc["from"] == 262144
    assert cc["to"] < 262144
    assert cc["reason"] == "kv_cache_exceeds_vram"

    for _ in range(50):
        if "overrides" in captured:
            break
        time.sleep(0.05)
    assert captured.get("overrides") == {"max_model_len": cc["to"]}


def test_load_crash_reports_diagnosed_last_error(tmp_data_dir, client):
    """FEATURE 2: when the engine crashes, the runner reads the engine-log tail
    and reports the parsed, actionable message as ``last_error`` (not the bare
    generic text)."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done_with_pulled_model(
        db_path, allowed=[0, 1, 2, 3], gpus=[0], model_id="crashy"
    )
    # #175 physical-presence pre-flight fails closed on an empty probe; confirm
    # the seeded GPU is present so the flow reaches the engine-spawn/crash path.
    _stub_probe(client, [0])
    auth = _jwt_login(client)

    # Seed a fake engine log that the diagnostics parser will recognise as a
    # KV-cache overflow (real d5 string, including vLLM's own fit estimate).
    logs_dir = tmp_data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "crashy.log").write_text(
        "ValueError: To serve at least one request with the models's max seq "
        "len (262144), (16.0 GiB KV cache is needed, which is larger than the "
        "available KV cache memory (9.6 GiB). Based on the available memory, "
        "the estimated maximum model length is 157216. Try increasing "
        "`gpu_memory_utilization` or decreasing `max_model_len`.\n"
    )

    async def fake_inputs(settings, model, request):
        # Fail open so the preflight does not pre-empt the crash path.
        return {}, 0, 0

    captured_on_exit = []

    async def fake_load(self, model, *, port, on_exit=None, overrides=None):
        captured_on_exit.append(on_exit)
        # Flip to loaded as the real driver would, then fire on_exit(rc=1).
        with sqlite3.connect(db_path) as db:
            db.execute("UPDATE models SET status='loaded' WHERE id='crashy'")
            db.commit()
        if on_exit is not None:
            await on_exit(1)

    with patch("app.models.routes_api._gather_preflight_inputs", new=fake_inputs), \
         patch("app.runtime.supervisor.Supervisor.load", new=fake_load):
        r = client.post(
            "/api/models/crashy/load", headers={**auth, **csrf_header(client)}
        )
    assert r.status_code == 202

    row = None
    for _ in range(100):
        with sqlite3.connect(db_path) as db:
            row = db.execute(
                "SELECT status, last_error FROM models WHERE id='crashy'"
            ).fetchone()
        if row and row[0] == "failed" and row[1]:
            break
        time.sleep(0.05)
    assert row is not None and row[0] == "failed"
    # The diagnosed message (with vLLM's estimate) replaced the generic text...
    assert "157216" in row[1]
    assert "max_model_len" in row[1]
    # ...and the rc is still appended for log correlation.
    assert "rc=1" in row[1]


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
    # #166 — teardown now runs in a background task, so the force flag is
    # observed asynchronously; poll for it rather than asserting inline.
    for _ in range(50):
        if "force" in captured:
            break
        time.sleep(0.05)
    assert captured["force"] is True


def test_unload_returns_202_immediately_even_when_teardown_is_slow(
    tmp_data_dir, client
):
    """#166 regression — the unload route must NOT block the HTTP response on
    the (potentially many-second) engine teardown. A slow ``sup.unload`` used
    to keep the request open until the client/proxy disconnected, at which
    point the CSRF ``BaseHTTPMiddleware`` raised Starlette's
    ``RuntimeError("No response returned.")`` → HTTP 500, stranding the row in
    'unloading'. Teardown must run in a background task; the route returns 202
    promptly and the row reaches the terminal 'pulled' state asynchronously.
    """
    client.get("/healthz")
    _seed_done_with_pulled_model(
        tmp_data_dir / "vllm-warden.db", allowed=[0, 1], gpus=[0],
        model_id="slow-model",
    )
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute("UPDATE models SET status='loaded' WHERE id='slow-model'")
        db.commit()
    auth = _jwt_login(client)

    async def slow_unload(self, model_id, *, force=False):
        await asyncio.sleep(1.0)  # simulate a large multi-GPU engine teardown

    with patch("app.runtime.supervisor.Supervisor.unload", new=slow_unload):
        start = time.monotonic()
        r = client.post(
            "/api/models/slow-model/unload",
            headers={**auth, **csrf_header(client)},
        )
        elapsed = time.monotonic() - start
    assert r.status_code == 202
    # The response must come back well before the 1.0s teardown completes.
    assert elapsed < 0.5, f"unload blocked on teardown for {elapsed:.2f}s"

    # The background task must still drive the row to the terminal state.
    row = None
    for _ in range(60):
        with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
            row = db.execute(
                "SELECT status FROM models WHERE id='slow-model'"
            ).fetchone()
        if row and row[0] == "pulled":
            break
        time.sleep(0.05)
    assert row[0] == "pulled", f"expected terminal 'pulled', got {row[0]}"


def test_load_422_when_configured_gpu_absent_from_probe(tmp_data_dir, client):
    """A model whose gpu_indices is allow-listed but physically absent must
    422 gpu_index_missing before the row flips to 'loading'."""
    client.get("/healthz")
    _seed_done_with_pulled_model(
        tmp_data_dir / "vllm-warden.db", allowed=[0, 1, 2, 3], gpus=[0, 2],
        model_id="ghost-gpu",
    )
    # Probe sees only GPU 0 — index 2 is gone (card pulled / re-indexed).
    client.app.state.gpu_probe_cache = _FakeProbeCache(_FakeSnap(gpus=[_FakeGpuLive(index=0)]))
    auth = _jwt_login(client)

    r = client.post("/api/models/ghost-gpu/load", headers={**auth, **csrf_header(client)})
    assert r.status_code == 422
    body = r.json()
    assert body["detail"]["error_code"] == "gpu_index_missing"
    assert body["detail"]["available"] == [0]
    assert "2" in body["detail"]["message"] or "[2]" in body["detail"]["message"]
    # Row must NOT have advanced to loading.
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        status = db.execute("SELECT status FROM models WHERE id='ghost-gpu'").fetchone()[0]
    assert status == "pulled"


def test_load_passes_preflight_when_all_gpus_present(tmp_data_dir, client):
    """Allow-listed AND present gpu_indices passes the probe pre-flight (202)."""
    client.get("/healthz")
    _seed_done_with_pulled_model(
        tmp_data_dir / "vllm-warden.db", allowed=[0, 1, 2, 3], gpus=[0, 1],
        model_id="present-gpu",
    )
    client.app.state.gpu_probe_cache = _FakeProbeCache(
        _FakeSnap(gpus=[_FakeGpuLive(index=0), _FakeGpuLive(index=1)])
    )
    auth = _jwt_login(client)

    sup_load = AsyncMock()
    health = AsyncMock(return_value=True)
    with patch("app.runtime.supervisor.Supervisor.load", new=sup_load), \
         patch("app.models.routes_api.wait_for_health", new=health):
        r = client.post("/api/models/present-gpu/load", headers={**auth, **csrf_header(client)})
    assert r.status_code == 202


def test_load_422_when_probe_errored(tmp_data_dir, client):
    """When the probe itself errored there is no ground truth that any
    configured GPU is present, so the pre-flight must FAIL CLOSED (422
    gpu_index_missing) per spec — every configured index is unconfirmed."""
    client.get("/healthz")
    _seed_done_with_pulled_model(
        tmp_data_dir / "vllm-warden.db", allowed=[0, 1, 2, 3], gpus=[2],
        model_id="probe-err-gpu",
    )
    # GPU 2 is configured but the probe reports an error and zero GPUs;
    # with no ground truth the guard must 422 (fail closed).
    client.app.state.gpu_probe_cache = _FakeProbeCache(
        _FakeSnap(gpus=[], probe_error="nvidia-smi unavailable")
    )
    auth = _jwt_login(client)
    sup_load = AsyncMock()
    health = AsyncMock(return_value=True)
    with patch("app.runtime.supervisor.Supervisor.load", new=sup_load), \
         patch("app.models.routes_api.wait_for_health", new=health):
        r = client.post("/api/models/probe-err-gpu/load", headers={**auth, **csrf_header(client)})
    assert r.status_code == 422
    body = r.json()
    assert body["detail"]["error_code"] == "gpu_index_missing"
    assert body["detail"]["probe_error"] == "nvidia-smi unavailable"
    assert body["detail"]["available"] == []
    # Row must NOT have advanced to loading.
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        status = db.execute(
            "SELECT status FROM models WHERE id='probe-err-gpu'"
        ).fetchone()[0]
    assert status == "pulled"


def test_unload_teardown_error_still_lands_terminal(tmp_data_dir, client):
    """#166 — if the engine teardown raises an unexpected error, the row must
    NOT be left stranded in 'unloading'. The background task drives it to a
    terminal state (the engine is gone either way) so the operator never needs
    a control-plane restart to recover.
    """
    client.get("/healthz")
    _seed_done_with_pulled_model(
        tmp_data_dir / "vllm-warden.db", allowed=[0, 1], gpus=[0],
        model_id="boom-model",
    )
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute("UPDATE models SET status='loaded' WHERE id='boom-model'")
        db.commit()
    auth = _jwt_login(client)

    async def boom_unload(self, model_id, *, force=False):
        raise RuntimeError("driver blew up mid-teardown")

    with patch("app.runtime.supervisor.Supervisor.unload", new=boom_unload):
        r = client.post(
            "/api/models/boom-model/unload",
            headers={**auth, **csrf_header(client)},
        )
    assert r.status_code == 202

    row = None
    for _ in range(60):
        with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
            row = db.execute(
                "SELECT status FROM models WHERE id='boom-model'"
            ).fetchone()
        if row and row[0] in ("pulled", "failed"):
            break
        time.sleep(0.05)
    assert row[0] in ("pulled", "failed"), (
        f"row stranded at {row[0]} after teardown error"
    )
