"""Tests for ``GET /api/header/metrics/stream`` — the live header
metrics SSE feeding the nav-bar instrument cluster.

We pin:

  * auth gate (SSE ticket required, JWT alone is rejected),
  * payload shape (the eight derived fields the FE consumes),
  * active-model lookup (DB join semantics),
  * cache reuse (the same ``_ProbeCache`` instance backs both this
    endpoint and ``/api/system/gpus``),
  * interval-seconds env clamp.

The SSE generator itself loops indefinitely (it's a live feed); rather
than coaxing the httpx TestClient into closing the stream mid-iter
(which previously hung the suite indefinitely — TestClient buffers
chunks until the response generator returns), we exercise the
generator's building blocks (``_payload``, ``_active_model``,
``_get_cache``, ``_interval_seconds``) as pure unit tests. This pins
the same observable contract — payload shape, active-model lookup,
cache reuse — without depending on TestClient streaming semantics.

Auth and ticket-mint paths are still driven through the real HTTP
client so the request-level gates are exercised end-to-end.
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

from app.header.routes_api import (
    _active_model,
    _get_cache,
    _interval_seconds,
    _payload,
)
from app.system.gpu import GpuLive, GpuSnapshot
from app.system.routes_gpus import _ProbeCache
from tests.conftest import jwt_login, seed_admin_user


def _seed_done(db_path) -> None:
    seed_admin_user(db_path, allowed_gpu_indices=[0, 1])


def _install_probe(client, snap: GpuSnapshot) -> dict[str, int]:
    counter: dict[str, int] = {"n": 0}

    async def fake_probe() -> GpuSnapshot:
        counter["n"] += 1
        return snap

    cache = _ProbeCache(ttl=2.0, clock=lambda: 0.0, probe=fake_probe)
    client.app.state.gpu_probe_cache = cache
    return counter


SNAP_TWO_GPUS = GpuSnapshot(
    gpus=[
        GpuLive(
            index=0, uuid="GPU-aaaa", name="NVIDIA RTX A4000",
            memory_total_mib=16376, memory_used_mib=12450,
            memory_free_mib=3926, utilization_pct=87,
        ),
        GpuLive(
            index=1, uuid="GPU-bbbb", name="NVIDIA RTX A4000",
            memory_total_mib=16376, memory_used_mib=100,
            memory_free_mib=16276, utilization_pct=0,
        ),
    ],
    apps=[],
    probe_error=None,
)


def test_header_metrics_stream_requires_ticket(tmp_data_dir, client):
    """JWT alone is not sufficient — the route requires a single-use
    SSE ticket (same as model log streams). Without a ticket query
    string the response should be rejected even with a valid Bearer.

    FastAPI returns 422 (Unprocessable Entity) when the required
    ``ticket`` query param is missing (validation layer fires before
    our dependency runs). With an *invalid* ticket the response is
    401. Either way the stream MUST NOT start — we assert both
    branches to pin the contract.
    """
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = jwt_login(client)
    # 1) Missing ticket → 422 from FastAPI Query(...) validation.
    r = client.get("/api/header/metrics/stream", headers=auth)
    assert r.status_code == 422, r.text
    # 2) Bogus ticket → 401 from require_sse_ticket consume().
    r = client.get(
        "/api/header/metrics/stream?ticket=not-a-real-ticket", headers=auth
    )
    assert r.status_code == 401, r.text


def test_header_metrics_stream_rejects_unauthenticated_ticket_mint(
    tmp_data_dir, client
):
    """No JWT → ticket mint 401s, so the operator can't even acquire
    the ticket needed to open the stream."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    r = client.post(
        "/api/auth/sse-ticket",
        json={"path": "/api/header/metrics/stream"},
    )
    assert r.status_code == 401


def test_header_metrics_ticket_mint_succeeds_with_jwt(tmp_data_dir, client):
    """The ticket-mint path itself (used by the FE singleton) must
    succeed when JWT-authenticated. This pins the contract the FE
    relies on without opening the streaming response."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = jwt_login(client)
    r = client.post(
        "/api/auth/sse-ticket",
        json={"path": "/api/header/metrics/stream"},
        headers=auth,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body.get("ticket"), str) and body["ticket"]


def test_payload_emits_all_required_fields_with_no_loaded_model():
    """``_payload`` derives the eight FE-consumed fields correctly and
    surfaces ``active_model: null`` when no model is loaded."""
    out = _payload(SNAP_TWO_GPUS, active_id=None, active_name=None)
    assert out["active_model"] is None
    assert out["active_model_id"] is None
    assert out["probe_error"] is None
    assert out["vram_used_mib"] == 12550  # 12450 + 100
    assert out["vram_total_mib"] == 32752  # 16376 * 2
    # max across GPUs.
    assert out["gpu_util_pct"] == 87
    # 12550 / 32752 ≈ 0.383 → 38 %.
    assert out["vram_pct"] == 38
    assert len(out["gpus"]) == 2
    assert out["gpus"][0]["name"] == "NVIDIA RTX A4000"
    assert out["gpus"][0]["memory_used_mib"] == 12450
    # The timestamp ends in Z (RFC-3339 UTC).
    assert out["ts"].endswith("Z")


def test_payload_surfaces_active_model_when_supplied():
    """When the active-model tuple is non-null, ``_payload`` echoes
    both ``active_model_id`` and ``active_model`` into the frame."""
    out = _payload(
        SNAP_TWO_GPUS, active_id="m-loaded", active_name="gpt-oss-20b"
    )
    assert out["active_model"] == "gpt-oss-20b"
    assert out["active_model_id"] == "m-loaded"


def test_payload_with_empty_snapshot_surfaces_probe_error():
    """An empty GPU snapshot with a probe_error string surfaces through
    the payload so the FE can degrade gracefully."""
    snap = GpuSnapshot(gpus=[], apps=[], probe_error="nvidia-smi unavailable")
    out = _payload(snap, active_id=None, active_name=None)
    assert out["probe_error"] == "nvidia-smi unavailable"
    assert out["gpus"] == []
    assert out["vram_total_mib"] == 0
    assert out["vram_used_mib"] == 0
    assert out["vram_pct"] == 0
    assert out["gpu_util_pct"] == 0


async def test_active_model_returns_none_when_no_loaded_row(
    tmp_data_dir, client
):
    """``_active_model`` returns ``(None, None)`` when no models row has
    status='loaded'."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path)
    settings = client.app.state.settings
    out = await _active_model(settings.db_path)
    assert out == (None, None, None)


async def test_active_model_returns_tuple_when_loaded_row_present(
    tmp_data_dir, client
):
    """When a models row is status='loaded' AND has a model_runtime
    sibling, ``_active_model`` returns ``(model_id, served_model_name)``.
    The join on ``model_runtime`` ensures the supervisor actually owns
    the loaded row (not just the DB)."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path)
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
            "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
            "gpu_memory_utilization, trust_remote_code, extra_args, status) "
            "VALUES ('m-loaded', 'gpt-oss-20b', 'o/r', 'main', '[0]', 1, "
            "NULL, NULL, 0.9, 0, '[]', 'loaded')"
        )
        db.execute(
            "INSERT INTO model_runtime(model_id, pid, port) "
            "VALUES ('m-loaded', 99999, 11000)"
        )
        db.commit()

    settings = client.app.state.settings
    out = await _active_model(settings.db_path)
    assert out == ("m-loaded", "gpt-oss-20b", "loaded")


async def test_active_model_skips_loaded_row_without_runtime(
    tmp_data_dir, client
):
    """A models row marked 'loaded' but with no model_runtime sibling
    must NOT surface — the supervisor doesn't own it. Pins the JOIN
    semantics (LEFT JOIN would have leaked stale state)."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path)
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
            "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
            "gpu_memory_utilization, trust_remote_code, extra_args, status) "
            "VALUES ('m-orphan', 'orphaned-model', 'o/r', 'main', '[0]', 1, "
            "NULL, NULL, 0.9, 0, '[]', 'loaded')"
        )
        db.commit()

    settings = client.app.state.settings
    out = await _active_model(settings.db_path)
    assert out == (None, None, None)


async def test_get_cache_reuses_system_gpus_cache(client):
    """``_get_cache`` returns ``app.state.gpu_probe_cache`` so a tab
    open on both views does not double the nvidia-smi load. Drive
    /api/system/gpus first to populate the cache, then assert the
    helper hands back the same instance (not a fresh ``_ProbeCache``)."""
    client.get("/healthz")
    # Force-install a known cache instance.
    counter = _install_probe(client, SNAP_TWO_GPUS)
    installed = client.app.state.gpu_probe_cache

    # Fake request that points at the app state — _get_cache reads
    # ``request.app.state.gpu_probe_cache``.
    fake_request = SimpleNamespace(app=client.app)
    got = _get_cache(fake_request)  # type: ignore[arg-type]
    assert got is installed

    # Driving the cache from the helper must reuse the same probe
    # function — calling .get() twice within the TTL should run the
    # probe exactly once. Pins the "no double probe" contract.
    await got.get()
    await got.get()
    assert counter["n"] == 1, f"expected single probe, got {counter['n']}"


def test_get_cache_lazy_initialises_when_absent(client):
    """If ``app.state.gpu_probe_cache`` was never set, the helper
    installs a fresh ``_ProbeCache`` on first call (and reuses it on
    subsequent calls — same identity)."""
    client.get("/healthz")
    # Wipe any prior probe cache the test client may have lazily
    # installed during /healthz. Starlette's State raises ``KeyError``
    # (not AttributeError) when the key is absent.
    try:
        del client.app.state.gpu_probe_cache
    except (AttributeError, KeyError):
        pass

    fake_request = SimpleNamespace(app=client.app)
    first = _get_cache(fake_request)  # type: ignore[arg-type]
    second = _get_cache(fake_request)  # type: ignore[arg-type]
    assert isinstance(first, _ProbeCache)
    assert first is second


def test_header_metrics_interval_clamps_to_floor():
    """``_interval_seconds`` floors at 0.5s so a misconfigured env can't
    spin the SSE loop hot."""
    import os

    saved = os.environ.get("VW_HEADER_METRICS_INTERVAL_S")
    try:
        os.environ["VW_HEADER_METRICS_INTERVAL_S"] = "0.01"
        assert _interval_seconds() == 0.5
        os.environ["VW_HEADER_METRICS_INTERVAL_S"] = "not-a-number"
        assert _interval_seconds() == 2.0
        os.environ.pop("VW_HEADER_METRICS_INTERVAL_S", None)
        assert _interval_seconds() == 2.0
        os.environ["VW_HEADER_METRICS_INTERVAL_S"] = "5.0"
        assert _interval_seconds() == 5.0
    finally:
        if saved is None:
            os.environ.pop("VW_HEADER_METRICS_INTERVAL_S", None)
        else:
            os.environ["VW_HEADER_METRICS_INTERVAL_S"] = saved


async def test_active_model_surfaces_loading_row_without_runtime(
    tmp_data_dir, client
):
    """A row still in 'loading' must surface so the header badge can say
    "loading" rather than falling through to "idle".

    A loading row has NO ``model_runtime`` sibling — the supervisor writes
    that only once the engine answers — so this case is exactly the one the
    strict JOIN used to drop. A 27B load takes minutes; reading "idle" for
    that whole window told the operator the opposite of the truth.
    """
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path)
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
            "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
            "gpu_memory_utilization, trust_remote_code, extra_args, status) "
            "VALUES ('m-loading', 'qwen3.8-27b', 'o/r', 'main', '[0]', 1, "
            "NULL, NULL, 0.9, 0, '[]', 'loading')"
        )
        db.commit()

    settings = client.app.state.settings
    out = await _active_model(settings.db_path)
    assert out == ("m-loading", "qwen3.8-27b", "loading")


async def test_active_model_prefers_loaded_over_loading(tmp_data_dir, client):
    """With both a serving row and a loading row present, the serving one
    wins the identity slot — the badge must not demote a live engine to
    "loading" because a second model started loading beside it."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path)
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
            "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
            "gpu_memory_utilization, trust_remote_code, extra_args, status) "
            "VALUES ('m-loading', 'incoming', 'o/r', 'main', '[0]', 1, "
            "NULL, NULL, 0.9, 0, '[]', 'loading')"
        )
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
            "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
            "gpu_memory_utilization, trust_remote_code, extra_args, status) "
            "VALUES ('m-loaded', 'serving', 'o/r', 'main', '[1]', 1, "
            "NULL, NULL, 0.9, 0, '[]', 'loaded')"
        )
        db.execute(
            "INSERT INTO model_runtime(model_id, pid, port) "
            "VALUES ('m-loaded', 99999, 11000)"
        )
        db.commit()

    settings = client.app.state.settings
    out = await _active_model(settings.db_path)
    assert out == ("m-loaded", "serving", "loaded")


def test_payload_surfaces_active_model_status():
    """``_payload`` echoes the status so the frontend can pick the label
    and dot colour without re-deriving state from the name alone."""
    out = _payload(
        SNAP_TWO_GPUS,
        active_id="m-loading",
        active_name="qwen3.8-27b",
        active_status="loading",
    )
    assert out["active_model"] == "qwen3.8-27b"
    assert out["active_model_id"] == "m-loading"
    assert out["active_model_status"] == "loading"


def test_payload_active_model_status_is_null_when_idle():
    """No model at all → the status field is present but null, so the
    frontend never has to distinguish "absent key" from "nothing loaded"."""
    out = _payload(
        SNAP_TWO_GPUS, active_id=None, active_name=None, active_status=None
    )
    assert out["active_model"] is None
    assert out["active_model_status"] is None


async def test_active_model_surfaces_failed_row(tmp_data_dir, client):
    """A crashed model must surface as 'failed' so the badge can go red.

    This is the state that matters most operationally and was the most
    invisible: after an engine death the row sits in 'failed' indefinitely
    (nothing auto-restores it), while the header cheerfully read "idle" —
    indistinguishable from a clean box with nothing loaded.
    """
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path)
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
            "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
            "gpu_memory_utilization, trust_remote_code, extra_args, status) "
            "VALUES ('m-failed', 'crashed-model', 'o/r', 'main', '[0]', 1, "
            "NULL, NULL, 0.9, 0, '[]', 'failed')"
        )
        db.commit()

    settings = client.app.state.settings
    out = await _active_model(settings.db_path)
    assert out == ("m-failed", "crashed-model", "failed")


async def test_active_model_prefers_loading_over_failed(tmp_data_dir, client):
    """Retrying a crashed model must flip the badge to 'loading', not leave
    it red — the operator's retry has to be visibly acknowledged."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path)
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
            "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
            "gpu_memory_utilization, trust_remote_code, extra_args, status) "
            "VALUES ('m-failed', 'crashed-model', 'o/r', 'main', '[0]', 1, "
            "NULL, NULL, 0.9, 0, '[]', 'failed')"
        )
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
            "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
            "gpu_memory_utilization, trust_remote_code, extra_args, status) "
            "VALUES ('m-loading', 'incoming', 'o/r', 'main', '[1]', 1, "
            "NULL, NULL, 0.9, 0, '[]', 'loading')"
        )
        db.commit()

    settings = client.app.state.settings
    out = await _active_model(settings.db_path)
    assert out == ("m-loading", "incoming", "loading")


async def test_active_model_prefers_loaded_over_failed(tmp_data_dir, client):
    """A live engine outranks an unrelated stale 'failed' row — one bad row
    must never paint a serving box red."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path)
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
            "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
            "gpu_memory_utilization, trust_remote_code, extra_args, status) "
            "VALUES ('m-failed', 'crashed-model', 'o/r', 'main', '[0]', 1, "
            "NULL, NULL, 0.9, 0, '[]', 'failed')"
        )
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
            "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
            "gpu_memory_utilization, trust_remote_code, extra_args, status) "
            "VALUES ('m-loaded', 'serving', 'o/r', 'main', '[1]', 1, "
            "NULL, NULL, 0.9, 0, '[]', 'loaded')"
        )
        db.execute(
            "INSERT INTO model_runtime(model_id, pid, port) "
            "VALUES ('m-loaded', 99999, 11000)"
        )
        db.commit()

    settings = client.app.state.settings
    out = await _active_model(settings.db_path)
    assert out == ("m-loaded", "serving", "loaded")
