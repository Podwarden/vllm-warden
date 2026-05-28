"""Integration tests for /api/models/{model_id}/settings.

Exercises both GET (full ModelRow surface) and PATCH (with the load-state
409 guard). Test pattern: same csrf/JWT bootstrap as
tests/integration/test_token_rotate_grace.py.
"""
import json
import sqlite3

import bcrypt
import httpx
import pytest
from httpx import ASGITransport


def _seed_admin(db_path):
    pw = bcrypt.hashpw(b"hunter2", bcrypt.gensalt()).decode()
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO users(username, password_hash) VALUES (?, ?)",
            ("admin", pw),
        )
        db.execute(
            "UPDATE setup_state SET step='done', draft=? WHERE id=1",
            (json.dumps({"allowed_gpu_indices": [0, 1, 2, 3]}),),
        )
        db.commit()


async def _login_and_prime_csrf(client):
    r = await client.post(
        "/api/auth/login", json={"username": "admin", "password": "hunter2"}
    )
    assert r.status_code == 200, r.text
    jwt = r.json()["access_token"]
    auth = {"Authorization": f"Bearer {jwt}"}

    r = await client.get("/api/csrf")
    assert r.status_code == 200, r.text
    csrf = r.json()["csrf"]
    mut = {**auth, "X-CSRF-Token": csrf}
    return auth, mut


async def _create_model(client, mut_headers, name="m1", gpus=None):
    """Create a model via POST /api/models. Returns the id."""
    body = {
        "served_model_name": name,
        "hf_repo": "meta-llama/Llama-3-8B",
        "hf_revision": "main",
        "gpu_indices": gpus or [0],
        "gpu_memory_utilization": 0.9,
    }
    r = await client.post("/api/models", json=body, headers=mut_headers)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _force_status(db_path, model_id, status):
    """Bypass the API and set models.status directly — needed to simulate 'loaded'."""
    with sqlite3.connect(db_path) as db:
        db.execute("UPDATE models SET status = ? WHERE id = ?", (status, model_id))
        db.commit()


@pytest.fixture
async def app_and_client(tmp_path, monkeypatch):
    monkeypatch.setenv("VW_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VW_HF_CACHE_DIR", str(tmp_path / "hf-cache"))
    monkeypatch.setenv("VW_COOKIE_SECRET", "test-secret-32-bytes-min-padding!")
    monkeypatch.setenv("VW_CONTAINER_GPU_COUNT", "4")

    from app.main import build_app

    app = build_app()
    lifespan_cm = app.router.lifespan_context(app)
    await lifespan_cm.__aenter__()
    try:
        _seed_admin(tmp_path / "vllm-warden.db")
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test", timeout=10.0
        ) as client:
            yield app, client, tmp_path
    finally:
        await lifespan_cm.__aexit__(None, None, None)


@pytest.mark.integration
async def test_get_model_settings_returns_full_modelrow(app_and_client):
    """GET returns every ModelRow field, including JSON-decoded list/dict columns."""
    _, client, _ = app_and_client
    auth, mut = await _login_and_prime_csrf(client)
    model_id = await _create_model(client, mut, name="surface-test", gpus=[0, 1])

    r = await client.get(f"/api/models/{model_id}/settings", headers=auth)
    assert r.status_code == 200, r.text
    body = r.json()
    for k in (
        "served_model_name", "hf_repo", "hf_revision",
        "gpu_indices", "tensor_parallel_size", "dtype", "max_model_len",
        "gpu_memory_utilization", "trust_remote_code",
        "extra_args", "extra_env",
    ):
        assert k in body, f"missing key {k}"
    assert body["served_model_name"] == "surface-test"
    assert body["gpu_indices"] == [0, 1]
    assert body["tensor_parallel_size"] == 2
    # status and lifecycle columns are also present (read-only on this endpoint).
    assert body["status"] == "registered"


@pytest.mark.integration
async def test_get_model_settings_404_for_unknown(app_and_client):
    _, client, _ = app_and_client
    auth, _ = await _login_and_prime_csrf(client)
    r = await client.get("/api/models/nonexistent-id/settings", headers=auth)
    assert r.status_code == 404, r.text


@pytest.mark.integration
async def test_patch_unloaded_model_succeeds(app_and_client):
    """PATCH on a model whose status != 'loaded' updates the row."""
    _, client, tmp_path = app_and_client
    _, mut = await _login_and_prime_csrf(client)
    model_id = await _create_model(client, mut, name="patch-ok")

    r = await client.patch(
        f"/api/models/{model_id}/settings",
        json={"max_model_len": 4096},
        headers=mut,
    )
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True

    # Verify the column was actually mutated.
    with sqlite3.connect(tmp_path / "vllm-warden.db") as db:
        row = db.execute(
            "SELECT max_model_len FROM models WHERE id = ?", (model_id,)
        ).fetchone()
    assert row[0] == 4096


@pytest.mark.integration
async def test_patch_loaded_model_returns_409(app_and_client):
    """PATCH on a loaded model is rejected — operator must unload first."""
    _, client, tmp_path = app_and_client
    _, mut = await _login_and_prime_csrf(client)
    model_id = await _create_model(client, mut, name="patch-locked")

    # Force the DB into "loaded" — supervisor would do this in real life.
    _force_status(tmp_path / "vllm-warden.db", model_id, "loaded")

    r = await client.patch(
        f"/api/models/{model_id}/settings",
        json={"max_model_len": 4096},
        headers=mut,
    )
    assert r.status_code == 409, r.text
    assert "unload" in r.text.lower()


@pytest.mark.integration
async def test_patch_unknown_field_returns_400(app_and_client):
    """Non-patchable / unknown fields are rejected as 400."""
    _, client, _ = app_and_client
    _, mut = await _login_and_prime_csrf(client)
    model_id = await _create_model(client, mut, name="patch-bad")

    # `status` is intentionally excluded from the patchable set.
    r = await client.patch(
        f"/api/models/{model_id}/settings",
        json={"status": "loaded"},
        headers=mut,
    )
    assert r.status_code == 400, r.text
    assert "status" in r.text

    # And a truly unknown key.
    r = await client.patch(
        f"/api/models/{model_id}/settings",
        json={"definitely_not_a_field": 1},
        headers=mut,
    )
    assert r.status_code == 400, r.text


@pytest.mark.integration
async def test_patch_404_for_unknown_model(app_and_client):
    _, client, _ = app_and_client
    _, mut = await _login_and_prime_csrf(client)
    r = await client.patch(
        "/api/models/nonexistent-id/settings",
        json={"max_model_len": 4096},
        headers=mut,
    )
    assert r.status_code == 404, r.text


@pytest.mark.integration
async def test_patch_json_field_roundtrips(app_and_client):
    """JSON-encoded columns (gpu_indices, extra_args, extra_env) round-trip through PATCH."""
    _, client, tmp_path = app_and_client
    _, mut = await _login_and_prime_csrf(client)
    model_id = await _create_model(client, mut, name="patch-json", gpus=[0])

    # Note: changing gpu_indices changes tensor_parallel_size too — patch both
    # so we don't violate any future consistency check at read time.
    r = await client.patch(
        f"/api/models/{model_id}/settings",
        json={
            "gpu_indices": [1, 2],
            "tensor_parallel_size": 2,
            "extra_args": ["--foo", "bar"],
            "extra_env": {"VLLM_LOGGING_LEVEL": "DEBUG"},
        },
        headers=mut,
    )
    assert r.status_code == 200, r.text

    r = await client.get(
        f"/api/models/{model_id}/settings",
        headers={"Authorization": mut["Authorization"]},
    )
    body = r.json()
    assert body["gpu_indices"] == [1, 2]
    assert body["tensor_parallel_size"] == 2
    assert body["extra_args"] == ["--foo", "bar"]
    assert body["extra_env"] == {"VLLM_LOGGING_LEVEL": "DEBUG"}
