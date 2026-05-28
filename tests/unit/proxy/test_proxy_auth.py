import json
import secrets
import sqlite3

from fastapi import Depends
from fastapi.testclient import TestClient

from app.proxy.auth import require_bearer
from tests.conftest import csrf_header


def _seed_done_with_token(db_path, plaintext):
    """Seed setup-done + insert a known token row."""
    from app.db.repos.tokens import hash_token
    with sqlite3.connect(db_path) as db:
        db.execute(
            "UPDATE setup_state SET step='done', draft=? WHERE id=1",
            (json.dumps({"allowed_gpu_indices": [0]}),),
        )
        tid = secrets.token_hex(16)
        db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope) VALUES (?, ?, ?, ?, ?)",
            (tid, "test", plaintext[:8], hash_token(plaintext), "inference"),
        )
        db.commit()
        return tid


def _build_test_app(settings):
    """Build a fresh FastAPI with a /protected route guarded by require_bearer.
    Reuses real settings/middleware via the main app's lifespan."""
    from app.main import build_app
    app = build_app()

    @app.post("/protected")
    async def protected(token=Depends(require_bearer)):
        return {"token_id": token.id}

    return app


def test_proxy_rejects_missing_bearer(tmp_data_dir):
    app = _build_test_app(tmp_data_dir)
    with TestClient(app) as client:
        client.get("/healthz")
        _seed_done_with_token(tmp_data_dir / "vllm-warden.db", "vw_unused1234567890abcdef")
        h = csrf_header(client)
        r = client.post("/protected", headers=h)
        assert r.status_code == 401


def test_proxy_rejects_unknown_bearer(tmp_data_dir):
    app = _build_test_app(tmp_data_dir)
    with TestClient(app) as client:
        client.get("/healthz")
        _seed_done_with_token(tmp_data_dir / "vllm-warden.db", "vw_unused1234567890abcdef")
        h = {**csrf_header(client), "Authorization": "Bearer vw_nonexistent99"}
        r = client.post("/protected", headers=h)
        assert r.status_code == 401


def test_proxy_rejects_revoked_bearer(tmp_data_dir):
    app = _build_test_app(tmp_data_dir)
    with TestClient(app) as client:
        client.get("/healthz")
        plaintext = "vw_validtoken1234567890abcdef12345"
        tid = _seed_done_with_token(tmp_data_dir / "vllm-warden.db", plaintext)
        # revoke it
        with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
            db.execute("UPDATE api_tokens SET revoked_at = datetime('now', '-1 second') WHERE id = ?", (tid,))
            db.commit()
        h = {**csrf_header(client), "Authorization": f"Bearer {plaintext}"}
        r = client.post("/protected", headers=h)
        assert r.status_code == 401


def test_proxy_accepts_valid_bearer_and_records_last_used(tmp_data_dir):
    app = _build_test_app(tmp_data_dir)
    with TestClient(app) as client:
        client.get("/healthz")
        plaintext = "vw_validtoken1234567890abcdef12345"
        tid = _seed_done_with_token(tmp_data_dir / "vllm-warden.db", plaintext)
        h = {**csrf_header(client), "Authorization": f"Bearer {plaintext}"}
        r = client.post("/protected", headers=h)
        assert r.status_code == 200
        assert r.json()["token_id"] == tid
        # last_used_at must now be populated
        with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
            cur = db.execute("SELECT last_used_at FROM api_tokens WHERE id = ?", (tid,))
            (last_used,) = cur.fetchone()
        assert last_used is not None


def test_proxy_rejects_non_vw_token(tmp_data_dir):
    app = _build_test_app(tmp_data_dir)
    with TestClient(app) as client:
        client.get("/healthz")
        _seed_done_with_token(tmp_data_dir / "vllm-warden.db", "vw_unused1234567890abcdef")
        h = {**csrf_header(client), "Authorization": "Bearer sk-openai-style"}
        r = client.post("/protected", headers=h)
        assert r.status_code == 401
