import json
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import bcrypt

from app.db.repos.tokens import hash_token


def _seed(db_path, *, model_status="loaded"):
    pw = bcrypt.hashpw(b"hunter2", bcrypt.gensalt()).decode()
    with sqlite3.connect(db_path) as db:
        db.execute("INSERT INTO users(username, password_hash) VALUES (?, ?)", ("admin", pw))
        db.execute(
            "UPDATE setup_state SET step='done', draft=? WHERE id=1",
            (json.dumps({"allowed_gpu_indices": [0, 1, 2, 3]}),),
        )
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization, "
            "trust_remote_code, extra_args, status, pulled_bytes, pulled_total, last_error) "
            "VALUES ('qwen','qwen','Qwen/Qwen3.5-9B','main',?,1,'auto',4096,0.9,0,'[]',?,0,NULL,NULL)",
            (json.dumps([0]), model_status),
        )
        plaintext = "vw_validtoken1234567890abcdef12345"
        db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope) VALUES (?, ?, ?, ?, ?)",
            ("tok1", "test", plaintext[:8], hash_token(plaintext), "inference"),
        )
        db.commit()
        return plaintext


def test_proxy_404_when_model_not_loaded(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed(tmp_data_dir / "vllm-warden.db", model_status="pulled")
    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {plaintext}"},
        json={"model": "qwen", "messages": []},
    )
    assert r.status_code == 404


def test_proxy_404_when_model_unknown(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed(tmp_data_dir / "vllm-warden.db")
    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {plaintext}"},
        json={"model": "ghost", "messages": []},
    )
    assert r.status_code == 404


def test_proxy_400_when_model_field_missing(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed(tmp_data_dir / "vllm-warden.db")
    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {plaintext}"},
        json={"messages": []},
    )
    assert r.status_code == 400


def test_proxy_forwards_to_subprocess_port(tmp_data_dir, client):
    """When model is loaded, requests route to its 127.0.0.1:<port>."""
    client.get("/healthz")
    plaintext = _seed(tmp_data_dir / "vllm-warden.db")
    # Register the supervisor's port for 'qwen' so _resolve_target finds it.
    client.app.state.supervisor._ports["qwen"] = 19099

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "application/json"}
    fake_resp.aread = AsyncMock(return_value=b'{"id":"x","model":"qwen","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":0}}')
    fake_resp.aclose = AsyncMock()

    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=fake_resp)) as send, \
         patch("app.proxy.routes._record_counters", new=AsyncMock()):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200
    sent_url = str(send.call_args.args[0].url)
    assert "127.0.0.1:19099" in sent_url
    assert "/v1/chat/completions" in sent_url


def _set_priority(db_path, priority):
    with sqlite3.connect(db_path) as db:
        db.execute("UPDATE api_tokens SET priority=? WHERE id='tok1'", (priority,))
        db.commit()


def test_proxy_injects_vllm_priority_into_forwarded_body(tmp_data_dir, client):
    """#173 part B — a non-zero warden priority must be pushed into the engine
    as ``priority = -warden_priority`` (vLLM orders ASCENDING, so negative puts
    high-priority traffic ahead of default-0 traffic)."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed(db_path)
    _set_priority(db_path, 7)
    client.app.state.supervisor._ports["qwen"] = 19099

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "application/json"}
    fake_resp.aread = AsyncMock(return_value=b'{"id":"x","model":"qwen","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":0}}')
    fake_resp.aclose = AsyncMock()

    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=fake_resp)) as send, \
         patch("app.proxy.routes._record_counters", new=AsyncMock()):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200
    forwarded = json.loads(send.call_args.args[0].content)
    assert forwarded["priority"] == -7


def test_proxy_does_not_inject_priority_when_zero(tmp_data_dir, client):
    """Warden priority 0 maps to vLLM's default (inert). The forwarded body must
    stay free of a ``priority`` field so an unprioritised request is byte-clean."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed(db_path)
    _set_priority(db_path, 0)
    client.app.state.supervisor._ports["qwen"] = 19099

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "application/json"}
    fake_resp.aread = AsyncMock(return_value=b'{"id":"x","model":"qwen","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":0}}')
    fake_resp.aclose = AsyncMock()

    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=fake_resp)) as send, \
         patch("app.proxy.routes._record_counters", new=AsyncMock()):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200
    forwarded = json.loads(send.call_args.args[0].content)
    assert "priority" not in forwarded


def test_proxy_does_not_override_client_supplied_priority(tmp_data_dir, client):
    """If the client already set ``priority`` in its request body, the proxy
    must leave it untouched — the explicit client value wins over the token map."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed(db_path)
    _set_priority(db_path, 7)
    client.app.state.supervisor._ports["qwen"] = 19099

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "application/json"}
    fake_resp.aread = AsyncMock(return_value=b'{"id":"x","model":"qwen","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":0}}')
    fake_resp.aclose = AsyncMock()

    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=fake_resp)) as send, \
         patch("app.proxy.routes._record_counters", new=AsyncMock()):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}], "priority": 3},
        )
    assert r.status_code == 200
    forwarded = json.loads(send.call_args.args[0].content)
    assert forwarded["priority"] == 3
