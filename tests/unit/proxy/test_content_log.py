"""Tests for the config-gated, per-token-scoped content logger (content_log.py).

Pure-function tests cover the gate and the truncation. Integration tests drive
a real proxied request through the FastAPI TestClient (mirroring
test_proxy_accounting.py: seed a loaded model + token, patch httpx.send) and
assert on the JSONL file the logger writes — for both a non-streaming and a
streaming completion, plus the disabled / not-allowlisted no-op paths and the
max-chars cap.
"""

import dataclasses
import json
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import bcrypt

from app.config import Settings
from app.db.repos.tokens import hash_token
from app.proxy import content_log

# ---------------------------------------------------------------------------
# Pure-function tests: should_log gate + _cap truncation
# ---------------------------------------------------------------------------

def _settings(**over) -> Settings:
    base = dict(
        data_dir=Path("/data"),
        hf_cache_dir=Path("/hf"),
        cookie_secret="x" * 32,
        container_gpu_count=0,
    )
    base.update(over)
    return Settings(**base)


def test_should_log_disabled_returns_false():
    s = _settings(content_log_enabled=False, content_log_tokens=frozenset({"tok1"}))
    assert content_log.should_log(s, "tok1") is False


def test_should_log_enabled_but_token_not_allowlisted_returns_false():
    s = _settings(content_log_enabled=True, content_log_tokens=frozenset({"other"}))
    assert content_log.should_log(s, "tok1") is False


def test_should_log_enabled_and_allowlisted_returns_true():
    s = _settings(content_log_enabled=True, content_log_tokens=frozenset({"tok1"}))
    assert content_log.should_log(s, "tok1") is True


def test_should_log_none_token_returns_false():
    s = _settings(content_log_enabled=True, content_log_tokens=frozenset({"tok1"}))
    assert content_log.should_log(s, None) is False


def test_cap_under_limit_unchanged():
    assert content_log._cap("hello", 40000) == "hello"


def test_cap_none_passthrough():
    assert content_log._cap(None, 10) is None


def test_cap_over_limit_truncates_with_marker():
    out = content_log._cap("abcdefghij", 4)
    assert out == "abcd…[truncated 6 chars]"


# ---------------------------------------------------------------------------
# Integration tests through the proxy forward path
# ---------------------------------------------------------------------------

def _seed_loaded(db_path):
    pw = bcrypt.hashpw(b"hunter2", bcrypt.gensalt()).decode()
    with sqlite3.connect(db_path) as db:
        db.execute("INSERT INTO users(username, password_hash) VALUES (?, ?)", ("admin", pw))
        db.execute(
            "UPDATE setup_state SET step='done', draft=? WHERE id=1",
            (json.dumps({"allowed_gpu_indices": [0]}),),
        )
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization, "
            "trust_remote_code, extra_args, status, pulled_bytes, pulled_total, last_error) "
            "VALUES ('qwen','qwen','Qwen/Qwen3.5-9B','main',?,1,'auto',4096,0.9,0,'[]','loaded',0,NULL,NULL)",
            (json.dumps([0]),),
        )
        plaintext = "vw_validtoken1234567890abcdef12345"
        db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope) VALUES (?, ?, ?, ?, ?)",
            ("tok1", "test", plaintext[:8], hash_token(plaintext), "inference"),
        )
        db.commit()
        return plaintext


def _make_fake_tokenizer(token_count_for):
    cache = MagicMock()
    cache.count = AsyncMock(side_effect=lambda repo, text, *, trust_remote_code: token_count_for(text))
    return cache


def _configure(client, *, path, enabled, tokens, max_chars=40000):
    """Replace app.state.settings (frozen) with a content-log-configured copy."""
    client.app.state.settings = dataclasses.replace(
        client.app.state.settings,
        content_log_enabled=enabled,
        content_log_tokens=frozenset(tokens),
        content_log_path=path,
        content_log_max_chars=max_chars,
    )


def _fake_nonstream_resp(body: dict):
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "application/json"}
    resp.aread = AsyncMock(return_value=json.dumps(body).encode())
    resp.aclose = AsyncMock()
    return resp


def _read_lines(path: Path):
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def test_disabled_writes_nothing(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _make_fake_tokenizer(lambda text: 5 if text else 0)
    log_path = tmp_data_dir / "content.jsonl"
    _configure(client, path=log_path, enabled=False, tokens={"tok1"})

    body = {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    }
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_fake_nonstream_resp(body))):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200
    assert _read_lines(log_path) == []


def test_enabled_but_token_not_allowlisted_writes_nothing(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _make_fake_tokenizer(lambda text: 5 if text else 0)
    log_path = tmp_data_dir / "content.jsonl"
    # Enabled, but the allowlist does NOT contain tok1.
    _configure(client, path=log_path, enabled=True, tokens={"someone-else"})

    body = {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    }
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_fake_nonstream_resp(body))):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200
    assert _read_lines(log_path) == []


def test_nonstream_allowlisted_writes_one_line(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _make_fake_tokenizer(lambda text: 5 if "hi" in text else 0)
    log_path = tmp_data_dir / "content.jsonl"
    _configure(client, path=log_path, enabled=True, tokens={"tok1"})

    body = {
        "choices": [{"message": {"content": "the answer is 42"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 4},
    }
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_fake_nonstream_resp(body))):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={
                "model": "qwen",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert r.status_code == 200

    lines = _read_lines(log_path)
    assert len(lines) == 1
    rec = lines[0]
    assert rec["token_id"] == "tok1"
    assert rec["model_id"] == "qwen"
    assert rec["served_name"] == "qwen"
    assert rec["stream"] is False
    assert rec["max_tokens"] == 128
    assert rec["prompt_tokens"] == 5
    assert rec["completion_tokens"] == 4
    assert rec["finish_reason"] == "stop"
    assert rec["prompt"] == "hi"
    assert rec["completion"] == "the answer is 42"
    assert isinstance(rec["ts"], int | float)


def test_stream_allowlisted_writes_one_line(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _make_fake_tokenizer(
        lambda text: len(text.split()) if text else 0
    )
    log_path = tmp_data_dir / "content.jsonl"
    _configure(client, path=log_path, enabled=True, tokens={"tok1"})

    sse_chunks = [
        b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\n',
        b'data: {"choices":[{"delta":{"content":" world"},"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
    ]

    async def aiter():
        for c in sse_chunks:
            yield c

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "text/event-stream"}
    fake_resp.aiter_bytes = aiter
    fake_resp.aclose = AsyncMock()

    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=fake_resp)):
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={
                "model": "qwen",
                "stream": True,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
            },
        ) as r:
            for _ in r.iter_bytes():
                pass
            assert r.status_code == 200

    lines = _read_lines(log_path)
    assert len(lines) == 1
    rec = lines[0]
    assert rec["token_id"] == "tok1"
    assert rec["stream"] is True
    assert rec["max_tokens"] == 64
    assert rec["prompt_tokens"] == 1  # "hi" -> 1 word
    assert rec["completion_tokens"] == 2  # "hello world" -> 2 words
    assert rec["finish_reason"] == "stop"
    assert rec["prompt"] == "hi"
    assert rec["completion"] == "hello world"


def test_max_chars_truncation_in_written_line(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _make_fake_tokenizer(lambda text: 5 if text else 0)
    log_path = tmp_data_dir / "content.jsonl"
    # Cap at 5 chars so the 10-char completion gets truncated.
    _configure(client, path=log_path, enabled=True, tokens={"tok1"}, max_chars=5)

    body = {
        "choices": [{"message": {"content": "abcdefghij"}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 4},
    }
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_fake_nonstream_resp(body))):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200

    lines = _read_lines(log_path)
    assert len(lines) == 1
    assert lines[0]["completion"] == "abcde…[truncated 5 chars]"
