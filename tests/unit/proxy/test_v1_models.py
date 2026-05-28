import json
import sqlite3

from app.db.repos.tokens import hash_token


def _seed(db_path):
    plaintext = "vw_validtoken1234567890abcdef12345"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "UPDATE setup_state SET step='done', draft=? WHERE id=1",
            (json.dumps({"allowed_gpu_indices": [0]}),),
        )
        db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope) VALUES (?, ?, ?, ?, ?)",
            ("tok1", "test", plaintext[:8], hash_token(plaintext), "inference"),
        )
        # 1 loaded, 1 pulled, 1 failed
        for mid, served, status in [
            ("qwen", "qwen", "loaded"),
            ("other", "other", "pulled"),
            ("dead", "dead", "failed"),
        ]:
            db.execute(
                "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
                "tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization, "
                "trust_remote_code, extra_args, status, pulled_bytes, pulled_total, last_error) "
                "VALUES (?,?,'r','main',?,1,'auto',4096,0.9,0,'[]',?,0,NULL,NULL)",
                (mid, served, json.dumps([0]), status),
            )
        db.commit()
        return plaintext


def test_v1_models_lists_only_loaded(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed(tmp_data_dir / "vllm-warden.db")
    r = client.get("/v1/models", headers={"Authorization": f"Bearer {plaintext}"})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    ids = [m["id"] for m in body["data"]]
    assert ids == ["qwen"]
    assert body["data"][0]["object"] == "model"
    assert body["data"][0]["owned_by"] == "vllm-warden"


def test_v1_models_requires_bearer(tmp_data_dir, client):
    client.get("/healthz")
    _seed(tmp_data_dir / "vllm-warden.db")
    r = client.get("/v1/models")
    assert r.status_code == 401
