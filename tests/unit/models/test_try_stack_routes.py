"""try-stack trial-and-error routes (#162).

Adapted to this repo's sync ``TestClient`` + JWT login fixtures (the plan's
async-client / ``make_model`` snippet does not match the real fixtures). A
local ``_make_model`` helper creates a registered model via the create-model
route and returns its id. Routes hang off ``/api/models``, so paths are
``/api/models/{id}/try-stack``.
"""
from tests.conftest import csrf_header, jwt_login, seed_admin_user


def _seed_done(db_path, allowed=None):
    seed_admin_user(db_path, allowed_gpu_indices=allowed)


def _jwt_login(client):
    return jwt_login(client, username="admin", password="hunter2")


def _make_model(client, auth, h, served="ts1"):
    r = client.post("/api/models", json={
        "served_model_name": served,
        "hf_repo": "o/r",
        "gpu_indices": [0],
    }, headers={**auth, **h})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def test_try_stack_records_attempt_and_sets_engine(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db", allowed=[0])
    auth = _jwt_login(client)
    h = csrf_header(client)
    mid = _make_model(client, auth, h, served="ts1")

    r = client.post(f"/api/models/{mid}/try-stack", json={
        "channel": "cuda-stable", "vllm_version": "0.20.0",
    }, headers={**auth, **h})
    assert r.status_code == 201, r.text
    attempt_id = r.json()["attempt_id"]

    # engine axis now set on the model
    body = client.get(f"/api/models/{mid}", headers=auth).json()
    assert body["engine"]["channel"] == "cuda-stable"

    # history shows the pending attempt
    hist = client.get(f"/api/models/{mid}/try-stack", headers=auth).json()
    assert hist["attempts"][0]["result"] == "pending"

    # post a failed result → classifier fills category + suggestion
    res = client.post(f"/api/models/{mid}/try-stack/{attempt_id}", json={
        "result": "failed",
        "error": "CUDA error: no kernel image is available (sm_86)",
    }, headers={**auth, **h})
    assert res.status_code == 200, res.text
    assert res.json()["category"] == "cuda_arch_unsupported"
    assert res.json()["suggestion"]
