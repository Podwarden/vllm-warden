import sqlite3


def test_lifespan_creates_db_with_schema(tmp_data_dir, client):
    """Calling any endpoint must trigger lifespan, which migrates the DB."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    assert db_path.exists()
    with sqlite3.connect(db_path) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "users" in tables
        assert "models" in tables


def test_lifespan_clears_runtime_table(tmp_data_dir, client):
    """Stale model_runtime rows must be wiped on startup."""
    client.get("/healthz")  # boot
    db_path = tmp_data_dir / "vllm-warden.db"
    with sqlite3.connect(db_path) as db:
        # Insert a model + runtime row, then re-boot via second client.
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, gpu_memory_utilization, trust_remote_code, extra_args, status, "
            "pulled_bytes) VALUES "
            "('m1','m1','o/r','main','[0]',1,0.9,0,'[]','loaded',0)"
        )
        db.execute(
            "INSERT INTO model_runtime(model_id, pid, port) VALUES ('m1', 9999, 10000)"
        )
        db.commit()

    # Reboot app
    from fastapi.testclient import TestClient

    from app.main import build_app
    with TestClient(build_app()) as c2:
        c2.get("/healthz")

    with sqlite3.connect(db_path) as db:
        (n,) = db.execute("SELECT COUNT(*) FROM model_runtime").fetchone()
        assert n == 0
        (status,) = db.execute("SELECT status FROM models WHERE id='m1'").fetchone()
        assert status == "failed"
