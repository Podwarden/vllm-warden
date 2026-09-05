import sqlite3

import pytest


@pytest.mark.fresh_db
def test_lifespan_creates_db_with_schema(tmp_data_dir, client):
    """Calling any endpoint must trigger lifespan, which migrates the DB.

    ``fresh_db``: the ``client`` fixture normally hands the lifespan a
    pre-migrated copy, which would make this test pass even if the lifespan
    stopped migrating. This is the one test that must watch it happen.
    """
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


def test_lifespan_reconciles_stranded_rows_before_marking_the_runtime_dead(
    tmp_data_dir, client
):
    """#236 -- the ORDER of the two boot sweeps is the contract.

    ``reconcile_stranded_models`` demotes a transient row with no backing
    process ('loading' with no was-serving flag: an interrupted first load)
    to something an operator can act on. ``mark_runtime_dead_on_startup``
    sweeps every transient row into 'failed' + prior_status. Run the second
    first and the reconcile finds nothing: the row lands in 'failed' with
    prior_status='loading', which ``wants_restart`` reads as a serving model
    to bring back -- a load that never came up, retried forever, instead of a
    'registered'/'pulled' row with a Load button. Every other test drives the
    two functions in isolation; only a real boot sees the order.
    """
    client.get("/healthz")  # first boot: migrated DB
    db_path = tmp_data_dir / "vllm-warden.db"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, gpu_memory_utilization, trust_remote_code, extra_args, status, "
            "pulled_bytes) VALUES "
            "('stranded','stranded','o/r','main','[0]',1,0.9,0,'[]','loading',0)"
        )
        db.commit()

    from fastapi.testclient import TestClient

    from app.main import build_app

    with TestClient(build_app()) as c2:
        c2.get("/healthz")

    with sqlite3.connect(db_path) as db:
        status, last_error, prior_status = db.execute(
            "SELECT status, last_error, prior_status FROM models WHERE id='stranded'"
        ).fetchone()
    # The HF cache is empty, so the operator-actionable status is 'registered'
    # ('pulled' would be the answer with weights on disk).
    assert status == "registered", (
        f"got {status!r} (last_error={last_error!r}): reconcile_stranded_models must "
        "run BEFORE mark_runtime_dead_on_startup, or the dead-marking sweep has already "
        "turned every transient row into 'failed' and the reconcile is a no-op"
    )
    assert "recovered from an interrupted loading" in (last_error or "")
    assert prior_status is None, (
        "an interrupted first load must not be flagged for automatic restart"
    )
