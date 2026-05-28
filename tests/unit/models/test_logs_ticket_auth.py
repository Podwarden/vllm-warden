import sqlite3

import bcrypt


def _seed_admin(db_path, username="admin", pw="hunter2"):
    h = bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO users(username, password_hash) VALUES (?, ?)",
            (username, h),
        )
        db.execute("UPDATE setup_state SET step='done' WHERE id=1")
        db.commit()


def test_logs_stream_rejects_without_ticket(tmp_data_dir, client):
    client.get("/healthz")
    _seed_admin(tmp_data_dir / "vllm-warden.db")  # mark setup done; otherwise gate_setup redirects
    r = client.get("/api/models/some-id/logs/stream")
    # Pinned: rejection happens at FastAPI's Query(...) validator layer
    # (missing required `ticket` query param) which returns 422. If a future
    # middleware short-circuits earlier (e.g. an auth layer returning 401),
    # this assertion will catch the silent layering change.
    assert r.status_code == 422


def test_logs_stream_rejects_wrong_path_ticket(tmp_data_dir, client):
    client.get("/healthz")
    _seed_admin(tmp_data_dir / "vllm-warden.db")
    # Mint a ticket bound to a DIFFERENT path; consuming it on /some-id/... must fail.
    ticket = client.app.state.sse_tickets.mint(
        "admin", "/api/models/OTHER/logs/stream"
    )
    r = client.get(f"/api/models/some-id/logs/stream?ticket={ticket}")
    assert r.status_code == 401


def test_logs_stream_rejects_bogus_ticket(tmp_data_dir, client):
    client.get("/healthz")
    _seed_admin(tmp_data_dir / "vllm-warden.db")
    r = client.get("/api/models/some-id/logs/stream?ticket=not-a-valid-ticket")
    assert r.status_code == 401
