import sqlite3

from tests.conftest import csrf_header, jwt_login, seed_admin_user


# #104 + #55 fix — route through the shared seed + login helpers.
def _seed_done(db_path):
    # Delegate to the shared helper (closes #104). The previous inline
    # implementation opened a sqlite3 connection in default rollback-journal
    # mode against a DB file the async TestClient lifespan had opened in
    # WAL mode, which raced under CI contention and surfaced as 401
    # "invalid credentials" on /api/auth/login (the seeded admin row
    # appeared to be missing until the WAL was checkpointed).
    seed_admin_user(db_path)


def _jwt_login(client, username="admin", password="hunter2"):
    # Delegate to the shared helper (closes #55). Adds a bounded
    # retry-on-401 on top of #104's WAL-checkpoint fix — defence in
    # depth for any residual aiosqlite read-after-write window or
    # future transient 401 not caused by seeding.
    return jwt_login(client, username=username, password=password)


def test_create_with_expires_in_days(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    h = csrf_header(client)
    r = client.post(
        "/api/tokens",
        json={"name": "ci-bot", "expires_in_days": 90},
        headers={**auth, **h},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["plaintext"].startswith("vw_")
    assert body["expires_at"] is not None


def test_create_with_expires_in_days_zero_never_expires(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    h = csrf_header(client)
    r = client.post(
        "/api/tokens",
        json={"name": "forever", "expires_in_days": 0},
        headers={**auth, **h},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["expires_at"] is None


def test_rotate_returns_new_plaintext_with_rotated_from(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    h = csrf_header(client)
    create = client.post("/api/tokens", json={"name": "ci-bot"}, headers={**auth, **h})
    assert create.status_code == 201, create.text
    old_id = create.json()["id"]

    r = client.post(
        f"/api/tokens/{old_id}/rotate",
        json={"grace_hours": 24},
        headers={**auth, **h},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["plaintext"].startswith("vw_")
    assert body["rotated_from"] == old_id
    assert body["id"] != old_id
    assert body["prefix"] == body["plaintext"][:8]
    # #150 — successor keeps original name; predecessor renamed to "(old 1)".
    assert body["name"] == "ci-bot"
    assert body["renamed_to"] == "ci-bot (old 1)"


def test_rotate_keeps_original_name_on_new_row_list(tmp_data_dir, client):
    """#150 — after rotate, the active row in /api/tokens lists shows the
    ORIGINAL name and the rotated row shows ``"{name} (old 1)"``."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    h = csrf_header(client)
    create = client.post("/api/tokens", json={"name": "prod-bot"}, headers={**auth, **h})
    old_id = create.json()["id"]

    rot = client.post(
        f"/api/tokens/{old_id}/rotate",
        json={"grace_hours": 24},
        headers={**auth, **h},
    )
    assert rot.status_code == 201, rot.text
    new_id = rot.json()["id"]

    r = client.get("/api/tokens", headers=auth)
    assert r.status_code == 200
    items = {it["id"]: it for it in r.json()["items"]}
    assert items[new_id]["name"] == "prod-bot"
    assert items[old_id]["name"] == "prod-bot (old 1)"


def test_rotate_second_time_uses_old_2(tmp_data_dir, client):
    """#150 — cascading rotation: second rotate must allocate ``(old 2)``."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    h = csrf_header(client)
    create = client.post("/api/tokens", json={"name": "prod-bot"}, headers={**auth, **h})
    old_id = create.json()["id"]

    rot1 = client.post(
        f"/api/tokens/{old_id}/rotate",
        json={"grace_hours": 24},
        headers={**auth, **h},
    )
    assert rot1.status_code == 201
    new1_id = rot1.json()["id"]
    assert rot1.json()["renamed_to"] == "prod-bot (old 1)"

    # Rotate the freshly-minted active row.
    rot2 = client.post(
        f"/api/tokens/{new1_id}/rotate",
        json={"grace_hours": 24},
        headers={**auth, **h},
    )
    assert rot2.status_code == 201
    assert rot2.json()["renamed_to"] == "prod-bot (old 2)"
    assert rot2.json()["name"] == "prod-bot"


def test_rotate_already_rotated_returns_409(tmp_data_dir, client):
    """#150 — rotating an already-rotated row is rejected with 409.

    The UI disables the button on rotated rows, but a direct API caller
    (curl, MCP, scripted client) must get a clean error instead of
    silently allocating ``(old 2)`` on a stale double-click.
    """
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    h = csrf_header(client)
    create = client.post("/api/tokens", json={"name": "ci-bot"}, headers={**auth, **h})
    old_id = create.json()["id"]

    first = client.post(
        f"/api/tokens/{old_id}/rotate",
        json={"grace_hours": 24},
        headers={**auth, **h},
    )
    assert first.status_code == 201

    # Same old_id, already rotated — must 409.
    second = client.post(
        f"/api/tokens/{old_id}/rotate",
        json={"grace_hours": 24},
        headers={**auth, **h},
    )
    assert second.status_code == 409, second.text
    assert "already rotated" in second.json()["detail"].lower()


def test_rotate_404_for_unknown_token(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    h = csrf_header(client)
    r = client.post(
        "/api/tokens/deadbeefdeadbeefdeadbeefdeadbeef/rotate",
        json={"grace_hours": 24},
        headers={**auth, **h},
    )
    assert r.status_code == 404, r.text


def test_list_includes_status_fields(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    h = csrf_header(client)
    client.post("/api/tokens", json={"name": "ci-bot"}, headers={**auth, **h})

    r = client.get("/api/tokens", headers=auth)
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 1
    item = items[0]
    for key in (
        "expires_at",
        "is_expired",
        "is_near_expiry",
        "rotated_at",
        "rotated_from",
        "successor_id",
        "revoked_at",
    ):
        assert key in item, f"missing key {key} in list item"
    assert item["is_expired"] is False
    assert item["is_near_expiry"] is False


def test_rotate_inherits_predecessor_expiry(tmp_data_dir, client):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path)
    auth = _jwt_login(client)
    h = csrf_header(client)
    create = client.post(
        "/api/tokens",
        json={"name": "ci-bot", "expires_in_days": 90},
        headers={**auth, **h},
    )
    old_id = create.json()["id"]
    old_expires = create.json()["expires_at"]
    assert old_expires is not None

    rot = client.post(
        f"/api/tokens/{old_id}/rotate",
        json={"grace_hours": 24},
        headers={**auth, **h},
    )
    assert rot.status_code == 201, rot.text
    new_id = rot.json()["id"]

    # Read both expires_at values straight out of sqlite.
    with sqlite3.connect(db_path) as db:
        rows = dict(
            db.execute(
                "SELECT id, expires_at FROM api_tokens WHERE id IN (?, ?)",
                (old_id, new_id),
            ).fetchall()
        )
    assert rows[new_id] == rows[old_id], (
        f"successor expiry {rows[new_id]!r} did not inherit predecessor {rows[old_id]!r}"
    )


def test_list_marks_successor_deleted_when_new_token_removed(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    h = csrf_header(client)
    create = client.post("/api/tokens", json={"name": "ci-bot"}, headers={**auth, **h})
    old_id = create.json()["id"]
    rot = client.post(
        f"/api/tokens/{old_id}/rotate",
        json={"grace_hours": 24},
        headers={**auth, **h},
    )
    new_id = rot.json()["id"]

    # Hard-delete the successor — predecessor row now points at a ghost.
    d = client.delete(f"/api/tokens/{new_id}", headers={**auth, **h})
    assert d.status_code == 204

    r = client.get("/api/tokens", headers=auth)
    assert r.status_code == 200
    items = {it["id"]: it for it in r.json()["items"]}
    assert old_id in items
    old = items[old_id]
    assert old["successor_id"] is None
    assert old["rotated_at"] is not None
    assert old["successor_deleted"] is True


def test_create_rejects_expires_in_days_above_max(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    h = csrf_header(client)
    r = client.post(
        "/api/tokens",
        json={"name": "ci-bot", "expires_in_days": 3651},
        headers={**auth, **h},
    )
    assert r.status_code == 422, r.text


def test_rotate_rejects_grace_hours_above_max(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    h = csrf_header(client)
    create = client.post("/api/tokens", json={"name": "ci-bot"}, headers={**auth, **h})
    old_id = create.json()["id"]
    r = client.post(
        f"/api/tokens/{old_id}/rotate",
        json={"grace_hours": 721},
        headers={**auth, **h},
    )
    assert r.status_code == 422, r.text


def test_list_includes_rotated_predecessor_during_grace(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    h = csrf_header(client)
    create = client.post("/api/tokens", json={"name": "ci-bot"}, headers={**auth, **h})
    old_id = create.json()["id"]
    rot = client.post(
        f"/api/tokens/{old_id}/rotate",
        json={"grace_hours": 24},
        headers={**auth, **h},
    )
    new_id = rot.json()["id"]

    r = client.get("/api/tokens", headers=auth)
    assert r.status_code == 200
    items = {it["id"]: it for it in r.json()["items"]}
    assert old_id in items, "predecessor missing during grace window"
    assert new_id in items, "successor missing"
    assert items[old_id]["rotated_at"] is not None
    assert items[old_id]["successor_id"] == new_id
    assert items[new_id]["rotated_from"] == old_id
