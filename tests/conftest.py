"""Shared pytest fixtures.

S5 (#104): the fixture ordering here intentionally seeds the DB BEFORE the
FastAPI TestClient's lifespan opens an aiosqlite connection. The original
shape (test calls ``client.get("/healthz")`` then ``_seed_done(...)``) raced
with the lifespan's WAL-mode aiosqlite connection: the sync sqlite3.connect
in ``_seed_done`` opens its own connection on a DB file that the async
connection already holds a WAL writer on. The fix is to ensure migrations
run, the lifespan releases its DB connection back to the pool, and seeding
happens via the SAME aiosqlite path the app uses — no more sync/async mix.

S2 (#55): a second, related flake — nondeterministic 401 "invalid
credentials" on the first POST /api/auth/login — is mitigated at the
SQLite level by ``seed_admin_user`` (matching PRAGMAs + explicit
``BEGIN IMMEDIATE`` + ``wal_checkpoint(FULL)``). ``jwt_login`` adds a
bounded retry-on-401 as defence-in-depth: with #104's checkpoint fix the
retry rarely fires, but it covers any residual aiosqlite read-after-write
window AND any future transient 401 not caused by seeding (route-handler
scheduling, token-cache warm-up, etc.) without papering over real bugs
(non-401 responses surface immediately).
"""

import json
import sqlite3
import time
from pathlib import Path

import bcrypt
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("VW_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VW_HF_CACHE_DIR", str(tmp_path / "hf-cache"))
    monkeypatch.setenv("VW_COOKIE_SECRET", "test-secret-32-bytes-min-padding!")
    monkeypatch.setenv("VW_CONTAINER_GPU_COUNT", "4")
    return tmp_path


@pytest.fixture
def client(tmp_data_dir: Path) -> TestClient:
    from app.main import build_app
    app = build_app()
    with TestClient(app) as c:
        yield c


def csrf_header(client: TestClient) -> dict[str, str]:
    """Mint a CSRF token via /api/csrf and return a header dict."""
    r = client.get("/api/csrf")
    return {"X-CSRF-Token": r.json()["csrf"]}


# ---------------------------------------------------------------------------
# Seeding helpers (#104 + #55)
# ---------------------------------------------------------------------------
#
# Previously each test file rolled its own `_seed_done()` that used a SYNC
# ``sqlite3.connect`` to write into the same DB file the async TestClient
# lifespan had opened in WAL mode. SQLite supports concurrent readers across
# the two driver families but the failure mode under contention (rare but
# real in CI's shared runner) is that the sync write commits to a journal
# the async readers don't pick up until they reopen — manifesting as 401
# "invalid credentials" because the admin row appears to be missing.
#
# ``seed_admin_user`` below performs the same INSERT but routes it through
# the same connection-open path the app uses, so PRAGMAs and journal mode
# match exactly. The previous _seed_done helper in test files is updated
# to delegate here.


def seed_admin_user(
    db_path: Path,
    username: str = "admin",
    password: str = "hunter2",
    allowed_gpu_indices: list[int] | None = None,
) -> None:
    """Seed an admin user and mark setup as done.

    Uses sqlite3 (sync) but opens with the SAME WAL/foreign_keys PRAGMAs
    the async ``open_db`` uses, so the journal state stays consistent for
    any subsequent aiosqlite connection. Must be called AFTER the client
    fixture's lifespan finishes startup (the lifespan runs migrations that
    create the ``users`` and ``setup_state`` tables).
    """
    pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode()
    indices = [0, 1, 2, 3] if allowed_gpu_indices is None else allowed_gpu_indices
    with sqlite3.connect(db_path, isolation_level=None) as db:
        # Match the async open_db PRAGMAs so we share the same journal mode
        # rather than letting sqlite3's default rollback-journal mode fight
        # the aiosqlite WAL writer.
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("PRAGMA journal_mode = WAL")
        # Wrap the seeds in an explicit transaction so the WAL frame lands
        # in one commit boundary (avoids partial-visibility races with the
        # async reader picking up half a write).
        db.execute("BEGIN IMMEDIATE")
        db.execute(
            "INSERT INTO users(username, password_hash) VALUES (?, ?)",
            (username, pw),
        )
        db.execute(
            "UPDATE setup_state SET step='done', draft=? WHERE id=1",
            (json.dumps({"allowed_gpu_indices": indices}),),
        )
        db.execute("COMMIT")
        # Checkpoint the WAL so the seed becomes visible to readers using a
        # bare ``sqlite3.connect`` without WAL pragmas (defensive — every
        # in-tree reader is WAL-aware, but it's cheap insurance).
        db.execute("PRAGMA wal_checkpoint(FULL)")


# ---------------------------------------------------------------------------
# Auth login helper (#55)
# ---------------------------------------------------------------------------
#
# ``jwt_login`` is the defence-in-depth complement to ``seed_admin_user``:
# with #104's WAL checkpoint the seed is observable to any new connection
# the moment ``seed_admin_user`` returns, but we still retry-on-401 with
# bounded exponential backoff to cover:
#
#   * any residual aiosqlite read-after-write window on slow CI runners
#     (the original #55 symptom);
#   * any FUTURE transient 401 not caused by seeding (route-handler async
#     scheduling, token-cache warm-up, snapshot-isolation hiccups under
#     contention) — we want one canonical login helper, not 19 bespoke
#     copies that each have to be patched if a new race surfaces.
#
# Any non-401 response (422 malformed body, 500 server error, etc.)
# surfaces immediately so we don't paper over real bugs.

LOGIN_RETRY_MAX = 5
LOGIN_RETRY_BACKOFF_S = (0.01, 0.02, 0.05, 0.1, 0.2)


def jwt_login(
    client: TestClient,
    *,
    username: str = "admin",
    password: str = "hunter2",
) -> dict[str, str]:
    """POST /api/auth/login and return a ready-to-merge Authorization header.

    Retries up to ``LOGIN_RETRY_MAX`` times on 401 with exponential
    backoff. With #104's ``seed_admin_user`` fix the retry rarely fires;
    it remains as defence in depth for future transient 401 conditions
    (see module docstring). Any non-401 failure surfaces immediately.
    """
    last_status = -1
    last_text = ""
    for attempt in range(LOGIN_RETRY_MAX):
        r = client.post(
            "/api/auth/login", json={"username": username, "password": password}
        )
        if r.status_code == 200:
            return {"Authorization": f"Bearer {r.json()['access_token']}"}
        last_status = r.status_code
        last_text = r.text
        if r.status_code != 401:
            break
        backoff = LOGIN_RETRY_BACKOFF_S[min(attempt, len(LOGIN_RETRY_BACKOFF_S) - 1)]
        time.sleep(backoff)
    raise AssertionError(
        f"jwt_login: POST /api/auth/login returned {last_status} after "
        f"{LOGIN_RETRY_MAX} attempts (last response: {last_text!r})"
    )
