import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.tokens import TokenRepo, hash_token


@pytest.fixture
async def db(tmp_data_dir):
    async with open_db(tmp_data_dir / "vllm-warden.db") as conn:
        await apply_migrations(conn)
        yield conn


async def test_create_lookup_revoke(db):
    repo = TokenRepo(db)
    await repo.create("tok1", "ci-bot", "vw_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    found = await repo.find_by_plaintext("vw_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    assert found.id == "tok1"
    assert found.prefix == "vw_aaaaa"

    await repo.revoke("tok1")
    found = await repo.find_by_plaintext("vw_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    assert found.revoked_at is not None


async def test_only_hash_stored(db):
    plaintext = "vw_secret_token_value_xyz"
    await TokenRepo(db).create("tok2", "n", plaintext)
    cur = await db.execute("SELECT hash FROM api_tokens WHERE id = ?", ("tok2",))
    (stored,) = await cur.fetchone()
    assert stored == hash_token(plaintext)
    assert plaintext not in stored
