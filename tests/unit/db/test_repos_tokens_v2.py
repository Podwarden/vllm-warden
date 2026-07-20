"""Repo-level coverage for the S5 tokens-v2 additions (#104):

  * TokenRepo.create accepts rate_limit_tps / priority
  * TokenRepo.rotate inherits both fields from the predecessor in a single
    transaction (verified by reading sqlite directly)
  * TokenRepo.update_limits is a PATCH-style sentinel-aware updater
  * TokenRepo.get returns the new fields
  * TokenUsageRepo.add / range / totals follow the half-open-interval contract
"""

import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.tokens import _UNSET, TokenRepo, TokenUsageRepo


@pytest.fixture
async def db(tmp_data_dir):
    async with open_db(tmp_data_dir / "vllm-warden.db") as conn:
        await apply_migrations(conn)
        yield conn


async def test_create_stores_rate_and_priority(db):
    repo = TokenRepo(db)
    await repo.create(
        "tok-rp", "rp", "vw_" + "a" * 32,
        rate_limit_tps=500, priority=7,
    )
    row = await repo.get("tok-rp")
    assert row is not None
    assert row.rate_limit_tps == 500
    assert row.priority == 7


async def test_create_defaults_unlimited_priority_5(db):
    repo = TokenRepo(db)
    await repo.create("tok-def", "def", "vw_" + "b" * 32)
    row = await repo.get("tok-def")
    assert row.rate_limit_tps is None
    assert row.priority == 5


async def test_rotate_inherits_rate_and_priority(db):
    repo = TokenRepo(db)
    await repo.create(
        "old", "n", "vw_" + "c" * 32,
        rate_limit_tps=250, priority=8,
    )
    # #150 — rotate() no longer takes new_name; successor keeps the
    # predecessor's name and the predecessor is renamed to "n (old 1)".
    new_id, _plaintext, renamed_to = await repo.rotate(old_id="old")
    new_row = await repo.get(new_id)
    assert new_row.rate_limit_tps == 250
    assert new_row.priority == 8
    assert new_row.name == "n"
    assert renamed_to == "n (old 1)"


async def test_update_limits_patches_rate_only(db):
    repo = TokenRepo(db)
    await repo.create("t", "n", "vw_" + "d" * 32, rate_limit_tps=100, priority=5)
    ok = await repo.update_limits("t", rate_limit_tps=200)
    assert ok is True
    row = await repo.get("t")
    assert row.rate_limit_tps == 200
    assert row.priority == 5  # untouched by the patch


async def test_update_limits_can_clear_rate_back_to_unlimited(db):
    repo = TokenRepo(db)
    await repo.create("t", "n", "vw_" + "e" * 32, rate_limit_tps=100)
    # None ≠ _UNSET — None explicitly clears.
    ok = await repo.update_limits("t", rate_limit_tps=None)
    assert ok is True
    row = await repo.get("t")
    assert row.rate_limit_tps is None


async def test_update_limits_unset_is_noop(db):
    repo = TokenRepo(db)
    await repo.create("t", "n", "vw_" + "f" * 32, rate_limit_tps=100, priority=4)
    ok = await repo.update_limits("t")  # both default to _UNSET
    assert ok is True  # noop is idempotent success
    row = await repo.get("t")
    assert row.rate_limit_tps == 100
    assert row.priority == 4


async def test_update_limits_returns_false_for_unknown_token(db):
    ok = await TokenRepo(db).update_limits("does-not-exist", priority=2)
    assert ok is False


async def test_unset_sentinel_is_a_real_singleton():
    """A regression would split _UNSET into multiple instances, breaking
    the isinstance() check inside update_limits."""
    from app.db.repos.tokens import _Unset
    assert _UNSET is _Unset()


async def test_token_usage_add_creates_then_increments(db):
    usage = TokenUsageRepo(db)
    await usage.add("tok-u", minute=1000, prompt_tokens=10, completion_tokens=20)
    await usage.add("tok-u", minute=1000, prompt_tokens=5, completion_tokens=3)
    requests, prompt, completion = await usage.totals("tok-u", 1000, 1001)
    assert requests == 2
    assert prompt == 15
    assert completion == 23


async def test_token_usage_range_is_ordered_and_half_open(db):
    usage = TokenUsageRepo(db)
    for m in (1000, 1001, 1005, 1010):
        await usage.add("tok-u", minute=m, prompt_tokens=1, completion_tokens=1)
    rows = await usage.range("tok-u", since_minute=1000, until_minute=1005)
    # 1005 must NOT be included (until is exclusive); 1000 IS included.
    minutes = [r[0] for r in rows]
    assert minutes == [1000, 1001]
    rows = await usage.range("tok-u", since_minute=1000, until_minute=1011)
    minutes = [r[0] for r in rows]
    assert minutes == [1000, 1001, 1005, 1010]


async def test_token_usage_totals_isolates_tokens(db):
    usage = TokenUsageRepo(db)
    await usage.add("A", 100, prompt_tokens=10, completion_tokens=20)
    await usage.add("B", 100, prompt_tokens=99, completion_tokens=99)
    a_totals = await usage.totals("A", 0, 200)
    b_totals = await usage.totals("B", 0, 200)
    assert a_totals == (1, 10, 20)
    assert b_totals == (1, 99, 99)


async def test_token_usage_totals_zero_for_empty_range(db):
    """Querying a token with no rows in the range must return all-zero,
    not raise — the UI relies on this to render 'no usage' cells."""
    totals = await TokenUsageRepo(db).totals("nope", 0, 1)
    assert totals == (0, 0, 0)
