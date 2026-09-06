"""app/stats/request_history -- the store, its queries, its pruning and its
statistics, against a real SQLite file.

Fixture names are neutral on purpose (model-a, key-a, 10.0.0.x): the repo was
swept for vendor and checkpoint names in test data.
"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.stats import request_history as rh
from app.stats.request_history import RequestHistoryStore


def _record(i: int, **over):
    base = {
        "id": f"req-{i}",
        "model": "model-a",
        "model_id": "id-model-a",
        "token_name": "key-a",
        "client_ip": "10.0.0.1",
        "prompt_tokens": 100 + i,
        "completion_tokens": 50,
        "duration_s": 2.0 + i,
        "ttft_s": 0.5,
        "finish_reason": "stop",
        "orphan": False,
        "started_iso": "2026-09-05T18:59:00Z",
    }
    base.update(over)
    return base


async def _migrated(tmp_path):
    db_path = tmp_path / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    return db_path


def _count(db_path) -> int:
    with sqlite3.connect(db_path) as db:
        return db.execute("SELECT COUNT(*) FROM request_history").fetchone()[0]


# ---- write side -----------------------------------------------------------


async def test_record_then_flush_lands_a_row(tmp_path):
    db_path = await _migrated(tmp_path)
    store = RequestHistoryStore(db_path)
    assert store.record(_record(1), finished_at=1000.0) is True
    assert store.pending() == 1
    assert await store.flush() == 1
    with sqlite3.connect(db_path) as db:
        row = db.execute(
            "SELECT id, finished_at, model_id, model, token_name, prompt_tokens, "
            "duration_s, ttft_s, finish_reason, orphan FROM request_history"
        ).fetchone()
    assert row == ("req-1", 1000.0, "id-model-a", "model-a", "key-a", 101, 3.0, 0.5, "stop", 0)
    assert store.written == 1


async def test_record_never_raises_and_never_blocks(tmp_path):
    """The proxy calls this in the streaming `finally`, before the slot is
    released. A full queue drops and counts; garbage is refused quietly."""
    db_path = await _migrated(tmp_path)
    store = RequestHistoryStore(db_path, queue_max=2)
    assert store.record(_record(1)) is True
    assert store.record(_record(2)) is True
    assert store.record(_record(3)) is False  # full: dropped, not blocked
    assert store.dropped == 1
    assert store.record({"no": "id"}) is False  # unusable: refused
    assert store.pending() == 2


async def test_a_bad_field_costs_one_value_not_the_batch(tmp_path):
    db_path = await _migrated(tmp_path)
    store = RequestHistoryStore(db_path)
    store.record(_record(1, prompt_tokens="not a number", ttft_s="?"), finished_at=1.0)
    store.record(_record(2), finished_at=2.0)
    assert await store.flush() == 2
    with sqlite3.connect(db_path) as db:
        rows = db.execute(
            "SELECT id, prompt_tokens, ttft_s FROM request_history ORDER BY id"
        ).fetchall()
    assert rows == [("req-1", 0, None), ("req-2", 102, 0.5)]


async def test_a_duplicate_id_is_ignored_not_an_error(tmp_path):
    db_path = await _migrated(tmp_path)
    store = RequestHistoryStore(db_path)
    store.record(_record(1), finished_at=1.0)
    store.record(_record(1), finished_at=2.0)
    assert await store.flush() == 2
    assert _count(db_path) == 1


async def test_a_failed_write_drops_the_batch_and_keeps_going(tmp_path):
    """No table (an unmigrated file) is the simplest broken volume."""
    db_path = tmp_path / "empty.db"
    store = RequestHistoryStore(db_path)
    store.record(_record(1))
    assert await store.flush() == 0
    assert store.pending() == 0
    # Still usable afterwards.
    assert store.record(_record(2)) is True


async def test_run_forever_batches_and_flushes_on_cancel(tmp_path):
    db_path = await _migrated(tmp_path)
    store = RequestHistoryStore(db_path, flush_interval_s=0.05)
    task = asyncio.create_task(store.run_forever())
    for i in range(5):
        store.record(_record(i), finished_at=float(i))
    for _ in range(50):
        await asyncio.sleep(0.02)
        if _count(db_path) == 5:
            break
    assert _count(db_path) == 5
    # Queued after the last batch, written by the cancellation flush.
    store.record(_record(99), finished_at=99.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert _count(db_path) == 6


# ---- reads ----------------------------------------------------------------


async def _seed(db_path, n: int, *, model_id="id-model-a", start=1000.0, step=1.0):
    store = RequestHistoryStore(db_path)
    for i in range(n):
        store.record(
            _record(i, id=f"{model_id}-{i}", model_id=model_id, model=model_id.removeprefix("id-")),
            finished_at=start + i * step,
        )
    assert await store.flush() == n


async def test_query_window_is_newest_first_and_bounded_by_since(tmp_path):
    db_path = await _migrated(tmp_path)
    await _seed(db_path, 10)  # finished_at 1000..1009
    async with open_db(db_path) as db:
        out = await rh.query_window(db, since=1005.0, model_ids=None, limit=100)
    assert out.total == 5
    assert out.stride == 1
    assert [r["finished_at"] for r in out.rows] == [1009.0, 1008.0, 1007.0, 1006.0, 1005.0]
    assert out.rows[0]["orphan"] is False


async def test_query_window_strides_rather_than_truncating(tmp_path):
    """A chart with a time axis must keep its span covered. The newest N
    would leave the left of the chart empty under a heading claiming the
    whole window."""
    db_path = await _migrated(tmp_path)
    await _seed(db_path, 100)  # 1000..1099
    async with open_db(db_path) as db:
        out = await rh.query_window(db, since=0.0, model_ids=None, limit=10)
    assert out.total == 100
    assert out.stride == 10
    assert len(out.rows) == 10
    at = [r["finished_at"] for r in out.rows]
    assert at[0] == 1099.0  # newest kept
    assert at[-1] == 1009.0  # reaches the far end of the window
    assert all(a - b == 10.0 for a, b in zip(at, at[1:], strict=False))


async def test_query_window_scopes_by_model_row_id(tmp_path):
    db_path = await _migrated(tmp_path)
    await _seed(db_path, 3, model_id="id-model-a")
    await _seed(db_path, 2, model_id="id-model-b", start=2000.0)
    async with open_db(db_path) as db:
        a = await rh.query_window(db, since=0.0, model_ids=["id-model-a"], limit=10)
        none = await rh.query_window(db, since=0.0, model_ids=[], limit=10)
        both = await rh.query_window(db, since=0.0, model_ids=None, limit=10)
    assert a.total == 3 and {r["model"] for r in a.rows} == {"model-a"}
    # An empty selection is NOT everything -- the widening every stats route
    # guards against.
    assert none.total == 0 and none.rows == []
    assert both.total == 5


async def test_query_last_ignores_time(tmp_path):
    db_path = await _migrated(tmp_path)
    await _seed(db_path, 10)
    async with open_db(db_path) as db:
        rows = await rh.query_last(db, n=3, model_ids=None)
    assert [r["finished_at"] for r in rows] == [1009.0, 1008.0, 1007.0]


async def test_earliest_is_store_wide_and_none_when_empty(tmp_path):
    db_path = await _migrated(tmp_path)
    async with open_db(db_path) as db:
        assert await rh.earliest_finished_at(db) is None
    await _seed(db_path, 3, start=500.0)
    async with open_db(db_path) as db:
        assert await rh.earliest_finished_at(db) == 500.0


# ---- pruning --------------------------------------------------------------


async def test_prune_by_age_then_by_count(tmp_path):
    db_path = await _migrated(tmp_path)
    await _seed(db_path, 20)  # 1000..1019
    async with open_db(db_path) as db:
        out = await rh.prune(db, cutoff=1005.0, max_rows=5)
        await db.commit()
    assert out == {"by_age": 5, "by_count": 10}
    with sqlite3.connect(db_path) as db:
        left = [r[0] for r in db.execute(
            "SELECT finished_at FROM request_history ORDER BY finished_at"
        )]
    # The five NEWEST survive the cap.
    assert left == [1015.0, 1016.0, 1017.0, 1018.0, 1019.0]


async def test_prune_with_no_cap_only_ages(tmp_path):
    db_path = await _migrated(tmp_path)
    await _seed(db_path, 5)
    async with open_db(db_path) as db:
        out = await rh.prune(db, cutoff=1002.0, max_rows=0)
        await db.commit()
    assert out == {"by_age": 2, "by_count": 0}
    assert _count(db_path) == 3


async def test_the_stats_pruner_prunes_history_with_the_configured_retention(tmp_path):
    from app.runtime.stats_pruner import prune_once

    db_path = await _migrated(tmp_path)
    import time

    now = time.time()
    store = RequestHistoryStore(db_path)
    store.record(_record(1), finished_at=now - 40 * 86400)  # older than 30d
    store.record(_record(2), finished_at=now - 10)
    await store.flush()

    class S:
        pass

    s = S()
    s.db_path = db_path
    s.request_history_retention_days = 30
    s.request_history_max_rows = 200_000
    out = await prune_once(s)
    assert out["request_history"] == 1
    assert _count(db_path) == 1


# ---- statistics (pure) ----------------------------------------------------


def test_quantile_is_exact_and_interpolated():
    assert rh.quantile([], 0.5) is None
    assert rh.quantile([3.0], 0.99) == 3.0
    assert rh.quantile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert rh.quantile([1.0, 2.0, 3.0, 4.0], 0.0) == 1.0
    assert rh.quantile([1.0, 2.0, 3.0, 4.0], 1.0) == 4.0


def test_histogram_is_cumulative_with_a_null_inf_bucket():
    h = rh.histogram([0.05, 0.5, 5.0, 1000.0], edges=(0.1, 1.0, 10.0))
    assert h["le"] == [0.1, 1.0, 10.0, None]
    assert h["counts"] == [1, 2, 3, 4]
    assert h["count"] == 4
    assert h["sum"] == 1005.55


def test_itl_mean_is_per_request_and_refuses_what_it_cannot_compute():
    ok = {"ttft_s": 1.0, "duration_s": 3.0, "completion_tokens": 5}
    assert rh.itl_mean_of(ok) == 0.5  # 2s of decode over 4 gaps
    assert rh.itl_mean_of({**ok, "ttft_s": None}) is None  # no first token
    assert rh.itl_mean_of({**ok, "completion_tokens": 1}) is None  # no gaps
    assert rh.itl_mean_of({**ok, "duration_s": 1.0}) is None  # no decode phase


def test_latency_summary_shapes_the_three_distributions():
    rows = [
        {"finished_at": 10.0, "ttft_s": 0.2, "duration_s": 2.2, "completion_tokens": 3},
        {"finished_at": 20.0, "ttft_s": None, "duration_s": 5.0, "completion_tokens": 0},
        {"finished_at": 30.0, "ttft_s": 0.4, "duration_s": 0.4, "completion_tokens": 10},
    ]
    s = rh.latency_summary(rows)
    assert s["count"] == 3
    assert s["span_s"] == 20.0
    assert s["oldest_epoch"] == 10.0 and s["newest_epoch"] == 30.0
    # TTFT: only the two that saw a first token.
    assert s["ttft"]["count"] == 2
    assert s["ttft"]["p50"] == pytest.approx(0.3)
    # Duration: every request.
    assert s["duration"]["count"] == 3
    assert s["duration"]["p99"] == pytest.approx(4.944)
    # ITL: only the first has a decode phase with gaps in it.
    assert s["itl"]["count"] == 1
    assert s["itl"]["mean"] == pytest.approx(1.0)
    assert s["itl"]["buckets"]["le"][-1] is None


def test_latency_summary_of_nothing_is_empty_not_zero():
    s = rh.latency_summary([])
    assert s["count"] == 0
    assert s["span_s"] is None
    for k in ("ttft", "itl", "duration"):
        assert s[k]["count"] == 0
        assert s[k]["p50"] is None
        assert s[k]["mean"] is None
