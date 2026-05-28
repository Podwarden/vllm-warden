import time

import pytest

from app.config import load_settings
from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.models import ModelRepo, ModelRow
from app.runtime.stats_pruner import RETENTION_MINUTES, prune_once


@pytest.fixture
async def settings_with_db(tmp_data_dir):
    settings = load_settings()
    async with open_db(settings.db_path) as conn:
        await apply_migrations(conn)
        await ModelRepo(conn).insert(ModelRow(
            id="m1", served_model_name="m1", hf_repo="o/r", hf_revision="main",
            gpu_indices=[0], tensor_parallel_size=1, dtype=None,
            max_model_len=None, gpu_memory_utilization=0.9, trust_remote_code=False,
            extra_args=[], extra_env={}, status="registered", pulled_bytes=0,
            pulled_total=None, last_error=None,
        ))
    return settings


async def test_prune_removes_rows_older_than_retention(settings_with_db):
    now_min = int(time.time() // 60)
    fresh = now_min
    stale = now_min - RETENTION_MINUTES - 1
    async with open_db(settings_with_db.db_path) as db:
        await db.execute(
            "INSERT INTO model_samples(model_id, minute, requests, prompt_tokens, completion_tokens) "
            "VALUES ('m1', ?, 1, 1, 1), ('m1', ?, 1, 1, 1)",
            (fresh, stale),
        )
        await db.execute(
            "INSERT INTO gpu_samples(gpu_index, minute, utilization_pct, memory_used_mib, memory_total_mib) "
            "VALUES (0, ?, 1, 1, 1), (0, ?, 1, 1, 1)",
            (fresh, stale),
        )
        await db.commit()

    deleted = await prune_once(settings_with_db)
    assert deleted["model_samples"] == 1
    assert deleted["gpu_samples"] == 1

    async with open_db(settings_with_db.db_path) as db:
        cur = await db.execute("SELECT COUNT(*) FROM model_samples")
        assert (await cur.fetchone())[0] == 1
        cur = await db.execute("SELECT COUNT(*) FROM gpu_samples")
        assert (await cur.fetchone())[0] == 1
