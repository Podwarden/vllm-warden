import aiosqlite

from app.db.database import open_db
from app.db.migrations import apply_migrations


async def test_migrations_create_schema_table_and_run_files(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in await cur.fetchall()}
        assert "schema_migrations" in tables
        assert "users" in tables


async def test_migrations_idempotent(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
        await apply_migrations(db)  # second call must be a no-op
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("SELECT COUNT(*) FROM schema_migrations")
        (count,) = await cur.fetchone()
        # 0001–0015 + 0017 (drop_bench; 0016 reserved) + 0018 (tokens-v2,
        # S5/#104) + 0019 (power_samples, S7/#124) + 0020 (landing_page_enabled
        # seed, #155) + 0021 (public_url doc-only no-op, #154) + 0022
        # (engine_templates + stack_attempts, #162) + 0023 (drop ghost
        # hf_cache_dir setting, 2026-06-15 ENOSPC follow-up).
        assert count == 22


async def test_migrations_create_all_v2_tables(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in await cur.fetchall()}
        for t in ["users", "setup_state", "models", "model_runtime"]:
            assert t in tables


async def test_migrations_create_full_v2_schema(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in await cur.fetchall()}
        for t in ["api_tokens", "counters", "model_samples", "gpu_samples"]:
            assert t in tables
        cur = await db.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indices = {row[0] for row in await cur.fetchall()}
        assert "idx_model_samples_minute" in indices
        assert "idx_gpu_samples_minute" in indices


async def test_0017_drops_bench_tables_and_is_idempotent(tmp_data_dir):
    """0017 must drop bench_run / bench_load_config_attempt / bench_cell —
    even when a pre-overhaul DB has 0012's tables already present, and
    even when migrations are re-applied (the DROP IF EXISTS in 0017
    becomes a no-op the second time around)."""
    db_path = tmp_data_dir / "vllm-warden.db"
    # First pass: pre-create the bench tables AND mark 0012 as already
    # applied so apply_migrations skips re-creating them. Simulates a
    # pre-overhaul DB.
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(
            "CREATE TABLE schema_migrations(filename TEXT PRIMARY KEY, applied_at TEXT);"
            "INSERT INTO schema_migrations(filename, applied_at) VALUES "
            "  ('0012_bench.sql', datetime('now'));"
            "CREATE TABLE bench_run(run_id TEXT PRIMARY KEY);"
            "CREATE TABLE bench_load_config_attempt(attempt_id TEXT PRIMARY KEY);"
            "CREATE TABLE bench_cell(cell_id TEXT PRIMARY KEY);"
        )
        await db.commit()
    async with open_db(db_path) as db:
        await apply_migrations(db)
        await apply_migrations(db)  # idempotency
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name LIKE 'bench_%'"
        )
        bench_tables = {row[0] for row in await cur.fetchall()}
        assert bench_tables == set(), bench_tables
