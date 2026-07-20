from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite


@asynccontextmanager
async def open_db(path: Path):
    async with aiosqlite.connect(path) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        await db.execute("PRAGMA journal_mode = WAL")
        # Background writers (stats sampler, pull poller) run their own
        # short-lived connections and can collide while a model download
        # saturates CPU/IO. A generous busy_timeout makes them wait out the
        # contention window instead of failing fast with "database is locked".
        await db.execute("PRAGMA busy_timeout = 30000")
        yield db
