from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite


@asynccontextmanager
async def open_db(path: Path):
    async with aiosqlite.connect(path) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        await db.execute("PRAGMA journal_mode = WAL")
        yield db
