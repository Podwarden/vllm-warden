from pathlib import Path

import aiosqlite

SQL_DIR = Path(__file__).parent / "sql"


async def apply_migrations(db: aiosqlite.Connection) -> None:
    await db.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "  filename TEXT PRIMARY KEY,"
        "  applied_at TEXT NOT NULL DEFAULT (datetime('now'))"
        ")"
    )
    await db.commit()

    cur = await db.execute("SELECT filename FROM schema_migrations")
    applied = {row[0] for row in await cur.fetchall()}

    files = sorted(p for p in SQL_DIR.glob("*.sql"))
    for path in files:
        if path.name in applied:
            continue
        sql = path.read_text(encoding="utf-8")
        await db.executescript(sql)
        await db.execute("INSERT INTO schema_migrations(filename) VALUES (?)", (path.name,))
        await db.commit()
