import aiosqlite


class CountersRepo:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    async def increment(
        self,
        model_id: str,
        token_id: str | None,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        # SQLite treats NULL as distinct in composite PRIMARY KEY, so ON CONFLICT
        # does not fire when token_id IS NULL. Use explicit SELECT + UPDATE/INSERT
        # to handle the NULL-token case correctly.
        cur = await self.db.execute(
            "SELECT 1 FROM counters WHERE model_id = ? "
            "AND (token_id = ? OR (token_id IS NULL AND ? IS NULL))",
            (model_id, token_id, token_id),
        )
        exists = await cur.fetchone()
        if exists:
            await self.db.execute(
                "UPDATE counters SET requests = requests + 1, "
                "prompt_tokens = prompt_tokens + ?, "
                "completion_tokens = completion_tokens + ? "
                "WHERE model_id = ? "
                "AND (token_id = ? OR (token_id IS NULL AND ? IS NULL))",
                (prompt_tokens, completion_tokens, model_id, token_id, token_id),
            )
        else:
            await self.db.execute(
                "INSERT INTO counters(model_id, token_id, requests, prompt_tokens, completion_tokens) "
                "VALUES (?, ?, 1, ?, ?)",
                (model_id, token_id, prompt_tokens, completion_tokens),
            )
        await self.db.commit()
