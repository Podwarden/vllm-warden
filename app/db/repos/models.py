import json
from dataclasses import dataclass

import aiosqlite


@dataclass
class ModelRow:
    id: str
    served_model_name: str
    hf_repo: str
    hf_revision: str
    gpu_indices: list[int]
    tensor_parallel_size: int
    dtype: str | None
    max_model_len: int | None
    gpu_memory_utilization: float
    trust_remote_code: bool
    extra_args: list[str]
    status: str
    pulled_bytes: int
    pulled_total: int | None
    last_error: str | None
    extra_env: dict[str, str]
    # New for #85 — see migration 0014. ``filename`` is None for legacy / whole-repo
    # pulls. ``parallelism_strategy`` and ``max_batch_size`` have defaults that
    # mirror the pre-#85 wizard's implicit behaviour so existing rows decode
    # to a sensible shape.
    filename: str | None = None
    parallelism_strategy: str = "auto"
    max_batch_size: int = 1
    # New for #106 — see migration 0015. ``hf_config_repo`` populates the
    # vLLM ``--hf-config-path`` flag for GGUF repos that omit ``config.json``
    # (common for unsloth republishes). ``tokenizer_repo`` populates
    # ``--tokenizer`` for the same upstream-vs-quant split. Both default to
    # None so non-GGUF rows and self-contained GGUF repos behave identically
    # to v17.17.
    hf_config_repo: str | None = None
    tokenizer_repo: str | None = None
    # Added for #114 — exposed on every SELECT so the cache-management GC
    # sweep can decide whether a ``status=failed`` row is "stale enough"
    # to collect WITHOUT an N+1 follow-up per row. Defaults to None for
    # ergonomic construction in tests; production reads from SQLite always
    # populate it (the column is NOT NULL with a ``datetime('now')`` default).
    updated_at: str | None = None
    # New for #162 — see migration 0022. Per-model engine axis. None on legacy
    # rows means the supervisor falls back to the in-container engine.
    engine_channel: str | None = None
    engine_vllm_version: str | None = None
    engine_image: str | None = None


# Column list shared by insert + every SELECT so we can't drift them.
# ``updated_at`` is appended at the end so the column-index map in
# ``_decode_row`` stays append-only — never reorder, the indices below
# pin the layout.
_MODEL_COLS = (
    "id, served_model_name, hf_repo, hf_revision, gpu_indices, "
    "tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization, "
    "trust_remote_code, extra_args, status, pulled_bytes, pulled_total, last_error, "
    "extra_env, filename, parallelism_strategy, max_batch_size, "
    "hf_config_repo, tokenizer_repo, updated_at, "
    "engine_channel, engine_vllm_version, engine_image"
)


def _decode_row(row: tuple) -> ModelRow:
    return ModelRow(
        id=row[0], served_model_name=row[1], hf_repo=row[2], hf_revision=row[3],
        gpu_indices=json.loads(row[4]), tensor_parallel_size=row[5], dtype=row[6],
        max_model_len=row[7], gpu_memory_utilization=row[8],
        trust_remote_code=bool(row[9]), extra_args=json.loads(row[10]), status=row[11],
        pulled_bytes=row[12], pulled_total=row[13], last_error=row[14],
        extra_env=json.loads(row[15]),
        filename=row[16],
        parallelism_strategy=row[17] if row[17] is not None else "auto",
        max_batch_size=row[18] if row[18] is not None else 1,
        hf_config_repo=row[19],
        tokenizer_repo=row[20],
        updated_at=row[21],
        engine_channel=row[22],
        engine_vllm_version=row[23],
        engine_image=row[24],
    )


class ModelRepo:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    async def insert(self, row: ModelRow) -> None:
        await self.db.execute(
            """INSERT INTO models(
                id, served_model_name, hf_repo, hf_revision, gpu_indices,
                tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization,
                trust_remote_code, extra_args, status, pulled_bytes, pulled_total, last_error,
                extra_env, filename, parallelism_strategy, max_batch_size,
                hf_config_repo, tokenizer_repo,
                engine_channel, engine_vllm_version, engine_image
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                row.id, row.served_model_name, row.hf_repo, row.hf_revision,
                json.dumps(row.gpu_indices),
                row.tensor_parallel_size, row.dtype, row.max_model_len, row.gpu_memory_utilization,
                int(row.trust_remote_code), json.dumps(row.extra_args), row.status, row.pulled_bytes,
                row.pulled_total, row.last_error, json.dumps(row.extra_env),
                row.filename, row.parallelism_strategy, row.max_batch_size,
                row.hf_config_repo, row.tokenizer_repo,
                row.engine_channel, row.engine_vllm_version, row.engine_image,
            ),
        )
        await self.db.commit()

    async def get(self, model_id: str) -> ModelRow | None:
        cur = await self.db.execute(
            f"SELECT {_MODEL_COLS} FROM models WHERE id = ?",
            (model_id,),
        )
        row = await cur.fetchone()
        if not row:
            return None
        return _decode_row(row)

    async def list_all(self) -> list[ModelRow]:
        cur = await self.db.execute(
            f"SELECT {_MODEL_COLS} FROM models ORDER BY created_at"
        )
        rows = await cur.fetchall()
        return [_decode_row(r) for r in rows]

    async def list_by_repo(self, hf_repo: str) -> list[ModelRow]:
        """Return every row whose ``hf_repo`` exactly matches ``hf_repo``.

        Used by the HF cache management API (vllm-warden#114) to decide
        whether a candidate cache directory is currently owned by one or
        more live model rows. The same ``hf_repo`` can legitimately back
        multiple rows (different served_model_names, different GPU
        mappings), so this MUST return a list and not an Optional.
        Empty list means the cache dir is orphaned and safe to delete.
        """
        cur = await self.db.execute(
            f"SELECT {_MODEL_COLS} FROM models WHERE hf_repo = ? ORDER BY created_at",
            (hf_repo,),
        )
        rows = await cur.fetchall()
        return [_decode_row(r) for r in rows]

    async def updated_at(self, model_id: str) -> str | None:
        """Return the ``updated_at`` timestamp (ISO text) for a row, or None.

        Pulled out so the GC sweep in app/cache/routes_api.py can decide
        whether a status=failed row is "stale enough" to garbage collect
        its cache without exposing the freshness signal to non-cache
        callers.
        """
        cur = await self.db.execute(
            "SELECT updated_at FROM models WHERE id = ?", (model_id,)
        )
        row = await cur.fetchone()
        return row[0] if row else None

    async def get_by_served_name(self, served: str) -> ModelRow | None:
        cur = await self.db.execute("SELECT id FROM models WHERE served_model_name = ?", (served,))
        row = await cur.fetchone()
        return await self.get(row[0]) if row else None

    async def update_status(
        self, model_id: str, status: str, last_error: str | None = None
    ) -> None:
        await self.db.execute(
            "UPDATE models SET status = ?, last_error = ?, updated_at = datetime('now') "
            "WHERE id = ?",
            (status, last_error, model_id),
        )
        await self.db.commit()

    async def update_pull_progress(
        self, model_id: str, pulled_bytes: int, pulled_total: int | None
    ) -> None:
        await self.db.execute(
            "UPDATE models SET pulled_bytes = ?, pulled_total = ?, updated_at = datetime('now') "
            "WHERE id = ?",
            (pulled_bytes, pulled_total, model_id),
        )
        await self.db.commit()

    async def delete(self, model_id: str) -> None:
        await self.db.execute("DELETE FROM models WHERE id = ?", (model_id,))
        await self.db.commit()

    async def mark_runtime_dead_on_startup(self) -> int:
        """Wipe any row whose status presumes a live backing process to
        ``failed`` on app startup. Returns count updated.

        Status set wiped: ``loaded``, ``loading``, ``unloading`` (live
        vLLM subprocess) and ``pulling`` (live HF download task). After a
        warden restart NONE of those processes exists — the supervisor
        and pull-task state lives in-process only — so leaving any of
        those statuses in the DB strands the row in a state with no
        operator-actionable affordance (#11).

        For ``pulling`` rows we additionally zero ``pulled_bytes`` and
        ``pulled_total`` so the UI does not show stale progress for a
        pull that no longer has a backing task. Re-pulling appends to a
        fresh counter from byte 0 (the pull-task does not resume — it
        re-fetches), so the zeroed counters reflect reality. Wiping
        progress on non-``pulling`` rows would be wrong (e.g. a
        ``loaded`` row's ``pulled_total`` is the persisted weights
        size); the CASE expression below guards that.
        """
        cur = await self.db.execute(
            "UPDATE models SET status = 'failed', "
            "last_error = 'process not running after restart', "
            "pulled_bytes = CASE WHEN status = 'pulling' THEN 0 "
            "                    ELSE pulled_bytes END, "
            "pulled_total = CASE WHEN status = 'pulling' THEN 0 "
            "                    ELSE pulled_total END, "
            "updated_at = datetime('now') "
            "WHERE status IN ('loaded', 'loading', 'unloading', 'pulling')"
        )
        await self.db.commit()
        return cur.rowcount
