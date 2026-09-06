"""Per-request history: the SQLite store behind the requests chart and the
latency distributions on the stats page.

WHY THIS EXISTS. Two operator complaints about the stats page had one cause:

  * "recently finished makes little sense" -- ten identical-looking rows,
    kept 15 minutes, nothing aggregated, so there was no question it answered.
  * "why are the latency charts only 5 minutes" -- the engine publishes only
    CUMULATIVE histograms, so the page kept bucket snapshots in React state and
    the "last 5 minutes" it drew was really "since you opened the tab, capped
    at 5 minutes".

Nothing persisted per-request history. This module does, and both panels read
from it. The columns are what ``app.proxy.request_registry.finished_record``
produces: the PROXY's own TTFT (first streamed frame) and duration, which exist
identically for every backend -- llama.cpp publishes no latency histogram at
all, and this is what gives GGUF models a latency panel for the first time.

WRITE PATH. ``RequestHistoryStore.record`` is called from the proxy's
``_deregister`` inside the streaming ``finally``, before the scheduler slot is
released. It therefore does NOT touch the database: it enqueues, and a
background writer drains the queue in batches. The proxy's existing guarantee
-- stats bookkeeping can never fail a request -- holds by construction here:
``record`` is synchronous, never raises, and a full queue drops the record
(counted, logged) rather than blocking a request on a busy /data volume. The
per-request accounting write (``_record_counters``) already opens its own
connection per request; this deliberately does not add a second one to the
slot-release path.

READ PATH. Module-level query functions take an open connection so the routes
can validate a selection and read on one connection, and so the queries can be
tested against a file with rows written by hand.

Everything statistical (quantiles, histograms) is pure and lives at the bottom.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite

from app.db.database import open_db

logger = logging.getLogger(__name__)

_COLUMNS = (
    "id",
    "finished_at",
    "model_id",
    "model",
    "token_name",
    "client_ip",
    "prompt_tokens",
    "completion_tokens",
    "duration_s",
    "ttft_s",
    "finish_reason",
    "orphan",
    "started_iso",
)
_SELECT = "SELECT " + ", ".join(_COLUMNS) + " FROM request_history"
# INSERT OR IGNORE: the id is the registry's uuid, and a record that somehow
# reaches the writer twice must not turn into an IntegrityError that costs the
# whole batch.
_INSERT = (
    "INSERT OR IGNORE INTO request_history(" + ", ".join(_COLUMNS) + ") "
    "VALUES (" + ", ".join("?" for _ in _COLUMNS) + ")"
)


def _row_params(record: dict[str, Any], finished_at: float) -> tuple[Any, ...]:
    """Coerce one ``finished_record`` dict into the INSERT's parameter tuple.

    Defensive on every field: the record is built on the proxy's hot path from
    values the engine and the client controlled, and a bad value here must
    cost this one row, not the batch it travels in.
    """
    def _int(v: Any) -> int:
        try:
            return int(v or 0)
        except (TypeError, ValueError):
            return 0

    def _float(v: Any) -> float | None:
        try:
            return None if v is None else float(v)
        except (TypeError, ValueError):
            return None

    return (
        str(record.get("id") or ""),
        float(finished_at),
        str(record.get("model_id") or ""),
        str(record.get("model") or ""),
        record.get("token_name"),
        record.get("client_ip"),
        _int(record.get("prompt_tokens")),
        _int(record.get("completion_tokens")),
        _float(record.get("duration_s")) or 0.0,
        _float(record.get("ttft_s")),
        record.get("finish_reason"),
        1 if record.get("orphan") else 0,
        str(record.get("started_iso") or ""),
    )


def _row_dict(row: Iterable[Any]) -> dict[str, Any]:
    d = dict(zip(_COLUMNS, row, strict=True))
    d["orphan"] = bool(d["orphan"])
    return d


class RequestHistoryStore:
    """Queue in front of the ``request_history`` table.

    ``record`` is the only method the proxy calls. It is synchronous and
    fail-open. ``run_forever`` is the writer the app starts at boot;
    ``flush`` is the same drain, callable directly by tests and by shutdown.
    """

    def __init__(
        self,
        db_path: Path,
        *,
        queue_max: int = 10_000,
        flush_interval_s: float = 1.0,
    ) -> None:
        self.db_path = db_path
        self.flush_interval_s = float(flush_interval_s)
        self._q: asyncio.Queue[tuple[Any, ...]] = asyncio.Queue(maxsize=max(1, int(queue_max)))
        #: Records refused because the queue was full. Exposed so a test -- or
        #: an operator reading the log -- can tell "nothing happened" from
        #: "the volume was too slow to keep up".
        self.dropped = 0
        #: Rows written since start, for the same reason.
        self.written = 0
        self._last_drop_log = 0.0

    # -- write side ---------------------------------------------------------

    def record(self, record: dict[str, Any], *, finished_at: float | None = None) -> bool:
        """Enqueue one finished-request record. Never raises, never blocks.

        Returns False when the record was dropped. ``finished_at`` defaults to
        now (wall clock) -- the moment ``_deregister`` runs is the moment the
        request ended.
        """
        try:
            params = _row_params(record, time.time() if finished_at is None else finished_at)
            if not params[0]:
                return False
            self._q.put_nowait(params)
            return True
        except asyncio.QueueFull:
            self.dropped += 1
            now = time.monotonic()
            if now - self._last_drop_log > 60.0:
                self._last_drop_log = now
                logger.warning(
                    "request history: queue full, dropped %d record(s) so far; "
                    "the data volume is not keeping up",
                    self.dropped,
                )
            return False
        except Exception:  # noqa: BLE001 -- bookkeeping must never fail a request
            logger.debug("request history: could not enqueue record", exc_info=True)
            return False

    def pending(self) -> int:
        return self._q.qsize()

    def _drain(self) -> list[tuple[Any, ...]]:
        batch: list[tuple[Any, ...]] = []
        while True:
            try:
                batch.append(self._q.get_nowait())
            except asyncio.QueueEmpty:
                return batch

    async def flush(self) -> int:
        """Write everything queued right now. Returns the number of rows sent.

        A failed write logs and drops the batch. Re-queueing would turn a
        persistently broken volume into an ever-growing queue and, once full,
        into the same drop -- with the log noise multiplied.
        """
        batch = self._drain()
        if not batch:
            return 0
        try:
            async with open_db(self.db_path) as db:
                await db.executemany(_INSERT, batch)
                await db.commit()
        except Exception:
            logger.exception("request history: dropped a batch of %d record(s)", len(batch))
            return 0
        self.written += len(batch)
        return len(batch)

    async def run_forever(self) -> None:
        """Drain the queue in batches, at most once per ``flush_interval_s``.

        Waits for the first record, then sleeps one interval so a burst lands
        as one transaction rather than one fsync per request. Cancellation
        performs a final flush so a clean shutdown loses nothing queued.
        """
        try:
            while True:
                first = await self._q.get()
                await asyncio.sleep(self.flush_interval_s)
                batch = [first, *self._drain()]
                try:
                    async with open_db(self.db_path) as db:
                        await db.executemany(_INSERT, batch)
                        await db.commit()
                    self.written += len(batch)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "request history: dropped a batch of %d record(s)", len(batch)
                    )
        except asyncio.CancelledError:
            try:
                await self.flush()
            except Exception:  # noqa: BLE001 -- shutdown must not hang on the volume
                logger.debug("request history: final flush failed", exc_info=True)
            raise


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def _model_clause(model_ids: Sequence[str] | None) -> tuple[str, tuple[str, ...]]:
    """`` AND model_id IN (...)`` for a selection, or nothing when unfiltered.

    Empty when unfiltered so the unfiltered path executes exactly the SQL it
    always would rather than a filter that happens to match everything.
    """
    if model_ids is None:
        return "", ()
    if not model_ids:
        # `IN ()` is a syntax error in SQLite and a bare 1=1 would silently
        # widen to every model -- the substitution every stats route exists to
        # prevent.
        return " AND 0", ()
    return f" AND model_id IN ({','.join('?' for _ in model_ids)})", tuple(model_ids)


@dataclass(frozen=True)
class WindowRows:
    rows: list[dict[str, Any]]
    #: How many rows the window holds for the selection, before sampling.
    total: int
    #: 1 when every row is returned; k when every k-th row (newest first) is.
    stride: int


async def query_window(
    db: aiosqlite.Connection,
    *,
    since: float,
    model_ids: Sequence[str] | None,
    limit: int,
) -> WindowRows:
    """Rows that finished at or after ``since``, newest first, at most ``limit``.

    When the window holds more than ``limit`` rows, every k-th row is returned
    rather than the newest ``limit``: this feeds a chart with a TIME axis, and
    a "newest N" cut would leave the left of the chart empty while claiming to
    show the whole window. Uniform striding keeps the span covered; the
    response says the stride so the panel can state it.
    """
    where, args = _model_clause(model_ids)
    cur = await db.execute(
        "SELECT COUNT(*) FROM request_history WHERE finished_at >= ?" + where,
        (since, *args),
    )
    total = int((await cur.fetchone() or (0,))[0] or 0)
    limit = max(1, int(limit))
    if total <= limit:
        cur = await db.execute(
            _SELECT + " WHERE finished_at >= ?" + where + " ORDER BY finished_at DESC, id DESC",
            (since, *args),
        )
        rows = await cur.fetchall()
        return WindowRows(rows=[_row_dict(r) for r in rows], total=total, stride=1)
    stride = math.ceil(total / limit)
    cur = await db.execute(
        "SELECT " + ", ".join(_COLUMNS) + " FROM ("
        "  SELECT " + ", ".join(_COLUMNS) + ", "
        "         ROW_NUMBER() OVER (ORDER BY finished_at DESC, id DESC) AS rn"
        "  FROM request_history WHERE finished_at >= ?" + where +
        ") WHERE (rn - 1) % ? = 0 ORDER BY rn ASC LIMIT ?",
        (since, *args, stride, limit),
    )
    rows = await cur.fetchall()
    return WindowRows(rows=[_row_dict(r) for r in rows], total=total, stride=stride)


async def query_window_latency(
    db: aiosqlite.Connection,
    *,
    since: float,
    model_ids: Sequence[str] | None,
) -> list[dict[str, Any]]:
    """Only the columns ``latency_summary`` needs, for EVERY row in the window.

    Deliberately uncapped: a distribution over a sample of the window is not
    the distribution of the window. Four columns a row keeps a full 7d at the
    retention cap well under a second.
    """
    where, args = _model_clause(model_ids)
    cur = await db.execute(
        "SELECT finished_at, ttft_s, duration_s, completion_tokens "
        "FROM request_history WHERE finished_at >= ?" + where,
        (since, *args),
    )
    return [
        {
            "finished_at": r[0],
            "ttft_s": r[1],
            "duration_s": r[2],
            "completion_tokens": r[3],
        }
        for r in await cur.fetchall()
    ]


async def query_last(
    db: aiosqlite.Connection,
    *,
    n: int,
    model_ids: Sequence[str] | None,
) -> list[dict[str, Any]]:
    """The newest ``n`` rows for the selection regardless of time, newest first.

    The operator's alternative basis for the latency distributions: a fixed
    number of requests does not go silent when traffic is thin, which a time
    window does.
    """
    where, args = _model_clause(model_ids)
    cur = await db.execute(
        _SELECT + " WHERE 1" + where + " ORDER BY finished_at DESC, id DESC LIMIT ?",
        (*args, max(1, int(n))),
    )
    return [_row_dict(r) for r in await cur.fetchall()]


async def earliest_finished_at(db: aiosqlite.Connection) -> float | None:
    """When history begins, across the whole store. None when empty.

    Store-wide rather than per selection on purpose: "history begins 3 h ago"
    is a statement about the store (it was added in a release, or pruned), and
    a selection with no rows is a different fact that the row count carries.
    """
    cur = await db.execute("SELECT MIN(finished_at) FROM request_history")
    row = await cur.fetchone()
    return float(row[0]) if row and row[0] is not None else None


async def prune(db: aiosqlite.Connection, *, cutoff: float, max_rows: int) -> dict[str, int]:
    """Delete rows older than ``cutoff`` and, after that, any beyond ``max_rows``.

    Age first, then count, so the count cap trims the OLDEST survivors. Does
    not commit -- the caller's transaction owns that, as with the other
    pruners.
    """
    cur = await db.execute("DELETE FROM request_history WHERE finished_at < ?", (cutoff,))
    by_age = cur.rowcount or 0
    by_count = 0
    if max_rows > 0:
        cur = await db.execute(
            "DELETE FROM request_history WHERE id IN ("
            "  SELECT id FROM request_history ORDER BY finished_at DESC, id DESC"
            "  LIMIT -1 OFFSET ?"
            ")",
            (int(max_rows),),
        )
        by_count = cur.rowcount or 0
    return {"by_age": by_age, "by_count": by_count}


# ---------------------------------------------------------------------------
# Statistics -- pure
# ---------------------------------------------------------------------------

# Bucket edges in seconds. TTFT and ITL reuse vLLM's own histogram boundaries
# so the bars line up with what vLLM's dashboards show for the same model;
# duration reuses its e2e set. The trailing +Inf bucket is encoded as ``None``
# on the wire, exactly as the engine histograms are (JSON has no infinity).
TTFT_EDGES: tuple[float, ...] = (
    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0,
    2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0,
)
ITL_EDGES: tuple[float, ...] = (
    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0,
    2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0,
)
DURATION_EDGES: tuple[float, ...] = (
    0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0,
    50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0,
)


def quantile(sorted_values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated quantile of an ascending sequence, or None if empty.

    Exact, from the samples themselves -- unlike ``hist_quantile`` over engine
    buckets, which can only interpolate inside a bucket. This is the reason to
    compute latency from per-request rows at all.
    """
    n = len(sorted_values)
    if n == 0:
        return None
    if n == 1:
        return float(sorted_values[0])
    pos = max(0.0, min(1.0, q)) * (n - 1)
    lo = int(math.floor(pos))
    hi = min(n - 1, lo + 1)
    frac = pos - lo
    return float(sorted_values[lo]) + frac * (float(sorted_values[hi]) - float(sorted_values[lo]))


def histogram(values: Iterable[float], edges: Sequence[float]) -> dict[str, Any]:
    """Cumulative buckets in the engine-histogram wire shape.

    ``le`` carries the edges plus ``None`` for +Inf; ``counts`` are cumulative
    along the boundaries, so the frontend's existing bucket renderer and
    quantile mirror consume this unchanged.
    """
    per_bucket = [0] * (len(edges) + 1)
    count = 0
    total = 0.0
    for v in values:
        count += 1
        total += v
        for i, edge in enumerate(edges):
            if v <= edge:
                per_bucket[i] += 1
                break
        else:
            per_bucket[len(edges)] += 1
    cumulative: list[int] = []
    running = 0
    for c in per_bucket:
        running += c
        cumulative.append(running)
    return {
        "le": [*edges, None],
        "counts": cumulative,
        "count": count,
        "sum": total,
    }


def distribution(values: Iterable[float], edges: Sequence[float]) -> dict[str, Any]:
    """Count, exact quantiles, mean and buckets for one latency series."""
    vals = sorted(float(v) for v in values)
    n = len(vals)
    return {
        "count": n,
        "p50": quantile(vals, 0.5),
        "p90": quantile(vals, 0.9),
        "p99": quantile(vals, 0.99),
        "mean": (sum(vals) / n) if n else None,
        "buckets": histogram(vals, edges),
    }


def itl_mean_of(row: dict[str, Any]) -> float | None:
    """Mean inter-token latency of ONE request, from the proxy's measurements.

    ``(duration - ttft) / (completion_tokens - 1)``: the decode phase divided by
    the gaps in it. This is a per-request MEAN, not the per-token distribution
    the engine's ITL histogram is -- contention inside a request averages out
    here. Every panel that shows it says "mean per request" for that reason.
    None when it cannot be computed: no first token, fewer than two tokens, or
    a decode phase of zero length (a non-streaming request has no first frame
    to time, so it lands here too).
    """
    ttft = row.get("ttft_s")
    dur = row.get("duration_s")
    n = row.get("completion_tokens") or 0
    if ttft is None or dur is None or n < 2:
        return None
    decode = float(dur) - float(ttft)
    if decode <= 0:
        return None
    return decode / (n - 1)


def latency_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The three distributions over a set of rows, plus their time span."""
    ttft = [float(r["ttft_s"]) for r in rows if r.get("ttft_s") is not None]
    durations = [float(r["duration_s"]) for r in rows if r.get("duration_s") is not None]
    itl = [v for v in (itl_mean_of(r) for r in rows) if v is not None]
    finished = [float(r["finished_at"]) for r in rows if r.get("finished_at") is not None]
    return {
        "count": len(rows),
        "oldest_epoch": min(finished) if finished else None,
        "newest_epoch": max(finished) if finished else None,
        "span_s": (max(finished) - min(finished)) if finished else None,
        "ttft": distribution(ttft, TTFT_EDGES),
        "itl": distribution(itl, ITL_EDGES),
        "duration": distribution(durations, DURATION_EDGES),
    }
