import time

from fastapi import APIRouter, Depends, HTTPException, Request

from app.auth.deps import require_jwt
from app.db.database import open_db

router = APIRouter()

_RANGE_TO_MINUTES = {"1h": 60, "6h": 360, "24h": 1440, "7d": 10080}


def _since_minute(range_: str) -> int:
    if range_ not in _RANGE_TO_MINUTES:
        raise HTTPException(status_code=400, detail="invalid range")
    now_min = int(time.time() // 60)
    return now_min - _RANGE_TO_MINUTES[range_]


def _validate_range(range_: str) -> int:
    """Return retention-bounded minutes for ``range_``. Raises 400 on unknown.

    Accepts ``1h``, ``6h``, ``24h``, ``7d``. S7 (#124) — shared by every
    ``/api/stats/v2/*`` endpoint so frontend range-picker errors hit a
    single, predictable validation site.
    """
    if range_ not in _RANGE_TO_MINUTES:
        raise HTTPException(
            status_code=400,
            detail=f"invalid range '{range_}'; allowed: {sorted(_RANGE_TO_MINUTES)}",
        )
    return _RANGE_TO_MINUTES[range_]


@router.get("/api/stats/models")
async def stats_models(
    request: Request, range: str = "24h", _user: str = Depends(require_jwt)
):
    since = _since_minute(range)
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        cur = await db.execute(
            "SELECT model_id, minute, requests, prompt_tokens, completion_tokens "
            "FROM model_samples WHERE minute >= ? ORDER BY minute ASC",
            (since,),
        )
        rows = await cur.fetchall()
    return [
        {"model_id": r[0], "minute": r[1], "requests": r[2],
         "prompt_tokens": r[3], "completion_tokens": r[4]}
        for r in rows
    ]


@router.get("/api/stats/gpus")
async def stats_gpus(
    request: Request, range: str = "24h", _user: str = Depends(require_jwt)
):
    since = _since_minute(range)
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        cur = await db.execute(
            "SELECT gpu_index, minute, utilization_pct, memory_used_mib, "
            "memory_total_mib, name "
            "FROM gpu_samples WHERE minute >= ? ORDER BY minute ASC, gpu_index ASC",
            (since,),
        )
        rows = await cur.fetchall()
    # `name` is NULL on rows written before migration 0013; UI must fall back
    # to "GPU N". Surface it explicitly so the frontend doesn't have to probe.
    return [
        {"gpu_index": r[0], "minute": r[1], "utilization_pct": r[2],
         "memory_used_mib": r[3], "memory_total_mib": r[4], "name": r[5]}
        for r in rows
    ]


# ============================================================================
# S7 (#124) — /api/stats/v2 endpoints. Coexist with v1 (CTO decision #7); v1
# stays untouched and is the existing dashboard's stable contract. v2 is the
# new richer shape consumed by the redesigned stats UI in dev-2's slice.
#
# Shape contract is the source of truth for the frontend handoff (see the MR
# description / dev-2 handoff note). Keys are stable; new fields may be added
# but existing ones won't be renamed or dropped without a follow-up issue.
# ============================================================================


@router.get("/api/stats/v2/overview")
async def stats_v2_overview(
    request: Request, range: str = "24h", _user: str = Depends(require_jwt)
):
    """Aggregate dashboard payload for the new stats page.

    Returns:
      {
        "range": "24h",
        "now_minute": int,
        "since_minute": int,
        "current": {
          "vram_used_mib": int,
          "vram_total_mib": int,
          "vram_pct": int,           # 0..100, rounded
          "gpu_util_pct": int,       # max across GPUs at most-recent minute
          "power_w": float | None,   # sum across GPUs at most-recent minute
          "tps": float,              # tokens-per-second over last full minute
                                     # (prompt + completion)
        },
        "active_models": [
          {"id": str, "served_model_name": str}, ...
        ],
        "series": {
          "vram": [{"minute": int, "used_mib": int, "total_mib": int}, ...],
          "util": [{"minute": int, "max_pct": int}, ...],
          "power": [{"minute": int, "watts": float}, ...],     # sum / minute
          "tokens": [{"minute": int, "prompt": int, "completion": int}, ...],
        }
      }
    """
    _validate_range(range)
    since = _since_minute(range)
    now_min = int(time.time() // 60)
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        # ---- series: GPU util + VRAM, aggregated across GPUs per minute -----
        cur = await db.execute(
            "SELECT minute, "
            "       MAX(utilization_pct) AS max_util, "
            "       SUM(memory_used_mib) AS used_mib, "
            "       SUM(memory_total_mib) AS total_mib "
            "FROM gpu_samples WHERE minute >= ? "
            "GROUP BY minute ORDER BY minute ASC",
            (since,),
        )
        gpu_rows = await cur.fetchall()

        # ---- series: power.draw, summed across GPUs per minute --------------
        # Each row in power_samples is (gpu_idx, minute, watts_sum, samples)
        # — to get per-minute average watts across the whole box we first
        # average within each GPU (watts_sum/samples) then sum across GPUs.
        cur = await db.execute(
            "SELECT minute, SUM(watts_sum / NULLIF(samples, 0)) AS box_watts "
            "FROM power_samples WHERE minute >= ? "
            "GROUP BY minute ORDER BY minute ASC",
            (since,),
        )
        power_rows = await cur.fetchall()

        # ---- series: tokens, summed across all api_tokens per minute --------
        cur = await db.execute(
            "SELECT minute, "
            "       COALESCE(SUM(prompt_tokens), 0) AS prompt, "
            "       COALESCE(SUM(completion_tokens), 0) AS completion "
            "FROM token_usage_minute WHERE minute >= ? "
            "GROUP BY minute ORDER BY minute ASC",
            (since,),
        )
        token_rows = await cur.fetchall()

        # ---- current snapshot (most recent minute in window) ----------------
        cur = await db.execute(
            "SELECT SUM(memory_used_mib), SUM(memory_total_mib), MAX(utilization_pct) "
            "FROM gpu_samples WHERE minute = ("
            "  SELECT MAX(minute) FROM gpu_samples WHERE minute >= ?"
            ")",
            (since,),
        )
        cur_gpu = await cur.fetchone() or (None, None, None)

        cur = await db.execute(
            "SELECT SUM(watts_sum / NULLIF(samples, 0)) "
            "FROM power_samples WHERE minute = ("
            "  SELECT MAX(minute) FROM power_samples WHERE minute >= ?"
            ")",
            (since,),
        )
        cur_power = (await cur.fetchone() or (None,))[0]

        cur = await db.execute(
            "SELECT COALESCE(SUM(prompt_tokens), 0) + COALESCE(SUM(completion_tokens), 0) "
            "FROM token_usage_minute WHERE minute = ("
            "  SELECT MAX(minute) FROM token_usage_minute WHERE minute >= ?"
            ")",
            (since,),
        )
        last_min_tokens = (await cur.fetchone() or (0,))[0] or 0

        # ---- active model list ---------------------------------------------
        # ``models.status='loaded'`` AND ``model_runtime`` row exists, same as
        # header/routes_api.py::_active_model — but emit ALL matching rows so
        # multi-model fleets (future) render correctly even though today the
        # supervisor enforces single-model loading.
        cur = await db.execute(
            "SELECT m.id, m.served_model_name "
            "FROM models m JOIN model_runtime r ON r.model_id = m.id "
            "WHERE m.status = 'loaded' "
            "ORDER BY m.id"
        )
        active_rows = await cur.fetchall()

    vram_used = int(cur_gpu[0] or 0)
    vram_total = int(cur_gpu[1] or 0)
    vram_pct = int(round(100.0 * vram_used / vram_total)) if vram_total else 0
    util_pct = int(cur_gpu[2] or 0)
    # TPS = total tokens in last full minute / 60s. Floor at 0.
    tps = float(last_min_tokens) / 60.0 if last_min_tokens else 0.0

    return {
        "range": range,
        "now_minute": now_min,
        "since_minute": since,
        "current": {
            "vram_used_mib": vram_used,
            "vram_total_mib": vram_total,
            "vram_pct": vram_pct,
            "gpu_util_pct": util_pct,
            "power_w": float(cur_power) if cur_power is not None else None,
            "tps": tps,
        },
        "active_models": [
            {"id": r[0], "served_model_name": r[1]} for r in active_rows
        ],
        "series": {
            "vram": [
                {"minute": r[0], "used_mib": int(r[2] or 0), "total_mib": int(r[3] or 0)}
                for r in gpu_rows
            ],
            "util": [
                {"minute": r[0], "max_pct": int(r[1] or 0)} for r in gpu_rows
            ],
            "power": [
                {"minute": r[0], "watts": float(r[1])} for r in power_rows
                if r[1] is not None
            ],
            "tokens": [
                {"minute": r[0], "prompt": int(r[1]), "completion": int(r[2])}
                for r in token_rows
            ],
        },
    }


@router.get("/api/stats/v2/tokens-per-key")
async def stats_v2_tokens_per_key(
    request: Request, range: str = "24h", _user: str = Depends(require_jwt)
):
    """Per-API-key token usage over the range.

    JOIN ``token_usage_minute`` onto ``api_tokens`` so the response carries
    the human-readable name (``name``) alongside the opaque ``token_id``.
    Tokens that haven't been used in the window are omitted (no row → no entry,
    keeps the response bounded by activity rather than the token catalog).

    Returns:
      {
        "range": "24h",
        "since_minute": int,
        "rows": [
          {
            "token_id": str,
            "name": str,           # api_tokens.name (or "(unknown)" for orphan)
            "prefix": str | None,  # api_tokens.prefix, helps disambiguate
            "requests": int,
            "prompt_tokens": int,
            "completion_tokens": int,
            "total_tokens": int,   # prompt + completion (server-computed)
          }, ...
        ]
      }

    Rows are sorted by total_tokens DESC so the heaviest keys are first.
    """
    _validate_range(range)
    since = _since_minute(range)
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        # LEFT JOIN: keep rows even if the api_tokens entry has been deleted —
        # surfacing the orphan token_id is more useful than silently dropping
        # historical usage that's still in the rollup.
        cur = await db.execute(
            "SELECT u.token_id, "
            "       COALESCE(t.name, '(unknown)') AS name, "
            "       t.prefix AS prefix, "
            "       SUM(u.requests) AS requests, "
            "       SUM(u.prompt_tokens) AS prompt_tokens, "
            "       SUM(u.completion_tokens) AS completion_tokens "
            "FROM token_usage_minute u "
            "LEFT JOIN api_tokens t ON t.id = u.token_id "
            "WHERE u.minute >= ? "
            "GROUP BY u.token_id, t.name, t.prefix "
            "ORDER BY (SUM(u.prompt_tokens) + SUM(u.completion_tokens)) DESC, "
            "         u.token_id ASC",
            (since,),
        )
        rows = await cur.fetchall()
    return {
        "range": range,
        "since_minute": since,
        "rows": [
            {
                "token_id": r[0],
                "name": r[1],
                "prefix": r[2],
                "requests": int(r[3] or 0),
                "prompt_tokens": int(r[4] or 0),
                "completion_tokens": int(r[5] or 0),
                "total_tokens": int((r[4] or 0) + (r[5] or 0)),
            }
            for r in rows
        ],
    }
