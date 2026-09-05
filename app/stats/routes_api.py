import json
import time

from fastapi import APIRouter, Depends, HTTPException, Request

from app.auth.deps import require_jwt
from app.db.database import open_db

router = APIRouter()

_RANGE_TO_MINUTES = {"1h": 60, "6h": 360, "24h": 1440, "7d": 10080}


def _parse_models(models: str | None) -> list[str] | None:
    """Parse the ``?models=`` selection. ``None`` means "no filter".

    The distinction that matters is ABSENT vs EMPTY, and they are not the same
    question:

      ``?models`` absent      the whole deployment -- every model, every card.
                              The pre-selection behaviour, unchanged.
      ``?models=a,b``         only those models, and only the cards they hold.
      ``?models=`` (empty)    400.

    An empty selection could plausibly be read as "everything" or as "nothing",
    and silently choosing one is how a UI bug becomes a wrong number instead of
    an error. The UI's own rule is that at least one model stays selected, so an
    empty string reaching here is a client defect and is reported as one.

    Order and duplicates are normalised away: the value is a set of ids, and
    two clients that spell the same selection differently must not produce two
    cache entries or two different answers.
    """
    if models is None:
        return None
    ids = sorted({m.strip() for m in models.split(",") if m.strip()})
    if not ids:
        raise HTTPException(
            status_code=400,
            detail=(
                "empty models selection; omit ?models entirely to include every "
                "model, or name at least one"
            ),
        )
    return ids


def _decode_gpu_indices(raw) -> list[int]:
    """``models.gpu_indices`` (JSON list stored as TEXT) -> list[int].

    Never raises. A malformed or absent value yields ``[]``, because this is
    read by a dashboard: a bad row must narrow what a chart covers, visibly,
    rather than 500 a page whose whole job is to report on the deployment.
    """
    try:
        parsed = json.loads(raw) if raw else []
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [g for g in parsed if isinstance(g, int)]


async def _resolve_selection(db, ids: list[str]) -> list[int]:
    """Validate ``ids`` against the models table; return their GPU indices.

    Raises 400 for an id this deployment does not have. A silently-dropped id
    would give the operator a chart that answers a narrower question than the
    one their checkboxes describe -- the worst possible failure for a page whose
    entire job is to say which models a number covers.

    ``models.gpu_indices`` is a JSON list stored as TEXT. The union of those
    lists is what makes a GPU-side filter meaningful: VRAM, utilisation and
    power are per-CARD facts with no model column, so "this selection's VRAM" is
    exactly "the VRAM of the cards this selection occupies".
    """
    placeholders = ",".join("?" for _ in ids)
    cur = await db.execute(
        f"SELECT id, gpu_indices FROM models WHERE id IN ({placeholders})",
        tuple(ids),
    )
    rows = await cur.fetchall()
    found = {r[0] for r in rows}
    missing = [i for i in ids if i not in found]
    if missing:
        raise HTTPException(
            status_code=400, detail=f"unknown model id(s): {', '.join(missing)}"
        )
    gpus: set[int] = set()
    for _mid, raw in rows:
        gpus.update(_decode_gpu_indices(raw))
    return sorted(gpus)


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
    request: Request,
    range: str = "24h",
    models: str | None = None,
    _user: str = Depends(require_jwt),
):
    """Aggregate dashboard payload for the stats page, for a selection of models.

    ``models`` is a comma-separated list of model ids. Absent means the whole
    deployment (the original behaviour, byte-for-byte). Present, it narrows
    EVERY number in the response -- see ``_parse_models``.

    WHAT "NARROWS" MEANS PER SERIES, because the three data sources carry
    different dimensions and pretending otherwise would produce a chart that
    silently answers a different question than its checkboxes:

      tokens / tps   per MODEL. ``model_samples`` has a model_id column, so the
                     filter is exact and a selection is a true partition:
                     selecting every model sums to the unfiltered total.
      vram / util /  per CARD. Those tables have no model column at all, so the
      power          selection is resolved to the union of the selected models'
                     ``gpu_indices`` and the cards are filtered by that. "This
                     model's VRAM" is, and can only be, "the VRAM of the cards
                     it occupies" -- which is also the number an operator asking
                     "will another model fit beside it" needs.

    A selected model holding no cards contributes none, so a selection of only
    such rows yields empty GPU series rather than silently widening to the box.

    Returns:
      {
        "range": "24h",
        "now_minute": int,
        "since_minute": int,
        "selected_model_ids": [str, ...] | None,   # echo; null when unfiltered
        "selected_gpu_indices": [int, ...] | None, # the cards it resolved to
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
          {"id": str, "served_model_name": str, "gpu_indices": [int, ...]}, ...
        ],
        "series": {
          "vram": [{"minute": int, "used_mib": int, "total_mib": int}, ...],
          "util": [{"minute": int, "max_pct": int}, ...],
          "power": [{"minute": int, "watts": float}, ...],     # sum / minute
          "tokens": [{"minute": int, "prompt": int, "completion": int}, ...],
        }
      }

    NOTE ON THE TOKEN SOURCE. The tokens series and ``tps`` come from
    ``model_samples``, not ``token_usage_minute``. They were the latter, which
    is keyed by API token and has no model dimension -- so it cannot be filtered
    by model at all, and using it for the unfiltered case while using
    ``model_samples`` for the filtered one would make "every model selected"
    disagree with "no filter". One source keeps the selection a partition. The
    numbers move slightly: ``model_samples`` also counts requests made without
    an API token, which ``token_usage_minute`` deliberately skips so the NULL
    key does not accumulate orphan rows. ``/api/stats/v2/tokens-per-key`` still
    reads ``token_usage_minute`` -- that endpoint asks a per-key question, which
    is the one that table answers.
    """
    _validate_range(range)
    since = _since_minute(range)
    now_min = int(time.time() // 60)
    settings = request.app.state.settings
    selected = _parse_models(models)
    async with open_db(settings.db_path) as db:
        # Resolved first: it validates the ids (400 on unknown) before any
        # aggregation runs, so a bad request never costs four table scans.
        gpu_indices = (
            await _resolve_selection(db, selected) if selected is not None else None
        )
        # Bound into every card-scoped query below. EMPTY when unfiltered, so
        # the unfiltered path executes exactly the SQL it always did rather
        # than a filtered query that happens to select everything.
        if gpu_indices is None:
            gpu_where, gpu_args = "", ()
        elif gpu_indices:
            gpu_where = f" AND gpu_index IN ({','.join('?' for _ in gpu_indices)})"
            gpu_args = tuple(gpu_indices)
        else:
            # A selection that occupies no card. `IN ()` is a syntax error in
            # SQLite, and a bare 1=1 would silently widen to the whole box --
            # the exact substitution this route exists to prevent.
            gpu_where, gpu_args = " AND 0", ()
        # power_samples names the column gpu_idx, not gpu_index.
        pwr_where = gpu_where.replace("gpu_index", "gpu_idx")
        if selected is None:
            model_where, model_args = "", ()
        else:
            model_where = f" AND model_id IN ({','.join('?' for _ in selected)})"
            model_args = tuple(selected)
        # ---- series: GPU util + VRAM, aggregated across the SELECTED cards --
        cur = await db.execute(
            "SELECT minute, "
            "       MAX(utilization_pct) AS max_util, "
            "       SUM(memory_used_mib) AS used_mib, "
            "       SUM(memory_total_mib) AS total_mib "
            "FROM gpu_samples WHERE minute >= ?" + gpu_where + " "
            "GROUP BY minute ORDER BY minute ASC",
            (since, *gpu_args),
        )
        gpu_rows = await cur.fetchall()

        # ---- series: power.draw, summed across the SELECTED cards ----------
        # Each row in power_samples is (gpu_idx, minute, watts_sum, samples)
        # — to get per-minute average watts we first average within each GPU
        # (watts_sum/samples) then sum across the cards in scope.
        cur = await db.execute(
            "SELECT minute, SUM(watts_sum / NULLIF(samples, 0)) AS box_watts "
            "FROM power_samples WHERE minute >= ?" + pwr_where + " "
            "GROUP BY minute ORDER BY minute ASC",
            (since, *gpu_args),
        )
        power_rows = await cur.fetchall()

        # ---- series: tokens, summed across the SELECTED models per minute ---
        cur = await db.execute(
            "SELECT minute, "
            "       COALESCE(SUM(prompt_tokens), 0) AS prompt, "
            "       COALESCE(SUM(completion_tokens), 0) AS completion "
            "FROM model_samples WHERE minute >= ?" + model_where + " "
            "GROUP BY minute ORDER BY minute ASC",
            (since, *model_args),
        )
        token_rows = await cur.fetchall()

        # ---- current snapshot (most recent minute in window) ----------------
        # The inner MAX(minute) carries the same filter as the outer query: the
        # "latest minute" has to be the latest minute THIS SELECTION reported,
        # or a card that stopped sampling would read as zero against a minute
        # only its neighbours reached.
        cur = await db.execute(
            "SELECT SUM(memory_used_mib), SUM(memory_total_mib), MAX(utilization_pct) "
            "FROM gpu_samples WHERE minute = ("
            "  SELECT MAX(minute) FROM gpu_samples WHERE minute >= ?" + gpu_where +
            ")" + gpu_where,
            (since, *gpu_args, *gpu_args),
        )
        cur_gpu = await cur.fetchone() or (None, None, None)

        cur = await db.execute(
            "SELECT SUM(watts_sum / NULLIF(samples, 0)) "
            "FROM power_samples WHERE minute = ("
            "  SELECT MAX(minute) FROM power_samples WHERE minute >= ?" + pwr_where +
            ")" + pwr_where,
            (since, *gpu_args, *gpu_args),
        )
        cur_power = (await cur.fetchone() or (None,))[0]

        cur = await db.execute(
            "SELECT COALESCE(SUM(prompt_tokens), 0) + COALESCE(SUM(completion_tokens), 0) "
            "FROM model_samples WHERE minute = ("
            "  SELECT MAX(minute) FROM model_samples WHERE minute >= ?" + model_where +
            ")" + model_where,
            (since, *model_args, *model_args),
        )
        last_min_tokens = (await cur.fetchone() or (0,))[0] or 0

        # ---- active model list ---------------------------------------------
        # ``models.status='loaded'`` AND ``model_runtime`` row exists, same as
        # header/routes_api.py::_active_models. NOT filtered by the selection:
        # this list is what the selector itself is built from, so narrowing it
        # would make a deselected model impossible to select again.
        cur = await db.execute(
            "SELECT m.id, m.served_model_name, m.gpu_indices "
            "FROM models m JOIN model_runtime r ON r.model_id = m.id "
            "WHERE m.status = 'loaded' "
            "ORDER BY m.served_model_name ASC"
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
        # Echoed so the page can state what its numbers cover instead of
        # relying on the checkboxes still matching the response in flight.
        # Null, not the full id list, when unfiltered: "every model" and "these
        # N models that happen to be all of them" answer different questions
        # about the cards, and the difference must survive the round trip.
        "selected_model_ids": selected,
        "selected_gpu_indices": gpu_indices,
        "current": {
            "vram_used_mib": vram_used,
            "vram_total_mib": vram_total,
            "vram_pct": vram_pct,
            "gpu_util_pct": util_pct,
            "power_w": float(cur_power) if cur_power is not None else None,
            "tps": tps,
        },
        "active_models": [
            {
                "id": r[0],
                "served_model_name": r[1],
                # So the selector can say WHICH cards a checkbox brings in --
                # the operator's next question after "why did VRAM drop when I
                # deselected that model".
                "gpu_indices": _decode_gpu_indices(r[2]),
            }
            for r in active_rows
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
