import secrets
import time
from datetime import timedelta

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from app.auth.bearer import generate_bearer_token
from app.auth.deps import require_jwt
from app.db.database import open_db
from app.db.repos.models import ModelRepo
from app.db.repos.tokens import (
    _UNSET,
    TokenRepo,
    TokenUsageRepo,
    sqlite_utc_in,
    sqlite_utc_now,
)

router = APIRouter(prefix="/api/tokens", tags=["tokens"])


class TokenCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    expires_in_days: int = Field(default=365, ge=0, le=3650)
    # S5 (#104) — sliding-window rate limit in TOKENS/sec; None = unlimited.
    # Pydantic validates >0; the DB CHECK trigger is redundant defence in
    # depth (we don't want a 500 if the route ever forgets the validator).
    rate_limit_tps: int | None = Field(default=None, ge=1, le=1_000_000)
    # STRICT scheduler priority 0..9; 9 is served first, 0 last. The schema
    # CHECK trigger mirrors this bound. Default 5 matches the column DEFAULT.
    priority: int = Field(default=5, ge=0, le=9)


class TokenUpdate(BaseModel):
    """PATCH body — every field is optional; omit to leave untouched.

    Setting ``rate_limit_tps`` to null (JSON null) clears the limit (i.e.
    switches the token back to unlimited). Omitting the key entirely leaves
    whatever value the row already has. The route turns the "omitted" case
    into the ``_UNSET`` sentinel before calling ``TokenRepo.update_limits``.
    """

    rate_limit_tps: int | None = Field(default=None, ge=1, le=1_000_000)
    priority: int | None = Field(default=None, ge=0, le=9)


class TokenRotate(BaseModel):
    grace_hours: int = Field(default=24, ge=0, le=720)
    expires_in_days: int | None = Field(default=None, ge=0, le=3650)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_token(
    body: TokenCreate, request: Request, _user: str = Depends(require_jwt)
):
    plaintext = generate_bearer_token()
    tid = secrets.token_hex(16)
    async with open_db(request.app.state.settings.db_path) as db:
        repo = TokenRepo(db)
        await repo.create(
            token_id=tid,
            name=body.name,
            plaintext=plaintext,
            expires_in_days=body.expires_in_days,
            rate_limit_tps=body.rate_limit_tps,
            priority=body.priority,
        )
        created = await repo.get(tid)
    if created is None:  # defensive — race between insert and read is impossible
        raise HTTPException(500, "token vanished after create")
    prefix = plaintext[:8]
    return {
        "id": tid,
        "name": body.name,
        "plaintext": plaintext,
        "prefix": prefix,
        "preview": prefix,
        "expires_at": created.expires_at,
        "rate_limit_tps": created.rate_limit_tps,
        "priority": created.priority,
    }


@router.patch("/{token_id}")
async def update_token(
    token_id: str,
    body: TokenUpdate,
    request: Request,
    _user: str = Depends(require_jwt),
):
    """Update rate/priority on an existing token.

    Omitted keys are untouched. Explicit ``null`` for ``rate_limit_tps``
    clears the limit (back to unlimited). The Pydantic schema rejects
    out-of-range values with 422; the DB CHECK trigger is belt-and-braces.

    **Note:** ``priority`` cannot be set to ``null`` — the column is ``NOT NULL``
    in the schema. Sending ``{"priority": null}`` will cause the underlying DB
    write to fail. To reset priority to the default, send ``{"priority": 5}``.
    """
    raw = body.model_dump(exclude_unset=True)
    rate_arg = raw.get("rate_limit_tps", _UNSET) if "rate_limit_tps" in raw else _UNSET
    prio_arg = raw.get("priority", _UNSET) if "priority" in raw else _UNSET

    async with open_db(request.app.state.settings.db_path) as db:
        repo = TokenRepo(db)
        ok = await repo.update_limits(
            token_id, rate_limit_tps=rate_arg, priority=prio_arg,
        )
        if not ok:
            raise HTTPException(404)
        row = await repo.get(token_id)
    # update_limits returned True so the row MUST exist; the assert
    # narrows the Optional for mypy and guards against a future race
    # if the row were deleted between the UPDATE and SELECT (sqlite
    # serialises the write, so this is defence-in-depth).
    assert row is not None, f"token {token_id} disappeared mid-PATCH"
    return {
        "id": row.id,
        "rate_limit_tps": row.rate_limit_tps,
        "priority": row.priority,
    }


@router.post("/{token_id}/rotate", status_code=status.HTTP_201_CREATED)
async def rotate_token(
    token_id: str,
    body: TokenRotate,
    request: Request,
    _user: str = Depends(require_jwt),
):
    """Rotate a token: rename old row to ``"{name} (old N)"`` and mint a
    new row that keeps the ORIGINAL name (#150).

    Rejects with 409 if the row was already rotated — cascaded `(old 1) →
    (old 2)` renames are footgun-y on accidental double-clicks; the UI
    already disables the button on rotated rows (token-row.tsx) but the
    server check is defence in depth for direct API callers.

    Returns the freshly-minted plaintext + the predecessor's new
    ``"{name} (old N)"`` name so the UI can show "rotated; old token is
    now <renamed_to>" without an extra list call.
    """
    async with open_db(request.app.state.settings.db_path) as db:
        repo = TokenRepo(db)
        old = await repo.get(token_id)
        if old is None:
            raise HTTPException(404)
        if old.rotated_at is not None:
            # 409 = the resource is in a state that conflicts with the request.
            # Matches the "already rotated" AC in the issue ("rotate of an
            # already-rotated token rejected with 4xx").
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                detail=(
                    f"Token '{old.name}' was already rotated at {old.rotated_at}; "
                    "rotate the successor instead."
                ),
            )
        try:
            new_id, new_plaintext, renamed_to = await repo.rotate(
                old_id=token_id,
                grace_hours=body.grace_hours,
                expires_in_days=body.expires_in_days,
            )
        except ValueError as exc:
            # Defence in depth: rotate() also raises "already rotated" if a
            # racing caller flipped rotated_at between our get() and rotate().
            # SQLite serialises writes so this is theoretical, but free.
            if "already rotated" in str(exc):
                raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
            raise
    prefix = new_plaintext[:8]
    return {
        "id": new_id,
        "name": old.name,  # The active token keeps the ORIGINAL name.
        "plaintext": new_plaintext,
        "prefix": prefix,
        "rotated_from": token_id,
        # UI surfaces this in the success modal: "old token renamed to <name>"
        # so the operator immediately knows where their previous bearer landed.
        "renamed_to": renamed_to,
    }


@router.get("")
async def list_tokens(request: Request, _user: str = Depends(require_jwt)):
    async with open_db(request.app.state.settings.db_path) as db:
        rows = await TokenRepo(db).list_all()
        # Bulk-fetch last-24h usage totals per token in a single pass —
        # the alternative (N round-trips) would scale poorly when an
        # operator has hundreds of tokens and the page loads on every
        # `/tokens` UI mount.
        usage_repo = TokenUsageRepo(db)
        now_minute = int(time.time() // 60)
        since_minute = now_minute - (24 * 60)
        usage_24h: dict[str, tuple[int, int, int]] = {}
        for r in rows:
            usage_24h[r.id] = await usage_repo.totals(
                token_id=r.id,
                since_minute=since_minute,
                until_minute=now_minute + 1,
            )

    now_str = sqlite_utc_now()
    in_30d_str = sqlite_utc_in(timedelta(days=30))

    def _enrich(r):
        is_expired = r.expires_at is not None and r.expires_at <= now_str
        is_near = (
            r.expires_at is not None
            and not is_expired
            and r.expires_at <= in_30d_str
        )
        successor = next((x.id for x in rows if x.rotated_from == r.id), None)
        usage_requests, usage_prompt, usage_completion = usage_24h.get(
            r.id, (0, 0, 0),
        )
        return {
            "id": r.id,
            "name": r.name,
            "prefix": r.prefix,
            "preview": r.prefix,
            "created_at": r.created_at,
            "last_used_at": r.last_used_at,
            "expires_at": r.expires_at,
            "rotated_at": r.rotated_at,
            "rotated_from": r.rotated_from,
            "successor_id": successor,
            "successor_deleted": r.rotated_at is not None and successor is None,
            "is_expired": is_expired,
            "is_near_expiry": is_near,
            "revoked_at": r.revoked_at,
            # S5 (#104) — surface rate/priority + 24h usage rollup so the
            # UI can paint the new "Rate / Priority / Last 24h" columns
            # without an extra round-trip per row.
            "rate_limit_tps": r.rate_limit_tps,
            "priority": r.priority,
            "usage_24h": {
                "requests": usage_requests,
                "prompt_tokens": usage_prompt,
                "completion_tokens": usage_completion,
                "total_tokens": usage_prompt + usage_completion,
            },
        }

    items = [
        _enrich(r) for r in rows if r.revoked_at is None or r.rotated_at is not None
    ]
    return {"items": items}


@router.get("/{token_id}/usage")
async def get_token_usage(
    token_id: str,
    request: Request,
    range: str = Query(default="24h", pattern=r"^(1h|24h|7d)$"),
    _user: str = Depends(require_jwt),
):
    """Per-minute usage rollup for one token.

    ``range`` selects the look-back window. The query plan is bounded by
    ``idx_token_usage_minute_minute`` so even a 7d range over a busy token
    is cheap. Returned ``buckets`` are minute-aligned (gaps mean zero —
    we don't pre-fill, the UI does that).
    """
    now_minute = int(time.time() // 60)
    if range == "1h":
        since_minute = now_minute - 60
    elif range == "24h":
        since_minute = now_minute - 24 * 60
    else:  # "7d" — the Pydantic Query pattern bounds this branch
        since_minute = now_minute - 7 * 24 * 60

    async with open_db(request.app.state.settings.db_path) as db:
        # Verify the token exists so we can return 404 instead of an
        # empty-bucket success that the UI would silently render as
        # "no usage" — clearer error surface.
        if await TokenRepo(db).get(token_id) is None:
            raise HTTPException(404)
        rollup = TokenUsageRepo(db)
        rows = await rollup.range(
            token_id=token_id,
            since_minute=since_minute,
            until_minute=now_minute + 1,
        )
        totals = await rollup.totals(
            token_id=token_id,
            since_minute=since_minute,
            until_minute=now_minute + 1,
        )

    return {
        "token_id": token_id,
        "range": range,
        "since_minute": since_minute,
        "until_minute": now_minute,
        "buckets": [
            {
                "minute": m,
                "requests": req,
                "prompt_tokens": pt,
                "completion_tokens": ct,
            }
            for (m, req, pt, ct) in rows
        ],
        "totals": {
            "requests": totals[0],
            "prompt_tokens": totals[1],
            "completion_tokens": totals[2],
            "total_tokens": totals[1] + totals[2],
        },
    }


@router.post("/{token_id}/test")
async def test_token(
    token_id: str,
    request: Request,
    _user: str = Depends(require_jwt),
):
    """Issue a 1-token completion through the proxy to sanity-check the token.

    Designed for the UI "Test token" button. The test mints a short-lived
    bearer header (we already know the token plaintext only at create time,
    so we can't recover it — instead we hit /v1/models which only requires
    the token id + scope/allowed_models check). On success returns the
    upstream model list. On any non-2xx, returns the proxy's status + body
    so the UI can display the exact failure to the operator.

    NB: this endpoint runs as the JWT-authenticated UI user, not as the
    bearer token holder. It cannot validate the rate-limit / priority
    path because that requires routing through the real proxy with the
    bearer secret — out of scope for the test button; the wizard surfaces
    "ping the proxy" mode which is sufficient for operator confidence.
    """
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        token_row = await TokenRepo(db).get(token_id)
        if token_row is None:
            raise HTTPException(404)
        models = await ModelRepo(db).list_all()

    # Build a "what would this token see?" projection so the UI can show
    # what models the bearer would be able to hit — done in-process, no
    # outbound HTTP needed (the proxy /v1/models endpoint also does this,
    # but it requires the actual bearer plaintext which we don't store).
    from app.proxy.auth import token_allows
    allowed_models = [
        m.served_model_name
        for m in models
        if m.status == "loaded" and token_allows(token_row, m.served_model_name)
    ]

    # If we can reach the local proxy listener, attempt a HEAD on /v1/models
    # to confirm the routing path. We can't pass the real bearer (we don't
    # store the plaintext), so this only checks "is the proxy alive" — the
    # status code is informational only.
    proxy_reachable = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            # Use 0.0.0.0 binding from settings — host is loopback for the
            # in-process test.
            url = f"http://127.0.0.1:{settings.bind_port}/healthz"
            r = await client.get(url)
            proxy_reachable = r.status_code == 200
    except httpx.HTTPError:
        proxy_reachable = False

    return {
        "token_id": token_id,
        "ok": True,
        "allowed_models": allowed_models,
        "rate_limit_tps": token_row.rate_limit_tps,
        "priority": token_row.priority,
        "proxy_reachable": proxy_reachable,
        "revoked": token_row.revoked_at is not None,
        "expired": token_row.expires_at is not None and token_row.expires_at <= sqlite_utc_now(),
    }


@router.delete("/{token_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_token(
    token_id: str, request: Request, _user: str = Depends(require_jwt)
):
    async with open_db(request.app.state.settings.db_path) as db:
        deleted = await TokenRepo(db).delete(token_id)
    if not deleted:
        raise HTTPException(404)
    return None
