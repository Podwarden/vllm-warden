import bcrypt
import jwt as pyjwt
from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from app.auth.deps import require_jwt
from app.auth.jwt import decode, mint_access, mint_refresh
from app.auth.origin import origin_check_dep
from app.db.database import open_db
from app.db.repos.users import UserRepo

# bcrypt.checkpw against this constant when the user is unknown, so the
# response time matches the "user exists, wrong password" path. Otherwise an
# attacker can enumerate usernames by timing /api/auth/login.
_DUMMY_HASH = bcrypt.hashpw(b"timing-equalizer", bcrypt.gensalt()).decode()

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginBody(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


@router.post("/login")
async def login(body: LoginBody, request: Request, response: Response):
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        user = await UserRepo(db).get_by_username(body.username)
    if user is None:
        bcrypt.checkpw(body.password.encode("utf-8"), _DUMMY_HASH.encode("utf-8"))
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid credentials")
    if not bcrypt.checkpw(
        body.password.encode("utf-8"), user.password_hash.encode("utf-8")
    ):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid credentials")
    secret = request.app.state.jwt_secret
    access_ttl = settings.session_access_ttl_minutes
    refresh_ttl = settings.session_refresh_ttl_days
    access = mint_access(user.username, secret, ttl_minutes=access_ttl)
    refresh = mint_refresh(user.username, secret, ttl_days=refresh_ttl)
    response.set_cookie(
        "vw_refresh", refresh,
        max_age=refresh_ttl * 86400,
        httponly=True, secure=True, samesite="strict",
        path="/api/auth",
    )
    return {"access_token": access, "expires_in": access_ttl * 60}


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT,
             dependencies=[Depends(origin_check_dep)])
async def logout(request: Request, response: Response,
                 user: str = Depends(require_jwt)):
    request.app.state.stream_registry.cancel_user(user)
    response.delete_cookie("vw_refresh", path="/api/auth")
    return None


@router.post("/refresh", dependencies=[Depends(origin_check_dep)])
async def refresh(
    request: Request,
    vw_refresh: str | None = Cookie(default=None),
):
    if vw_refresh is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing refresh cookie")
    secret = request.app.state.jwt_secret
    try:
        claims = decode(vw_refresh, secret)
    except pyjwt.PyJWTError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid refresh token") from exc
    if claims.get("typ") != "refresh":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "wrong token type")
    sub = claims.get("sub")
    if not sub:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing subject")
    access_ttl = request.app.state.settings.session_access_ttl_minutes
    access = mint_access(sub, secret, ttl_minutes=access_ttl)
    return {"access_token": access, "expires_in": access_ttl * 60}


class TicketBody(BaseModel):
    path: str


@router.post("/sse-ticket")
async def mint_sse_ticket(
    body: TicketBody, request: Request, user: str = Depends(require_jwt)
):
    # God mode is the one stream that can be switched off at runtime. The
    # browser's EventSource can never read the stream endpoint's own 409
    # (`onerror` exposes no HTTP status), so the viewer classifies the
    # disabled state from THIS mint response. Refuse a god-mode ticket up
    # front when the feature is off. Scoped to the god-mode path only — every
    # other stream (e.g. model logs) mints unconditionally as before.
    from app.proxy.routes_godmode import STREAM_PATH as GODMODE_STREAM_PATH

    if body.path == GODMODE_STREAM_PATH:
        settings = request.app.state.settings
        hub = getattr(request.app.state, "godmode_hub", None)
        if hub is None or not settings.godmode_enabled:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                "god mode is disabled (VW_GODMODE_ENABLED)",
            )

    return {"ticket": request.app.state.sse_tickets.mint(user, body.path)}
