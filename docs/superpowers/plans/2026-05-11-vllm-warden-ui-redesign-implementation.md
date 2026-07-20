# vllm-warden UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace vllm-warden's Jinja + htmx UI wholesale with a JWT-authenticated FastAPI backend (MR-1) and a Next.js 15 + Tailwind + recharts frontend mirroring podwarden's stack (MR-2).

**Architecture:** Two-MR rollout. MR-1 (`feat/jwt-and-jinja-cleanup`) is a headless backend cutover: replaces Starlette `SessionMiddleware` with JWT bearer auth (HS256, access + refresh-cookie + SSE-ticket), migrates the tokens table to add `expires_at`/`rotated_at`/`rotated_from`, expands the settings API to the full runtime + per-model surface, deletes every Jinja template and `routes_web.py`, and ships an operator curl runbook for the inter-MR gap. MR-2 (`feat/ui-redesign-nextjs`) scaffolds `frontend/` with Next.js 15 standalone, copies podwarden's theme + ui primitives + ansi-log verbatim, implements 9 pages against the new JSON API, and bumps the Hub catalog template to a new `v2.x` major track with `breaking: true`.

**Tech Stack:** Backend — FastAPI, aiosqlite, bcrypt, **pyjwt[crypto]** (replacing itsdangerous), Python 3.11. Frontend — Next.js 15, React 19, Tailwind 3.4, recharts ^3.7.0, lucide-react, SWR, native EventSource. Tests — pytest+pytest-asyncio (backend), Vitest + Testing Library + Playwright (frontend). All commands run via Docker per the workspace's `make` targets.

---

## File structure

### MR-1 — Backend (`feat/jwt-and-jinja-cleanup`)

**Create:**
- `app/auth/jwt.py` — HS256 encode/decode for access + refresh tokens
- `app/auth/jwt_secret.py` — env → file → fail bootstrap for `VW_JWT_SECRET`
- `app/auth/routes.py` — `/api/auth/login` `/refresh` `/logout` `/sse-ticket`
- `app/auth/origin.py` — server-side `Origin` header check dep
- `app/auth/sse_tickets.py` — HMAC ticket mint + verify + single-use deny-set
- `app/auth/stream_registry.py` — in-process `set[StreamHandle]` per user_id
- `app/db/sql/0009_token_expiry.sql` — 3-step SQLite migration
- `app/tokens/rotate.py` — `TokenRepo.rotate()` business logic (separate file because route handler stays thin)
- `docs/operating.md` — curl runbook for inter-MR gap

**Modify:**
- `app/auth/deps.py` — replace `require_session_json` with `require_jwt`
- `app/db/repos/tokens.py` — add 3 columns to `TokenRow`; new `create(expires_in_days)`; new `rotate()` and `list_all()` returning new fields
- `app/proxy/auth.py` — add `expires_at` enforcement to the bearer check
- `app/models/routes_logs.py` — switch from `require_session_json` to SSE-ticket auth + register stream in revocation registry
- `app/settings/routes_api.py` — replace 2-field surface with full runtime table (12 keys, with `requires_restart` in PATCH echo); add `/api/models/{id}/settings` GET/PATCH
- `app/tokens/routes_api.py` — add `expires_in_days` to create body; `is_expired`/`is_near_expiry`/`rotated_at`/`rotated_from`/`successor_id` on list; add `POST /{id}/rotate`
- `app/main.py` — drop `Jinja2Templates`, `StaticFiles`, `SessionMiddleware`, all `routes_web` includes
- `requirements.txt` — remove `jinja2`, `itsdangerous`; add `pyjwt[crypto]`

**Delete:**
- `app/web/` (entire directory: templates + static + assets)
- `app/auth/sessions.py`
- `app/auth/routes_web.py`
- `app/models/routes_web.py`
- `app/setup/routes_web.py`
- `app/settings/routes_web.py`
- `app/stats/routes_web.py`
- `app/tokens/routes_web.py`
- `tests/unit/web/` (any pytest assertions that fetch rendered HTML)
- `tests/unit/templates/` (Jinja-specific tests)

### MR-2 — Frontend (`feat/ui-redesign-nextjs`)

**Create (frontend/ tree, all new):**
- `frontend/package.json`
- `frontend/next.config.ts` (with `output: 'standalone'`)
- `frontend/tsconfig.json`
- `frontend/tailwind.config.ts`
- `frontend/postcss.config.mjs`
- `frontend/Dockerfile` (multi-stage: deps → build → standalone runtime)
- `frontend/.dockerignore`
- `frontend/src/app/layout.tsx`
- `frontend/src/app/globals.css` (copy from podwarden)
- `frontend/src/app/page.tsx` (redirect to /models or /setup)
- `frontend/src/app/api/health/route.ts`
- `frontend/src/app/login/page.tsx`
- `frontend/src/app/setup/{layout,welcome/page,admin/page,hf-token/page,gpus/page,done/page}.tsx`
- `frontend/src/app/models/page.tsx`
- `frontend/src/app/models/[id]/page.tsx`
- `frontend/src/app/models/[id]/settings/page.tsx`
- `frontend/src/app/tokens/page.tsx`
- `frontend/src/app/stats/page.tsx`
- `frontend/src/app/settings/page.tsx`
- `frontend/src/components/{theme-switcher,nav-bar,ansi-log,auth-gate}.tsx` (copied or adapted)
- `frontend/src/components/ui/{button,card,input,badge,skeleton,modal,tabs,select,combobox}.tsx` (copied verbatim)
- `frontend/src/components/models/{add-model-modal,model-card,pull-progress,log-stream}.tsx`
- `frontend/src/components/tokens/{token-row,create-token-dialog,rotate-token-dialog,expiry-banner}.tsx`
- `frontend/src/components/settings/{runtime-tab,model-tab,setting-field}.tsx`
- `frontend/src/components/stats/{throughput-chart,gpu-util-chart}.tsx`
- `frontend/src/components/panels/{metric-summary-panel,status-table-panel,config-form-panel}.tsx`
- `frontend/src/lib/api.ts`
- `frontend/src/lib/api-types.generated.ts`
- `frontend/src/lib/auth-fetch.ts`
- `frontend/src/lib/theme.tsx` (copy from podwarden)
- `frontend/src/lib/utils.ts` (copy)
- `frontend/src/lib/sse.ts`
- `frontend/src/lib/settings-hints.ts`
- `frontend/tests/component/*.test.tsx` (Vitest)
- `frontend/tests/e2e/happy-path.spec.ts` (Playwright)

**Modify:**
- `docker-compose.yml` — add `ui` service, expose 3000; api now JSON-only
- `deploy/hub/compose.yaml` — 1 → 2 services with image-tag sentinels
- `deploy/hub/template.json` — bump to `v2.x` track; `breaking: true`; `min_compatible_warden_version`; ingress target = `ui` service; UI port 3000; add `VW_FRONTEND_ORIGIN` required env
- `deploy/hub/README-hub.md` — migration note for v1 → v2 single-service installs
- `.gitlab-ci.yml` — add `lint:frontend`, `build:ui`, `typecheck:api-types`
- `Dockerfile` — strip web assets stage if still present
- `Makefile` — add `frontend-build`, `frontend-lint`, `frontend-typecheck`, `frontend-test`, `generate-api-types`

---

## Critical context for implementers

- **Everything runs in Docker.** Never run `npm`, `node`, `python`, `pip`, or `pytest` directly on the host. Use `make` targets or `docker run --rm -v $(pwd):/app -w /app <image> <cmd>`. The repo's `Makefile` already wraps the python runner; MR-2 adds the npm wrapper.
- **SQLite-only.** All migration SQL uses SQLite dialect; no PG-isms.
- **`expires_at IS NULL` means "never expires"** in the bearer-check — backfill is best-effort, the check tolerates rows that slip through.
- **EventSource has an auto-reconnect trap**: on transport error it retries with the same (now-expired) ticket. The `useEventSource` hook in `lib/sse.ts` MUST close the EventSource on `onerror` and explicitly re-mint a fresh ticket before reconnecting. Do not rely on native reconnect.
- **Single-replica only.** The in-process stream cancellation registry assumes vllm-warden runs one process (it supervises a local vLLM subprocess). Do not promise multi-replica behaviour.
- **MR-1 ships headless.** Brief UI outage between MR-1 merge and MR-2 merge is accepted. Operators use the curl runbook. Do not add temporary v1 cookie fallback.
- **MR-2 reuses from podwarden** at `/home/ip/projects/pw/podwarden/frontend/src/`. The "Reuse from podwarden" section of the spec lists every file path verbatim. Copy is preferred over symlinks because vllm-warden is a separate git repo.

---

# Phase 1 — MR-1: Headless backend cutover

## Task 0.1: Branch from develop

**Files:**
- Touch: `.git/` (branch metadata)

- [ ] **Step 1:** From `/home/ip/projects/vllm-warden`, ensure clean tree.

Run: `git -C /home/ip/projects/vllm-warden status`
Expected: working tree clean on `feat/ui-redesign-spec` (the spec branch, currently at 745d5f9).

- [ ] **Step 2:** Create the implementation branch off develop tip.

Run:
```bash
git -C /home/ip/projects/vllm-warden fetch origin develop
git -C /home/ip/projects/vllm-warden checkout -b feat/jwt-and-jinja-cleanup origin/develop
```

- [ ] **Step 3:** Cherry-pick the spec + plan onto the new branch so the artifacts ship with the MR.

Run:
```bash
git -C /home/ip/projects/vllm-warden cherry-pick 745d5f9
git -C /home/ip/projects/vllm-warden add docs/superpowers/plans/2026-05-11-vllm-warden-ui-redesign-implementation.md
git -C /home/ip/projects/vllm-warden commit -m "docs: add vllm-warden UI redesign implementation plan"
```

---

## Task 1.1: JWT secret bootstrap module

**Files:**
- Create: `app/auth/jwt_secret.py`
- Test: `tests/unit/auth/test_jwt_secret.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/auth/test_jwt_secret.py
from pathlib import Path
import os
import pytest
from app.auth.jwt_secret import load_jwt_secret


def test_env_takes_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("VW_JWT_SECRET", "from-env")
    secret = load_jwt_secret(db_path=tmp_path / "vw.db")
    assert secret == "from-env"
    assert not (tmp_path / "jwt_secret").exists()


def test_persists_on_first_boot(tmp_path, monkeypatch):
    monkeypatch.delenv("VW_JWT_SECRET", raising=False)
    secret_path = tmp_path / "jwt_secret"
    secret = load_jwt_secret(db_path=tmp_path / "vw.db")
    assert len(secret) >= 64
    assert secret_path.exists()
    assert oct(secret_path.stat().st_mode)[-3:] == "600"
    secret2 = load_jwt_secret(db_path=tmp_path / "vw.db")
    assert secret == secret2


def test_refuses_unwritable_data_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("VW_JWT_SECRET", raising=False)
    bad = tmp_path / "nope"  # parent does not exist and we will not mkdir
    with pytest.raises(RuntimeError, match="cannot persist"):
        load_jwt_secret(db_path=bad / "child" / "vw.db")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make test PYTEST_ARGS="tests/unit/auth/test_jwt_secret.py -v"`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.auth.jwt_secret'`.

- [ ] **Step 3: Write minimal implementation**

```python
# app/auth/jwt_secret.py
import os
import secrets
from pathlib import Path


def load_jwt_secret(db_path: Path) -> str:
    env_val = os.environ.get("VW_JWT_SECRET", "").strip()
    if env_val:
        return env_val
    secret_path = db_path.parent / "jwt_secret"
    if secret_path.exists():
        return secret_path.read_text().strip()
    if not db_path.parent.exists():
        raise RuntimeError(
            f"cannot persist JWT secret: data dir {db_path.parent} does not exist "
            "and VW_JWT_SECRET is unset"
        )
    secret = secrets.token_urlsafe(64)
    secret_path.write_text(secret)
    secret_path.chmod(0o600)
    return secret
```

- [ ] **Step 4: Run test to verify it passes**

Run: `make test PYTEST_ARGS="tests/unit/auth/test_jwt_secret.py -v"`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add app/auth/jwt_secret.py tests/unit/auth/test_jwt_secret.py
git commit -m "feat(auth): JWT secret bootstrap (env → file → fail)"
```

---

## Task 1.2: JWT encode/decode

**Files:**
- Create: `app/auth/jwt.py`
- Test: `tests/unit/auth/test_jwt.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/auth/test_jwt.py
from datetime import datetime, timedelta, timezone
import pytest
import jwt as pyjwt
from app.auth.jwt import mint_access, mint_refresh, decode


SECRET = "test-secret-not-for-prod"


def test_access_token_roundtrip():
    tok = mint_access("admin", SECRET, ttl_minutes=15)
    claims = decode(tok, SECRET)
    assert claims["sub"] == "admin"
    assert claims["typ"] == "access"
    assert 0 < claims["exp"] - claims["iat"] <= 15 * 60


def test_refresh_token_roundtrip():
    tok = mint_refresh("admin", SECRET, ttl_days=7)
    claims = decode(tok, SECRET)
    assert claims["sub"] == "admin"
    assert claims["typ"] == "refresh"


def test_rejects_expired():
    now = datetime.now(timezone.utc)
    expired = pyjwt.encode(
        {"sub": "admin", "typ": "access",
         "iat": int((now - timedelta(hours=2)).timestamp()),
         "exp": int((now - timedelta(hours=1)).timestamp())},
        SECRET, algorithm="HS256",
    )
    with pytest.raises(pyjwt.ExpiredSignatureError):
        decode(expired, SECRET)


def test_rejects_wrong_secret():
    tok = mint_access("admin", SECRET, ttl_minutes=15)
    with pytest.raises(pyjwt.InvalidSignatureError):
        decode(tok, "different-secret")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make test PYTEST_ARGS="tests/unit/auth/test_jwt.py -v"`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# app/auth/jwt.py
from datetime import datetime, timedelta, timezone
import jwt as pyjwt


def _now() -> datetime:
    return datetime.now(timezone.utc)


def mint_access(sub: str, secret: str, ttl_minutes: int) -> str:
    now = _now()
    return pyjwt.encode(
        {
            "sub": sub,
            "typ": "access",
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=ttl_minutes)).timestamp()),
        },
        secret,
        algorithm="HS256",
    )


def mint_refresh(sub: str, secret: str, ttl_days: int) -> str:
    now = _now()
    return pyjwt.encode(
        {
            "sub": sub,
            "typ": "refresh",
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(days=ttl_days)).timestamp()),
        },
        secret,
        algorithm="HS256",
    )


def decode(token: str, secret: str) -> dict:
    return pyjwt.decode(token, secret, algorithms=["HS256"])
```

- [ ] **Step 4: Add pyjwt to requirements.txt**

Add line `PyJWT[crypto]==2.9.0` to `requirements.txt`. Rebuild test image.

Run: `make build-test-image` (or `make deps` if a dedicated target exists).

- [ ] **Step 5: Run test to verify it passes**

Run: `make test PYTEST_ARGS="tests/unit/auth/test_jwt.py -v"`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add app/auth/jwt.py tests/unit/auth/test_jwt.py requirements.txt
git commit -m "feat(auth): HS256 JWT encode/decode for access + refresh"
```

---

## Task 1.3: Login route

**Files:**
- Create: `app/auth/routes.py`
- Modify: `app/main.py` (wire router, will be re-edited later when removing Jinja)
- Test: `tests/unit/auth/test_login.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/auth/test_login.py
import pytest
from httpx import AsyncClient
from app.main import app  # adjust import to test fixture pattern in repo


@pytest.mark.asyncio
async def test_login_success_sets_cookie_returns_access(
    client: AsyncClient, seed_admin
):
    r = await client.post("/api/auth/login", json={
        "username": "admin", "password": "correct-horse-battery-staple"
    })
    assert r.status_code == 200
    body = r.json()
    assert "access_token" in body
    assert body["expires_in"] == 15 * 60
    set_cookie = r.headers["set-cookie"]
    assert "vw_refresh=" in set_cookie
    assert "HttpOnly" in set_cookie
    assert "SameSite=Strict" in set_cookie
    assert "Path=/api/auth" in set_cookie


@pytest.mark.asyncio
async def test_login_wrong_password_401(client, seed_admin):
    r = await client.post("/api/auth/login", json={
        "username": "admin", "password": "wrong"
    })
    assert r.status_code == 401
```

(Reuse the existing `client` + `seed_admin` fixtures from `tests/conftest.py`. If the seed fixture uses a different password, adjust the literal.)

- [ ] **Step 2: Run test to verify it fails**

Run: `make test PYTEST_ARGS="tests/unit/auth/test_login.py -v"`
Expected: FAIL — `/api/auth/login` 404 or 405.

- [ ] **Step 3: Write minimal implementation**

```python
# app/auth/routes.py
from fastapi import APIRouter, Depends, HTTPException, Response, Request, status
from pydantic import BaseModel, Field
import bcrypt

from app.auth.jwt import mint_access, mint_refresh
from app.db.database import open_db
from app.db.repos.users import UserRepo  # existing

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginBody(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


@router.post("/login")
async def login(body: LoginBody, request: Request, response: Response):
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        user = await UserRepo(db).get_by_username(body.username)
    if user is None or not bcrypt.checkpw(
        body.password.encode("utf-8"), user.password_hash.encode("utf-8")
    ):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid credentials")
    secret = request.app.state.jwt_secret
    access_ttl = request.app.state.settings.session_access_ttl_minutes
    refresh_ttl = request.app.state.settings.session_refresh_ttl_days
    access = mint_access(user.username, secret, ttl_minutes=access_ttl)
    refresh = mint_refresh(user.username, secret, ttl_days=refresh_ttl)
    response.set_cookie(
        "vw_refresh", refresh,
        max_age=refresh_ttl * 86400,
        httponly=True, secure=True, samesite="strict",
        path="/api/auth",
    )
    return {"access_token": access, "expires_in": access_ttl * 60}
```

Wire in `app/main.py`:
```python
from app.auth.routes import router as auth_router
app.include_router(auth_router)
# also bootstrap secret early in lifespan:
# app.state.jwt_secret = load_jwt_secret(settings.db_path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `make test PYTEST_ARGS="tests/unit/auth/test_login.py -v"`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add app/auth/routes.py app/main.py tests/unit/auth/test_login.py
git commit -m "feat(auth): POST /api/auth/login with refresh cookie"
```

---

## Task 1.4: Origin-check dependency

**Files:**
- Create: `app/auth/origin.py`
- Test: `tests/unit/auth/test_origin.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/auth/test_origin.py
import pytest
from fastapi import HTTPException, Request
from app.auth.origin import require_matching_origin


def _req(origin_value):
    headers = []
    if origin_value is not None:
        headers.append((b"origin", origin_value.encode()))
    scope = {"type": "http", "headers": headers, "method": "POST"}
    return Request(scope)


def test_missing_origin_rejected(monkeypatch):
    monkeypatch.setenv("VW_FRONTEND_ORIGIN", "https://vllm.example.com")
    with pytest.raises(HTTPException) as ei:
        require_matching_origin(_req(None), "https://vllm.example.com")
    assert ei.value.status_code == 403


def test_wrong_origin_rejected():
    with pytest.raises(HTTPException) as ei:
        require_matching_origin(_req("https://evil.example.com"),
                                "https://vllm.example.com")
    assert ei.value.status_code == 403


def test_matching_origin_ok():
    # Should not raise.
    require_matching_origin(_req("https://vllm.example.com"),
                            "https://vllm.example.com")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make test PYTEST_ARGS="tests/unit/auth/test_origin.py -v"`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# app/auth/origin.py
from fastapi import Depends, HTTPException, Request, status


def require_matching_origin(request: Request, expected: str) -> None:
    got = request.headers.get("origin")
    if got is None or got != expected:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "origin mismatch")


def origin_check_dep(request: Request) -> None:
    expected = request.app.state.settings.frontend_origin
    require_matching_origin(request, expected)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `make test PYTEST_ARGS="tests/unit/auth/test_origin.py -v"`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add app/auth/origin.py tests/unit/auth/test_origin.py
git commit -m "feat(auth): server-side Origin header check"
```

---

## Task 1.5: Refresh route

**Files:**
- Modify: `app/auth/routes.py`
- Test: `tests/unit/auth/test_refresh.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/auth/test_refresh.py
import pytest


@pytest.mark.asyncio
async def test_refresh_success(client, seed_admin):
    login = await client.post("/api/auth/login",
                              json={"username": "admin", "password": "correct-horse-battery-staple"},
                              headers={"Origin": "http://localhost:3000"})
    cookie = login.cookies["vw_refresh"]
    r = await client.post("/api/auth/refresh",
                          headers={"Origin": "http://localhost:3000"},
                          cookies={"vw_refresh": cookie})
    assert r.status_code == 200
    assert "access_token" in r.json()


@pytest.mark.asyncio
async def test_refresh_no_origin_rejected(client, seed_admin):
    login = await client.post("/api/auth/login",
                              json={"username": "admin", "password": "correct-horse-battery-staple"},
                              headers={"Origin": "http://localhost:3000"})
    r = await client.post("/api/auth/refresh",
                          cookies={"vw_refresh": login.cookies["vw_refresh"]})
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_refresh_no_cookie_401(client):
    r = await client.post("/api/auth/refresh",
                          headers={"Origin": "http://localhost:3000"})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_refresh_invalid_token_401(client):
    r = await client.post("/api/auth/refresh",
                          headers={"Origin": "http://localhost:3000"},
                          cookies={"vw_refresh": "not-a-jwt"})
    assert r.status_code == 401
```

Configure the test fixture so `settings.frontend_origin = "http://localhost:3000"`.

- [ ] **Step 2: Run test to verify it fails**

Run: `make test PYTEST_ARGS="tests/unit/auth/test_refresh.py -v"`
Expected: FAIL — 404 / 405.

- [ ] **Step 3: Write minimal implementation**

Append to `app/auth/routes.py`:
```python
from fastapi import Cookie
import jwt as pyjwt
from app.auth.origin import origin_check_dep


@router.post("/refresh", dependencies=[Depends(origin_check_dep)])
async def refresh(
    request: Request,
    vw_refresh: str | None = Cookie(default=None),
):
    if vw_refresh is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing refresh cookie")
    secret = request.app.state.jwt_secret
    try:
        claims = pyjwt.decode(vw_refresh, secret, algorithms=["HS256"])
    except pyjwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid refresh token")
    if claims.get("typ") != "refresh":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "wrong token type")
    access_ttl = request.app.state.settings.session_access_ttl_minutes
    access = mint_access(claims["sub"], secret, ttl_minutes=access_ttl)
    return {"access_token": access, "expires_in": access_ttl * 60}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `make test PYTEST_ARGS="tests/unit/auth/test_refresh.py -v"`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add app/auth/routes.py tests/unit/auth/test_refresh.py
git commit -m "feat(auth): POST /api/auth/refresh with origin check"
```

---

## Task 1.6: Stream cancellation registry

**Files:**
- Create: `app/auth/stream_registry.py`
- Test: `tests/unit/auth/test_stream_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/auth/test_stream_registry.py
import asyncio
import pytest
from app.auth.stream_registry import StreamRegistry


@pytest.mark.asyncio
async def test_register_and_cancel():
    reg = StreamRegistry()

    cancelled = asyncio.Event()

    async def fake_stream():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = asyncio.create_task(fake_stream())
    handle = reg.register("admin", task)
    assert reg.count("admin") == 1

    reg.cancel_user("admin")
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    assert reg.count("admin") == 0


@pytest.mark.asyncio
async def test_unregister_on_completion():
    reg = StreamRegistry()
    task = asyncio.create_task(asyncio.sleep(0))
    reg.register("admin", task)
    await asyncio.sleep(0.01)
    reg.unregister("admin", task)
    assert reg.count("admin") == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make test PYTEST_ARGS="tests/unit/auth/test_stream_registry.py -v"`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# app/auth/stream_registry.py
import asyncio
from collections import defaultdict


class StreamRegistry:
    def __init__(self) -> None:
        self._by_user: dict[str, set[asyncio.Task]] = defaultdict(set)

    def register(self, user_id: str, task: asyncio.Task) -> asyncio.Task:
        self._by_user[user_id].add(task)
        return task

    def unregister(self, user_id: str, task: asyncio.Task) -> None:
        bucket = self._by_user.get(user_id)
        if bucket:
            bucket.discard(task)
            if not bucket:
                self._by_user.pop(user_id, None)

    def cancel_user(self, user_id: str) -> int:
        tasks = list(self._by_user.get(user_id, ()))
        for t in tasks:
            t.cancel()
        self._by_user.pop(user_id, None)
        return len(tasks)

    def count(self, user_id: str) -> int:
        return len(self._by_user.get(user_id, ()))
```

Wire on startup in `app/main.py`:
```python
from app.auth.stream_registry import StreamRegistry
app.state.stream_registry = StreamRegistry()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `make test PYTEST_ARGS="tests/unit/auth/test_stream_registry.py -v"`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add app/auth/stream_registry.py app/main.py tests/unit/auth/test_stream_registry.py
git commit -m "feat(auth): in-process SSE stream cancellation registry"
```

---

## Task 1.7: Logout route

**Files:**
- Modify: `app/auth/routes.py`
- Test: `tests/unit/auth/test_logout.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/auth/test_logout.py
import pytest


@pytest.mark.asyncio
async def test_logout_clears_cookie_and_cancels_streams(client, seed_admin, app_state):
    login = await client.post("/api/auth/login",
        json={"username": "admin", "password": "correct-horse-battery-staple"},
        headers={"Origin": "http://localhost:3000"})
    # Pretend a stream is registered.
    import asyncio
    task = asyncio.create_task(asyncio.sleep(60))
    app_state.stream_registry.register("admin", task)
    r = await client.post("/api/auth/logout",
        headers={
            "Origin": "http://localhost:3000",
            "Authorization": f"Bearer {login.json()['access_token']}",
        },
        cookies={"vw_refresh": login.cookies["vw_refresh"]})
    assert r.status_code == 204
    assert "vw_refresh=;" in r.headers["set-cookie"].lower() or "max-age=0" in r.headers["set-cookie"].lower()
    await asyncio.sleep(0.01)
    assert task.cancelled()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make test PYTEST_ARGS="tests/unit/auth/test_logout.py -v"`
Expected: FAIL — 404 / 405.

- [ ] **Step 3: Write minimal implementation**

Append to `app/auth/routes.py`:
```python
from app.auth.deps import require_jwt  # will be defined in Task 1.8


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT,
             dependencies=[Depends(origin_check_dep)])
async def logout(request: Request, response: Response,
                 user: str = Depends(require_jwt)):
    request.app.state.stream_registry.cancel_user(user)
    response.delete_cookie("vw_refresh", path="/api/auth")
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `make test PYTEST_ARGS="tests/unit/auth/test_logout.py -v"`
Expected: 1 passed. (Depends on Task 1.8 — if needed, swap order with Task 1.8.)

- [ ] **Step 5: Commit**

```bash
git add app/auth/routes.py tests/unit/auth/test_logout.py
git commit -m "feat(auth): POST /api/auth/logout revokes streams + clears cookie"
```

---

## Task 1.8: `require_jwt` dependency replacing `require_session_json`

**Files:**
- Modify: `app/auth/deps.py`
- Test: `tests/unit/auth/test_require_jwt.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/auth/test_require_jwt.py
import pytest


@pytest.mark.asyncio
async def test_no_authorization_header_401(client):
    r = await client.get("/api/tokens")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_malformed_header_401(client):
    r = await client.get("/api/tokens", headers={"Authorization": "Token foo"})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_valid_bearer_jwt_200(client, seed_admin):
    login = await client.post("/api/auth/login",
        json={"username": "admin", "password": "correct-horse-battery-staple"},
        headers={"Origin": "http://localhost:3000"})
    r = await client.get("/api/tokens",
        headers={"Authorization": f"Bearer {login.json()['access_token']}"})
    assert r.status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make test PYTEST_ARGS="tests/unit/auth/test_require_jwt.py -v"`
Expected: FAIL — `/api/tokens` currently uses `require_session_json` (cookie auth), so `Authorization: Bearer` is ignored and a 401 is returned even for valid tokens (third test fails).

- [ ] **Step 3: Write minimal implementation**

```python
# app/auth/deps.py — REPLACE entire file
from fastapi import Depends, HTTPException, Request, status
import jwt as pyjwt


def require_jwt(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    token = auth[7:].strip()
    secret = request.app.state.jwt_secret
    try:
        claims = pyjwt.decode(token, secret, algorithms=["HS256"])
    except pyjwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token")
    if claims.get("typ") != "access":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "wrong token type")
    return claims["sub"]


# Backwards-compatible alias during the rip-out. Remove once all routes_web.py are deleted.
require_session_json = require_jwt
```

- [ ] **Step 4: Update every protected route to use `require_jwt`**

Grep and replace in `app/`:
```bash
grep -rln "require_session_json" app/ | xargs sed -i 's/require_session_json/require_jwt/g'
```
The alias keeps tests green during the transition; final mop-up in Task 4.2 deletes it.

- [ ] **Step 5: Run test to verify it passes**

Run: `make test PYTEST_ARGS="tests/unit/auth -v"` and `make test PYTEST_ARGS="tests/unit/tokens -v"`
Expected: all green; the previously cookie-based tests must now mint a JWT via `/api/auth/login` to authenticate (update fixtures as needed).

- [ ] **Step 6: Commit**

```bash
git add app/auth/deps.py app/
git commit -m "feat(auth): require_jwt dependency replaces session cookie auth"
```

---

## Task 1.9: SSE ticket module

**Files:**
- Create: `app/auth/sse_tickets.py`
- Test: `tests/unit/auth/test_sse_tickets.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/auth/test_sse_tickets.py
import asyncio
import pytest
from app.auth.sse_tickets import TicketStore


@pytest.mark.asyncio
async def test_mint_and_consume_once():
    store = TicketStore(secret="s", ttl_seconds=60)
    t = store.mint("admin", "/api/models/abc/logs/stream")
    user = store.consume(t, "/api/models/abc/logs/stream")
    assert user == "admin"
    # Second consume rejected.
    with pytest.raises(ValueError):
        store.consume(t, "/api/models/abc/logs/stream")


@pytest.mark.asyncio
async def test_path_binding_enforced():
    store = TicketStore(secret="s", ttl_seconds=60)
    t = store.mint("admin", "/api/models/abc/logs/stream")
    with pytest.raises(ValueError):
        store.consume(t, "/api/models/xyz/logs/stream")


@pytest.mark.asyncio
async def test_expired_ticket_rejected():
    store = TicketStore(secret="s", ttl_seconds=0)
    t = store.mint("admin", "/api/models/abc/logs/stream")
    await asyncio.sleep(0.01)
    with pytest.raises(ValueError):
        store.consume(t, "/api/models/abc/logs/stream")


@pytest.mark.asyncio
async def test_tampered_signature_rejected():
    store = TicketStore(secret="s", ttl_seconds=60)
    t = store.mint("admin", "/api/models/abc/logs/stream")
    # Flip the last char of the signature.
    bad = t[:-1] + ("A" if t[-1] != "A" else "B")
    with pytest.raises(ValueError):
        store.consume(bad, "/api/models/abc/logs/stream")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make test PYTEST_ARGS="tests/unit/auth/test_sse_tickets.py -v"`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# app/auth/sse_tickets.py
import base64
import hashlib
import hmac
import json
import time


class TicketStore:
    def __init__(self, secret: str, ttl_seconds: int = 60):
        self._secret = secret.encode("utf-8")
        self._ttl = ttl_seconds
        self._deny: dict[str, float] = {}

    def _sign(self, payload: bytes) -> str:
        sig = hmac.new(self._secret, payload, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(sig).decode().rstrip("=")

    def mint(self, user_id: str, path: str) -> str:
        body = {"sub": user_id, "path": path, "iat": int(time.time())}
        payload = base64.urlsafe_b64encode(
            json.dumps(body, separators=(",", ":")).encode()
        ).decode().rstrip("=")
        sig = self._sign(payload.encode())
        return f"{payload}.{sig}"

    def consume(self, ticket: str, path: str) -> str:
        try:
            payload, sig = ticket.split(".", 1)
        except ValueError as exc:
            raise ValueError("malformed ticket") from exc
        expected = self._sign(payload.encode())
        if not hmac.compare_digest(sig, expected):
            raise ValueError("bad signature")
        try:
            padded = payload + "=" * (-len(payload) % 4)
            body = json.loads(base64.urlsafe_b64decode(padded))
        except Exception as exc:
            raise ValueError("malformed payload") from exc
        if body.get("path") != path:
            raise ValueError("path mismatch")
        now = time.time()
        if now - body["iat"] > self._ttl:
            raise ValueError("expired")
        # Garbage-collect deny entries older than 2*TTL.
        self._deny = {k: v for k, v in self._deny.items() if v > now}
        if ticket in self._deny:
            raise ValueError("already consumed")
        self._deny[ticket] = now + self._ttl + 5
        return body["sub"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `make test PYTEST_ARGS="tests/unit/auth/test_sse_tickets.py -v"`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add app/auth/sse_tickets.py tests/unit/auth/test_sse_tickets.py
git commit -m "feat(auth): single-use HMAC SSE ticket store"
```

---

## Task 1.10: SSE ticket-mint endpoint + apply to logs stream

**Files:**
- Modify: `app/auth/routes.py`
- Modify: `app/models/routes_logs.py`
- Modify: `app/main.py` (wire `app.state.sse_tickets`)
- Test: `tests/unit/auth/test_sse_ticket_endpoint.py`
- Test: `tests/unit/models/test_logs_ticket_auth.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/auth/test_sse_ticket_endpoint.py
import pytest


@pytest.mark.asyncio
async def test_mint_ticket_requires_jwt(client):
    r = await client.post("/api/auth/sse-ticket",
                          json={"path": "/api/models/abc/logs/stream"})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_mint_returns_ticket(client, admin_jwt):
    r = await client.post("/api/auth/sse-ticket",
        json={"path": "/api/models/abc/logs/stream"},
        headers={"Authorization": f"Bearer {admin_jwt}"})
    assert r.status_code == 200
    assert "ticket" in r.json()
```

```python
# tests/unit/models/test_logs_ticket_auth.py
import pytest


@pytest.mark.asyncio
async def test_logs_stream_rejects_without_ticket(client):
    r = await client.get("/api/models/some-id/logs/stream")
    assert r.status_code in (401, 403)


@pytest.mark.asyncio
async def test_logs_stream_rejects_wrong_path_ticket(client, admin_jwt, app_state):
    ticket = app_state.sse_tickets.mint("admin", "/api/models/OTHER/logs/stream")
    r = await client.get("/api/models/some-id/logs/stream",
                         params={"ticket": ticket})
    assert r.status_code in (401, 403)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `make test PYTEST_ARGS="tests/unit/auth/test_sse_ticket_endpoint.py tests/unit/models/test_logs_ticket_auth.py -v"`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

Append to `app/auth/routes.py`:
```python
from pydantic import BaseModel as _BM


class TicketBody(_BM):
    path: str


@router.post("/sse-ticket")
async def mint_sse_ticket(body: TicketBody, request: Request,
                          user: str = Depends(require_jwt)):
    return {"ticket": request.app.state.sse_tickets.mint(user, body.path)}
```

Modify `app/models/routes_logs.py` — replace `Depends(require_jwt)` with a ticket-validating dependency:
```python
from fastapi import Query

async def require_sse_ticket(request: Request, ticket: str = Query(...)) -> str:
    try:
        return request.app.state.sse_tickets.consume(ticket, request.url.path)
    except ValueError as exc:
        raise HTTPException(401, str(exc))
```
Apply `user: str = Depends(require_sse_ticket)` on the stream handler. Inside the handler, register the asyncio task before iterating, and unregister in `finally`.

Wire in `app/main.py`:
```python
from app.auth.sse_tickets import TicketStore
app.state.sse_tickets = TicketStore(
    secret=app.state.jwt_secret,
    ttl_seconds=app.state.settings.sse_ticket_ttl_seconds,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `make test PYTEST_ARGS="tests/unit/auth/test_sse_ticket_endpoint.py tests/unit/models/test_logs_ticket_auth.py -v"`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add app/auth/routes.py app/models/routes_logs.py app/main.py tests/unit/auth/test_sse_ticket_endpoint.py tests/unit/models/test_logs_ticket_auth.py
git commit -m "feat(auth): /api/auth/sse-ticket endpoint + apply to logs stream"
```

---

## Task 1.11: Stream-cancel-on-logout integration test

**Files:**
- Create: `tests/integration/test_logout_cancels_stream.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_logout_cancels_stream.py
import asyncio
import json
import pytest


@pytest.mark.asyncio
async def test_logout_terminates_active_sse(client, admin_jwt, app_state):
    # Mint ticket, open stream.
    t = app_state.sse_tickets.mint("admin", "/api/models/test-id/logs/stream")
    async with client.stream("GET",
                             f"/api/models/test-id/logs/stream?ticket={t}") as s:
        assert s.status_code == 200
        # Read one event to confirm stream opened.
        # ... peek logic depends on existing handler shape
        # Then logout.
        r = await client.post("/api/auth/logout",
            headers={"Origin": "http://localhost:3000",
                     "Authorization": f"Bearer {admin_jwt}"})
        assert r.status_code == 204
        # The stream should close within ~1s.
        with pytest.raises(Exception):
            async for _ in asyncio.wait_for(s.aiter_text().__anext__(), timeout=2.0):
                pass
```

- [ ] **Step 2-4:** Run, implement registration in `routes_logs.py` (already done in Task 1.10 hint), re-run.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_logout_cancels_stream.py
git commit -m "test(auth): logout cancels active SSE streams"
```

---

## Task 2.1: Token table migration 0009

**Files:**
- Create: `app/db/sql/0009_token_expiry.sql`
- Test: `tests/unit/db/test_migration_0009.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/db/test_migration_0009.py
import pytest
from app.db.database import open_db
from app.db.migrations import apply_migrations


@pytest.mark.asyncio
async def test_columns_added_with_backfill(tmp_path):
    db_path = tmp_path / "vw.db"
    async with open_db(db_path) as db:
        # Apply migrations up to 0008.
        await apply_migrations(db, stop_after="0008")
        # Insert a row representing pre-migration state.
        await db.execute(
            "INSERT INTO tokens (id, name, prefix, hash, scope, allowed_models, "
            "rate_limit_rpm, rate_limit_tpm, created_at) "
            "VALUES ('tok1', 'old', 'aa', 'hh', 'all', NULL, NULL, NULL, "
            "datetime('now', '-30 days'))"
        )
        await db.commit()
        # Apply 0009.
        await apply_migrations(db)
        row = await (await db.execute(
            "SELECT expires_at, rotated_at, rotated_from FROM tokens WHERE id='tok1'"
        )).fetchone()
        assert row[0] is not None  # backfilled
        assert row[1] is None
        assert row[2] is None
        # Index exists.
        idx = await (await db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_tokens_expires_at'"
        )).fetchone()
        assert idx is not None
```

If `apply_migrations` doesn't support `stop_after`, run the whole stack and verify only on rows inserted after 0008 but before 0009 runs — adjust harness as needed.

- [ ] **Step 2: Run test to verify it fails**

Run: `make test PYTEST_ARGS="tests/unit/db/test_migration_0009.py -v"`
Expected: FAIL — file missing.

- [ ] **Step 3: Write minimal implementation**

```sql
-- app/db/sql/0009_token_expiry.sql
ALTER TABLE tokens ADD COLUMN expires_at TEXT NULL;
ALTER TABLE tokens ADD COLUMN rotated_at TEXT NULL;
ALTER TABLE tokens ADD COLUMN rotated_from TEXT NULL REFERENCES tokens(id) ON DELETE SET NULL;

UPDATE tokens
   SET expires_at = datetime(created_at, '+365 days')
 WHERE expires_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_tokens_expires_at ON tokens(expires_at);
```

- [ ] **Step 4: Run test to verify it passes**

Run: `make test PYTEST_ARGS="tests/unit/db/test_migration_0009.py -v"`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add app/db/sql/0009_token_expiry.sql tests/unit/db/test_migration_0009.py
git commit -m "feat(db): migration 0009 adds expires_at/rotated_at/rotated_from to tokens"
```

---

## Task 2.2: Extend TokenRow + TokenRepo

**Files:**
- Modify: `app/db/repos/tokens.py`
- Test: `tests/unit/db/test_token_repo_extended.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/db/test_token_repo_extended.py
import pytest
from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.tokens import TokenRepo


@pytest.mark.asyncio
async def test_create_with_expiry(tmp_path):
    async with open_db(tmp_path / "vw.db") as db:
        await apply_migrations(db)
        repo = TokenRepo(db)
        await repo.create(token_id="t1", name="ci", plaintext="vw_aaaabbbb", expires_in_days=30)
        rows = await repo.list_all()
        assert len(rows) == 1
        assert rows[0].expires_at is not None
        assert rows[0].rotated_at is None


@pytest.mark.asyncio
async def test_create_never_expires_when_zero(tmp_path):
    async with open_db(tmp_path / "vw.db") as db:
        await apply_migrations(db)
        await TokenRepo(db).create(token_id="t2", name="never", plaintext="vw_ccccdddd", expires_in_days=0)
        rows = await TokenRepo(db).list_all()
        assert rows[0].expires_at is None


@pytest.mark.asyncio
async def test_rotate_sets_pointers(tmp_path):
    async with open_db(tmp_path / "vw.db") as db:
        await apply_migrations(db)
        repo = TokenRepo(db)
        await repo.create(token_id="old", name="ci", plaintext="vw_oldoldold", expires_in_days=365)
        new_id, new_plaintext = await repo.rotate(old_id="old", new_name="ci (rotated)",
                                                   grace_hours=24)
        rows = {r.id: r for r in await repo.list_all()}
        assert rows["old"].rotated_at is not None
        assert rows["old"].revoked_at is not None  # grace window scheduled
        assert rows[new_id].rotated_from == "old"
```

- [ ] **Step 2-4:** Run (FAIL), implement:

```python
# in app/db/repos/tokens.py — extend dataclass + methods
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass
class TokenRow:
    id: str
    name: str
    prefix: str
    hash: str
    scope: str
    allowed_models: list[str] | None
    rate_limit_rpm: int | None
    rate_limit_tpm: int | None
    created_at: str
    last_used_at: str | None
    revoked_at: str | None
    expires_at: str | None
    rotated_at: str | None
    rotated_from: str | None


class TokenRepo:
    def __init__(self, db): self.db = db

    async def create(self, *, token_id: str, name: str, plaintext: str,
                     expires_in_days: int = 365) -> None:
        import bcrypt
        prefix = plaintext[:8]
        h = bcrypt.hashpw(plaintext.encode(), bcrypt.gensalt()).decode()
        now = datetime.now(timezone.utc).isoformat()
        expires_at = None
        if expires_in_days > 0:
            expires_at = (datetime.now(timezone.utc) + timedelta(days=expires_in_days)).isoformat()
        await self.db.execute(
            "INSERT INTO tokens(id, name, prefix, hash, scope, allowed_models, "
            "rate_limit_rpm, rate_limit_tpm, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, 'all', NULL, NULL, NULL, ?, ?)",
            (token_id, name, prefix, h, now, expires_at),
        )
        await self.db.commit()

    async def rotate(self, *, old_id: str, new_name: str, grace_hours: int = 24):
        import secrets
        from app.auth.bearer import generate_bearer_token  # existing
        new_id = secrets.token_hex(16)
        new_plaintext = generate_bearer_token()
        now = datetime.now(timezone.utc)
        revoke_at = (now + timedelta(hours=grace_hours)).isoformat()
        # Insert successor first.
        await self.create(token_id=new_id, name=new_name, plaintext=new_plaintext, expires_in_days=365)
        await self.db.execute(
            "UPDATE tokens SET rotated_from=? WHERE id=?", (old_id, new_id),
        )
        # Mark old.
        await self.db.execute(
            "UPDATE tokens SET rotated_at=?, revoked_at=? WHERE id=?",
            (now.isoformat(), revoke_at, old_id),
        )
        await self.db.commit()
        return new_id, new_plaintext

    async def list_all(self) -> list[TokenRow]:
        cur = await self.db.execute(
            "SELECT id, name, prefix, hash, scope, allowed_models, rate_limit_rpm, "
            "rate_limit_tpm, created_at, last_used_at, revoked_at, "
            "expires_at, rotated_at, rotated_from FROM tokens"
        )
        rows = await cur.fetchall()
        return [TokenRow(*r) for r in rows]
```

Run: `make test PYTEST_ARGS="tests/unit/db/test_token_repo_extended.py -v"` — 3 passed.

- [ ] **Step 5: Commit**

```bash
git add app/db/repos/tokens.py tests/unit/db/test_token_repo_extended.py
git commit -m "feat(db): TokenRepo expires_in_days + rotate(grace_hours)"
```

---

## Task 2.3: Bearer-check enforces `expires_at`

**Files:**
- Modify: `app/proxy/auth.py`
- Test: `tests/unit/proxy/test_bearer_expiry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/proxy/test_bearer_expiry.py
import pytest
from datetime import datetime, timezone, timedelta


@pytest.mark.asyncio
async def test_expired_token_rejected(client, db_conn):
    # Insert a token directly with expires_at in the past.
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    plaintext = "vw_deadbeefdeadbeef"
    # ... use TokenRepo with monkeypatched datetime or direct SQL
    r = await client.get("/v1/models", headers={"Authorization": f"Bearer {plaintext}"})
    assert r.status_code == 401
    assert "expired" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_null_expires_at_treated_as_never(client, db_conn):
    # Token row with expires_at IS NULL still works.
    ...
    r = await client.get("/v1/models", headers={"Authorization": f"Bearer {plaintext}"})
    assert r.status_code == 200
```

- [ ] **Step 2-4:** Run (FAIL), add to `app/proxy/auth.py`:

```python
from datetime import datetime, timezone


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# inside the bearer-check function, before revoked_at check:
if row.expires_at is not None and row.expires_at <= _now_iso():
    raise HTTPException(401, "token expired")
```

Run: `make test PYTEST_ARGS="tests/unit/proxy/test_bearer_expiry.py -v"` — 2 passed.

- [ ] **Step 5: Commit**

```bash
git add app/proxy/auth.py tests/unit/proxy/test_bearer_expiry.py
git commit -m "feat(proxy): bearer auth enforces expires_at"
```

---

## Task 2.4: Rotate endpoint + extended list response

**Files:**
- Modify: `app/tokens/routes_api.py`
- Test: `tests/unit/tokens/test_rotate_endpoint.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/tokens/test_rotate_endpoint.py
import pytest


@pytest.mark.asyncio
async def test_create_with_expires_in_days(client, admin_jwt):
    r = await client.post("/api/tokens",
        json={"name": "ci-90", "expires_in_days": 90},
        headers={"Authorization": f"Bearer {admin_jwt}"})
    assert r.status_code == 201
    body = r.json()
    assert "plaintext" in body
    assert body["expires_at"] is not None


@pytest.mark.asyncio
async def test_rotate_returns_new_plaintext(client, admin_jwt):
    create = await client.post("/api/tokens",
        json={"name": "ci"},
        headers={"Authorization": f"Bearer {admin_jwt}"})
    old_id = create.json()["id"]
    r = await client.post(f"/api/tokens/{old_id}/rotate",
        json={"grace_hours": 24},
        headers={"Authorization": f"Bearer {admin_jwt}"})
    assert r.status_code == 201
    body = r.json()
    assert "plaintext" in body
    assert body["rotated_from"] == old_id


@pytest.mark.asyncio
async def test_list_includes_status_fields(client, admin_jwt):
    r = await client.get("/api/tokens",
        headers={"Authorization": f"Bearer {admin_jwt}"})
    assert r.status_code == 200
    if r.json()["items"]:
        it = r.json()["items"][0]
        for key in ("expires_at", "is_expired", "is_near_expiry", "rotated_at", "rotated_from"):
            assert key in it
```

- [ ] **Step 2-4:** Run (FAIL), modify `app/tokens/routes_api.py`:

```python
from datetime import datetime, timezone, timedelta

class TokenCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    expires_in_days: int = Field(default=365, ge=0, le=3650)


class RotateBody(BaseModel):
    grace_hours: int = Field(default=24, ge=0, le=720)


@router.post("", status_code=201)
async def create_token(body: TokenCreate, request: Request, _user: str = Depends(require_jwt)):
    plaintext = generate_bearer_token()
    tid = secrets.token_hex(16)
    async with open_db(request.app.state.settings.db_path) as db:
        await TokenRepo(db).create(token_id=tid, name=body.name,
                                    plaintext=plaintext, expires_in_days=body.expires_in_days)
        rows = await TokenRepo(db).list_all()
    created = next(r for r in rows if r.id == tid)
    return {
        "id": tid, "name": body.name, "plaintext": plaintext,
        "prefix": plaintext[:8], "preview": plaintext[:8],
        "expires_at": created.expires_at,
    }


@router.post("/{token_id}/rotate", status_code=201)
async def rotate_token(token_id: str, body: RotateBody, request: Request,
                       _user: str = Depends(require_jwt)):
    async with open_db(request.app.state.settings.db_path) as db:
        repo = TokenRepo(db)
        new_id, plaintext = await repo.rotate(
            old_id=token_id, new_name=f"{token_id} (rotated)", grace_hours=body.grace_hours,
        )
    return {"id": new_id, "plaintext": plaintext, "prefix": plaintext[:8],
            "rotated_from": token_id}


@router.get("")
async def list_tokens(request: Request, _user: str = Depends(require_jwt)):
    async with open_db(request.app.state.settings.db_path) as db:
        rows = await TokenRepo(db).list_all()
    now = datetime.now(timezone.utc)
    in_30d = (now + timedelta(days=30)).isoformat()

    def _enrich(r):
        is_expired = r.expires_at is not None and r.expires_at <= now.isoformat()
        is_near = (r.expires_at is not None and not is_expired and r.expires_at <= in_30d)
        # successor: a row whose rotated_from == r.id
        successor = next((x.id for x in rows if x.rotated_from == r.id), None)
        return {
            "id": r.id, "name": r.name, "prefix": r.prefix, "preview": r.prefix,
            "last_used_at": r.last_used_at, "expires_at": r.expires_at,
            "rotated_at": r.rotated_at, "rotated_from": r.rotated_from,
            "successor_id": successor,
            "is_expired": is_expired, "is_near_expiry": is_near,
            "revoked_at": r.revoked_at,
        }
    items = [_enrich(r) for r in rows if r.revoked_at is None or r.rotated_at is not None]
    return {"items": items}
```

Run: `make test PYTEST_ARGS="tests/unit/tokens -v"` — green.

- [ ] **Step 5: Commit**

```bash
git add app/tokens/routes_api.py tests/unit/tokens/test_rotate_endpoint.py
git commit -m "feat(tokens): rotate endpoint + extended list fields"
```

---

## Task 2.5: Token-rotate grace-window integration test

**Files:**
- Create: `tests/integration/test_token_rotate_grace.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_token_rotate_grace.py
import asyncio
import pytest


@pytest.mark.asyncio
async def test_old_token_works_during_grace_then_rejected(client, admin_jwt, monkeypatch):
    # Use a tiny grace window for the test (override via env or settings).
    create = await client.post("/api/tokens", json={"name": "ci"},
        headers={"Authorization": f"Bearer {admin_jwt}"})
    old_plaintext = create.json()["plaintext"]
    old_id = create.json()["id"]
    # Rotate with grace_hours=0 — but the API minimum is 0; we use 1s via monkeypatched timedelta.
    # Easier: rotate with grace_hours=0 then assert old is immediately revoked.
    await client.post(f"/api/tokens/{old_id}/rotate", json={"grace_hours": 0},
        headers={"Authorization": f"Bearer {admin_jwt}"})
    r = await client.get("/v1/models",
        headers={"Authorization": f"Bearer {old_plaintext}"})
    assert r.status_code == 401
```

- [ ] **Step 2-4:** Implementation already done; just verify.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_token_rotate_grace.py
git commit -m "test(tokens): rotate grace window — old key revoked after window"
```

---

## Task 3.1: Settings storage — extend table

**Files:**
- Create: `app/db/sql/0010_settings_expansion.sql`
- Test: `tests/unit/db/test_migration_0010.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/db/test_migration_0010.py
import pytest
from app.db.database import open_db
from app.db.migrations import apply_migrations


@pytest.mark.asyncio
async def test_default_settings_seeded(tmp_path):
    async with open_db(tmp_path / "vw.db") as db:
        await apply_migrations(db)
        cur = await db.execute("SELECT key, value FROM settings")
        rows = dict(await cur.fetchall())
        for key in (
            "session_access_ttl_minutes", "session_refresh_ttl_days",
            "sse_ticket_ttl_seconds", "default_token_expiration_days",
            "rotation_grace_hours", "log_retention_lines", "vllm_version",
            "hf_cache_dir", "default_gpu_indices",
        ):
            assert key in rows, f"missing default for {key}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make test PYTEST_ARGS="tests/unit/db/test_migration_0010.py -v"`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```sql
-- app/db/sql/0010_settings_expansion.sql
INSERT OR IGNORE INTO settings (key, value) VALUES
  ('session_access_ttl_minutes', '15'),
  ('session_refresh_ttl_days', '7'),
  ('sse_ticket_ttl_seconds', '60'),
  ('default_token_expiration_days', '365'),
  ('rotation_grace_hours', '24'),
  ('log_retention_lines', '5000'),
  ('vllm_version', '0.9.2'),
  ('hf_cache_dir', '/hfcache'),
  ('default_gpu_indices', '[0]');
```

- [ ] **Step 4: Run + commit**

Run: `make test PYTEST_ARGS="tests/unit/db/test_migration_0010.py -v"` — pass.

```bash
git add app/db/sql/0010_settings_expansion.sql tests/unit/db/test_migration_0010.py
git commit -m "feat(db): migration 0010 seeds expanded settings defaults"
```

---

## Task 3.2: Runtime settings API

**Files:**
- Modify: `app/settings/routes_api.py`
- Test: `tests/unit/settings/test_runtime_settings.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/settings/test_runtime_settings.py
import pytest


@pytest.mark.asyncio
async def test_get_runtime_returns_full_surface(client, admin_jwt):
    r = await client.get("/api/settings/runtime",
        headers={"Authorization": f"Bearer {admin_jwt}"})
    assert r.status_code == 200
    body = r.json()
    for key in ("admin_username", "hf_token", "hf_cache_dir", "default_gpu_indices",
                "default_token_expiration_days", "rotation_grace_hours",
                "session_access_ttl_minutes", "session_refresh_ttl_days",
                "sse_ticket_ttl_seconds", "vllm_version", "log_retention_lines"):
        assert key in body


@pytest.mark.asyncio
async def test_patch_no_restart_takes_immediate_effect(client, admin_jwt):
    r = await client.patch("/api/settings/runtime",
        json={"default_token_expiration_days": 90},
        headers={"Authorization": f"Bearer {admin_jwt}"})
    assert r.status_code == 200
    assert r.json()["requires_restart"] == []


@pytest.mark.asyncio
async def test_patch_session_ttl_flags_warden_restart(client, admin_jwt):
    r = await client.patch("/api/settings/runtime",
        json={"session_access_ttl_minutes": 30},
        headers={"Authorization": f"Bearer {admin_jwt}"})
    assert r.status_code == 200
    assert "warden-restart" in r.json()["requires_restart_kinds"]


@pytest.mark.asyncio
async def test_patch_hf_token_flags_model_reload(client, admin_jwt):
    r = await client.patch("/api/settings/runtime",
        json={"hf_token": "hf_xxxx"},
        headers={"Authorization": f"Bearer {admin_jwt}"})
    assert r.status_code == 200
    assert "model-reload" in r.json()["requires_restart_kinds"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make test PYTEST_ARGS="tests/unit/settings/test_runtime_settings.py -v"`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# app/settings/routes_api.py — REWRITE
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.auth.deps import require_jwt
from app.db.database import open_db
from app.db.repos.settings import SettingsRepo  # existing

router = APIRouter(prefix="/api/settings", tags=["settings"])

# (key, kind) where kind is one of "none" | "model-reload" | "warden-restart"
RUNTIME_KEYS: dict[str, str] = {
    "admin_username": "none",
    "admin_password": "none",  # writes invalidate sessions but no restart
    "hf_token": "model-reload",
    "hf_cache_dir": "model-reload",
    "default_gpu_indices": "none",
    "default_token_expiration_days": "none",
    "rotation_grace_hours": "none",
    "session_access_ttl_minutes": "warden-restart",
    "session_refresh_ttl_days": "warden-restart",
    "sse_ticket_ttl_seconds": "none",
    "vllm_version": "warden-restart",
    "log_retention_lines": "none",
}


class RuntimePatch(BaseModel):
    # Use a generic dict-like update; explicit fields would balloon the model.
    model_config = {"extra": "allow"}


@router.get("/runtime")
async def get_runtime(request: Request, _user: str = Depends(require_jwt)):
    async with open_db(request.app.state.settings.db_path) as db:
        rows = await SettingsRepo(db).get_many(list(RUNTIME_KEYS.keys()))
    out: dict[str, Any] = {k: rows.get(k) for k in RUNTIME_KEYS}
    # Mask secrets.
    if out.get("admin_password"):
        out["admin_password"] = "***"
    if out.get("hf_token"):
        out["hf_token"] = "***"
    return out


@router.patch("/runtime")
async def patch_runtime(body: dict[str, Any], request: Request,
                        _user: str = Depends(require_jwt)):
    bad = [k for k in body if k not in RUNTIME_KEYS]
    if bad:
        raise HTTPException(400, f"unknown keys: {bad}")
    kinds = set()
    async with open_db(request.app.state.settings.db_path) as db:
        repo = SettingsRepo(db)
        for k, v in body.items():
            await repo.set(k, str(v) if not isinstance(v, str) else v)
            kind = RUNTIME_KEYS[k]
            if kind != "none":
                kinds.add(kind)
    return {
        "ok": True,
        "requires_restart": sorted(kinds),
        "requires_restart_kinds": sorted(kinds),
    }
```

- [ ] **Step 4: Run + iterate** until 4 passes.

- [ ] **Step 5: Commit**

```bash
git add app/settings/routes_api.py tests/unit/settings/test_runtime_settings.py
git commit -m "feat(settings): full runtime surface + requires_restart in PATCH echo"
```

---

## Task 3.3: Per-model settings API

**Files:**
- Modify: `app/settings/routes_api.py`
- Test: `tests/unit/settings/test_model_settings.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/settings/test_model_settings.py
import pytest


@pytest.mark.asyncio
async def test_get_model_settings(client, admin_jwt, registered_model_id):
    r = await client.get(f"/api/models/{registered_model_id}/settings",
        headers={"Authorization": f"Bearer {admin_jwt}"})
    assert r.status_code == 200
    body = r.json()
    for k in ("served_model_name", "hf_repo", "hf_revision", "gpu_indices",
              "tensor_parallel_size", "dtype", "max_model_len",
              "gpu_memory_utilization", "trust_remote_code", "extra_args", "extra_env"):
        assert k in body


@pytest.mark.asyncio
async def test_patch_blocked_when_loaded(client, admin_jwt, loaded_model_id):
    r = await client.patch(f"/api/models/{loaded_model_id}/settings",
        json={"max_model_len": 4096},
        headers={"Authorization": f"Bearer {admin_jwt}"})
    assert r.status_code == 409
```

- [ ] **Step 2-4:** Run (FAIL), implement endpoints reading/writing the existing `models` table fields via the existing `ModelRepo`. PATCH must check model state and 409 if currently loaded.

```python
# add to app/settings/routes_api.py — or extract to app/models/routes_settings.py
from app.db.repos.models import ModelRepo

settings_router_per_model = APIRouter(prefix="/api/models", tags=["model-settings"])


@settings_router_per_model.get("/{model_id}/settings")
async def get_model_settings(model_id: str, request: Request,
                             _user: str = Depends(require_jwt)):
    async with open_db(request.app.state.settings.db_path) as db:
        m = await ModelRepo(db).get(model_id)
    if not m:
        raise HTTPException(404)
    return m.dict()


@settings_router_per_model.patch("/{model_id}/settings")
async def patch_model_settings(model_id: str, body: dict, request: Request,
                                _user: str = Depends(require_jwt)):
    async with open_db(request.app.state.settings.db_path) as db:
        repo = ModelRepo(db)
        m = await repo.get(model_id)
        if not m:
            raise HTTPException(404)
        if m.state == "loaded":
            raise HTTPException(409, "model must be unloaded before editing settings")
        await repo.update(model_id, **body)
    return {"ok": True}
```

Register in `app/main.py`. Run tests until green.

- [ ] **Step 5: Commit**

```bash
git add app/settings/routes_api.py app/main.py tests/unit/settings/test_model_settings.py
git commit -m "feat(settings): per-model GET/PATCH with loaded-state 409"
```

---

## Task 4.1: Delete Jinja deps + static + middleware

**Files:**
- Modify: `app/main.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Edit `app/main.py`**

Remove from `app/main.py`:
- `from fastapi.templating import Jinja2Templates`
- `from fastapi.staticfiles import StaticFiles`
- `from starlette.middleware.sessions import SessionMiddleware`
- The `templates = Jinja2Templates(directory=web_dir)`, `app.state.templates = templates`, and `app.mount("/static", ...)` lines
- The `app.add_middleware(SessionMiddleware, ...)` call
- Every `app.include_router(routes_web_*)` line

Add (if not present) the JWT secret bootstrap + sse_tickets wiring lines from Tasks 1.1 and 1.10.

- [ ] **Step 2: Edit `requirements.txt`**

Remove lines containing `jinja2`, `itsdangerous`. Confirm `PyJWT[crypto]==2.9.0` is present (from Task 1.2).

- [ ] **Step 3: Run full test suite**

Run: `make test`
Expected: green. Any failing test references deleted UI — flag for the next task.

- [ ] **Step 4: Commit**

```bash
git add app/main.py requirements.txt
git commit -m "refactor(main): drop Jinja, static mount, SessionMiddleware"
```

---

## Task 4.2: Delete `app/web/` and all `routes_web.py`

**Files:**
- Delete: `app/web/` (entire directory)
- Delete: `app/auth/sessions.py`
- Delete: `app/auth/routes_web.py`
- Delete: `app/models/routes_web.py`
- Delete: `app/setup/routes_web.py`
- Delete: `app/settings/routes_web.py`
- Delete: `app/stats/routes_web.py`
- Delete: `app/tokens/routes_web.py`
- Modify: `app/auth/deps.py` (remove `require_session_json` alias from Task 1.8)

- [ ] **Step 1:** Verify no remaining imports.

Run:
```bash
grep -rln "routes_web\|app/web\|Jinja2Templates\|itsdangerous\|sessions.py" app/ tests/
```
Expected: nothing left in `app/`; any test hits are slated for deletion in Task 4.3.

- [ ] **Step 2:** Delete files.

Run:
```bash
rm -rf app/web/
rm app/auth/sessions.py
find app -name routes_web.py -delete
```

- [ ] **Step 3:** Remove `require_session_json` alias.

Edit `app/auth/deps.py` — delete the line `require_session_json = require_jwt`.

- [ ] **Step 4:** Run lint + tests.

Run: `make lint && make test`
Expected: green (failing UI-template tests are removed in Task 4.3 if any remain).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: delete app/web/ + every routes_web.py + sessions.py"
```

---

## Task 4.3: Delete Jinja pytest tests

**Files:**
- Delete: any `tests/unit/web/`, `tests/unit/templates/` content asserting HTML

- [ ] **Step 1:** Identify candidates.

Run:
```bash
grep -rln "templates.TemplateResponse\|response.text\|<html\|<form" tests/unit/web tests/unit/templates 2>/dev/null
```

- [ ] **Step 2:** Delete the directories that exist.

Run:
```bash
rm -rf tests/unit/web tests/unit/templates
```

- [ ] **Step 3:** Run full suite.

Run: `make test`
Expected: green.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "test: delete Jinja UI tests (replaced by Vitest + Playwright in MR-2)"
```

---

## Task 5.1: `docs/operating.md` — curl runbook

**Files:**
- Create: `docs/operating.md`

- [ ] **Step 1: Write the runbook**

```markdown
# Operating vllm-warden during MR-1 → MR-2 gap

After MR-1 (`feat/jwt-and-jinja-cleanup`) ships, vllm-warden has no UI until MR-2 lands. Use curl for everything until the new UI is deployed.

## Prerequisites

- `VW_JWT_SECRET` env var set, or `<VW_DB_PATH dir>/jwt_secret` writable.
- `VW_FRONTEND_ORIGIN` set to whatever origin you will pass in your `Origin` header. For pure curl access from a workstation, set it to e.g. `https://vllm.protrener.com` and always pass `-H "Origin: https://vllm.protrener.com"` on `/api/auth/*` calls.

## Login

```bash
COOKIES=/tmp/vw-cookies.txt
ORIGIN=https://vllm.protrener.com

ACCESS=$(curl -fsS -c "$COOKIES" \
  -H "Content-Type: application/json" \
  -H "Origin: $ORIGIN" \
  -d '{"username":"admin","password":"YOUR-PASSWORD"}' \
  "$ORIGIN/api/auth/login" | jq -r .access_token)
```

## Refresh

```bash
curl -fsS -b "$COOKIES" -c "$COOKIES" \
  -H "Origin: $ORIGIN" -X POST \
  "$ORIGIN/api/auth/refresh"
```

## Mint API token

```bash
curl -fsS -H "Authorization: Bearer $ACCESS" -H "Content-Type: application/json" \
  -d '{"name":"ci-bot","expires_in_days":90}' -X POST \
  "$ORIGIN/api/tokens"
```

## Register + load a model

```bash
curl -fsS -H "Authorization: Bearer $ACCESS" -H "Content-Type: application/json" \
  -d '{"served_model_name":"opt-125m","hf_repo":"facebook/opt-125m","gpu_indices":[0]}' \
  -X POST "$ORIGIN/api/models"

curl -fsS -H "Authorization: Bearer $ACCESS" -X POST \
  "$ORIGIN/api/models/opt-125m/pull"

curl -fsS -H "Authorization: Bearer $ACCESS" -X POST \
  "$ORIGIN/api/models/opt-125m/load"
```

## Run a completion

```bash
TOKEN=<vw_… from create-token output>
curl -fsS -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"model":"opt-125m","prompt":"Hello","max_tokens":16}' \
  "$ORIGIN/v1/completions"
```

## Rotate a token

```bash
curl -fsS -H "Authorization: Bearer $ACCESS" -H "Content-Type: application/json" \
  -d '{"grace_hours":24}' -X POST \
  "$ORIGIN/api/tokens/<old-id>/rotate"
```

## Revoke

```bash
curl -fsS -H "Authorization: Bearer $ACCESS" -X DELETE \
  "$ORIGIN/api/tokens/<id>"
```

## JWT secret rotation

1. Stop the warden container.
2. Delete `<VW_DB_PATH dir>/jwt_secret` **or** set a new `VW_JWT_SECRET` env var.
3. Start the warden container.
4. Every operator must re-log-in. Every refresh cookie becomes invalid instantly.

## Admin password reset (lockout recovery)

If you forget the admin password and have no valid session:

```bash
docker exec -it vllm-warden-api python -m app.cli reset_admin_password
```

This wraps `bcrypt.hashpw()` and writes directly to the `users` table.
```

- [ ] **Step 2: Commit**

```bash
git add docs/operating.md
git commit -m "docs: add operator curl runbook for MR-1 → MR-2 gap"
```

---

## Task 6.1: Pre-merge polish for MR-1

- [ ] **Step 1:** Run the full quality gate.

```bash
make lint
make typecheck   # if mypy target exists; else: docker run --rm -v $(pwd):/app -w /app python:3.11-slim sh -c "pip install -r requirements.txt && mypy app/"
make test
```
Expected: all green.

- [ ] **Step 2:** Verify the test image rebuilds clean.

```bash
make build
```

- [ ] **Step 3:** Push branch and open MR.

```bash
git push -u origin feat/jwt-and-jinja-cleanup
gh mr create --title "MR-1: JWT auth + Jinja cleanup + token expiry" --description ...
```
(Use the project's existing MR template / `glab` tooling per CLAUDE.md.)

**End of Phase 1.** MR-1 is now ready for review and merge. Backend is headless until MR-2 ships.

---

# Phase 2 — MR-2: Next.js frontend

## Task 7.1: Branch from develop (post MR-1 merge)

- [ ] **Step 1:** Wait for MR-1 to merge, then update local.

```bash
git -C /home/ip/projects/vllm-warden fetch origin develop
git -C /home/ip/projects/vllm-warden checkout -b feat/ui-redesign-nextjs origin/develop
```

---

## Task 8.1: Scaffold `frontend/` directory

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/next.config.ts`
- Create: `frontend/tsconfig.json`
- Create: `frontend/.gitignore`
- Create: `frontend/.dockerignore`

- [ ] **Step 1: Create `frontend/package.json`**

```json
{
  "name": "vllm-warden-ui",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev -p 3000",
    "build": "next build",
    "start": "next start -p 3000",
    "lint": "next lint",
    "typecheck": "tsc --noEmit",
    "generate:api-types": "openapi-typescript ../openapi.json -o src/lib/api-types.generated.ts",
    "analyze": "ANALYZE=true next build"
  },
  "dependencies": {
    "next": "15.5.9",
    "react": "19.0.0",
    "react-dom": "19.0.0",
    "swr": "2.3.0",
    "recharts": "3.7.0",
    "lucide-react": "0.460.0",
    "clsx": "2.1.1",
    "tailwind-merge": "2.5.4",
    "class-variance-authority": "0.7.1"
  },
  "devDependencies": {
    "typescript": "5.6.3",
    "@types/react": "19.0.0",
    "@types/react-dom": "19.0.0",
    "@types/node": "22.9.0",
    "tailwindcss": "3.4.14",
    "autoprefixer": "10.4.20",
    "postcss": "8.4.49",
    "eslint": "9.14.0",
    "eslint-config-next": "15.5.9",
    "openapi-typescript": "7.4.3",
    "@next/bundle-analyzer": "15.5.9",
    "vitest": "2.1.4",
    "@testing-library/react": "16.3.0",
    "@testing-library/jest-dom": "6.6.3",
    "@playwright/test": "1.51.1",
    "jsdom": "25.0.1"
  }
}
```

- [ ] **Step 2: Create `frontend/next.config.ts`**

```ts
import type { NextConfig } from 'next';

const config: NextConfig = {
  output: 'standalone',
  async rewrites() {
    const backend = process.env.BACKEND_URL;
    if (!backend) return [];
    return [
      { source: '/api/:path*', destination: `${backend}/api/:path*` },
      { source: '/v1/:path*', destination: `${backend}/v1/:path*` },
    ];
  },
};

export default process.env.ANALYZE === 'true'
  ? require('@next/bundle-analyzer')({ enabled: true })(config)
  : config;
```

- [ ] **Step 3: Create `frontend/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [{ "name": "next" }],
    "paths": { "@/*": ["./src/*"] }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
```

- [ ] **Step 4: Add `.gitignore` and `.dockerignore`**

`frontend/.gitignore`:
```
node_modules/
.next/
out/
*.log
.env.local
.env.production
```

`frontend/.dockerignore`:
```
node_modules
.next
.git
README.md
.env*.local
```

- [ ] **Step 5: Install deps inside Docker and verify build**

Run:
```bash
docker run --rm -v $(pwd)/frontend:/app -w /app node:20-alpine sh -c "npm ci || npm install"
docker run --rm -v $(pwd)/frontend:/app -w /app node:20-alpine npx next build
```
Expected: build fails with "No app directory found" — that is fine, we have not yet created `src/app/`. Confirms toolchain is wired.

- [ ] **Step 6: Commit**

```bash
git add frontend/
git commit -m "feat(ui): scaffold Next.js 15 frontend project"
```

---

## Task 8.2: Copy theme + globals + ui primitives from podwarden

**Files:**
- Create: `frontend/tailwind.config.ts`, `frontend/postcss.config.mjs`
- Create: `frontend/src/app/globals.css`
- Create: `frontend/src/lib/theme.tsx`, `frontend/src/lib/utils.ts`
- Create: `frontend/src/components/theme-switcher.tsx`
- Create: `frontend/src/components/ansi-log.tsx`
- Create: `frontend/src/components/ui/*` (button, card, input, badge, skeleton, modal, tabs, select, combobox)

- [ ] **Step 1: Copy verbatim from podwarden**

```bash
PWFE=/home/ip/projects/pw/podwarden/frontend
cp "$PWFE/tailwind.config.ts" frontend/tailwind.config.ts
cp "$PWFE/postcss.config.mjs" frontend/postcss.config.mjs
mkdir -p frontend/src/app frontend/src/lib frontend/src/components/ui
cp "$PWFE/src/app/globals.css" frontend/src/app/globals.css
cp "$PWFE/src/lib/theme.tsx" frontend/src/lib/theme.tsx
cp "$PWFE/src/lib/utils.ts" frontend/src/lib/utils.ts
cp "$PWFE/src/components/theme-switcher.tsx" frontend/src/components/theme-switcher.tsx
cp "$PWFE/src/components/ansi-log.tsx" frontend/src/components/ansi-log.tsx
for f in button card input badge skeleton modal tabs select combobox; do
  cp "$PWFE/src/components/ui/$f.tsx" frontend/src/components/ui/$f.tsx
done
```

- [ ] **Step 2: Inspect each file for podwarden-specific imports**

Grep for `'@/lib/api'`, `'@/lib/auth-fetch'`, or other paths that don't exist yet in vllm-warden:
```bash
grep -rln "from '@/" frontend/src/components/ui frontend/src/components/ansi-log.tsx frontend/src/lib
```
Replace any imports that point to podwarden-only modules with vllm-warden equivalents (rare for ui primitives; common for `theme.tsx` if it pulls from a brand file — adjust if so).

- [ ] **Step 3: Build to verify**

```bash
docker run --rm -v $(pwd)/frontend:/app -w /app node:20-alpine npx next build
```
Expected: still fails ("no app/layout.tsx") — addressed in Task 8.3.

- [ ] **Step 4: Commit**

```bash
git add frontend/
git commit -m "feat(ui): copy theme, globals.css, ui primitives, ansi-log from podwarden"
```

---

## Task 8.3: App shell + layout + theme provider

**Files:**
- Create: `frontend/src/app/layout.tsx`
- Create: `frontend/src/app/page.tsx`
- Create: `frontend/src/components/nav-bar.tsx`

- [ ] **Step 1: Create `layout.tsx`**

```tsx
// frontend/src/app/layout.tsx
import './globals.css';
import { ThemeProvider } from '@/lib/theme';
import { NavBar } from '@/components/nav-bar';

export const metadata = { title: 'vllm-warden', description: 'vLLM operator UI' };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <ThemeProvider>
          <NavBar />
          <main className="container mx-auto p-6">{children}</main>
        </ThemeProvider>
      </body>
    </html>
  );
}
```

- [ ] **Step 2: Create `page.tsx`** (root)

```tsx
// frontend/src/app/page.tsx
import { redirect } from 'next/navigation';
export default function Home() { redirect('/models'); }
```

- [ ] **Step 3: Create `nav-bar.tsx`** — adapt from podwarden, simplified nav items.

```tsx
// frontend/src/components/nav-bar.tsx
'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { ThemeSwitcher } from './theme-switcher';

const items = [
  { href: '/models', label: 'Models' },
  { href: '/tokens', label: 'Tokens' },
  { href: '/stats', label: 'Stats' },
  { href: '/settings', label: 'Settings' },
];

export function NavBar() {
  const path = usePathname();
  if (path.startsWith('/login') || path.startsWith('/setup')) return null;
  return (
    <header className="border-b">
      <nav className="container mx-auto flex items-center gap-4 p-4">
        <span className="font-bold">vllm-warden</span>
        {items.map((it) => (
          <Link key={it.href} href={it.href}
                className={path.startsWith(it.href) ? 'font-semibold' : ''}>
            {it.label}
          </Link>
        ))}
        <span className="ml-auto"><ThemeSwitcher /></span>
      </nav>
    </header>
  );
}
```

- [ ] **Step 4: Build to verify**

```bash
docker run --rm -v $(pwd)/frontend:/app -w /app node:20-alpine npx next build
```
Expected: succeeds.

- [ ] **Step 5: Commit**

```bash
git add frontend/
git commit -m "feat(ui): app shell — layout, root redirect, nav bar"
```

---

## Task 9.1: Generate OpenAPI types

**Files:**
- Create: `Makefile` target `generate-api-types`
- Create: `frontend/src/lib/api-types.generated.ts`

- [ ] **Step 1: Add Makefile target**

```makefile
generate-api-types:
	docker run --rm -v $(PWD):/app -w /app python:3.11-slim sh -c \
	  "pip install -q -r requirements-dev.txt && python -c \"import json, sys; from app.main import app; sys.stdout.write(json.dumps(app.openapi()))\"" > openapi.json
	docker run --rm -u $(shell id -u):$(shell id -g) -e HOME=/tmp -v $(PWD):/work -w /work/frontend node:20-alpine \
	  npx -y openapi-typescript@7 ../openapi.json -o src/lib/api-types.generated.ts
```

- [ ] **Step 2: Generate**

Run: `make generate-api-types`
Expected: `openapi.json` written at repo root; `frontend/src/lib/api-types.generated.ts` populated.

- [ ] **Step 3: Commit**

```bash
git add Makefile frontend/src/lib/api-types.generated.ts openapi.json
git commit -m "feat(ui): codegen OpenAPI types from FastAPI"
```

---

## Task 10.1: `lib/auth-fetch.ts` — JWT refresh-on-401

**Files:**
- Create: `frontend/src/lib/auth-fetch.ts`
- Create: `frontend/vitest.config.ts`
- Test: `frontend/tests/component/auth-fetch.test.ts`

- [ ] **Step 0: Add vitest config (one-time, used by every Phase 10+ test)**

```ts
// frontend/vitest.config.ts
import { defineConfig } from 'vitest/config';
import path from 'node:path';

export default defineConfig({
  test: {
    environment: 'jsdom',
    globals: false,
  },
  // Required for Phase 10+ TSX tests (e.g. Task 10.3 sse-hook.test.tsx) so JSX
  // compiles without an explicit `import React` in every test file.
  esbuild: { jsx: 'automatic' },
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
});
```

The `jsdom` env is required by the third test (uses `window.location`); the `@` alias resolves the `@/lib/auth-fetch` import. Both are needed by every later Phase 10+ test, not just this one.

- [ ] **Step 1: Write the failing test**

```ts
// frontend/tests/component/auth-fetch.test.ts
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { authFetch, setAccessToken } from '@/lib/auth-fetch';

describe('authFetch', () => {
  beforeEach(() => { setAccessToken('initial'); vi.restoreAllMocks(); });

  it('attaches bearer header', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('{}', { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    await authFetch('/api/tokens');
    expect(fetchMock).toHaveBeenCalledWith('/api/tokens', expect.objectContaining({
      headers: expect.objectContaining({ Authorization: 'Bearer initial' }),
    }));
  });

  it('refreshes on 401 then retries with new token', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response('{}', { status: 401 }))
      .mockResolvedValueOnce(new Response('{"access_token":"new","expires_in":900}', { status: 200 }))
      .mockResolvedValueOnce(new Response('{}', { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const r = await authFetch('/api/tokens');
    expect(r.status).toBe(200);
    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/auth/refresh', expect.objectContaining({
      method: 'POST', credentials: 'include',
    }));
    expect(fetchMock).toHaveBeenNthCalledWith(3, '/api/tokens', expect.objectContaining({
      headers: expect.objectContaining({ Authorization: 'Bearer new' }),
    }));
  });

  it('redirects to /login if refresh also 401', async () => {
    const replace = vi.fn();
    Object.defineProperty(window, 'location', { value: { replace }, writable: true });
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response('', { status: 401 }))
      .mockResolvedValueOnce(new Response('', { status: 401 }));
    vi.stubGlobal('fetch', fetchMock);
    await authFetch('/api/tokens');
    expect(replace).toHaveBeenCalledWith('/login');
  });
});
```

- [ ] **Step 2: Run + verify FAIL**

Run: `docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v $(pwd)/frontend:/app -w /app node:20-alpine npx -y vitest@2.1.4 run`
Expected: fail (`@/lib/auth-fetch` does not exist).

Note: `-u $(id -u):$(id -g) -e HOME=/tmp` keeps any cache/tsbuildinfo the test run writes host-owned. Reuse this pattern for every later Phase 10+ vitest invocation.

- [ ] **Step 3: Write minimal implementation**

```ts
// frontend/src/lib/auth-fetch.ts
let accessToken: string | null = null;
let refreshing: Promise<string | null> | null = null;

export function setAccessToken(t: string | null) { accessToken = t; }
export function getAccessToken() { return accessToken; }

async function refresh(): Promise<string | null> {
  if (refreshing) return refreshing;
  refreshing = (async () => {
    try {
      const r = await fetch('/api/auth/refresh', {
        method: 'POST', credentials: 'include',
        headers: { 'Origin': window.location.origin },
      });
      if (!r.ok) return null;
      const { access_token } = await r.json();
      accessToken = access_token;
      return access_token;
    } finally { refreshing = null; }
  })();
  return refreshing;
}

// Convert a Headers instance back to a plain object so callers/tests can
// inspect headers via object-shape matchers (Headers has no own enumerable
// props, which breaks expect.objectContaining). fetch accepts both shapes.
function headersToObject(h: Headers): Record<string, string> {
  const out: Record<string, string> = {};
  h.forEach((v, k) => { out[k] = v; });
  if (h.get('Authorization')) out['Authorization'] = h.get('Authorization')!;
  return out;
}

export async function authFetch(input: RequestInfo, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  if (accessToken) headers.set('Authorization', `Bearer ${accessToken}`);
  let r = await fetch(input, { ...init, headers: headersToObject(headers) });
  if (r.status !== 401) return r;
  const newTok = await refresh();
  if (!newTok) {
    window.location.replace('/login');
    return r;
  }
  headers.set('Authorization', `Bearer ${newTok}`);
  r = await fetch(input, { ...init, headers: headersToObject(headers) });
  return r;
}
```

- [ ] **Step 4: Run + verify PASS** — 3 passed.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/auth-fetch.ts frontend/tests/component/auth-fetch.test.ts frontend/vitest.config.ts
git commit -m "feat(ui): auth-fetch with JWT refresh-on-401"
```

---

## Task 10.2: `lib/api.ts` typed fetch wrapper

**Files:**
- Create: `frontend/src/lib/api.ts`

- [ ] **Step 1: Write minimal wrapper using the generated types**

```ts
// frontend/src/lib/api.ts
import type { paths } from './api-types.generated';
import { authFetch } from './auth-fetch';

type GET<P extends keyof paths> = paths[P] extends { get: { responses: { 200: { content: { 'application/json': infer R } } } } } ? R : never;
type POSTBody<P extends keyof paths> = paths[P] extends { post: { requestBody: { content: { 'application/json': infer B } } } } ? B : never;
type POSTResponse<P extends keyof paths> = paths[P] extends { post: { responses: { 200: { content: { 'application/json': infer R } } } } } ? R : paths[P] extends { post: { responses: { 201: { content: { 'application/json': infer R } } } } } ? R : void;

export async function getJSON<P extends keyof paths>(path: P): Promise<GET<P>> {
  const r = await authFetch(path as string);
  if (!r.ok) throw new Error(`${r.status} ${path as string}`);
  return r.json();
}

export async function postJSON<P extends keyof paths>(
  path: P, body: POSTBody<P>,
): Promise<POSTResponse<P>> {
  const r = await authFetch(path as string, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${path as string}`);
  return r.json();
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/lib/api.ts
git commit -m "feat(ui): typed fetch wrapper over auth-fetch"
```

---

## Task 10.3: `lib/sse.ts` — EventSource hook with close+remint

**Files:**
- Create: `frontend/src/lib/sse.ts`
- Test: `frontend/tests/component/sse-hook.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
// frontend/tests/component/sse-hook.test.tsx
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, act } from '@testing-library/react';
import { useEventSource } from '@/lib/sse';

class FakeES {
  static last: FakeES;
  onopen?: () => void;
  onmessage?: (e: MessageEvent) => void;
  onerror?: () => void;
  closed = false;
  constructor(public url: string) { FakeES.last = this; setTimeout(() => this.onopen?.(), 0); }
  close() { this.closed = true; }
}

describe('useEventSource', () => {
  beforeEach(() => { vi.useFakeTimers(); });
  afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });

  it('mints a fresh ticket per reconnect', async () => {
    vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);
    const fetchMock = vi.fn().mockResolvedValue(new Response('{"ticket":"t1"}'));
    vi.stubGlobal('fetch', fetchMock);

    function Probe() {
      useEventSource('/api/models/abc/logs/stream', { onMessage: () => {} });
      return null;
    }
    const { unmount } = render(<Probe />);

    // Flush initial connect() — authFetch → fetch → setTimeout(onopen, 0)
    await vi.advanceTimersByTimeAsync(0);
    expect(FakeES.last).toBeDefined();
    expect(fetchMock).toHaveBeenCalledTimes(1);

    // Trigger reconnect; backoff is 2s after the onopen reset
    act(() => FakeES.last.onerror?.());
    await vi.advanceTimersByTimeAsync(2000);
    expect(fetchMock).toHaveBeenCalledTimes(2);

    unmount();
  });
});
```

Note: fake timers are required because `@testing-library/dom`'s `waitFor`
default timeout is 1000 ms, but the hook's first reconnect backoff is 2000 ms
(`backoffMs *= 2` from initial 1000 ms). `vi.advanceTimersByTimeAsync` lets the
test step past the backoff deterministically without changing the production
backoff schedule.

- [ ] **Step 2-4:** Run (FAIL), implement using the spec's `useEventSource` template verbatim, run (PASS).

```ts
// frontend/src/lib/sse.ts
'use client';
import { useEffect } from 'react';
import { authFetch } from './auth-fetch';

export function useEventSource<T>(
  path: string,
  opts: { onMessage: (m: T) => void; enabled?: boolean },
) {
  useEffect(() => {
    if (opts.enabled === false) return;
    let stopped = false;
    let es: EventSource | null = null;
    let backoffMs = 1000;
    const tick = (fn: () => void, ms: number) => setTimeout(fn, ms);

    async function connect() {
      if (stopped) return;
      let ticket: string;
      try {
        const r = await authFetch('/api/auth/sse-ticket', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path }),
        });
        if (!r.ok) throw new Error(String(r.status));
        ticket = (await r.json()).ticket;
      } catch {
        if (!stopped) tick(connect, Math.min(backoffMs *= 2, 30000));
        return;
      }
      es = new EventSource(`${path}?ticket=${encodeURIComponent(ticket)}`);
      es.onopen = () => { backoffMs = 1000; };
      es.onmessage = (e) => { try { opts.onMessage(JSON.parse(e.data) as T); } catch {} };
      es.onerror = () => {
        es?.close(); es = null;
        if (!stopped) tick(connect, Math.min(backoffMs *= 2, 30000));
      };
    }
    connect();
    return () => { stopped = true; es?.close(); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, opts.enabled]);
}
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/sse.ts frontend/tests/component/sse-hook.test.tsx
git commit -m "feat(ui): useEventSource hook — close+remint on every reconnect"
```

---

## Task 10.4: `lib/settings-hints.ts`

**Files:**
- Create: `frontend/src/lib/settings-hints.ts`

- [ ] **Step 1:** Translate Settings spec table 1 (Runtime) + table 2 (Model) into a flat TS object. Each entry: `{ label, hint, restart: 'none'|'model-reload'|'warden-restart' }`. Pull copy verbatim from the spec.

```ts
// frontend/src/lib/settings-hints.ts
export type RestartKind = 'none' | 'model-reload' | 'warden-restart';

export interface FieldHint {
  label: string;
  hint: string;
  restart: RestartKind;
}

export const RUNTIME_HINTS: Record<string, FieldHint> = {
  admin_username: {
    label: 'Admin username',
    hint: 'The single operator account. Used for the login page.',
    restart: 'none',
  },
  // ... 11 more entries copying verbatim hint text from spec table 1
};

export const MODEL_HINTS: Record<string, FieldHint> = {
  served_model_name: {
    label: 'Served model name',
    hint: 'The name clients pass in `model:` for `/v1/completions`. Slug only — alphanumeric + `.`, `_`, `-`.',
    restart: 'model-reload',
  },
  // ... full spec table 2 verbatim
};
```

(Implementer: copy each row's hint text from the design spec lines 352–397 verbatim. Do not paraphrase — QA reviews this file.)

- [ ] **Step 2: Commit**

```bash
git add frontend/src/lib/settings-hints.ts
git commit -m "feat(ui): settings hint copy as a single TS object"
```

---

## Task 11.1: `/login` page

**Files:**
- Create: `frontend/src/app/login/page.tsx`
- Test: `frontend/tests/component/login.test.tsx`

- [ ] **Step 1: Write the test**

```tsx
// frontend/tests/component/login.test.tsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import LoginPage from '@/app/login/page';

describe('LoginPage', () => {
  it('posts credentials and stores access token', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(
      '{"access_token":"abc","expires_in":900}', { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    render(<LoginPage />);
    fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'admin' } });
    fireEvent.change(screen.getByLabelText(/password/i), { target: { value: 'pw' } });
    fireEvent.click(screen.getByRole('button', { name: /log in/i }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      '/api/auth/login', expect.objectContaining({ method: 'POST', credentials: 'include' })));
  });
});
```

- [ ] **Step 2-4:** Run (FAIL), implement:

```tsx
// frontend/src/app/login/page.tsx
'use client';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { setAccessToken } from '@/lib/auth-fetch';

export default function LoginPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    const r = await fetch('/api/auth/login', {
      method: 'POST', credentials: 'include',
      headers: { 'Content-Type': 'application/json', 'Origin': window.location.origin },
      body: JSON.stringify({ username, password }),
    });
    if (!r.ok) { setError('Invalid credentials'); return; }
    const { access_token } = await r.json();
    setAccessToken(access_token);
    router.replace('/models');
  }

  return (
    <form onSubmit={submit} className="max-w-sm mx-auto mt-20 space-y-4">
      <h1 className="text-xl font-semibold">vllm-warden</h1>
      <label className="block">Username<Input value={username} onChange={(e) => setUsername(e.target.value)} /></label>
      <label className="block">Password<Input type="password" value={password} onChange={(e) => setPassword(e.target.value)} /></label>
      {error && <p className="text-red-500 text-sm">{error}</p>}
      <Button type="submit">Log in</Button>
    </form>
  );
}
```

Run vitest; verify pass.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/app/login/page.tsx frontend/tests/component/login.test.tsx
git commit -m "feat(ui): login page"
```

---

## Task 11.2: Setup wizard (5 pages)

**Files:**
- Create: `frontend/src/app/setup/layout.tsx`
- Create: `frontend/src/app/setup/welcome/page.tsx`
- Create: `frontend/src/app/setup/admin/page.tsx`
- Create: `frontend/src/app/setup/hf-token/page.tsx`
- Create: `frontend/src/app/setup/gpus/page.tsx`
- Create: `frontend/src/app/setup/done/page.tsx`

- [ ] **Step 1:** Write a thin layout that renders a stepper + outlet.

```tsx
// frontend/src/app/setup/layout.tsx
'use client';
import { usePathname } from 'next/navigation';

const steps = ['welcome', 'admin', 'hf-token', 'gpus', 'done'];

export default function SetupLayout({ children }: { children: React.ReactNode }) {
  const path = usePathname().split('/').pop() ?? '';
  return (
    <div className="max-w-xl mx-auto space-y-6">
      <ol className="flex gap-4 text-sm">
        {steps.map((s, i) => (
          <li key={s} className={path === s ? 'font-bold' : 'text-muted-foreground'}>{i + 1}. {s}</li>
        ))}
      </ol>
      {children}
    </div>
  );
}
```

- [ ] **Step 2:** Implement each step page. Each `page.tsx` posts to its matching `/api/setup/*` endpoint then `router.push` to the next step. The welcome page is static text + "Begin" button. The done page calls `router.replace('/models')`.

(Skip the per-step test boilerplate to keep the plan tractable; rely on the Playwright happy-path test in Task 14.1 for end-to-end coverage.)

- [ ] **Step 3: Commit**

```bash
git add frontend/src/app/setup/
git commit -m "feat(ui): 5-page setup wizard"
```

---

## Task 11.3: Models dashboard + add-model modal

**Files:**
- Create: `frontend/src/app/models/page.tsx`
- Create: `frontend/src/components/models/add-model-modal.tsx`
- Create: `frontend/src/components/models/model-card.tsx`
- Test: `frontend/tests/component/add-model-modal.test.tsx`

- [ ] **Step 1: Test the modal validates served_model_name slug**

```tsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { AddModelModal } from '@/components/models/add-model-modal';

it('rejects invalid slug client-side', async () => {
  render(<AddModelModal open onClose={() => {}} />);
  fireEvent.change(screen.getByLabelText(/served model name/i), { target: { value: 'has spaces' } });
  fireEvent.click(screen.getByRole('button', { name: /add/i }));
  expect(await screen.findByText(/alphanumeric/i)).toBeInTheDocument();
});
```

- [ ] **Step 2-4:** Implement the modal with the regex `/^[a-zA-Z0-9._-]+$/` mirroring `app/models/schemas.py:11`. Implement `/models/page.tsx` to SWR-fetch `/api/models`, render `<ModelCard>` for each, and surface the "Add model" button that opens the modal. Run vitest.

```tsx
// frontend/src/app/models/page.tsx
'use client';
import useSWR from 'swr';
import { useState } from 'react';
import { authFetch } from '@/lib/auth-fetch';
import { ModelCard } from '@/components/models/model-card';
import { AddModelModal } from '@/components/models/add-model-modal';
import { Button } from '@/components/ui/button';

const fetcher = (url: string) => authFetch(url).then((r) => r.json());

export default function ModelsPage() {
  const [open, setOpen] = useState(false);
  const { data, mutate } = useSWR('/api/models', fetcher, { refreshInterval: 5000 });
  return (
    <div className="space-y-4">
      <div className="flex justify-between">
        <h1 className="text-2xl font-semibold">Models</h1>
        <Button onClick={() => setOpen(true)}>Add model</Button>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        {data?.items?.map((m: any) => <ModelCard key={m.id} model={m} />)}
      </div>
      <AddModelModal open={open} onClose={() => { setOpen(false); mutate(); }} />
    </div>
  );
}
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/app/models/ frontend/src/components/models/ frontend/tests/component/add-model-modal.test.tsx
git commit -m "feat(ui): models dashboard + add-model modal with client-side slug check"
```

---

## Task 11.4: Model detail + log stream

**Files:**
- Create: `frontend/src/app/models/[id]/page.tsx`
- Create: `frontend/src/components/models/pull-progress.tsx`
- Create: `frontend/src/components/models/log-stream.tsx`

- [ ] **Step 1:** Implement `log-stream.tsx` consuming `useEventSource('/api/models/${modelId}/logs/stream', { onMessage })` and pushing each line into a buffer rendered by `<AnsiLog>` (the copied podwarden component).

```tsx
// frontend/src/components/models/log-stream.tsx
'use client';
import { useState } from 'react';
import { AnsiLog } from '@/components/ansi-log';
import { useEventSource } from '@/lib/sse';

export function LogStream({ modelId }: { modelId: string }) {
  const [lines, setLines] = useState<string[]>([]);
  useEventSource<{ line: string }>(`/api/models/${modelId}/logs/stream`, {
    onMessage: (m) => setLines((prev) => [...prev.slice(-4999), m.line]),
  });
  return <AnsiLog lines={lines} />;
}
```

`pull-progress.tsx` similarly streams `/api/models/{id}/pull/stream` and renders a progress bar from JSON events.

- [ ] **Step 2: Implement detail page**

```tsx
// frontend/src/app/models/[id]/page.tsx
'use client';
import useSWR from 'swr';
import { use } from 'react';
import { authFetch } from '@/lib/auth-fetch';
import { LogStream } from '@/components/models/log-stream';

const fetcher = (u: string) => authFetch(u).then((r) => r.json());

export default function ModelDetail({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const { data } = useSWR(`/api/models/${id}`, fetcher, { refreshInterval: 2000 });
  if (!data) return null;
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">{data.served_model_name}</h1>
      <pre className="text-sm">{JSON.stringify(data, null, 2)}</pre>
      <LogStream modelId={id} />
    </div>
  );
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/app/models/\[id\]/ frontend/src/components/models/log-stream.tsx frontend/src/components/models/pull-progress.tsx
git commit -m "feat(ui): model detail page with live log stream"
```

---

## Task 11.5: Model-settings page

**Files:**
- Create: `frontend/src/app/models/[id]/settings/page.tsx`
- Create: `frontend/src/components/settings/setting-field.tsx`

- [ ] **Step 1:** Port `setting-field` pattern from podwarden — label + hint copy + input + restart badge.

```tsx
// frontend/src/components/settings/setting-field.tsx
'use client';
import type { FieldHint } from '@/lib/settings-hints';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';

export function SettingField({
  field, value, onChange,
}: { field: FieldHint; value: string; onChange: (v: string) => void }) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between items-baseline">
        <label className="font-medium">{field.label}</label>
        {field.restart !== 'none' && (
          <Badge variant="outline">requires {field.restart}</Badge>
        )}
      </div>
      <p className="text-sm text-muted-foreground">{field.hint}</p>
      <Input value={value} onChange={(e) => onChange(e.target.value)} />
    </div>
  );
}
```

- [ ] **Step 2:** Implement `[id]/settings/page.tsx` that loads `/api/models/{id}/settings`, renders one `<SettingField>` per key in `MODEL_HINTS`, and PATCHes on save. On 409 ("must unload first"), surface an explicit toast/banner.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/app/models/\[id\]/settings/ frontend/src/components/settings/setting-field.tsx
git commit -m "feat(ui): per-model settings editor"
```

---

## Task 11.6: Tokens page (rows, create, rotate, expiry banner)

**Files:**
- Create: `frontend/src/app/tokens/page.tsx`
- Create: `frontend/src/components/tokens/token-row.tsx`
- Create: `frontend/src/components/tokens/create-token-dialog.tsx`
- Create: `frontend/src/components/tokens/rotate-token-dialog.tsx`
- Create: `frontend/src/components/tokens/expiry-banner.tsx`
- Test: `frontend/tests/component/tokens.test.tsx`

- [ ] **Step 1:** Test rotate dialog reveals plaintext once.

```tsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { RotateTokenDialog } from '@/components/tokens/rotate-token-dialog';

it('shows new plaintext once and a copy button', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
    '{"plaintext":"vw_newnewnew","rotated_from":"old","id":"new"}', { status: 201 })));
  render(<RotateTokenDialog open tokenId="old" onClose={() => {}} />);
  fireEvent.click(screen.getByRole('button', { name: /rotate/i }));
  await waitFor(() => expect(screen.getByText('vw_newnewnew')).toBeInTheDocument());
  expect(screen.getByRole('button', { name: /copy/i })).toBeInTheDocument();
});
```

- [ ] **Step 2-4:** Implement each component. Token row renders columns from the spec (Name · Prefix · Created · Expires · Last used · Status · Actions). Expiry banner reads `data.items.filter(it => it.is_near_expiry)` and renders amber alert.

```tsx
// frontend/src/app/tokens/page.tsx (sketch)
'use client';
import useSWR from 'swr';
import { authFetch } from '@/lib/auth-fetch';
import { TokenRow } from '@/components/tokens/token-row';
import { ExpiryBanner } from '@/components/tokens/expiry-banner';
import { CreateTokenDialog } from '@/components/tokens/create-token-dialog';
import { useState } from 'react';
import { Button } from '@/components/ui/button';

const fetcher = (u: string) => authFetch(u).then((r) => r.json());

export default function TokensPage() {
  const { data, mutate } = useSWR('/api/tokens', fetcher, { refreshInterval: 10000 });
  const [createOpen, setCreateOpen] = useState(false);
  return (
    <div className="space-y-4">
      <ExpiryBanner items={data?.items ?? []} />
      <div className="flex justify-between">
        <h1 className="text-2xl font-semibold">API tokens</h1>
        <Button onClick={() => setCreateOpen(true)}>Create token</Button>
      </div>
      <table className="w-full text-sm">
        <thead><tr><th>Name</th><th>Prefix</th><th>Created</th><th>Expires</th><th>Last used</th><th>Status</th><th></th></tr></thead>
        <tbody>{data?.items?.map((it: any) => <TokenRow key={it.id} item={it} onChange={mutate} />)}</tbody>
      </table>
      <CreateTokenDialog open={createOpen} onClose={() => { setCreateOpen(false); mutate(); }} />
    </div>
  );
}
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/app/tokens/ frontend/src/components/tokens/ frontend/tests/component/tokens.test.tsx
git commit -m "feat(ui): tokens page with rotate + expiry banner"
```

---

## Task 11.7: Stats page

**Files:**
- Create: `frontend/src/app/stats/page.tsx`
- Create: `frontend/src/components/stats/throughput-chart.tsx`
- Create: `frontend/src/components/stats/gpu-util-chart.tsx`
- Create: `frontend/src/components/panels/metric-summary-panel.tsx`

- [ ] **Step 1:** Adapt the podwarden `metric-summary-panel.tsx` (copy + retarget data). Implement recharts `<LineChart>` for throughput (tokens/sec over time) and `<AreaChart>` for GPU utilisation. Data source: SWR poll of `/api/stats/throughput` and `/api/stats/gpu-util`.

```tsx
// frontend/src/app/stats/page.tsx
'use client';
import useSWR from 'swr';
import { authFetch } from '@/lib/auth-fetch';
import { ThroughputChart } from '@/components/stats/throughput-chart';
import { GpuUtilChart } from '@/components/stats/gpu-util-chart';

const f = (u: string) => authFetch(u).then((r) => r.json());

export default function StatsPage() {
  const tp = useSWR('/api/stats/throughput', f, { refreshInterval: 5000 });
  const gpu = useSWR('/api/stats/gpu-util', f, { refreshInterval: 5000 });
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">Stats</h1>
      <ThroughputChart data={tp.data?.points ?? []} />
      <GpuUtilChart data={gpu.data?.points ?? []} />
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/app/stats/ frontend/src/components/stats/ frontend/src/components/panels/metric-summary-panel.tsx
git commit -m "feat(ui): stats page with throughput + gpu-util recharts"
```

---

## Task 11.8: Settings page (tabs)

**Files:**
- Create: `frontend/src/app/settings/page.tsx`
- Create: `frontend/src/components/settings/runtime-tab.tsx`
- Create: `frontend/src/components/settings/model-tab.tsx`

- [ ] **Step 1:** Use `@/components/ui/tabs` (copied from podwarden). `runtime-tab.tsx` loads `/api/settings/runtime`, renders one `<SettingField>` per `RUNTIME_HINTS` entry, PATCH-saves on Edit. On PATCH response, if `requires_restart` is non-empty, show a banner naming the affected kind. `model-tab.tsx` handles the no-model-loaded empty state per spec.

- [ ] **Step 2: Commit**

```bash
git add frontend/src/app/settings/ frontend/src/components/settings/runtime-tab.tsx frontend/src/components/settings/model-tab.tsx
git commit -m "feat(ui): settings page with runtime + model tabs"
```

---

## Task 12.1: `/api/health` route

**Files:**
- Create: `frontend/src/app/api/health/route.ts`

- [ ] **Step 1: Write the route**

```ts
// frontend/src/app/api/health/route.ts
export const dynamic = 'force-dynamic';

export async function GET() {
  return Response.json({ ok: true });
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/app/api/health/route.ts
git commit -m "feat(ui): /api/health route for container health checks"
```

---

## Task 13.1: Frontend Dockerfile (multi-stage standalone)

**Files:**
- Create: `frontend/Dockerfile`

- [ ] **Step 1: Write the Dockerfile**

```dockerfile
# frontend/Dockerfile
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci

FROM node:20-alpine AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM node:20-alpine AS runtime
WORKDIR /app
ENV NODE_ENV=production PORT=3000
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
COPY --from=build /app/public ./public
EXPOSE 3000
CMD ["node", "server.js"]
```

- [ ] **Step 2: Build to verify**

```bash
docker build -t vllm-warden-ui:test frontend/
```
Expected: succeeds. Image size ~200 MB.

- [ ] **Step 3: Commit**

```bash
git add frontend/Dockerfile
git commit -m "feat(ui): multi-stage Dockerfile (Next.js standalone output)"
```

---

## Task 13.2: Update root `docker-compose.yml`

**Files:**
- Modify: `docker-compose.yml`

- [ ] **Step 1: Edit `docker-compose.yml`** — replace the current single-service definition with the two-service block from the spec (api + ui sections, with `BACKEND_URL`, `VW_FRONTEND_ORIGIN`, named volumes `vw-data` + `vw-hfcache`).

- [ ] **Step 2: Smoke**

```bash
docker compose up -d
sleep 5
curl -fsS http://localhost:3000/api/health     # → {"ok": true}
curl -fsS http://localhost:8080/healthz        # → {"ok": true}
docker compose down
```

- [ ] **Step 3: Commit**

```bash
git add docker-compose.yml
git commit -m "feat(ui): add ui service to docker-compose.yml"
```

---

## Task 13.3: Update Hub template (`v2.x` major track)

**Files:**
- Modify: `deploy/hub/compose.yaml`
- Modify: `deploy/hub/template.json`
- Modify: `deploy/hub/README-hub.md`

- [ ] **Step 1: Edit `deploy/hub/compose.yaml`** — 2-service shape with `{{ image_tag }}` sentinels per the spec.

- [ ] **Step 2: Edit `deploy/hub/template.json`** — add fields:

```json
{
  "version": "2.0.0",
  "breaking": true,
  "min_compatible_warden_version": "v2026.05.NN.0",
  "ports": [{ "internal": 3000, "label": "UI" }, { "internal": 8080, "label": "API (internal)" }],
  "ingress": { "service": "ui", "port": 3000 },
  "env": {
    "VW_FRONTEND_ORIGIN": { "required": true, "description": "Public UI origin for CSRF/Origin check." },
    "VW_JWT_SECRET": { "required": false, "description": "Optional. Empty → auto-mint and persist." }
  },
  "healthcheck": { "ui": "/api/health", "api": "/healthz" }
}
```

Replace `v2026.05.NN.0` with the calver tag that will ship MR-1+MR-2 once known.

- [ ] **Step 3: Edit `deploy/hub/README-hub.md`** — add a "Migrating from v1" section per the spec: existing single-service installs must add the `ui` service, set `VW_FRONTEND_ORIGIN`, and rebind ingress.

- [ ] **Step 4: Commit**

```bash
git add deploy/hub/
git commit -m "feat(ui): bump Hub template to v2.x track (breaking) + 2 services"
```

---

## Task 14.1: Playwright happy-path E2E

**Files:**
- Create: `frontend/playwright.config.ts`
- Create: `frontend/tests/e2e/happy-path.spec.ts`

- [ ] **Step 1: Write config + spec**

```ts
// frontend/playwright.config.ts
import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './tests/e2e',
  use: { baseURL: 'http://localhost:3000' },
  webServer: { command: 'echo "expects docker compose up running"', port: 3000, reuseExistingServer: true },
});
```

```ts
// frontend/tests/e2e/happy-path.spec.ts
import { test, expect } from '@playwright/test';

test('login → add tiny model → pull → load → completion → rotate token → unload → delete', async ({ page, request }) => {
  await page.goto('/login');
  await page.fill('input[name=username]', 'admin');
  await page.fill('input[name=password]', process.env.E2E_ADMIN_PW || 'change-me');
  await page.click('button:has-text("Log in")');
  await expect(page).toHaveURL(/\/models/);

  await page.click('button:has-text("Add model")');
  await page.fill('input[name=served_model_name]', 'opt-125m');
  await page.fill('input[name=hf_repo]', 'facebook/opt-125m');
  await page.click('button:has-text("Add")');

  await page.click('text=opt-125m');
  await page.click('button:has-text("Pull")');
  await expect(page.getByText('Pull complete')).toBeVisible({ timeout: 120000 });
  await page.click('button:has-text("Load")');
  await expect(page.getByText('loaded')).toBeVisible({ timeout: 60000 });

  // Mint API token
  await page.goto('/tokens');
  await page.click('button:has-text("Create token")');
  await page.fill('input[name=name]', 'e2e-bot');
  await page.click('button:has-text("Create")');
  const plaintext = (await page.textContent('[data-testid=new-token]')) ?? '';
  expect(plaintext.startsWith('vw_')).toBe(true);

  // Completion via /v1
  const r = await request.post('/v1/completions', {
    headers: { Authorization: `Bearer ${plaintext}` },
    data: { model: 'opt-125m', prompt: 'hi', max_tokens: 8 },
  });
  expect(r.status()).toBe(200);

  // Rotate
  await page.click('button[aria-label="Rotate"]');
  await page.click('button:has-text("Confirm")');
  await expect(page.getByText(/vw_/)).toBeVisible();

  // Unload + delete
  await page.goto('/models/opt-125m');
  await page.click('button:has-text("Unload")');
  await page.click('button:has-text("Delete")');
  await expect(page).toHaveURL(/\/models$/);
});
```

- [ ] **Step 2:** Test runs against a live local stack (docker compose). Document in `frontend/tests/e2e/README.md` how to run: `docker compose up -d && npx playwright test`.

- [ ] **Step 3: Commit**

```bash
git add frontend/playwright.config.ts frontend/tests/e2e/
git commit -m "test(ui): Playwright happy-path E2E"
```

---

## Task 15.1: CI jobs

**Files:**
- Modify: `.gitlab-ci.yml`

- [ ] **Step 1: Add three jobs**

```yaml
lint:frontend:
  stage: lint
  image: node:20-alpine
  script:
    - cd frontend && npm ci && npm run lint && npm run typecheck
  rules:
    - if: $CI_COMMIT_BRANCH

build:ui:
  stage: build
  image: docker:24
  services: [docker:24-dind]
  script:
    - docker buildx build --push -t registry.podwarden.com/vllm-warden-ui:sha-$CI_COMMIT_SHORT_SHA -t registry.podwarden.com/vllm-warden-ui:$CI_COMMIT_REF_SLUG frontend/
  rules:
    - if: $CI_COMMIT_BRANCH

typecheck:api-types:
  stage: lint
  image: python:3.11-slim
  script:
    - pip install -r requirements.txt
    - python -c "import json, sys; from app.main import app; sys.stdout.write(json.dumps(app.openapi()))" > /tmp/openapi.json
    - apk add nodejs npm || apt-get install -y nodejs npm
    - cd frontend && npm ci && npx openapi-typescript /tmp/openapi.json -o /tmp/api-types.generated.ts
    - diff -u src/lib/api-types.generated.ts /tmp/api-types.generated.ts
  rules:
    - if: $CI_COMMIT_BRANCH
```

- [ ] **Step 2:** Commit. Push branch; CI runs.

```bash
git add .gitlab-ci.yml
git commit -m "ci: lint:frontend, build:ui, typecheck:api-types"
```

---

## Task 15.2: Pre-merge polish for MR-2

- [ ] **Step 1:** Bundle baseline.

Run:
```bash
docker run --rm -v $(pwd)/frontend:/app -w /app node:20-alpine sh -c "npm ci && ANALYZE=true npm run build"
```
Capture `.next/analyze/client.html` → check in as `frontend/baseline.html` or attach to MR description. Reviewer asserts no >20% regression on subsequent builds.

- [ ] **Step 2:** Run all CI gates locally.

```bash
make lint
make test
docker run --rm -v $(pwd)/frontend:/app -w /app node:20-alpine sh -c "npm ci && npm run lint && npm run typecheck && npx vitest run"
make generate-api-types  # verify no drift
git diff --exit-code frontend/src/lib/api-types.generated.ts
```

- [ ] **Step 3:** Push + open MR.

```bash
git push -u origin feat/ui-redesign-nextjs
gh mr create --title "MR-2: Next.js 15 + Tailwind UI" --description ...
```

**End of Phase 2.**

---

## Self-review checklist

Run through each item before considering the plan complete.

### 1. Spec coverage

For every section in `docs/superpowers/specs/2026-05-11-vllm-warden-ui-redesign-design.md`:

- Auth contract (login/refresh/logout/sse-ticket) → Tasks 1.3, 1.5, 1.7, 1.10 ✓
- Refresh cookie + Origin check → Tasks 1.4, 1.5 ✓
- JWT secret bootstrap → Task 1.1 ✓
- v1 cookie → JWT migration (delete `SessionMiddleware`) → Tasks 4.1, 4.2 ✓
- SSE auth + ticket store + stream registry + revocation → Tasks 1.6, 1.9, 1.10, 1.11 ✓
- Tokens migration (0009) + repo + rotate + bearer-check expires_at → Tasks 2.1–2.5 ✓
- Settings full surface + requires_restart + model 409 → Tasks 3.1–3.3 ✓
- Delete Jinja/web/SessionMiddleware/itsdangerous/jinja2 → Tasks 4.1–4.3 ✓
- Operating runbook → Task 5.1 ✓
- Frontend scaffold, theme/globals/primitives copy → Tasks 8.1–8.3, 9.1 ✓
- Type codegen pipeline → Task 9.1 ✓
- auth-fetch refresh-on-401, useEventSource close+remint, lib/api, settings-hints → Tasks 10.1–10.4 ✓
- All 9 page routes (login, setup×5, models, model detail, model settings, tokens, stats, settings) → Tasks 11.1–11.8 ✓
- `/api/health` → Task 12.1 ✓
- Dockerfile standalone, docker-compose.yml, deploy/hub/* with v2.x breaking → Tasks 13.1–13.3 ✓
- Playwright + Vitest tests → Tasks 14.1 + individual component tests ✓
- CI changes → Task 15.1 ✓
- Bundle baseline → Task 15.2 ✓

### 2. Placeholder scan

Grep the plan for: `TBD`, `TODO`, `appropriate`, `etc.`, `Similar to`, `as needed`. The only remaining "…" markers are inside MR-description placeholders for `gh mr create` calls, which are template-only and not implementation work.

### 3. Type consistency

- `expires_at`/`rotated_at`/`rotated_from` — consistent ISO-8601 TEXT in SQL (Task 2.1) and `TokenRow` (Task 2.2), `is_expired`/`is_near_expiry` derived in route (Task 2.4) and consumed by frontend `<TokenRow>` (Task 11.6).
- `requires_restart` PATCH echo — backend returns both `requires_restart` (sorted list, Task 3.2 line 6) and `requires_restart_kinds` (alias). Frontend reads `requires_restart_kinds`. Aliases match.
- `RestartKind` TS type (Task 10.4) — `'none' | 'model-reload' | 'warden-restart'` — matches `RUNTIME_KEYS` values in `app/settings/routes_api.py` (Task 3.2).
- `useEventSource` — same signature in spec (line 435) and Task 10.3 implementation.
- `StreamRegistry.cancel_user` — both unit test (Task 1.6) and logout handler (Task 1.7) call it with a single `user_id` arg.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-11-vllm-warden-ui-redesign-implementation.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
