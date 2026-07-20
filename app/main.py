import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.auth.csrf import csrf_check, ensure_csrf_id
from app.config import load_settings
from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.models import ModelRepo
from app.db.repos.runtime import RuntimeRepo
from app.runtime.port_alloc import PortAllocator
from app.runtime.supervisor import Supervisor


def _build_engine_driver(settings):
    """Pick the engine driver from settings (#160). Default is the
    in-container subprocess driver. The docker driver is opt-in and, until
    image-channel resolution lands in P2 (#161), requires an explicit
    VLLM_ENGINE_IMAGE so it never guesses an image."""
    from pathlib import Path

    if settings.engine_driver == "docker":
        import os

        import docker

        from app.runtime.engine.docker_socket import DockerSocketDriver

        image = os.environ.get("VLLM_ENGINE_IMAGE")
        if not image:
            raise RuntimeError(
                "VW_ENGINE_DRIVER=docker requires VLLM_ENGINE_IMAGE to be set "
                "(the engine container image). Channel-based image resolution "
                "lands in P2/#161; until then the image must be explicit."
            )
        # Pass the same per-model logs dir the subprocess driver uses so the
        # docker driver mirrors the engine container's stdout+stderr into
        # <data_dir>/logs/<model_id>.log — the only place routes_logs.py reads
        # from. Without it the UI Live-logs panel is stale/empty under the
        # docker driver. (#177 follow-up)
        return DockerSocketDriver(
            client=docker.from_env(),
            image=image,
            log_dir=str(Path(settings.data_dir) / "logs"),
        )
    from app.runtime.engine.local_subprocess import LocalSubprocessDriver

    return LocalSubprocessDriver(log_dir=str(Path(settings.data_dir) / "logs"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.hf_cache_dir.mkdir(parents=True, exist_ok=True)
    app.state.settings = settings

    from app.auth.jwt_secret import load_jwt_secret

    app.state.jwt_secret = load_jwt_secret(settings.db_path)

    from app.auth.sse_tickets import TicketStore

    app.state.sse_tickets = TicketStore(
        secret=app.state.jwt_secret,
        ttl_seconds=settings.sse_ticket_ttl_seconds,
    )

    from app.auth.stream_registry import StreamRegistry

    app.state.stream_registry = StreamRegistry()

    async with open_db(settings.db_path) as db:
        await apply_migrations(db)
        await ModelRepo(db).mark_runtime_dead_on_startup()
        await RuntimeRepo(db).clear_all()

    app.state.supervisor = Supervisor(
        app.state.settings,
        driver=_build_engine_driver(app.state.settings),
    )
    app.state.port_allocator = PortAllocator(start=10000, end=10999)

    from app.proxy.tokenizers import TokenizerCache

    app.state.tokenizers = TokenizerCache()

    # S5 (#104) — sliding-window rate limiter + STRICT priority scheduler.
    # Both are in-process singletons because the warden runs a single
    # uvicorn worker per pod; if we ever scale workers, swap for Redis-
    # backed implementations (see app/proxy/scheduler.py module docstring).
    from app.proxy.scheduler import PriorityScheduler, TokenRateLimiter

    app.state.rate_limiter = TokenRateLimiter()
    app.state.scheduler = PriorityScheduler()

    # S8 (#117) — chat playground singletons. ``playground_store`` caches
    # the `vw-playground` bearer plaintext server-side (browser never sees
    # it). ``chat_active_requests`` is a counter the Playwright suite polls
    # to verify abort-cleanup. Both are process-local for the same reason
    # rate_limiter is — single uvicorn worker.
    from app.chat.active_requests import ActiveRequestCounter
    from app.chat.playground_store import PlaygroundStore

    app.state.playground_store = PlaygroundStore()
    app.state.chat_active_requests = ActiveRequestCounter()

    # Live request registry (feature/live-stats-dashboard, Plane B). In-process,
    # single-worker, lock-light — tracks every in-flight /v1 request with token
    # name + client IP + context tokens for GET /api/stats/requests. dev-2 hooks
    # register/deregister into app/proxy/routes.py::_forward (fail-open).
    from app.proxy.request_registry import RequestRegistry

    app.state.request_registry = RequestRegistry()

    from app.runtime.stats_pruner import run_pruner_forever
    from app.runtime.stats_sampler import run_sampler_forever

    sampler_task = asyncio.create_task(run_sampler_forever(settings))
    pruner_task = asyncio.create_task(run_pruner_forever(settings))

    try:
        yield
    finally:
        sampler_task.cancel()
        pruner_task.cancel()
        await asyncio.gather(
            sampler_task,
            pruner_task,
            return_exceptions=True,
        )


def build_app() -> FastAPI:
    app = FastAPI(title="vllm-warden", lifespan=lifespan)

    from app.setup import routes_api as setup_routes_api

    app.include_router(setup_routes_api.router)

    from app.models import routes_api as models_routes_api

    app.include_router(models_routes_api.router)

    from app.models import routes_logs as models_routes_logs

    app.include_router(models_routes_logs.router)

    # #177: engine-version dropdown — GET /api/templates/engine-versions.
    # Backs the try-stack vLLM-version field with the published
    # vllm/vllm-openai semver tags (6h family-keyed cache over Docker Hub).
    from app.templates import routes_api as templates_routes_api

    app.include_router(templates_routes_api.router)

    from app.auth.routes import router as auth_router

    app.include_router(auth_router)

    from app.tokens import routes_api as tokens_routes_api

    app.include_router(tokens_routes_api.router)

    from app.stats import routes_api as stats_routes_api

    app.include_router(stats_routes_api.router)

    # Live realtime stats dashboard (feature/live-stats-dashboard). Two
    # independent planes behind a fixed API contract (docs/live-stats-spec.md):
    #   Plane A — engine /metrics scraper SSE:  GET /api/stats/live
    #   Plane B — live per-request registry:     GET /api/stats/requests
    # Registered here up front so the two backend slices never collide in
    # main.py. Both surface on the NEW /ui/stats/live page; /stats is untouched.
    from app.stats import live_engine as stats_live_engine
    from app.stats import live_requests as stats_live_requests

    app.include_router(stats_live_engine.router)
    app.include_router(stats_live_requests.router)

    # HF cache management — vllm-warden#114. Lives next to stats because
    # the UI surfaces it as a section on /stats; the routes are
    # JWT-gated like every other /api/*.
    from app.cache import routes_api as cache_routes_api

    app.include_router(cache_routes_api.router)

    from app.settings import routes_api as settings_routes_api

    app.include_router(settings_routes_api.router)
    app.include_router(settings_routes_api.model_settings_router)

    from app.proxy import routes as proxy_routes

    app.include_router(proxy_routes.router)

    from app.system import routes_version as system_routes_version

    app.include_router(system_routes_version.router)

    from app.system import routes_gpus as system_routes_gpus

    app.include_router(system_routes_gpus.router)

    # #177: active engine-driver capability (can it swap the engine image to a
    # pinned vLLM version?). Consumed by the Try-stack panel to disable +
    # explain the version selector under the in-container subprocess driver.
    from app.system import routes_engine as system_routes_engine

    app.include_router(system_routes_engine.router)

    # #148: static system inventory (CPU/RAM/GPU/OS/Docker) consumed by
    # the /stats System Configuration panel. 60s in-process cache lives
    # on app.state.system_info_cache (lazy-init on first request).
    from app.system import routes_info as system_routes_info

    app.include_router(system_routes_info.router)

    from app.header import routes_api as header_routes_api

    app.include_router(header_routes_api.router)

    # S4: built-in tuning presets ("Apply preset" dropdown on /settings).
    # Read-only — FE applies preset.settings via the existing PATCH
    # /api/models/{id}/settings endpoint, no new write path.
    from app.presets import routes_api as presets_routes_api

    app.include_router(presets_routes_api.router)

    # S8 (#117): /chat playground — JWT-authed SSE proxy + admin
    # active-requests diagnostic. Mounts /api/chat/* and
    # /api/admin/active-requests; routes are in app/chat/routes_api.py.
    from app.chat import routes_api as chat_routes_api

    app.include_router(chat_routes_api.router)

    # #155 — Unified-port architecture: public landing page at /_landing
    # served behind Caddy's `handle /` rewrite. Route is intentionally
    # NOT JWT-gated (the whole point is that an anonymous browser hitting
    # the unified-port root sees a useful page). Opt-out via the
    # `landing_page_enabled` runtime setting → route returns 404.
    from app.landing import routes as landing_routes

    app.include_router(landing_routes.router)

    # Middleware registration order matters: in Starlette the LAST-added decorator
    # is the OUTERMOST wrapper (first to run on every request).
    #
    # Desired request-path order:
    #   ensure_csrf_id  (outer — populates request.state.csrf_id / csrf_token)
    #   csrf_check      (inner — validates X-CSRF-Token after csrf_id is set)
    #
    # Therefore: csrf_check is added first (→ innermost), ensure_csrf_id last (→ outermost).

    @app.middleware("http")
    async def _csrf_check(request: Request, call_next):
        return await csrf_check(request, call_next)

    @app.middleware("http")
    async def _ensure_csrf_id(request: Request, call_next):
        return await ensure_csrf_id(request, call_next)

    @app.get("/api/csrf")
    async def get_csrf_token(request: Request) -> dict:
        return {"csrf": request.state.csrf_token}

    @app.get("/healthz")
    async def healthz() -> dict[str, bool]:
        return {"ok": True}

    return app


app = build_app()
