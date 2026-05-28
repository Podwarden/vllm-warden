import asyncio
import json
import time
from pathlib import Path

import aiofiles
import aiofiles.os  # noqa: F401  # populates aiofiles.os.path for async stat
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from app.utils.sse import sse_headers

router = APIRouter(prefix="/api/models", tags=["logs"])


# v2026.05.15.5: keepalive cadence for the SSE stream. We emit an SSE
# comment line (": keepalive\n\n") whenever this many seconds elapse
# with no log activity. The comment is silently ignored by EventSource
# / browsers but resets idle timers on intermediate proxies
# (nginx/Caddy default 60s) and on EventSource's own heuristics.
# Without it, a model that takes >60s to produce its first stdout line
# (e.g. vLLM loading a 20GB weight set) would have its stream silently
# severed by the proxy before the first byte ever flowed.
#
# Module-level so tests can monkeypatch a shorter value rather than
# waiting 15s of real time. Float so sub-second values work in tests.
KEEPALIVE_INTERVAL_S: float = 15.0

# Internal poll cadence for the tail follower. Must be <= keepalive
# interval so the gen() loop gets a chance to emit a keepalive within
# one interval of an idle period. 0.5s is the long-standing value and
# matches the pre-v15.5 behaviour exactly.
TAIL_POLL_S: float = 0.5


async def require_sse_ticket(
    request: Request, ticket: str = Query(...)
) -> str:
    try:
        return request.app.state.sse_tickets.consume(ticket, request.url.path)
    except ValueError as exc:
        raise HTTPException(401, str(exc)) from exc


# Sentinel object yielded by _tail every TAIL_POLL_S seconds when no log
# line is available. The outer generator uses these ticks to decide
# whether enough idle time has passed to emit a keepalive comment.
# Using a module-level sentinel rather than `None` so a future change
# that lets _tail yield literal `None` (e.g. for empty log lines) can't
# silently collide with the tick semantics.
_TAIL_TICK: object = object()


async def _tail(path: Path, request: Request):
    """Yield log lines as they're appended to `path`, plus periodic
    ``_TAIL_TICK`` sentinels during idle windows. Stops on client
    disconnect or file removal.

    The tick is what lets the outer generator drive the keepalive
    cadence without resorting to ``asyncio.wait_for`` around an async
    iterator (which corrupts generator state on timeout). One tick per
    ``TAIL_POLL_S`` seconds when no line is available.
    """
    async with aiofiles.open(path) as f:
        await f.seek(0, 2)
        while True:
            if await request.is_disconnected():
                return
            line = await f.readline()
            if line:
                yield line.rstrip()
            else:
                # aiofiles.os.path.exists offloads the stat to a thread
                # pool — keeps the event loop free during the tail
                # poll loop and silences ASYNC240 (no blocking
                # pathlib.Path.exists in async code).
                if not await aiofiles.os.path.exists(path):
                    return
                # Idle window — yield a tick so the consumer can run
                # its keepalive bookkeeping, then sleep.
                yield _TAIL_TICK
                await asyncio.sleep(TAIL_POLL_S)


@router.get("/{model_id}/logs/stream")
async def stream_logs(
    model_id: str, request: Request, user: str = Depends(require_sse_ticket)
):
    settings = request.app.state.settings
    log_path = Path(settings.data_dir) / "logs" / f"{model_id}.log"

    # v2026.05.15.5: open-or-create. Previously this returned 404 when
    # the log file didn't exist, which surfaced through useEventSource's
    # ticket-mint preflight as a terminal-error — the UI stuck on
    # "stream unavailable" even though the subprocess would create the
    # file moments later. We now match the supervisor's behaviour
    # (app/runtime/supervisor.py opens with O_CREAT | O_APPEND): touch
    # the file if missing so the _tail follower has something to read.
    # If the file is empty, the connected-but-empty placeholder in
    # LogStream carries the UX. If the parent directory doesn't exist,
    # that IS a real misconfiguration and should still 500.
    #
    # Path.touch(exist_ok=True) is the canonical single-syscall
    # open-or-create: no fd to leak if anything throws (signal,
    # cancellation, oom) between open and close, and no TOCTOU window
    # between an is_dir() probe and the create — both prior failure
    # modes of the os.open + os.close prologue we replaced. We let
    # FileNotFoundError (missing parent) translate to the same curated
    # 500 the old is_dir() pre-check produced, preserving the existing
    # error message contract.
    try:
        log_path.touch(mode=0o600, exist_ok=True)
    except FileNotFoundError as exc:
        # Hard fail — data_dir/logs is created at app bootstrap. A
        # missing parent is a deployment bug, not a "subprocess hasn't
        # started" race, so we don't paper over it by mkdir-p'ing.
        raise HTTPException(
            500, f"log directory does not exist: {log_path.parent}"
        ) from exc

    registry = request.app.state.stream_registry

    async def gen():
        current = asyncio.current_task()
        registry.register(user, current)
        try:
            async with aiofiles.open(log_path) as f:
                content = await f.read()
            # Emit each log line as a JSON event so the FE useEventSource hook
            # (which JSON.parse's e.data and silently swallows non-JSON) can
            # consume them as typed {line: string} messages. Raw-text events
            # would parse-fail and silently drop on the client.
            for line in content.splitlines()[-200:]:
                yield f"data: {json.dumps({'line': line})}\n\n"

            # Track the wall-clock time of the most recent yield. After
            # KEEPALIVE_INTERVAL_S of silence we emit an SSE comment
            # line (`: keepalive\n\n`) — silently ignored by
            # EventSource but visible to intermediate proxies, which
            # is what stops them from declaring the connection dead.
            last_yield_at = time.monotonic()
            async for item in _tail(log_path, request):
                if item is _TAIL_TICK:
                    # Idle tick. If enough time has elapsed since the
                    # last real or keepalive yield, emit one.
                    if time.monotonic() - last_yield_at >= KEEPALIVE_INTERVAL_S:
                        yield ": keepalive\n\n"
                        last_yield_at = time.monotonic()
                else:
                    yield f"data: {json.dumps({'line': item})}\n\n"
                    last_yield_at = time.monotonic()
        finally:
            registry.unregister(user, current)

    # Anti-buffering headers (X-Accel-Buffering, Cache-Control) live in
    # app.utils.sse so all SSE endpoints get them uniformly — see #50.
    # Belt-and-suspenders alongside the LogStream re-subscribe fix
    # (commit 7df62a3 / #50): even if a proxy buffers data through a
    # status transition, the FE will re-handshake on the next status
    # flip and pick up the backfilled lines.
    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers=sse_headers(),
    )
