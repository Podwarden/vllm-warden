"""God mode live-stream SSE endpoint — ``GET /api/admin/godmode/stream``.

Mirrors the model-loading log stream (``app/models/routes_logs.py``): guarded by
the same single-use SSE ticket (minted by the ``require_jwt``-gated
``POST /api/auth/sse-ticket``), so EventSource — which cannot send an
Authorization header — composes with the existing proxy/Caddy SSE path. There is
no viewer/operator/admin RBAC in this project; the warden's single privileged UI
session (a valid access JWT → a ticket) is the gate.

On connect: replay the hub's ring snapshot as ``data:`` frames, then stream live
events from the subscriber queue. Idle windows emit a ``: keepalive`` comment so
intermediate proxies don't sever the connection. The subscriber is removed in a
``finally`` on disconnect / cancellation / close.
"""

import asyncio
import base64
import binascii
import json
import re

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from app.auth.deps import require_jwt
from app.models.routes_logs import require_sse_ticket
from app.utils.sse import sse_headers

router = APIRouter(prefix="/api/admin/godmode", tags=["godmode"])

# Full path of the live-stream endpoint. Exported so the shared SSE-ticket
# mint (``POST /api/auth/sse-ticket``) can recognise a god-mode ticket request
# and refuse it up front when the feature is off — the browser's EventSource
# can never read the stream endpoint's own 409 (``onerror`` carries no HTTP
# status), so the viewer classifies the disabled state from the *mint*
# response. Keep this in sync with the router prefix + route below.
STREAM_PATH = "/api/admin/godmode/stream"

# Keepalive cadence: emit an SSE comment line after this many idle seconds so
# intermediate proxies (nginx/Caddy default 60s) and EventSource's own timers
# don't tear down a quiet stream. Module-level so tests can monkeypatch a
# sub-second value rather than waiting real time. Mirrors routes_logs.py.
KEEPALIVE_INTERVAL_S: float = 15.0

# Media ids are secrets.token_hex(8) -> exactly 16 lowercase hex chars. The
# regex is defense-in-depth (the store is a flat dict, so no traversal is
# possible anyway) and lets us 404 junk without a store lookup.
_MEDIA_ID_RE = re.compile(r"^[0-9a-f]{16}\Z")


@router.get("/media/{media_id}")
async def godmode_media(
    media_id: str, request: Request, user: str = Depends(require_jwt)
):
    """Serve one stored god-mode image (spec 2026-08-03, §5). Plain authed
    fetch (NOT EventSource), so require_jwt applies directly — no SSE ticket.
    Decoding happens here, off the proxy hot path."""
    settings = request.app.state.settings
    store = getattr(request.app.state, "godmode_media", None)
    if store is None or not settings.godmode_enabled:
        raise HTTPException(409, "god mode is disabled (VW_GODMODE_ENABLED)")
    if not _MEDIA_ID_RE.match(media_id):
        raise HTTPException(404, "unknown media id")
    item = store.get(media_id)
    if item is None:
        raise HTTPException(404, "media evicted or unknown")
    mime, b64 = item
    try:
        data = base64.b64decode(b64, validate=True)
    except (ValueError, binascii.Error) as err:
        raise HTTPException(404, "media payload undecodable") from err
    return Response(
        content=data,
        media_type=mime,
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.get("/stream")
async def godmode_stream(
    request: Request, user: str = Depends(require_sse_ticket)
):
    settings = request.app.state.settings
    hub = getattr(request.app.state, "godmode_hub", None)
    # Clear "disabled" signal (not a dead stream) so the UI can render a
    # "god mode is disabled (VW_GODMODE_ENABLED)" placeholder rather than
    # spinning on an empty EventSource.
    if hub is None or not settings.godmode_enabled:
        raise HTTPException(409, "god mode is disabled (VW_GODMODE_ENABLED)")

    queue, snapshot = hub.subscribe()
    registry = request.app.state.stream_registry

    async def gen():
        current = asyncio.current_task()
        registry.register(user, current)
        try:
            # Replay the ring snapshot first so a freshly-opened page
            # immediately shows the last few requests.
            for event in snapshot:
                yield f"data: {json.dumps(event)}\n\n"
            # Live loop: block on the queue up to the keepalive interval; on
            # timeout emit a comment frame and re-check the client connection.
            while True:
                if await request.is_disconnected():
                    return
                try:
                    event = await asyncio.wait_for(
                        queue.get(), timeout=KEEPALIVE_INTERVAL_S
                    )
                    yield f"data: {json.dumps(event)}\n\n"
                except TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            registry.unregister(user, current)
            hub.unsubscribe(queue)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers=sse_headers(),
    )
