"""Live header metrics SSE: ``GET /api/header/metrics/stream``.

Powers the compact VRAM% + GPU% + active-model badge mounted in the nav
chrome (frontend/src/components/header-metrics.tsx). Emits one JSON event
every ``VW_HEADER_METRICS_INTERVAL_S`` seconds (default 2.0).

The payload is intentionally a small superset of what the badge renders
today so future iterations (e.g. tooltip with per-GPU breakdown) don't
need a new endpoint::

    {
      "ts": "2026-05-23T19:01:02.345Z",
      "gpus": [{"index": 0, "name": "NVIDIA RTX A4000",
                "memory_used_mib": 12450, "memory_total_mib": 16376,
                "utilization_pct": 87}],
      "vram_used_mib": 12450,
      "vram_total_mib": 16376,
      "vram_pct": 76,                   # int 0-100, derived
      "gpu_util_pct": 87,               # max across GPUs, derived
      "active_model": "gpt-oss-20b",    # served_model_name of loaded row, or null
      "active_model_id": "61c82bbc...", # or null
      "probe_error": null               # passthrough from nvidia-smi probe
    }

Auth: SSE ticket (same pattern as model log streams). The widget is
hidden on /login and /setup on the client; this endpoint stays JWT-gated
via ticket mint so the contract is symmetric — no GPU info leaks to
unauthenticated callers.

Data source: shares ``app.state.gpu_probe_cache`` with
``/api/system/gpus`` (2s TTL absorbs burst polling so multiple tabs
collapse to one ``nvidia-smi`` invocation). Active-model lookup goes
through ``model_runtime`` joined to ``models`` — only ``status='loaded'``
rows surface in the badge.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.db.database import open_db
from app.models.routes_logs import require_sse_ticket
from app.system.routes_gpus import _ProbeCache
from app.utils.sse import sse_headers

router = APIRouter(prefix="/api/header", tags=["header"])


# Default emit cadence. Override via ``VW_HEADER_METRICS_INTERVAL_S`` env var
# (float seconds). Floor at 0.5s so a misconfigured env can't pin a CPU.
def _interval_seconds() -> float:
    raw = os.environ.get("VW_HEADER_METRICS_INTERVAL_S")
    if not raw:
        return 2.0
    try:
        v = float(raw)
    except ValueError:
        return 2.0
    return max(0.5, v)


# SSE keepalive cadence — same rationale as routes_logs.KEEPALIVE_INTERVAL_S.
# Header metrics emit every 2s by default so the keepalive practically never
# fires, but a slow probe (e.g. nvidia-smi cold start) could still leave a
# gap long enough for an idle proxy to drop the connection.
KEEPALIVE_INTERVAL_S: float = 15.0


def _get_cache(request: Request) -> _ProbeCache:
    """Reuse the same probe cache as ``/api/system/gpus``.

    The cache lives on ``app.state.gpu_probe_cache`` (lazy-init in
    ``app/system/routes_gpus._get_cache``). We avoid importing the
    private helper directly so a future relocation of the cache owner
    doesn't break this module silently.
    """
    cache = getattr(request.app.state, "gpu_probe_cache", None)
    if cache is None:
        cache = _ProbeCache()
        request.app.state.gpu_probe_cache = cache
    return cache


async def _active_model(db_path: str) -> tuple[str | None, str | None]:
    """Return ``(model_id, served_model_name)`` of the currently-loaded model.

    "Loaded" = ``models.status='loaded'`` AND ``model_runtime`` row exists.
    Returns the first match — the supervisor enforces single-model loading,
    so multiple rows would be a defect we don't paper over here.
    """
    async with open_db(db_path) as db:
        cur = await db.execute(
            "SELECT m.id, m.served_model_name "
            "FROM models m JOIN model_runtime r ON r.model_id = m.id "
            "WHERE m.status = 'loaded' "
            "LIMIT 1"
        )
        row = await cur.fetchone()
    if row is None:
        return (None, None)
    return (row[0], row[1])


def _payload(snap, active_id: str | None, active_name: str | None) -> dict:
    gpus = [
        {
            "index": g.index,
            "name": g.name,
            "memory_used_mib": g.memory_used_mib,
            "memory_total_mib": g.memory_total_mib,
            "utilization_pct": g.utilization_pct,
        }
        for g in snap.gpus
    ]
    used = sum(g.memory_used_mib for g in snap.gpus)
    total = sum(g.memory_total_mib for g in snap.gpus)
    vram_pct = int(round(100.0 * used / total)) if total else 0
    util = max((g.utilization_pct for g in snap.gpus), default=0)
    return {
        "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "gpus": gpus,
        "vram_used_mib": used,
        "vram_total_mib": total,
        "vram_pct": vram_pct,
        "gpu_util_pct": util,
        "active_model": active_name,
        "active_model_id": active_id,
        "probe_error": snap.probe_error,
    }


@router.get("/metrics/stream")
async def stream_metrics(
    request: Request, _user: str = Depends(require_sse_ticket)
):
    """Stream live VRAM%, GPU utilisation, and active-model name as SSE events."""
    cache = _get_cache(request)
    settings = request.app.state.settings
    interval = _interval_seconds()

    async def gen():
        last_yield_at = time.monotonic()
        # Emit one immediate frame so the consumer doesn't sit blank for
        # ``interval`` seconds waiting on the first tick.
        try:
            snap = await cache.get()
            active_id, active_name = await _active_model(settings.db_path)
            yield f"data: {json.dumps(_payload(snap, active_id, active_name))}\n\n"
            last_yield_at = time.monotonic()
        except Exception:  # noqa: BLE001
            # If the very first probe errors, fall through to the loop
            # below; the next attempt will surface a probe_error payload.
            pass

        while True:
            if await request.is_disconnected():
                return
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                return
            if await request.is_disconnected():
                return
            try:
                snap = await cache.get()
                active_id, active_name = await _active_model(settings.db_path)
                yield f"data: {json.dumps(_payload(snap, active_id, active_name))}\n\n"
                last_yield_at = time.monotonic()
            except Exception as exc:  # noqa: BLE001
                # A probe / DB failure shouldn't kill the stream — emit a
                # probe_error event and keep ticking. The consumer renders
                # the badge in a "degraded" state on probe_error != null.
                err_payload = {
                    "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "gpus": [],
                    "vram_used_mib": 0,
                    "vram_total_mib": 0,
                    "vram_pct": 0,
                    "gpu_util_pct": 0,
                    "active_model": None,
                    "active_model_id": None,
                    "probe_error": str(exc) or exc.__class__.__name__,
                }
                yield f"data: {json.dumps(err_payload)}\n\n"
                last_yield_at = time.monotonic()
            # Belt-and-suspenders keepalive — when the emit interval is
            # tuned shorter than KEEPALIVE_INTERVAL_S (the default 2s
            # case) this never fires; it exists for the operator who
            # bumps the env var to 30s for a quiet dashboard.
            if time.monotonic() - last_yield_at >= KEEPALIVE_INTERVAL_S:
                yield ": keepalive\n\n"
                last_yield_at = time.monotonic()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers=sse_headers(),
    )
