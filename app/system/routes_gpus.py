"""Live GPU probe: ``GET /api/system/gpus``.

Hits ``nvidia-smi`` directly each call (with a short in-process cache to
absorb burst polling) and returns per-GPU memory + a list of "holders" — the
PIDs currently using GPU memory, labelled with the owning model_id when the
supervisor recognises them.

The chart-friendly historical endpoint remains ``GET /api/stats/gpus`` in
``app/stats/routes_api.py``; this endpoint is the 5–10 s live gauge feed.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Request

from app.auth.deps import require_jwt
from app.db.database import open_db
from app.db.repos.models import ModelRepo
from app.system.gpu import GpuSnapshot, query_gpu_snapshot
from app.system.pid_attribution import attribute_pid_to_model

logger = logging.getLogger(__name__)
router = APIRouter()

# Burst-poll absorbtion. The frontend gauges poll every 5–10 s, but multiple
# tabs / fast scrubbing should not stack subprocesses. Two seconds is short
# enough to feel live and long enough to drop double-clicks.
PROBE_CACHE_TTL_SECONDS = 2.0


class _ProbeCache:
    """Single in-process cache for the nvidia-smi snapshot.

    A single asyncio.Lock guards both the cache read and the underlying probe,
    so concurrent requests inside the TTL window collapse to one nvidia-smi
    invocation. Lives on ``app.state.gpu_probe_cache`` so we can hand it a
    settable clock + probe fn for tests.
    """

    def __init__(
        self,
        *,
        ttl: float = PROBE_CACHE_TTL_SECONDS,
        clock=time.monotonic,
        probe=query_gpu_snapshot,
    ) -> None:
        self._ttl = ttl
        self._clock = clock
        self._probe = probe
        self._lock = asyncio.Lock()
        self._cached: GpuSnapshot | None = None
        self._cached_at: float = -1e9
        self.invocations = 0  # exposed for tests

    async def get(self) -> GpuSnapshot:
        async with self._lock:
            now = self._clock()
            if self._cached is not None and (now - self._cached_at) < self._ttl:
                return self._cached
            self.invocations += 1
            snap = await self._probe()
            self._cached = snap
            self._cached_at = now
            return snap


def _get_cache(request: Request) -> _ProbeCache:
    cache = getattr(request.app.state, "gpu_probe_cache", None)
    if cache is None:
        cache = _ProbeCache()
        request.app.state.gpu_probe_cache = cache
    return cache


async def _build_model_label_map(db_path: str) -> dict[str, str]:
    """Return {model_id: served_model_name} for every model currently in DB."""
    async with open_db(db_path) as db:
        rows = await ModelRepo(db).list_all()
    return {r.id: r.served_model_name for r in rows}


def _holder_payload(
    app,
    *,
    parent_pid_to_model: dict[int, str],
    label_for: dict[str, str],
    pid: int,
    process_name: str,
    memory_mib: int,
) -> dict[str, Any]:
    model_id = attribute_pid_to_model(pid, parent_pid_to_model)
    if model_id is not None:
        return {
            "pid": pid,
            "memory_mib": memory_mib,
            "process": process_name,
            "kind": "model",
            "model_id": model_id,
            "label": label_for.get(model_id),
        }
    return {
        "pid": pid,
        "memory_mib": memory_mib,
        "process": process_name,
        "kind": "external",
        "model_id": None,
        "label": None,
    }


@router.get("/api/system/gpus")
async def system_gpus(request: Request, _user: str = Depends(require_jwt)) -> dict[str, Any]:
    """Live nvidia-smi snapshot with per-GPU memory and PID-attributed holders.

    Response shape (locked for Phase 2 frontend consumption)::

        {
          "probed_at": "<iso8601 UTC>",
          "probe_error": null | "nvidia-smi unavailable" | ...,
          "gpus": [
            {
              "index": 0,
              "name": "NVIDIA RTX A4000",
              "memory_total_mib": 16376,
              "memory_used_mib": 12450,
              "memory_free_mib": 3926,
              "utilization_pct": 87,
              "holders": [
                {
                  "pid": 12345, "memory_mib": 12400,
                  "process": "vllm-worker",
                  "kind": "model",                 # we own this PID
                  "model_id": "61c82bbc55d5147b",
                  "label": "gpt-oss-20b"
                },
                {
                  "pid": 67890, "memory_mib": 50,
                  "process": "Xorg",
                  "kind": "external",              # unknown owner
                  "model_id": null,
                  "label": null
                }
              ]
            }
          ]
        }

    When ``nvidia-smi`` is not on PATH (dev box without NVIDIA) the endpoint
    still returns HTTP 200 with ``gpus: []`` and ``probe_error`` populated so
    the UI can render an empty state rather than crash.
    """
    snap = await _get_cache(request).get()
    parent_pid_to_model = request.app.state.supervisor.parent_pid_to_model()
    label_for = await _build_model_label_map(request.app.state.settings.db_path)

    uuid_to_index: dict[str, int] = {g.uuid: g.index for g in snap.gpus}
    holders_by_index: dict[int, list[dict[str, Any]]] = {g.index: [] for g in snap.gpus}
    for app in snap.apps:
        idx = uuid_to_index.get(app.gpu_uuid)
        if idx is None:
            # nvidia-smi returned a holder for a UUID we didn't see in the
            # per-GPU output — shouldn't happen but don't drop it on the
            # floor; log and skip.
            logger.warning("compute-app on unknown gpu_uuid=%s pid=%d", app.gpu_uuid, app.pid)
            continue
        holders_by_index[idx].append(
            _holder_payload(
                request.app,
                parent_pid_to_model=parent_pid_to_model,
                label_for=label_for,
                pid=app.pid,
                process_name=app.process_name,
                memory_mib=app.memory_mib,
            )
        )

    gpus_payload: list[dict[str, Any]] = []
    for g in snap.gpus:
        holders = sorted(holders_by_index[g.index], key=lambda h: h["memory_mib"], reverse=True)
        gpus_payload.append({
            "index": g.index,
            "name": g.name,
            "memory_total_mib": g.memory_total_mib,
            "memory_used_mib": g.memory_used_mib,
            "memory_free_mib": g.memory_free_mib,
            "utilization_pct": g.utilization_pct,
            # #176 — CUDA compute capability (e.g. 8.6 / sm_86). null when the
            # driver doesn't report it. The Add Model wizard crosses this with
            # the candidate quant/dtype in fit-preview to warn on emulated
            # FP8 / unsupported builds.
            "compute_cap": g.compute_cap,
            "holders": holders,
        })

    return {
        "probed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "probe_error": snap.probe_error,
        "gpus": gpus_payload,
    }
