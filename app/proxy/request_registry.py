"""In-process live request registry (Plane B).

Owner: dev-2. See docs/live-stats-spec.md § "Plane B".

Tracks every in-flight /v1 proxy request with token name, client IP, model,
context tokens (prompt + streamed completion) vs max_model_len, elapsed, phase
(prefill/decode), and an orphan flag (client disconnected but forward still
draining). Single uvicorn worker; lock-light — a short ``asyncio.Lock`` guards
insert/delete only, field updates during streaming are plain attribute writes
directly on the ``LiveRequest`` the caller holds (last-write-wins tolerated by
the reader, same rationale as ``ActiveRequestCounter.count()``). Absorbs
ActiveRequestCounter's ``count()`` so ``GET /api/admin/active-requests`` and its
Playwright test stay green.

Registration must be FAIL-OPEN: no registry error may ever break a proxied
request — every hook call in ``app/proxy/routes.py`` wraps this in try/except.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass
class LiveRequest:
    """One in-flight /v1 request. Metadata only — never a token secret.

    ``completion_tokens`` / ``phase`` / ``orphan`` are mutated in place by the
    streaming loop in ``_forward`` without taking the registry lock (the reader
    tolerates a one-tick-stale field). ``prompt_tokens`` and identity fields are
    fixed at register time.
    """

    id: str
    token_id: str | None
    token_name: str | None
    client_ip: str | None
    model: str
    path: str
    prompt_tokens: int
    max_model_len: int | None
    started_monotonic: float
    started_iso: str
    completion_tokens: int = 0
    phase: str = "prefill"
    orphan: bool = False


class RequestRegistry:
    """Live in-flight /v1 request registry.

    The dict is the single source of truth. ``register`` / ``deregister`` take
    a short lock so an insert and a delete that land in the same tick can't
    corrupt the mapping; ``snapshot`` / ``count`` are lock-free reads (a racy
    snapshot may be one entry off mid-insert, which the ~1.5s poller tolerates).
    Live field updates are done by mutating the returned ``LiveRequest`` object
    directly, no registry method required.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._reqs: dict[str, LiveRequest] = {}

    async def register(self, req: LiveRequest) -> None:
        async with self._lock:
            self._reqs[req.id] = req

    async def deregister(self, req_id: str) -> None:
        async with self._lock:
            self._reqs.pop(req_id, None)

    def get(self, req_id: str) -> LiveRequest | None:
        return self._reqs.get(req_id)

    def count(self) -> int:
        """Back-compat with ActiveRequestCounter — number in flight."""
        return len(self._reqs)

    def snapshot(self) -> list[LiveRequest]:
        """Lock-free copy of the current in-flight requests."""
        return list(self._reqs.values())
