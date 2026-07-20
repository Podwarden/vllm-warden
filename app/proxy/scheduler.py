"""Per-token sliding-window rate limiter + STRICT priority scheduler.

This module sits between the proxy `require_bearer` step and the upstream
vLLM forward. Two responsibilities, intentionally split into two classes so
test fixtures can exercise them in isolation:

  1. ``TokenRateLimiter`` — per-token sliding-window in *tokens-per-second*.
     ``rate_limit_tps`` from the api_tokens row sets the budget; the window
     length is 10 seconds by default (env-overridable via
     ``VW_RATE_LIMIT_WINDOW_S``). Returns ``False`` when the burst would
     exceed the budget; caller raises 429 (NOT 503 — OpenAI client retry
     logic distinguishes the two: 429 is retried with backoff, 503 is
     considered "model unavailable" and surfaced to the user).

  2. ``PriorityScheduler`` — a *per-engine, multi-slot* admission gate in
     front of each vLLM engine. Each engine (keyed by an opaque string the
     proxy supplies — the model id) admits up to ``VW_PROXY_MAX_INFLIGHT``
     requests concurrently (default 16); vLLM's own continuous batching
     does the real work once admitted, so the cap exists only to bound the
     queue depth we hand the engine, not to serialise. (#173: the original
     design was a single global slot — ONE request talked to vLLM at a
     time, held for the whole SSE stream — which throttled the product path
     to ~40 tok/s while the engines could sustain >1000 tok/s aggregate.)

     Admission is STRICT-priority *ordered*: when an engine is at capacity,
     waiters queue in a per-engine heap ordered first by priority (high →
     low), then by enqueue time. Priority-9 is admitted before any waiting
     priority-0..8; heavy priority-9 traffic CAN starve lower priorities
     indefinitely. That trade-off is intentional and locked by CTO decision
     #3 of the 2026-05 overhaul plan — operators surface the risk in the UI
     (see tooltip on the Priority column). Priority is *also* pushed down
     into vLLM itself via ``--scheduling-policy priority`` + a per-request
     ``priority`` field (#173 part B, see app/proxy/routes.py), so ordering
     is honoured by the engine's batch scheduler once many requests are
     admitted concurrently — not just at our admission boundary.

     Engines are independent (separate GPUs): a saturated engine A never
     blocks a request bound for an idle engine B — they have separate
     queues and in-flight counters.

Both classes are async-safe but NOT process-safe — the warden runs a
single uvicorn worker per pod (app/main.py), so a single in-memory
instance per app is sufficient. If we ever scale to multiple workers
we'd need a Redis-backed implementation; that's deferred.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


def _default_window_s() -> float:
    """Window length in seconds for the sliding-window rate limiter.

    Defaults to 10s per the master plan. Bumping or shrinking via
    ``VW_RATE_LIMIT_WINDOW_S`` is intended for tests and for operators
    tuning latency-vs-burstiness in production. Values <= 0 are
    treated as 'use the default' — better than 0/0 inside the budget
    arithmetic below.
    """
    try:
        v = float(os.environ.get("VW_RATE_LIMIT_WINDOW_S", "10.0"))
        return v if v > 0 else 10.0
    except ValueError:
        return 10.0


def _default_max_inflight() -> int:
    """Per-engine concurrency cap for the priority scheduler (#173).

    Defaults to 16 — comfortably inside the per-engine sweet spot we measured
    on 4×A4000 (throughput keeps climbing through concurrency 16 with no error
    rate), while still bounding the queue depth handed to any one engine.
    ``VW_PROXY_MAX_INFLIGHT`` overrides it; values <= 0 fall back to the
    default rather than dead-locking every engine at zero admissions.
    """
    try:
        v = int(os.environ.get("VW_PROXY_MAX_INFLIGHT", "16"))
        return v if v > 0 else 16
    except ValueError:
        return 16


@dataclass
class _Bucket:
    """A per-token sliding-window bucket.

    Stores (epoch_seconds, tokens_charged) for every request inside the
    window. The window is purged on every check_and_charge() call so the
    deque cannot grow unbounded for a steadily-used token.
    """

    samples: deque[tuple[float, int]] = field(default_factory=deque)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class TokenRateLimiter:
    """Per-token sliding-window rate limiter, sized in *tokens/sec*.

    Usage:

        ok = await limiter.check_and_charge(token_id, rate_limit_tps, n_tokens=42)
        if not ok:
            raise HTTPException(429, "rate limit exceeded")

    ``rate_limit_tps`` of None means unlimited — the limiter immediately
    returns True without recording anything (no memory cost for the
    NULL-rate-limit common case).

    The "budget" for a window of length W seconds and a rate of R tokens/s
    is W * R tokens. A request asking for ``n_tokens`` is admitted iff
    sum(samples_in_window) + n_tokens <= budget.

    Implementation note: this is a *charge-on-success* limiter. If the
    request is rejected we do NOT record it (operators want their burst
    headroom unaffected by failed bursts). If you want a "leaky bucket
    on failure too" change the policy here — there's a test that asserts
    the current behaviour.
    """

    def __init__(self, window_s: float | None = None) -> None:
        self._window_s = window_s if window_s is not None else _default_window_s()
        # Token id → bucket. Buckets are NEVER evicted (a long-lived
        # warden process accumulates one entry per ever-seen token);
        # this is fine because the warden's deployment lifecycle bounds
        # the token count (operators do not mint thousands of tokens per
        # process). If that assumption changes, add an LRU evictor here.
        self._buckets: dict[str, _Bucket] = {}
        self._buckets_lock = asyncio.Lock()

    @property
    def window_s(self) -> float:
        return self._window_s

    async def _get_bucket(self, token_id: str) -> _Bucket:
        # Outer lock guards bucket creation; inner per-bucket lock guards
        # the deque mutation. Two-level locking lets two unrelated tokens
        # proceed in parallel without contending on a single global lock.
        bucket = self._buckets.get(token_id)
        if bucket is not None:
            return bucket
        async with self._buckets_lock:
            bucket = self._buckets.get(token_id)
            if bucket is None:
                bucket = _Bucket()
                self._buckets[token_id] = bucket
            return bucket

    async def check_and_charge(
        self,
        token_id: str,
        rate_limit_tps: int | None,
        n_tokens: int,
        *,
        now: float | None = None,
    ) -> bool:
        """Return True if the request fits the budget; False to reject (429).

        ``now`` is injectable for deterministic tests — production callers
        pass None to use the wall clock.
        """
        if rate_limit_tps is None or rate_limit_tps <= 0:
            return True  # NULL or non-positive ⇒ unlimited (the schema's CHECK
            # trigger guarantees we never see <= 0 from the DB but the proxy
            # may pass a positional 0 by accident — fail-open here, not
            # fail-closed, to avoid a thundering-herd 429 on operator typo)

        # Charging is symbolic — we count "tokens" in the LLM sense (prompt
        # tokens) but it's an opaque integer to this class. Anything <= 0
        # is normalized to 1 so an empty-prompt request still consumes a slot.
        charge = max(1, int(n_tokens))
        budget = self._window_s * float(rate_limit_tps)
        t = time.monotonic() if now is None else now
        cutoff = t - self._window_s

        bucket = await self._get_bucket(token_id)
        async with bucket.lock:
            # Drop expired samples.
            while bucket.samples and bucket.samples[0][0] < cutoff:
                bucket.samples.popleft()
            current = sum(s[1] for s in bucket.samples)
            if current + charge > budget:
                return False
            bucket.samples.append((t, charge))
            return True


# ---------------------------------------------------------------------------
# Priority scheduler
# ---------------------------------------------------------------------------


@dataclass(order=True)
class _Waiter:
    """One queued request waiting for the scheduler slot.

    The ``sort_index`` tuple gives the strict ordering:
      - ``-priority`` so HIGHER priority sorts FIRST,
      - ``enqueue_seq`` (a monotonically increasing counter, ints only)
        breaks ties so same-priority requests are FIFO inside their tier.
    asyncio.PriorityQueue is heap-backed; the heap invariant only needs
    a < comparison on the tuple, so the ``Event`` is excluded via
    ``compare=False`` (otherwise Event has no __lt__).
    """

    sort_index: tuple[int, int]
    event: asyncio.Event = field(compare=False)


_DEFAULT_ENGINE_KEY = "__default__"


class PriorityScheduler:
    """Per-engine, multi-slot priority admission gate in front of vLLM.

    Each engine (keyed by an opaque ``engine_key`` the caller supplies —
    in production the model id) admits up to ``max_inflight`` requests
    concurrently. When an engine is at capacity, further acquirers wait in
    that engine's heap ordered by priority (9 first), FIFO inside a tier.

    Priority 9 is highest. There is NO aging or anti-starvation — priority-0
    traffic CAN wait indefinitely behind a hot priority-9 client. Document
    this in the UI (Priority column tooltip) and in docs/operating.md.

    Usage:

        async with scheduler.acquire(priority=token.priority, engine_key=model.id):
            return await forward_to_vllm(...)

    ``engine_key`` defaults to a shared key so callers (and tests) that don't
    distinguish engines still get a single shared pool. A cancelled waiter
    (client disconnect) is tombstoned and skipped on the next release — see
    acquire()'s CancelledError handler and ``_release``.
    """

    def __init__(self, max_inflight: int | None = None) -> None:
        self._max_inflight = (
            max_inflight if max_inflight is not None else _default_max_inflight()
        )
        # Per-engine admission heap. asyncio.PriorityQueue's heap-pop is
        # O(log n) and async-safe. Created lazily per engine_key.
        self._queues: dict[str, asyncio.PriorityQueue[_Waiter]] = {}
        # Per-engine count of requests currently admitted ("in flight").
        self._inflight: dict[str, int] = {}
        # One lock guards all the bookkeeping dicts. The critical sections
        # are tiny dict ops; contention between engines is negligible and a
        # single lock keeps the admit/release invariants trivially correct.
        self._lock = asyncio.Lock()
        # Monotonically increasing — used for FIFO tie-break inside a tier.
        self._seq = 0
        # Toggle for tests: when True, acquire() does NOT release the slot
        # automatically on context-manager exit; the test must call
        # ``_release_for_test`` to advance the queue. Operations code never
        # touches this.
        self._manual_release = False

    @property
    def max_inflight(self) -> int:
        return self._max_inflight

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _queue_for(self, engine_key: str) -> asyncio.PriorityQueue[_Waiter]:
        q = self._queues.get(engine_key)
        if q is None:
            q = asyncio.PriorityQueue()
            self._queues[engine_key] = q
        return q

    @asynccontextmanager
    async def acquire(
        self, priority: int, engine_key: str = _DEFAULT_ENGINE_KEY
    ) -> AsyncIterator[None]:
        """Acquire an admission slot for ``engine_key``, ordered by priority.

        Fast path — if the engine is below ``max_inflight`` and nobody is
        already queued for it, admit immediately without touching the heap
        (most production traffic IS idle most of the time). Otherwise enqueue
        in the engine's heap and wait for a holder to release.
        """
        # Sanity-clamp priority to 0..9 to match the DB CHECK. We trust
        # the DB but defending in depth costs nothing here.
        prio = max(0, min(9, int(priority)))

        # Decide fast-path vs enqueue atomically under the lock so two
        # acquirers can't both observe spare capacity and overshoot it, and
        # so we never jump the queue ahead of an already-waiting request.
        async with self._lock:
            inflight = self._inflight.get(engine_key, 0)
            q = self._queues.get(engine_key)
            queue_empty = q is None or q.empty()
            if inflight < self._max_inflight and queue_empty:
                self._inflight[engine_key] = inflight + 1
                fast_path = True
            else:
                fast_path = False
                waiter = _Waiter(
                    sort_index=(-prio, self._next_seq()),
                    event=asyncio.Event(),
                )
                await self._queue_for(engine_key).put(waiter)

        if not fast_path:
            try:
                await waiter.event.wait()
            except asyncio.CancelledError:
                # Client disconnected while waiting. Mark the waiter as
                # cancelled so the next release() skips it; can't remove
                # it from PriorityQueue cleanly without rebuilding the
                # heap, so we use the tombstone pattern (event.set + a
                # cancelled flag check in _release).
                waiter.cancelled = True  # type: ignore[attr-defined]
                waiter.event.set()
                raise

        try:
            yield
        finally:
            if not self._manual_release:
                await self._release(engine_key)

    async def _release(self, engine_key: str = _DEFAULT_ENGINE_KEY) -> None:
        """Hand this engine's freed slot to its next waiter, or free it.

        Skips tombstoned (cancelled) waiters at the head. If a real waiter
        takes the slot, the in-flight count is unchanged (one holder swapped
        for another); otherwise the count is decremented and empty
        bookkeeping is dropped so a long-lived warden doesn't accumulate one
        dict entry per ever-seen engine.
        """
        async with self._lock:
            q = self._queues.get(engine_key)
            while q is not None and not q.empty():
                nxt = q.get_nowait()
                if getattr(nxt, "cancelled", False):
                    continue
                # Hand the slot off; in-flight count stays the same until
                # the new holder exits its `async with` and releases in turn.
                nxt.event.set()
                return
            cur = self._inflight.get(engine_key, 0)
            if cur > 0:
                cur -= 1
            if cur <= 0:
                self._inflight.pop(engine_key, None)
                if q is not None and q.empty():
                    self._queues.pop(engine_key, None)
            else:
                self._inflight[engine_key] = cur

    # ----- Test helpers (NOT for production code) ----------------------

    async def _release_for_test(self, engine_key: str = _DEFAULT_ENGINE_KEY) -> None:
        """Public-for-tests release. The test sets _manual_release=True,
        runs N acquirers (each of which sleeps in their `async with`),
        then calls this once per acquirer to step the heap. Lets the
        scheduler ordering test pin down "9 fires before 0" without
        having to control coroutine scheduling via asyncio.sleep.
        """
        await self._release(engine_key)

    def _queue_size_for_test(self, engine_key: str = _DEFAULT_ENGINE_KEY) -> int:
        q = self._queues.get(engine_key)
        return q.qsize() if q is not None else 0

    def _inflight_for_test(self, engine_key: str = _DEFAULT_ENGINE_KEY) -> int:
        return self._inflight.get(engine_key, 0)
