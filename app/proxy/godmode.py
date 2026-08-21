"""God mode — a process-local, in-memory pub/sub broadcast hub.

The hub taps the proxy hot path (``app/proxy/routes.py::_forward``) and fans
prompt/output events out to the warden's privileged UI stream
(``GET /api/admin/godmode/stream``). It is deliberately tiny and dependency-free:

  * **No DB, no disk.** Everything lives in a bounded in-memory ring; the ring
    dies on pod restart. Prompt/output content is NEVER persisted (see the
    god-mode spec, §2 "History depth").
  * **Never backpressures the proxy.** ``publish`` is *sync* and non-awaiting so
    the tap can call it without ``await`` on the hot path. A slow/full
    subscriber queue drops its OLDEST event rather than blocking ``publish`` —
    god mode must never break, delay, or backpressure a proxied response.
  * **Heap-bounded.** The ring is capped by BOTH an event count AND a total
    char budget; the oldest events are evicted when either bound is exceeded so
    a burst of huge runaway completions can't blow the heap. The single newest
    event is always kept even if it alone exceeds the char budget (so an
    oversized completion is still visible, not silently swallowed).

The raw bearer token / Authorization header is never placed on an event — the
tap passes only a ``token_label`` (token name, or id prefix) and ``token_id``.
"""

from __future__ import annotations

import asyncio
from collections import deque

# How many events a single subscriber queue holds before drop-oldest kicks in.
# Bounded so one slow SSE client can't pin unbounded memory; the frontend keys
# rows on the monotonic ``seq`` so a gap from a dropped event is recoverable
# (the client can show a "lagged" hint).
DEFAULT_QUEUE_MAXSIZE = 1000


def _event_chars(event: dict) -> int:
    """Approximate an event's heap cost as the sum of its string field lengths.

    The prompt/delta text dominates; ints (seq, token counts, ts) are ignored.
    Used only for ring-eviction accounting, so an approximation is fine.
    """
    return sum(len(v) for v in event.values() if isinstance(v, str))


class GodModeHub:
    """In-memory pub/sub with a bounded replay ring. Instantiated once on
    ``app.state.godmode_hub`` at startup."""

    def __init__(
        self,
        *,
        ring_events: int = 2000,
        ring_chars: int = 4_000_000,
        queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE,
    ) -> None:
        self._ring_events = ring_events
        self._ring_chars = ring_chars
        self._queue_maxsize = queue_maxsize
        # deque of (event, char_cost) so eviction can decrement the running
        # total in O(1) without re-measuring the evicted event.
        self._ring: deque[tuple[dict, int]] = deque()
        self._ring_chars_total = 0
        self._seq = 0
        self._subscribers: set[asyncio.Queue] = set()

    # -- publish (sync, non-awaiting — safe to call on the proxy hot path) --

    def publish(self, event: dict) -> int:
        """Assign a monotonic ``seq``, append to the ring (evicting as needed),
        then non-blocking-offer to every subscriber. Returns the seq.

        Never awaits, never blocks, never raises on a full subscriber queue.
        """
        self._seq += 1
        event["seq"] = self._seq
        cost = _event_chars(event)
        self._ring.append((event, cost))
        self._ring_chars_total += cost

        # Evict oldest past the event-count bound.
        while len(self._ring) > self._ring_events:
            _old, old_cost = self._ring.popleft()
            self._ring_chars_total -= old_cost
        # Evict oldest past the char budget, but always keep the newest event
        # (a single runaway completion bigger than the whole budget must still
        # be visible rather than evicting itself into the void).
        while len(self._ring) > 1 and self._ring_chars_total > self._ring_chars:
            _old, old_cost = self._ring.popleft()
            self._ring_chars_total -= old_cost

        for q in self._subscribers:
            self._offer(q, event)
        return self._seq

    @staticmethod
    def _offer(q: asyncio.Queue, event: dict) -> None:
        """Non-blocking put; on a full queue drop the OLDEST item for that
        subscriber and mark it lagged. Never blocks, never raises."""
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass
            q._godmode_lagged = True  # type: ignore[attr-defined]

    # -- subscribe / unsubscribe --

    def subscribe(self) -> tuple[asyncio.Queue, list[dict]]:
        """Register a new subscriber. Returns ``(queue, ring_snapshot)`` where
        the snapshot is a shallow copy of the current ring for replay-on-connect
        (later publishes do not mutate a returned snapshot list)."""
        q: asyncio.Queue = asyncio.Queue(maxsize=self._queue_maxsize)
        q._godmode_lagged = False  # type: ignore[attr-defined]
        self._subscribers.add(q)
        snapshot = [event for event, _cost in self._ring]
        return q, snapshot

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    # -- introspection (tests / diagnostics) --

    def subscriber_count(self) -> int:
        return len(self._subscribers)

    @staticmethod
    def is_lagged(q: asyncio.Queue) -> bool:
        return bool(getattr(q, "_godmode_lagged", False))


# Mime allowlist for stored media: an explicit raster set, not a generic
# image/* pattern. Stored images are rendered via blob: URLs, which inherit
# the WARDEN PAGE'S OWN ORIGIN (unlike a plain <img src>) — so a scripted
# format opened directly (e.g. "open image in new tab") would run same-origin
# against the admin API. image/svg+xml is exactly that: SVG can carry
# <script>, so it is excluded even though it matches image/*. See spec
# 2026-08-03 §5.
_MEDIA_MIME_ALLOW = frozenset({
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
    "image/avif",
    "image/bmp",
})


class GodModeMediaStore:
    """Bounded in-memory blob store for god-mode images (spec 2026-08-03, §3).

    Same contract family as :class:`GodModeHub`: sync, non-raising,
    heap-bounded, oldest-first eviction, nothing persisted. Values are the
    base64 payload *string* sliced straight out of the data URI — decoding
    happens at serve time in the endpoint, never on the proxy hot path.
    """

    def __init__(
        self,
        *,
        store_chars: int = 64_000_000,
        max_item_chars: int = 14_000_000,
    ) -> None:
        self._store_chars = store_chars
        self._max_item_chars = max_item_chars
        # dict preserves insertion order -> oldest-first eviction via iteration.
        self._items: dict[str, tuple[str, str]] = {}
        self._chars_total = 0

    def put(self, media_id: str, mime: str, b64: str) -> bool:
        """Store one image payload. Returns False (storing nothing) on any
        rejected/garbage input. Never raises — same hot-path contract as
        ``GodModeHub.publish``."""
        try:
            if not (
                isinstance(media_id, str)
                and isinstance(mime, str)
                and isinstance(b64, str)
            ):
                return False
            if mime not in _MEDIA_MIME_ALLOW:
                return False
            if len(b64) > self._max_item_chars:
                return False
            old = self._items.pop(media_id, None)
            if old is not None:
                self._chars_total -= len(old[1])
            self._items[media_id] = (mime, b64)
            self._chars_total += len(b64)
            # Oldest-first eviction past the total budget; the newest item
            # always survives (mirrors the hub ring's newest-kept rule).
            while len(self._items) > 1 and self._chars_total > self._store_chars:
                oldest_id = next(iter(self._items))
                _mime, old_b64 = self._items.pop(oldest_id)
                self._chars_total -= len(old_b64)
            return True
        except Exception:
            return False

    def get(self, media_id: str) -> tuple[str, str] | None:
        try:
            return self._items.get(media_id)
        except Exception:
            return None

    def size_chars(self) -> int:
        return self._chars_total
