"""Tests for the sliding-window rate limiter + STRICT priority scheduler
(``app.proxy.scheduler``, introduced in S5 / closes #104).
"""

import asyncio

import pytest

from app.proxy.scheduler import PriorityScheduler, TokenRateLimiter

# These tests deliberately busy-wait on an internal queue-size invariant
# (sched._queue_size_for_test) so we can deterministically observe that a
# previous task has reached its enqueue point before launching the next.
# An asyncio.Event won't work here — we're waiting for the *internal*
# heap state of asyncio.PriorityQueue.put(), not for a signal the
# waiter coroutine can hand us. ASYNC110 is suppressed at the line of
# each loop with a short pointer back to this paragraph.


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


async def test_rate_limiter_admits_within_budget():
    """A token with budget = window_s * tps should admit a request that
    fits inside the window without rejecting."""
    rl = TokenRateLimiter(window_s=10.0)
    # 100 tps × 10s window = 1000 token budget. One 500-token charge fits.
    assert await rl.check_and_charge("tok-A", 100, n_tokens=500, now=1000.0) is True


async def test_rate_limiter_rejects_when_burst_exceeds_budget():
    rl = TokenRateLimiter(window_s=10.0)
    # 100 tps × 10s = 1000 budget; 600 + 600 = 1200 > 1000 → second must reject.
    assert await rl.check_and_charge("tok-B", 100, n_tokens=600, now=1000.0) is True
    assert await rl.check_and_charge("tok-B", 100, n_tokens=600, now=1000.1) is False


async def test_rate_limiter_rejected_request_does_not_consume_budget():
    """Charge-on-success policy — a rejected request must NOT shrink the
    remaining headroom. Otherwise a steady stream of oversized prompts
    would lock a token out indefinitely.
    """
    rl = TokenRateLimiter(window_s=10.0)
    # Budget 1000. Spend 700.
    assert await rl.check_and_charge("tok-C", 100, n_tokens=700, now=1000.0) is True
    # 400 doesn't fit (would total 1100) → rejected; budget unchanged.
    assert await rl.check_and_charge("tok-C", 100, n_tokens=400, now=1000.0) is False
    # 200 should still fit (700 + 200 = 900 ≤ 1000).
    assert await rl.check_and_charge("tok-C", 100, n_tokens=200, now=1000.0) is True


async def test_rate_limiter_window_slides_forward():
    rl = TokenRateLimiter(window_s=10.0)
    assert await rl.check_and_charge("tok-D", 100, n_tokens=900, now=1000.0) is True
    # 100 more right now → still inside budget (total 1000).
    assert await rl.check_and_charge("tok-D", 100, n_tokens=100, now=1000.0) is True
    # 1 more right now → rejected (would exceed 1000).
    assert await rl.check_and_charge("tok-D", 100, n_tokens=1, now=1000.0) is False
    # 11 seconds later the window has moved past the original 900+100;
    # the new sample of 50 should be accepted.
    assert await rl.check_and_charge("tok-D", 100, n_tokens=50, now=1011.0) is True


async def test_rate_limiter_null_means_unlimited():
    """A token with NULL rate_limit_tps must NEVER be rejected, regardless
    of charge size — the schema's CHECK trigger guarantees we never see
    <= 0; None is the sentinel for 'no limit configured'."""
    rl = TokenRateLimiter(window_s=10.0)
    assert await rl.check_and_charge("tok-E", None, n_tokens=10**9) is True
    # Repeated giant requests must continue to pass.
    for _ in range(100):
        assert await rl.check_and_charge("tok-E", None, n_tokens=10**6) is True


async def test_rate_limiter_isolates_tokens():
    """One token blowing through its budget must NOT affect another token."""
    rl = TokenRateLimiter(window_s=10.0)
    # Saturate tok-X.
    assert await rl.check_and_charge("tok-X", 10, n_tokens=100, now=1000.0) is True
    assert await rl.check_and_charge("tok-X", 10, n_tokens=10, now=1000.0) is False
    # tok-Y is independent.
    assert await rl.check_and_charge("tok-Y", 10, n_tokens=100, now=1000.0) is True


# ---------------------------------------------------------------------------
# Priority scheduler
# ---------------------------------------------------------------------------


async def test_scheduler_fast_path_when_idle():
    """First acquirer when nobody's holding goes through the fast path
    without touching the heap."""
    sched = PriorityScheduler()
    async with sched.acquire(priority=5):
        assert sched._queue_size_for_test() == 0


async def test_scheduler_strict_priority_ordering():
    """When multiple waiters are queued, the highest priority MUST be
    served first — even if a lower-priority waiter was enqueued earlier.

    Forced to a single slot (max_inflight=1) so the engine saturates after
    the holder and every other acquirer must queue — admission ordering
    only manifests under contention.
    """
    sched = PriorityScheduler(max_inflight=1)
    sched._manual_release = True  # tests drive the release manually

    # Holder enters first to force everyone else to queue.
    holder_done = asyncio.Event()
    holder_release_signal = asyncio.Event()

    async def holder():
        async with sched.acquire(priority=5):
            holder_done.set()
            await holder_release_signal.wait()

    holder_task = asyncio.create_task(holder())
    await holder_done.wait()  # holder is inside the CM, others will queue

    # Now enqueue 10x priority-0 BEFORE the single priority-9, so FIFO
    # would serve them first. STRICT priority must defeat FIFO here.
    served: list[int] = []

    async def waiter(prio: int):
        async with sched.acquire(priority=prio):
            served.append(prio)

    waiters = [asyncio.create_task(waiter(0)) for _ in range(10)]
    # Wait until they're all inside the queue (sched._queue_size becomes 10).
    while sched._queue_size_for_test() < 10:  # noqa: ASYNC110 — see module docstring
        await asyncio.sleep(0)
    # Enqueue the priority-9 waiter LAST in time.
    high = asyncio.create_task(waiter(9))
    while sched._queue_size_for_test() < 11:  # noqa: ASYNC110 — see module docstring
        await asyncio.sleep(0)

    # Release holder → next-highest-priority waiter (the priority-9) fires.
    holder_release_signal.set()
    await holder_task
    # Manually step the heap one slot.
    await sched._release_for_test()
    # Allow the woken priority-9 waiter to enter its CM and append.
    await high
    # Now drain the rest of the heap so the test cleans up.
    for _ in range(10):
        await sched._release_for_test()
    await asyncio.gather(*waiters)

    # The first served (after the seed holder) must be the priority-9.
    assert served[0] == 9, f"strict priority violated: served order = {served}"
    # The remaining 10 are all priority-0 (FIFO inside the tier — order
    # of enqueue is preserved, but we don't assert that here because the
    # tier-internal ordering is covered by the FIFO test below).
    assert served[1:] == [0] * 10


async def test_scheduler_fifo_within_same_priority_tier():
    """Same-priority waiters must be served in enqueue order."""
    sched = PriorityScheduler(max_inflight=1)
    sched._manual_release = True
    holder_done = asyncio.Event()
    holder_release_signal = asyncio.Event()

    async def holder():
        async with sched.acquire(priority=5):
            holder_done.set()
            await holder_release_signal.wait()

    holder_task = asyncio.create_task(holder())
    await holder_done.wait()

    served: list[str] = []

    async def waiter(label: str):
        async with sched.acquire(priority=3):
            served.append(label)

    labels = list("ABCDE")
    tasks = []
    for label in labels:
        t = asyncio.create_task(waiter(label))
        tasks.append(t)
        # Spin until this waiter has actually enqueued before launching
        # the next — otherwise the enqueue order is non-deterministic.
        while sched._queue_size_for_test() < len(tasks):  # noqa: ASYNC110 — see module docstring
            await asyncio.sleep(0)

    holder_release_signal.set()
    await holder_task
    for _ in range(len(labels)):
        await sched._release_for_test()
    await asyncio.gather(*tasks)

    assert served == labels


async def test_scheduler_cancelled_waiter_skipped():
    """A waiter that's cancelled mid-wait must not block the next live
    waiter from being woken — the release() loop drains tombstones at
    the queue head before signalling a real event."""
    sched = PriorityScheduler(max_inflight=1)
    sched._manual_release = True
    holder_done = asyncio.Event()
    holder_release_signal = asyncio.Event()

    async def holder():
        async with sched.acquire(priority=5):
            holder_done.set()
            await holder_release_signal.wait()

    holder_task = asyncio.create_task(holder())
    await holder_done.wait()

    served: list[str] = []

    async def cancellable_waiter():
        async with sched.acquire(priority=8):  # higher than other
            served.append("cancelled-should-never-append")

    async def live_waiter():
        async with sched.acquire(priority=2):
            served.append("live")

    c = asyncio.create_task(cancellable_waiter())
    while sched._queue_size_for_test() < 1:  # noqa: ASYNC110 — see module docstring
        await asyncio.sleep(0)
    live = asyncio.create_task(live_waiter())
    while sched._queue_size_for_test() < 2:  # noqa: ASYNC110 — see module docstring
        await asyncio.sleep(0)

    # Cancel the high-priority waiter before holder releases.
    c.cancel()
    with pytest.raises(asyncio.CancelledError):
        await c

    holder_release_signal.set()
    await holder_task
    # Step the scheduler — first release must skip the tombstone for the
    # cancelled priority-8 and hand the slot to the priority-2 live waiter.
    await sched._release_for_test()
    await live

    assert served == ["live"]


async def test_scheduler_clamps_priority_out_of_range():
    """A buggy caller passing priority=99 must not crash the scheduler —
    we clamp to 0..9 as defence-in-depth alongside the DB CHECK."""
    sched = PriorityScheduler()
    async with sched.acquire(priority=99):
        pass
    async with sched.acquire(priority=-5):
        pass


# ---------------------------------------------------------------------------
# Per-engine multi-slot admission (#173 part A)
# ---------------------------------------------------------------------------


async def test_scheduler_admits_up_to_max_inflight_concurrently():
    """With max_inflight=N, the first N acquirers proceed concurrently
    without ever touching the heap; only the (N+1)th queues. This is the
    core of #173: the proxy must let vLLM batch many requests at once
    instead of serialising them one-at-a-time."""
    sched = PriorityScheduler(max_inflight=3)
    sched._manual_release = True

    inside = asyncio.Event()
    release = asyncio.Event()
    entered = 0

    async def holder():
        nonlocal entered
        async with sched.acquire(priority=5):
            entered += 1
            if entered == 3:
                inside.set()
            await release.wait()

    holders = [asyncio.create_task(holder()) for _ in range(3)]
    await inside.wait()  # all 3 admitted simultaneously
    assert sched._inflight_for_test() == 3
    assert sched._queue_size_for_test() == 0  # none had to queue

    # A 4th acquirer must now queue — capacity is full.
    async def fourth():
        async with sched.acquire(priority=5):
            pass

    q = asyncio.create_task(fourth())
    while sched._queue_size_for_test() < 1:  # noqa: ASYNC110 — see module docstring
        await asyncio.sleep(0)
    assert sched._queue_size_for_test() == 1

    # Release the holders; step the heap once to admit the queued 4th.
    release.set()
    await asyncio.gather(*holders)
    await sched._release_for_test()
    await q


async def test_scheduler_engines_are_isolated():
    """A saturated engine A must NOT block an acquirer bound for engine B —
    they are different GPUs with independent queues and counters."""
    sched = PriorityScheduler(max_inflight=1)
    sched._manual_release = True

    a_inside = asyncio.Event()
    a_release = asyncio.Event()

    async def hold_a():
        async with sched.acquire(priority=5, engine_key="engine-A"):
            a_inside.set()
            await a_release.wait()

    a_task = asyncio.create_task(hold_a())
    await a_inside.wait()  # engine-A is now at capacity (1/1)

    # engine-B must admit immediately via its own fast path despite A being full.
    b_served = asyncio.Event()

    async def hit_b():
        async with sched.acquire(priority=0, engine_key="engine-B"):
            b_served.set()

    b_task = asyncio.create_task(hit_b())
    await asyncio.wait_for(b_served.wait(), timeout=1.0)
    assert sched._inflight_for_test("engine-A") == 1
    await b_task

    a_release.set()
    await a_task


async def test_scheduler_inflight_bookkeeping_drops_idle_engines():
    """After every holder releases, the engine's bookkeeping is removed so a
    long-lived warden doesn't accumulate one dict entry per ever-seen
    engine."""
    sched = PriorityScheduler(max_inflight=2)
    async with sched.acquire(priority=5, engine_key="ephemeral"):
        assert sched._inflight_for_test("ephemeral") == 1
    assert sched._inflight_for_test("ephemeral") == 0
    assert sched._queue_size_for_test("ephemeral") == 0


def test_scheduler_max_inflight_from_env(monkeypatch):
    """VW_PROXY_MAX_INFLIGHT sets the per-engine cap; non-positive / garbage
    values fall back to the default (16) rather than dead-locking at zero."""
    monkeypatch.setenv("VW_PROXY_MAX_INFLIGHT", "32")
    assert PriorityScheduler().max_inflight == 32
    monkeypatch.setenv("VW_PROXY_MAX_INFLIGHT", "0")
    assert PriorityScheduler().max_inflight == 16
    monkeypatch.setenv("VW_PROXY_MAX_INFLIGHT", "not-a-number")
    assert PriorityScheduler().max_inflight == 16
    monkeypatch.delenv("VW_PROXY_MAX_INFLIGHT", raising=False)
    assert PriorityScheduler().max_inflight == 16
    # An explicit constructor arg always wins over the env.
    monkeypatch.setenv("VW_PROXY_MAX_INFLIGHT", "32")
    assert PriorityScheduler(max_inflight=4).max_inflight == 4
