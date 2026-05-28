import inspect
import time
from unittest.mock import AsyncMock, patch

import pytest

from app.runtime.supervisor import DEFAULT_HEALTH_TIMEOUT_S, wait_for_health


@pytest.mark.asyncio
async def test_wait_for_health_returns_when_endpoint_200():
    responses = [Exception("conn refused"), Exception("conn refused"), 200]

    async def fake_get(url, timeout):  # noqa: ASYNC109
        v = responses.pop(0)
        if isinstance(v, Exception):
            raise v
        class R:
            status_code = v
        return R()

    with patch("app.runtime.supervisor._http_get", new=AsyncMock(side_effect=fake_get)):
        ok = await wait_for_health(port=10001, timeout_s=5, interval_s=0.05)
    assert ok is True


@pytest.mark.asyncio
async def test_wait_for_health_times_out():
    with patch("app.runtime.supervisor._http_get",
               new=AsyncMock(side_effect=Exception("never"))):
        ok = await wait_for_health(port=10001, timeout_s=0.2, interval_s=0.05)
    assert ok is False


def test_wait_for_health_default_timeout_matches_module_constant():
    """#99 — the default value of ``timeout_s`` must be sourced from the
    module-level ``DEFAULT_HEALTH_TIMEOUT_S`` constant, not a hard-coded
    literal at the function signature. The constant is the single source
    of truth so callers can reference it (e.g. settings layer) without
    parroting the magic number — and so tests can assert the contract
    without coupling to ``600.0``.
    """
    sig = inspect.signature(wait_for_health)
    assert sig.parameters["timeout_s"].default == DEFAULT_HEALTH_TIMEOUT_S


@pytest.mark.asyncio
async def test_wait_for_health_respects_injected_timeout_s():
    """#99 — passing a tiny timeout_s MUST cause the loop to give up
    quickly even though the module default is 600s. Regression guard
    against a future refactor that would hard-code 600.0 inside the
    loop and ignore the parameter.
    """
    call_count = {"n": 0}

    async def slow_fake(url, timeout):  # noqa: ASYNC109
        call_count["n"] += 1
        raise Exception("conn refused")

    start = time.monotonic()
    with patch("app.runtime.supervisor._http_get", new=AsyncMock(side_effect=slow_fake)):
        ok = await wait_for_health(port=10001, timeout_s=0.1, interval_s=0.02)
    elapsed = time.monotonic() - start
    assert ok is False
    # Should have given up well before the module default would have us
    # waiting; a generous upper bound to avoid CI flakes is 1 s.
    assert elapsed < 1.0, f"expected fast failure with timeout_s=0.1, took {elapsed:.3f}s"
    assert call_count["n"] >= 1
