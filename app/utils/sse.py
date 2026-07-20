"""Helpers for Server-Sent Events (SSE) responses.

Centralised here so every ``StreamingResponse(media_type="text/event-stream")``
in the codebase gets the same anti-buffering headers without each route
re-inventing them — a regression that would let one new endpoint silently
re-introduce the proxy-buffering bug fixed for log streams in #50.

See ``sse_headers`` below for the rationale on each header.
"""

from __future__ import annotations


def sse_headers() -> dict[str, str]:
    """Return headers that defeat reverse-proxy buffering of SSE streams.

    nginx (and Caddy/Traefik, which honour the same hint) buffer
    ``text/event-stream`` responses by default, causing the client to see
    a wall of buffered chunks at proxy-buffer intervals rather than the
    intended real-time stream. ``X-Accel-Buffering: no`` disables this on
    nginx and its lookalikes; ``Cache-Control: no-cache`` is a defensive
    cover for CDNs/proxies that don't recognise the SSE media type and
    would otherwise apply default caching to the response body.

    Callers should pass the returned dict to ``StreamingResponse`` via
    ``headers=sse_headers()``. If the endpoint needs additional headers,
    merge with ``{**sse_headers(), **extra}`` so the buffering hints
    cannot be accidentally clobbered.
    """
    return {
        "X-Accel-Buffering": "no",
        "Cache-Control": "no-cache",
    }
