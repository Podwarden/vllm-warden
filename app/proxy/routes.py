import asyncio
import json
import time
from datetime import UTC, datetime
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from app.db.database import open_db
from app.db.repos.counters import CountersRepo
from app.db.repos.models import ModelRepo
from app.db.repos.samples import SamplesRepo
from app.db.repos.tokens import TokenRow, TokenUsageRepo
from app.proxy.auth import require_bearer, token_allows
from app.proxy.envelope_hint import enrich_5xx_from_db
from app.proxy.request_registry import LiveRequest

router = APIRouter(prefix="/v1", tags=["proxy"])

# Live-stats registry throttle: refresh a streaming request's coarse
# completion-token estimate at most this often, to keep the hot path cheap
# (no per-delta tokenize — see docs/live-stats-spec.md § "Plane B").
_LIVE_UPDATE_INTERVAL_S = 0.5


def _client_ip(request: Request) -> str | None:
    """First X-Forwarded-For hop, else X-Real-Ip, else the socket peer."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip() or None
    xri = request.headers.get("x-real-ip")
    if xri:
        return xri.strip() or None
    return request.client.host if request.client else None


async def _resolve_target(request: Request, served_name: str):
    settings = request.app.state.settings
    sup = request.app.state.supervisor
    async with open_db(settings.db_path) as db:
        model = await ModelRepo(db).get_by_served_name(served_name)
    if not model or model.status != "loaded":
        raise HTTPException(404, f"model '{served_name}' is not loaded")
    port = sup.get_port(model.id)
    if port is None:
        raise HTTPException(404, f"model '{served_name}' not running")
    # The driver owns where the engine listens: loopback for the
    # in-container subprocess, the engine container's DNS name for the
    # docker driver. Loopback fallback covers a driver without get_host.
    host = sup.get_host(model.id) or "127.0.0.1"
    return model, host, port


def _extract_prompt(body_json) -> str:
    def _text(c):
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            parts = []
            for blk in c:
                if isinstance(blk, dict) and blk.get("type") == "text":
                    parts.append(blk.get("text", ""))
            return "\n".join(parts)
        return ""

    if "messages" in body_json:
        return "\n".join(
            _text(m.get("content", "")) for m in body_json["messages"] if isinstance(m, dict)
        )
    return body_json.get("prompt", "") or ""


def _parse_sse_delta(line: bytes) -> str | None:
    if not line.startswith(b"data:"):
        return None
    payload = line[5:].strip()
    if payload == b"[DONE]":
        return None
    try:
        ev = json.loads(payload)
        return ev["choices"][0].get("delta", {}).get("content") or None
    except Exception:
        return None


async def _record_counters(
    request: Request,
    model,
    token_id: str | None,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    settings = request.app.state.settings
    minute = int(time.time() // 60)
    async with open_db(settings.db_path) as db:
        await CountersRepo(db).increment(
            model_id=model.id,
            token_id=token_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        await SamplesRepo(db).add_model_sample(
            model_id=model.id,
            minute=minute,
            delta_requests=1,
            delta_prompt=prompt_tokens,
            delta_completion=completion_tokens,
        )
        # S5 (#104) — per-token minute-bucket rollup feeding /api/tokens/{id}/usage.
        # Only count when token_id is set (skip system-token-less requests so the
        # NULL key doesn't accumulate orphan rows). Same minute integer as
        # SamplesRepo above for cross-table joins later (Stats v2).
        if token_id is not None:
            await TokenUsageRepo(db).add(
                token_id=token_id,
                minute=minute,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )


async def _forward(request: Request, model, host: str, port: int, path: str, token: TokenRow):
    body = await request.body()
    body_json = json.loads(body) if body else {}
    is_stream = bool(body_json.get("stream"))
    tok_cache = request.app.state.tokenizers
    token_id = token.id
    # Wall-clock origin for the server-side reaper (see settings.request_max_wall_s).
    request_start = time.monotonic()

    prompt_text = _extract_prompt(body_json)
    prompt_tokens = await tok_cache.count(model.hf_repo, prompt_text, trust_remote_code=bool(model.trust_remote_code))

    # S5 (#104) — sliding-window rate limit (per-token; NULL means unlimited).
    # MUST be 429 NOT 503 — OpenAI-compatible clients (openai-python, Vercel AI
    # SDK, LangChain) retry-with-backoff on 429 and surface "model unavailable"
    # on 503. The proxy must not be mistaken for an outage.
    rate_limiter = request.app.state.rate_limiter
    if not await rate_limiter.check_and_charge(
        token_id, token.rate_limit_tps, prompt_tokens,
    ):
        raise HTTPException(
            status_code=429,
            detail=(
                f"rate limit exceeded: token configured for "
                f"{token.rate_limit_tps} tokens/sec over a "
                f"{rate_limiter.window_s:g}s window"
            ),
        )

    # S5 — STRICT priority scheduler in front of vLLM. Acquired here so the
    # slot is held for the full duration of the upstream call (including the
    # streaming-response body), and released in the StreamingResponse's
    # finally block via the `_release` callback we pass through.
    scheduler = request.app.state.scheduler
    # Per-engine admission (#173 part A): key on the model id so each engine
    # admits up to VW_PROXY_MAX_INFLIGHT requests concurrently and a hot
    # engine never blocks a request bound for an idle one. Priority is
    # preserved as the admission ordering within the engine's queue, and
    # pushed into vLLM itself (#173 part B) via the per-request priority field.
    slot_cm = scheduler.acquire(priority=token.priority, engine_key=model.id)
    await slot_cm.__aenter__()
    slot_released = False

    # #173 part B — push the token's priority into the engine itself. vLLM's
    # priority scheduler orders the waiting queue by (priority, arrival) ASCENDING
    # — LOWER value is scheduled first — while the warden convention is 9=highest.
    # Map ``vllm_priority = -warden_priority`` so warden's default 0 stays at
    # vLLM's default 0 (inert: behaves like FCFS), and higher-priority tokens
    # sort ahead of default traffic. Only inject when non-zero so unprioritised
    # requests forward a byte-identical body. Re-serialize so the upstream send
    # carries the field. Never override a priority the client set explicitly.
    if token.priority and "priority" not in body_json:
        body_json["priority"] = -token.priority
        body = json.dumps(body_json).encode()

    async def _release_slot() -> None:
        nonlocal slot_released
        if slot_released:
            return
        slot_released = True
        # Mirror the contextmanager's exit: pass no exception (we already
        # propagated upstream errors before reaching here).
        await slot_cm.__aexit__(None, None, None)

    # Live-stats registry (Plane B): register this in-flight request so
    # GET /api/stats/requests can surface it. FAIL-OPEN — any registry error
    # is swallowed so it can never break the proxied request. ``live_req`` is
    # None if registration failed; every later hook guards on that.
    registry = getattr(request.app.state, "request_registry", None)
    live_req: LiveRequest | None = None
    try:
        if registry is not None:
            live_req = LiveRequest(
                id=uuid4().hex,
                token_id=token.id,
                token_name=token.name,
                client_ip=_client_ip(request),
                model=model.served_model_name,
                path=path,
                prompt_tokens=prompt_tokens,
                max_model_len=model.max_model_len,
                started_monotonic=time.monotonic(),
                started_iso=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            )
            await registry.register(live_req)
    except Exception:
        live_req = None

    async def _deregister() -> None:
        if live_req is None or registry is None:
            return
        try:
            await registry.deregister(live_req.id)
        except Exception:
            pass

    try:
        url = f"http://{host}:{port}{path}"
        client = httpx.AsyncClient(timeout=None)
        upstream = client.build_request(
            request.method, url, content=body,
            headers={k: v for k, v in request.headers.items()
                     if k.lower() not in ("host", "authorization", "content-length")},
        )
        resp = await client.send(upstream, stream=is_stream)
    except BaseException:
        # On send failure release the slot immediately so the next priority
        # waiter is not blocked. Re-raise to FastAPI.
        await _deregister()
        await _release_slot()
        raise

    if is_stream:
        # Server-side reaper budget (0.0 = disabled). Guarantees no single
        # request pins its scheduler slot + KV blocks indefinitely when the
        # downstream client abandons the request but its TCP connection stays
        # transport-alive (writes keep draining at the token rate, so no
        # http.disconnect ever reaches uvicorn and Starlette never cancels
        # this body iterator). See config.Settings.request_max_wall_s.
        max_wall = getattr(request.app.state.settings, "request_max_wall_s", 0.0) or 0.0

        async def gen():
            buf = b""
            accumulated = ""
            # Coarse completion-token estimate for the live registry: count SSE
            # content deltas (vLLM streams ~1 token/delta for chat completions)
            # rather than tokenizing on the hot path. Pushed into the registry
            # at most every _LIVE_UPDATE_INTERVAL_S. The exact count is still
            # recomputed once at the end for accounting (below).
            delta_count = 0
            last_live_update = 0.0
            last_disconnect_check = 0.0
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
                    buf += chunk

                    # --- Reaper + disconnect propagation --------------------
                    # Run EVERY iteration, independent of whether this chunk
                    # carried a parseable content delta (the reasoning phase
                    # emits reasoning_content deltas that _parse_sse_delta
                    # returns None for — the old check lived inside `if delta:`
                    # and was unreachable for that entire phase). On trigger,
                    # break out: the `finally` below aclose()s the upstream
                    # response + httpx client, which makes vLLM abort the
                    # generation and free its KV blocks, and releases the slot.
                    now = time.monotonic()
                    # (a) wall-clock backstop — pure arithmetic, checked always.
                    if max_wall > 0.0 and (now - request_start) >= max_wall:
                        if live_req is not None:
                            try:
                                live_req.orphan = True
                            except Exception:
                                pass
                        break
                    # (b) downstream client gone — throttled receive-poll.
                    if now - last_disconnect_check >= _LIVE_UPDATE_INTERVAL_S:
                        last_disconnect_check = now
                        try:
                            if await request.is_disconnected():
                                if live_req is not None:
                                    try:
                                        live_req.orphan = True
                                    except Exception:
                                        pass
                                break
                        except Exception:
                            pass

                    while b"\n\n" in buf:
                        line, buf = buf.split(b"\n\n", 1)
                        # Any SSE `data:` frame means the engine has left prefill
                        # and is emitting tokens (content OR reasoning_content),
                        # so flip the dashboard phase to decode here rather than
                        # only on content deltas — otherwise a request that is
                        # still streaming reasoning renders as "prefill 0%".
                        if (
                            live_req is not None
                            and live_req.phase != "decode"
                            and line.lstrip().startswith(b"data:")
                        ):
                            try:
                                live_req.phase = "decode"
                            except Exception:
                                pass
                        delta = _parse_sse_delta(line)
                        if delta:
                            accumulated += delta
                            delta_count += 1
                            if live_req is not None:
                                try:
                                    if now - last_live_update >= _LIVE_UPDATE_INTERVAL_S:
                                        live_req.completion_tokens = delta_count
                                        last_live_update = now
                                except Exception:
                                    pass
            finally:
                await _deregister()
                await resp.aclose()
                await client.aclose()
                completion_tokens = await tok_cache.count(model.hf_repo, accumulated, trust_remote_code=bool(model.trust_remote_code))
                await _record_counters(request, model, token_id, prompt_tokens, completion_tokens)
                # Release the scheduler slot only after the body is fully
                # drained — otherwise we'd hand the slot to the next waiter
                # while still hammering vLLM with our SSE chunks.
                await _release_slot()

        return StreamingResponse(
            gen(), status_code=resp.status_code,
            headers={"content-type": resp.headers.get("content-type", "text/event-stream")},
        )

    try:
        # Server-side wall-clock backstop for the non-stream path too (the
        # config contract is "streaming or not"). httpx runs with timeout=None,
        # so a hung upstream that keeps the socket transport-alive would pin the
        # slot forever. When request_max_wall_s > 0, bound the read by the time
        # already spent since request_start; on expiry asyncio.TimeoutError
        # unwinds into the `finally` below, which aclose()s the upstream socket
        # (vLLM aborts + frees KV) and releases the slot. 0.0 = disabled.
        max_wall = getattr(request.app.state.settings, "request_max_wall_s", 0.0) or 0.0
        if max_wall > 0.0:
            remaining = max_wall - (time.monotonic() - request_start)
            content = await asyncio.wait_for(resp.aread(), timeout=max(remaining, 0.0))
        else:
            content = await resp.aread()
        completion_tokens = 0
        try:
            completion_tokens = json.loads(content).get("usage", {}).get("completion_tokens", 0)
        except Exception:
            pass
        # Live registry: reflect the final completion count for the brief
        # window before deregister (non-stream has no observable decode phase).
        if live_req is not None:
            try:
                live_req.completion_tokens = completion_tokens
            except Exception:
                pass
        await _record_counters(request, model, token_id, prompt_tokens, completion_tokens)
    except TimeoutError:
        # Wall-clock backstop fired: the upstream read outlived
        # request_max_wall_s. The ``finally`` below tears down the upstream
        # socket (vLLM aborts + frees KV) and releases the slot; surface a
        # clean 504 to the still-connected client rather than a raw 500.
        raise HTTPException(
            status_code=504, detail="upstream request timed out"
        ) from None
    finally:
        # Close the upstream response/client in ``finally`` so a client
        # disconnect — which raises CancelledError out of ``resp.aread()``
        # above — still tears down the warden->vLLM socket synchronously. That
        # makes vLLM abort the generation and free its KV blocks immediately,
        # rather than leaving them pinned until the httpx objects are GC'd.
        # ``aclose()`` is idempotent, so calling it on the success path is
        # safe; this mirrors the streaming path's teardown. (#184)
        await _deregister()
        await resp.aclose()
        await client.aclose()
        await _release_slot()

    # Error enrichment: on upstream 5xx, attempt to attach a last_error
    # hint from the models row to the JSON error envelope. Pure-function
    # gates inside ``enrich_5xx_from_db`` (status >= 500, JSON envelope
    # shape, last_error present) keep this cheap when nothing matches.
    if resp.status_code >= 500:
        try:
            gpu_set = ",".join(str(i) for i in sorted(model.gpu_indices or []))
            asked = {
                "concurrency": body_json.get("n") or body_json.get("best_of"),
                "max_new": body_json.get("max_tokens"),
            }
            enriched = await enrich_5xx_from_db(
                db_path=request.app.state.settings.db_path,
                model_id=model.id,
                gpu_set=gpu_set,
                status_code=resp.status_code,
                body_bytes=content,
                asked=asked,
            )
            if enriched is not None:
                content = enriched
        except Exception:
            pass

    return Response(
        content=content, status_code=resp.status_code,
        headers={"content-type": resp.headers.get("content-type", "application/json")},
    )


@router.post("/chat/completions")
async def chat_completions(request: Request, token: TokenRow = Depends(require_bearer)):
    body_bytes = await request.body()
    body_json = json.loads(body_bytes) if body_bytes else {}
    served_name = body_json.get("model")
    if not served_name:
        raise HTTPException(400, "missing 'model' field")
    if not token_allows(token, served_name):
        raise HTTPException(403, f"token not allowed for model '{served_name}'")
    model, host, port = await _resolve_target(request, served_name)

    # Re-set the body on the request so _forward can re-read it.
    async def _receive():
        return {"type": "http.request", "body": body_bytes, "more_body": False}
    request._receive = _receive

    return await _forward(request, model, host, port, "/v1/chat/completions", token)


@router.post("/completions")
async def completions(request: Request, token: TokenRow = Depends(require_bearer)):
    body_bytes = await request.body()
    body_json = json.loads(body_bytes) if body_bytes else {}
    served_name = body_json.get("model")
    if not served_name:
        raise HTTPException(400, "missing 'model' field")
    if not token_allows(token, served_name):
        raise HTTPException(403, f"token not allowed for model '{served_name}'")
    model, host, port = await _resolve_target(request, served_name)

    async def _receive():
        return {"type": "http.request", "body": body_bytes, "more_body": False}
    request._receive = _receive

    return await _forward(request, model, host, port, "/v1/completions", token)


@router.get("/models")
async def list_models(request: Request, token: TokenRow = Depends(require_bearer)):
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        rows = await ModelRepo(db).list_all()
    loaded = [r for r in rows if r.status == "loaded" and token_allows(token, r.served_model_name)]
    return {
        "object": "list",
        "data": [
            {"id": r.served_model_name, "object": "model", "owned_by": "vllm-warden"}
            for r in loaded
        ],
    }
