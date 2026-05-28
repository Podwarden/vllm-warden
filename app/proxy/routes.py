import json
import time

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

router = APIRouter(prefix="/v1", tags=["proxy"])


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
        await _release_slot()
        raise

    if is_stream:
        async def gen():
            buf = b""
            accumulated = ""
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
                    buf += chunk
                    while b"\n\n" in buf:
                        line, buf = buf.split(b"\n\n", 1)
                        delta = _parse_sse_delta(line)
                        if delta:
                            accumulated += delta
            finally:
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
        content = await resp.aread()
        await resp.aclose()
        await client.aclose()
        completion_tokens = 0
        try:
            completion_tokens = json.loads(content).get("usage", {}).get("completion_tokens", 0)
        except Exception:
            pass
        await _record_counters(request, model, token_id, prompt_tokens, completion_tokens)
    finally:
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
