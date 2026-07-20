"""Tiny aiohttp server mimicking vLLM /v1/* and /health for integration tests."""
import argparse
import json
import time

from aiohttp import web


async def health(req):
    return web.Response(status=200, text="ok")


async def models(req):
    name = req.app["served_name"]
    return web.json_response({"data": [{"id": name, "object": "model"}]})


async def chat_completions(req):
    body = await req.json()
    msg = body["messages"][-1]["content"]
    if body.get("stream"):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(req)
        for tok in ["hi", " there"]:
            payload = {
                "choices": [{"delta": {"content": tok}}],
                "model": req.app["served_name"],
            }
            await resp.write(f"data: {json.dumps(payload)}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        return resp
    return web.json_response({
        "id": "fake-1",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.app["served_name"],
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": f"echo: {msg}"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": len(msg), "completion_tokens": 4, "total_tokens": len(msg) + 4},
    })


async def completions(req):
    """OpenAI-compat /v1/completions used by integration tests.

    Always streams (the caller sets stream=true). Yields a short canned
    response with a brief sleep between chunks so TTFT is measurable in
    the integration test."""
    import asyncio
    body = await req.json()
    prompt = body.get("prompt", "")
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else ""
    if body.get("stream"):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(req)
        # Tiny inter-chunk delay so the runner can measure non-zero TTFT
        # without slowing the test to a crawl. ~2 ms per chunk × 3 chunks.
        for tok in ["hi", " there", "!"]:
            await asyncio.sleep(0.002)
            payload = {
                "choices": [{"text": tok, "index": 0}],
                "model": req.app["served_name"],
            }
            await resp.write(f"data: {json.dumps(payload)}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        return resp
    return web.json_response({
        "id": "fake-1",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.app["served_name"],
        "choices": [{
            "index": 0,
            "text": f"echo: {prompt[:50]}",
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": len(prompt),
            "completion_tokens": 4,
            "total_tokens": len(prompt) + 4,
        },
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18001)
    ap.add_argument("--served-model-name", default="fake-model")
    args = ap.parse_args()

    app = web.Application()
    app["served_name"] = args.served_model_name
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", chat_completions)
    app.router.add_post("/v1/completions", completions)
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
