import asyncio
import sys

import httpx


async def test_fake_vllm_health_and_completions():
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "tests.fakes.fake_vllm",
        "--port", "18001",
        "--served-model-name", "fake-model",
    )
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            for _ in range(50):
                try:
                    r = await c.get("http://127.0.0.1:18001/health")
                    if r.status_code == 200:
                        break
                except Exception:
                    await asyncio.sleep(0.1)
            else:
                raise RuntimeError("fake_vllm did not become healthy")

            r = await c.get("http://127.0.0.1:18001/v1/models")
            assert r.status_code == 200
            assert r.json()["data"][0]["id"] == "fake-model"

            r = await c.post(
                "http://127.0.0.1:18001/v1/chat/completions",
                json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert r.status_code == 200
            data = r.json()
            assert data["choices"][0]["message"]["content"]
            assert data["usage"]["prompt_tokens"] >= 1
    finally:
        proc.terminate()
        await proc.wait()
