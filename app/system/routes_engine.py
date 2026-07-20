"""GET /api/system/engine — the active engine driver's capability.

The frontend Try-stack panel (#177) needs to know whether the running
deployment can swap the engine container image to a pinned vLLM version. Under
the default in-container subprocess driver it CANNOT — the vLLM version is
fixed by the warden image — so the version selector must be disabled and the
operator told why, instead of silently launching the warden-baked version.

``vllm_version`` is the vLLM package version baked into THIS image. We read it
cheaply via ``importlib.metadata`` (no ``import vllm`` — that drags in torch
and is far too heavy for a metadata read) and cache it at import time. It is
``None`` in environments where vLLM isn't installed (dev shell, pytest).
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from fastapi import APIRouter, Depends, Request

from app.auth.deps import require_jwt

try:
    _VLLM_VERSION: str | None = version("vllm")
except PackageNotFoundError:
    _VLLM_VERSION = None

router = APIRouter()


@router.get("/api/system/engine")
async def get_engine(request: Request, _user: str = Depends(require_jwt)) -> dict:
    driver = getattr(request.app.state, "supervisor", None)
    driver = getattr(driver, "_driver", None)
    # Unknown/test stand-in drivers default to capable so we never wrongly
    # block them — mirrors the defensive read in Supervisor.load.
    supports = getattr(driver, "supports_engine_image", True)
    return {
        "driver": "docker" if supports else "subprocess",
        "supports_version_select": supports,
        "vllm_version": _VLLM_VERSION,
    }
