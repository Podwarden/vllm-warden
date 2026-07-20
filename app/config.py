import os
from dataclasses import dataclass
from pathlib import Path

_TRUTHY = {"1", "true", "yes", "on"}


def _truthy(raw: str) -> bool:
    return raw.strip().lower() in _TRUTHY


def _parse_origins(raw: str) -> tuple[str, ...]:
    out = []
    for part in raw.split(","):
        cleaned = part.strip().rstrip("/")
        if cleaned:
            out.append(cleaned)
    return tuple(out)


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    hf_cache_dir: Path
    cookie_secret: str
    container_gpu_count: int
    bind_host: str = "0.0.0.0"
    bind_port: int = 8080
    load_timeout_s: float = 600.0
    # Max time the warmup verification probe waits for a successful
    # POST /v1/completions before marking the load failed. Closes the
    # window between /health 200 and actual serving readiness (e.g.
    # Qwen3-VL's _warmup_mm_processor). Configurable via env override.
    warmup_probe_timeout_s: float = 60.0
    # Server-side wall-clock backstop for a single proxied request (streaming
    # or not). 0.0 = disabled (default; behaviour-identical to pre-reaper).
    # When > 0, a streamed generation that has run longer than this many
    # seconds is torn down: the upstream vLLM socket is closed so the engine
    # aborts the generation and frees its KV blocks, and the scheduler slot is
    # released. This is the guaranteed reaper for the "client abandoned the
    # request but the TCP connection stayed transport-alive" case, where no
    # http.disconnect ever reaches uvicorn and Starlette never cancels the
    # body iterator — so nothing else can reclaim the slot. Set to a real
    # value (e.g. 600) in the deployment env to bound worst-case slot pinning.
    request_max_wall_s: float = 0.0
    session_access_ttl_minutes: int = 15
    session_refresh_ttl_days: int = 7
    allowed_origins: tuple[str, ...] = ("http://localhost:3000",)
    trust_proxy_origin: bool = False
    sse_ticket_ttl_seconds: int = 60
    # #160 — which engine driver Supervisor uses to run the vLLM engine.
    # "local" = in-container subprocess (default, behaviour-identical to
    # pre-#160); "docker" = sibling container via the host docker socket
    # (requires VLLM_ENGINE_IMAGE; image-channel resolution lands in P2/#161).
    engine_driver: str = "local"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "vllm-warden.db"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def hf_token_path(self) -> Path:
        return self.data_dir / "hf-token"


def load_settings() -> Settings:
    data_dir = Path(os.environ.get("VW_DATA_DIR", "/data"))
    # HF model cache lives on its own PVC in production so the data PVC can
    # stay small. Compose mounts the cache at /root/.cache/huggingface to
    # match the HF library default; VW_HF_CACHE_DIR overrides that path.
    hf_cache_dir = Path(
        os.environ.get("VW_HF_CACHE_DIR", "/root/.cache/huggingface")
    )
    secret = os.environ.get("VW_COOKIE_SECRET")
    if not secret or len(secret) < 32:
        raise RuntimeError("VW_COOKIE_SECRET must be set and >=32 chars")
    gpu_count = int(os.environ.get("VW_CONTAINER_GPU_COUNT", "0"))
    raw_origins = os.environ.get("VW_FRONTEND_ORIGIN", "http://localhost:3000")
    # An explicitly-empty VW_FRONTEND_ORIGIN ("" or whitespace/commas only)
    # parses to () and falls back to the localhost default here — by design,
    # so a blank-but-set env var doesn't silently lock admins out of refresh.
    # This is NOT a fail-open hole: the fail-closed contract lives at the
    # Settings level — constructing Settings(allowed_origins=()) directly (with
    # trust_proxy_origin=False) still blocks every origin. Env-parse leniency
    # and the Settings-level fail-closed guarantee are deliberately separate.
    allowed_origins = _parse_origins(raw_origins) or ("http://localhost:3000",)
    trust_proxy_origin = _truthy(os.environ.get("VW_TRUST_PROXY_ORIGIN", ""))
    warmup_probe_timeout_s = float(
        os.environ.get("VW_WARMUP_PROBE_TIMEOUT_S", "60.0")
    )
    engine_driver = os.environ.get("VW_ENGINE_DRIVER", "local")
    request_max_wall_s = float(os.environ.get("VW_REQUEST_MAX_WALL_S", "0.0"))
    return Settings(
        data_dir=data_dir,
        hf_cache_dir=hf_cache_dir,
        cookie_secret=secret,
        container_gpu_count=gpu_count,
        allowed_origins=allowed_origins,
        trust_proxy_origin=trust_proxy_origin,
        warmup_probe_timeout_s=warmup_probe_timeout_s,
        engine_driver=engine_driver,
        request_max_wall_s=request_max_wall_s,
    )
