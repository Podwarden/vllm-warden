import os
from dataclasses import dataclass
from pathlib import Path

_TRUTHY = {"1", "true", "yes", "on"}

# Recognised VW_RUNAWAY_MODE values; anything else falls back to the off-safe
# default so a typo can never silently arm enforcement on shared prod.
_RUNAWAY_MODES = {"off", "log", "enforce"}


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
    # God mode (live prompt/output viewer) — OFF by default. When false the
    # proxy hot path is byte-identical to pre-godmode: no hub call happens at
    # any tap. All bounds are read once at startup into GodModeHub.
    godmode_enabled: bool = False
    godmode_ring_events: int = 2000
    godmode_ring_chars: int = 4_000_000
    godmode_max_prompt_chars: int = 16000
    # Display-only capture window: when a prompt exceeds godmode_max_prompt_chars
    # the tap keeps the head PLUS this many chars from the tail (joined by an
    # elision marker), so the newest turn — which lives at the tail — survives
    # even under a giant repeated system prompt. Never affects the forwarded
    # request. 0 = pure head-slice (pre-#tail behaviour).
    godmode_prompt_tail_chars: int = 4000
    # God-mode media store (spec 2026-08-03 — inline images). Bounds for the
    # out-of-band image blob store; only consulted when godmode_enabled.
    godmode_media_store_chars: int = 64_000_000
    godmode_max_image_chars: int = 14_000_000
    godmode_max_images_per_req: int = 16
    # Content logging (diagnostic) — see app/proxy/content_log.py. Disabled by
    # default; when off the proxy forward path is byte-identical to a build
    # without it. Hard-scoped to an explicit token allowlist so this can run on
    # a shared server without ever logging all traffic: a request is logged
    # only when content_log_enabled is True AND its token id is in
    # content_log_tokens. The prompt/completion fields are capped at
    # content_log_max_chars so the file can't grow unbounded on very long
    # generations.
    content_log_enabled: bool = False
    content_log_tokens: frozenset[str] = frozenset()
    content_log_path: Path = Path("/data/logs/content.jsonl")
    content_log_max_chars: int = 40000
    # Runaway-generation detector (app/proxy/runaway.py) — see
    # docs/superpowers/specs/2026-07-21-runaway-detector-design.md. Off by
    # default: when runaway_mode == "off" the proxy forward path is
    # byte-identical to a build without the detector — no detector is
    # constructed and NO upstream stream-forcing happens. "log" runs the
    # detector and records a would-trip incident via the content-logger sink
    # without interrupting; "enforce" tears the generation down on a trip. The
    # three thresholds carry conservative defaults so a confident pathology
    # signal is required before any teardown.
    runaway_mode: str = "off"
    runaway_think_budget: int = 24000
    runaway_repeat_max: int = 6
    runaway_hard_max: int = 96000

    # --- Engine watchdog (2026-08-17) --------------------------------------
    # _watch_exit waits on the `vllm serve` WRAPPER, but vLLM v1 runs EngineCore
    # and the TP workers underneath it. Four times on 2026-08-17 EngineCore died
    # while the wrapper stayed alive, so the wait never returned, the row kept
    # status='loaded', and the proxy forwarded to a dead engine indefinitely.
    # The watchdog probes the engine's OWN /health instead.
    watchdog_enabled: bool = True
    watchdog_interval_s: float = 30.0
    watchdog_failure_threshold: int = 3      # ~90s before declaring death
    watchdog_max_restarts: int = 3           # per window, then leave it failed
    watchdog_restart_window_s: float = 3600.0
    watchdog_crash_keep: int = 20            # crash dirs retained per model
    # Bytes of engine log copied into each crash dir. Was the WHOLE log, which
    # made /data occupancy `watchdog_crash_keep x current log size` -- a moving
    # target that grows with uptime, not a ceiling. Measured 2026-08-18: 20
    # crash dirs x a 6.8 MB log = 131 MB, all of the volume's usage. The log is
    # append-only across loads and never rotated, so a longer-lived engine
    # simply makes every copy bigger. A tail is what anyone reads anyway -- the
    # crash is at the END of the file.
    watchdog_crash_log_bytes: int = 4 * 1024 * 1024
    # Rotate the live engine log once it passes this, keeping one .1 alongside.
    # /data also holds the SQLite DB, and filling it takes the database down
    # with it -- the 2026-06-15 ENOSPC incident.
    engine_log_max_bytes: int = 32 * 1024 * 1024

    @property
    def db_path(self) -> Path:
        return self.data_dir / "vllm-warden.db"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def crashes_dir(self) -> Path:
        return self.data_dir / "crashes"

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
    watchdog_enabled = _truthy(os.environ.get("VW_WATCHDOG_ENABLED", "1"))
    watchdog_interval_s = float(os.environ.get("VW_WATCHDOG_INTERVAL_S", "30.0"))
    watchdog_failure_threshold = int(
        os.environ.get("VW_WATCHDOG_FAILURE_THRESHOLD", "3")
    )
    watchdog_max_restarts = int(os.environ.get("VW_WATCHDOG_MAX_RESTARTS", "3"))
    watchdog_restart_window_s = float(
        os.environ.get("VW_WATCHDOG_RESTART_WINDOW_S", "3600.0")
    )
    watchdog_crash_keep = int(os.environ.get("VW_WATCHDOG_CRASH_KEEP", "20"))
    watchdog_crash_log_bytes = int(
        os.environ.get("VW_WATCHDOG_CRASH_LOG_BYTES", str(4 * 1024 * 1024))
    )
    engine_log_max_bytes = int(
        os.environ.get("VW_ENGINE_LOG_MAX_BYTES", str(32 * 1024 * 1024))
    )
    godmode_enabled = _truthy(os.environ.get("VW_GODMODE_ENABLED", ""))
    godmode_ring_events = int(os.environ.get("VW_GODMODE_RING_EVENTS", "2000"))
    godmode_ring_chars = int(os.environ.get("VW_GODMODE_RING_CHARS", "4000000"))
    godmode_max_prompt_chars = int(
        os.environ.get("VW_GODMODE_MAX_PROMPT_CHARS", "16000")
    )
    godmode_prompt_tail_chars = int(
        os.environ.get("VW_GODMODE_PROMPT_TAIL_CHARS", "4000")
    )
    godmode_media_store_chars = int(
        os.environ.get("VW_GODMODE_MEDIA_STORE_CHARS", "64000000")
    )
    godmode_max_image_chars = int(
        os.environ.get("VW_GODMODE_MAX_IMAGE_CHARS", "14000000")
    )
    godmode_max_images_per_req = int(
        os.environ.get("VW_GODMODE_MAX_IMAGES_PER_REQ", "16")
    )
    request_max_wall_s = float(os.environ.get("VW_REQUEST_MAX_WALL_S", "0.0"))
    content_log_enabled = _truthy(os.environ.get("VW_CONTENT_LOG_ENABLED", ""))
    # Comma-separated token ids; blanks dropped. Empty/unset => log nothing
    # (the gate in content_log.should_log also requires the flag above).
    content_log_tokens = frozenset(
        t.strip()
        for t in os.environ.get("VW_CONTENT_LOG_TOKENS", "").split(",")
        if t.strip()
    )
    content_log_path = Path(
        os.environ.get("VW_CONTENT_LOG_PATH", "/data/logs/content.jsonl")
    )
    content_log_max_chars = int(os.environ.get("VW_CONTENT_LOG_MAX_CHARS", "40000"))
    # Unknown / typo'd modes fall back to "off" so the forward path stays
    # byte-identical rather than silently arming enforcement on shared prod.
    runaway_mode = os.environ.get("VW_RUNAWAY_MODE", "off").strip().lower()
    if runaway_mode not in _RUNAWAY_MODES:
        runaway_mode = "off"
    runaway_think_budget = int(os.environ.get("VW_RUNAWAY_THINK_BUDGET", "24000"))
    runaway_repeat_max = int(os.environ.get("VW_RUNAWAY_REPEAT_MAX", "6"))
    runaway_hard_max = int(os.environ.get("VW_RUNAWAY_HARD_MAX", "96000"))
    return Settings(
        data_dir=data_dir,
        hf_cache_dir=hf_cache_dir,
        cookie_secret=secret,
        container_gpu_count=gpu_count,
        allowed_origins=allowed_origins,
        trust_proxy_origin=trust_proxy_origin,
        warmup_probe_timeout_s=warmup_probe_timeout_s,
        engine_driver=engine_driver,
        watchdog_enabled=watchdog_enabled,
        watchdog_interval_s=watchdog_interval_s,
        watchdog_failure_threshold=watchdog_failure_threshold,
        watchdog_max_restarts=watchdog_max_restarts,
        watchdog_restart_window_s=watchdog_restart_window_s,
        watchdog_crash_keep=watchdog_crash_keep,
        watchdog_crash_log_bytes=watchdog_crash_log_bytes,
        engine_log_max_bytes=engine_log_max_bytes,
        godmode_enabled=godmode_enabled,
        godmode_ring_events=godmode_ring_events,
        godmode_ring_chars=godmode_ring_chars,
        godmode_max_prompt_chars=godmode_max_prompt_chars,
        godmode_prompt_tail_chars=godmode_prompt_tail_chars,
        godmode_media_store_chars=godmode_media_store_chars,
        godmode_max_image_chars=godmode_max_image_chars,
        godmode_max_images_per_req=godmode_max_images_per_req,
        request_max_wall_s=request_max_wall_s,
        content_log_enabled=content_log_enabled,
        content_log_tokens=content_log_tokens,
        content_log_path=content_log_path,
        content_log_max_chars=content_log_max_chars,
        runaway_mode=runaway_mode,
        runaway_think_budget=runaway_think_budget,
        runaway_repeat_max=runaway_repeat_max,
        runaway_hard_max=runaway_hard_max,
    )
