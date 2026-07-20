# Changelog

All notable changes to vLLM Warden are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) once the first
release ships.

## [Unreleased]

## [v2026.07.19.3] — 2026-07-19

### Added
- **Server-side request reaper for the proxy** (`VW_REQUEST_MAX_WALL_S`,
  default `0.0` = disabled). When set `> 0`, a proxied request that runs longer
  than this many seconds is torn down: the proxy stops draining the upstream,
  `aclose()`s the upstream response + httpx client (so vLLM aborts the generation
  and frees its KV blocks), and releases the scheduler slot. Covers both paths —
  streaming (per-chunk wall-clock check in the SSE loop) and non-streaming
  (`asyncio.wait_for` around the upstream read, returning `504` on expiry). Closes
  the "abandoned but transport-alive client" hole where the downstream connection
  keeps draining tokens (or a hung upstream keeps the socket alive), no
  `http.disconnect` ever reaches uvicorn, Starlette never cancels the body
  iterator, and the request pins its slot + KV indefinitely. Recommended
  production value: `600`. Priority scheduling and the auth/token model are
  unchanged.

### Fixed
- **Streaming proxy now detects downstream disconnect during the reasoning
  phase.** The `request.is_disconnected()` check was nested inside `if delta:`
  (content deltas only), so it was unreachable while the engine streamed
  `reasoning_content` — a request abandoned mid-reasoning was never noticed. The
  disconnect and wall-clock checks now run every loop iteration, independent of
  whether the chunk carried a parseable content delta.
- **Live-stats dashboard no longer mislabels reasoning as "prefill 0%".** The
  phase flips to `decode` on any SSE `data:` frame (content *or*
  `reasoning_content`), not only content deltas; token accounting stays
  content-only so billing is unaffected.

## [v2026.07.19.2] — 2026-07-19

### Added
- **New realtime stats dashboard at `/ui/stats/live`** (the old `/stats` page is
  unchanged, gaining only a "Live view" link). Surfaces live vLLM engine +
  per-request telemetry the aggregate `/stats` page could not: requests
  running/waiting (by wait reason), KV-cache pressure with absolute used/total
  tokens and preemptions/s, throughput and latency percentiles, prefix-cache hit
  rate and MFU, finished-reason mix, and a live per-request table (token · client
  IP · model · phase · context-window fill vs `max_model_len` · elapsed · orphan
  flag) aggregated by warden token and client IP. Built as two independent data
  planes behind a fixed contract (`docs/live-stats-spec.md`):
  - **Engine plane** — `GET /api/stats/live` (SSE, ticket-auth): scrapes the vLLM
    aggregate Prometheus `/metrics`, ~1.5s cache, rates as frame deltas, histogram
    percentiles from `_bucket`, absolute KV tokens from `cache_config_info`.
  - **Proxy plane** — `GET /api/stats/requests` (JWT poll): in-process
    `RequestRegistry` hooked fail-open into the proxy `_forward` path; the only
    source of per-request context (aggregate `/metrics` cannot provide it).

## [v2026.07.19.1] — 2026-07-19

### Fixed
- **Non-streaming proxy path now closes the upstream vLLM socket on client
  disconnect (#184).** In `app/proxy/routes.py` the `resp.aclose()` /
  `client.aclose()` calls sat after `await resp.aread()` inside the `try`, so a
  client disconnect (which raises `CancelledError` out of `aread()`) skipped
  them — the warden→vLLM socket stayed open until GC, leaving vLLM generating
  and holding KV blocks for a request nobody was reading. Moved both `aclose()`
  calls into the `finally` (mirroring the already-correct streaming path) so the
  abort reaches vLLM synchronously and KV blocks free immediately. A latent KV
  amplifier under load on VRAM-constrained engines; no behaviour change on the
  success path (`aclose()` is idempotent).

## [v2026.07.18.4] — 2026-07-18

### Fixed
- **All model loads on 0.25.1 fixed — `vllm-gguf-plugin` 0.0.4
  `override_quantization_method` signature patched.** vLLM 0.25.1's
  `_verify_quantization` (`vllm/config/model.py`) calls
  `override_quantization_method(quant_cfg, user_quant, hf_config=...)` for
  **every registered quantization method** during ModelConfig init — core's
  base signature grew a third `hf_config` param. The out-of-tree plugin (0.0.4
  is the latest on PyPI) still had the old 2-arg classmethod signature, so the
  `hf_config=` kwarg raised `TypeError` and broke **100%** of loads, even the
  non-GGUF FP8 production row. This was a second, pre-existing blocker masked
  behind the nixl_ep segfault below — it only surfaced (as a catchable `rc=1`)
  once the segfault was fixed. The Dockerfile now patches the plugin
  classmethod's signature at build time to accept and ignore the extra kwarg
  (its body already discards its inputs, so this is behaviour-preserving). The
  patch is applied in one build step and **verified in a separate `python3`
  process**: resolving the module path imports the plugin package (loading the
  config module at the old source into `sys.modules`), so an in-process
  re-import returns a stale compiled code object and the signature check fails
  spuriously — a fresh interpreter compiles the patched source, exactly as the
  runtime `vllm serve` subprocess does. Keep the `s2 != s` guard: any base-image
  or plugin bump can re-break the match and must fail the build loudly.

## [v2026.07.18.3] — 2026-07-18

### Fixed
- **gguf-plugin `override_quantization_method` signature patched for vLLM
  0.25.1 — superseded the same day by v2026.07.18.4.** Intermediate hotfix: it
  applied the signature patch (see v2026.07.18.4) but validated it in the *same*
  `python3` interpreter that had already imported the plugin, so the build-time
  check read a stale compiled code object and passed spuriously — the shipped
  image did not actually carry the working patch. Corrected in v2026.07.18.4 by
  running the verification in a separate interpreter.

## [v2026.07.18.2] — 2026-07-18

### Fixed
- **Model loads no longer SIGSEGV (rc=-11) at engine startup on non-AVX CPUs.**
  The v0.25.1 base image bundles NVIDIA's `nixl_ep` (cross-node expert-parallel
  all2all) packages, whose compiled CUDA extension dlopens an AVX-built UCX
  library at *import* time. On a CPU without AVX (the pw_prod GPU node runs a
  QEMU vCPU with none) UCX's load-time feature check aborts the process —
  `FATAL: UCX library was compiled with avx but CPU does not support it` — a
  C-level segfault, not a catchable exception, so *every* model load crashed on
  0.25.1. vLLM guards this import only with `has_nixl_ep()` (a bare
  `importlib.find_spec` presence check), so the Dockerfile now removes the
  `nixl_ep*` packages: the guard goes False and the AVX-UCX path is never
  touched. `nixl_ep` is cross-node expert-parallel, irrelevant to our
  single-node TP deployments; base `nixl` is left intact.

## [v2026.07.18.1] — 2026-07-18

### Changed
- **Base image bumped `vllm/vllm-openai` v0.20.0 → v0.25.1** (digest-pinned).
  Motivation: the 2026-07-17 production outage — vLLM 0.20.0's ngram
  speculative-decoding + prefix-caching block-accounting bug
  (`AssertionError: num_required_blocks < len(req_blocks)` in
  `single_type_kv_cache_manager.py`) crash-looped the engine; 0.25.1 carries
  the upstream fix, so `--speculative-config` can be re-enabled on affected
  model rows after this ships.
- **GGUF support now comes from the `vllm-gguf-plugin` (pinned 0.0.4)**
  instead of vLLM core. Upstream moved GGUF loading out of core in the 0.25.x
  line (`model_loader/gguf_loader.py` no longer exists), so the three
  sed-patches we carried against the 0.20.0 base for Qwen3.5/3.6 GGUF tensor
  renames (#107/#108/#115, upstream vllm PR #38140 — still open/unmerged) are
  removed with it. The plugin uses an adapter-based loader with no rename
  table; **Qwen3.5/3.6 GGUF loadability under the plugin is unverified** — if
  a Qwen3.x GGUF row regresses after this release, start there. Safetensors
  rows (including the production FP8 model) are unaffected.
- Curated template engine pins (`app/templates/registry.py`,
  `docs/tested-stacks.md`) deliberately stay on their tested 0.20.0 stacks —
  they describe docker-driver try-stack combos that were validated on 0.20.0,
  not the baked engine of this image.

### Fixed
- **Live log tail no longer unsticks mid-stream during a fast vLLM startup
  burst.** The per-model log viewer (`models/log-stream.tsx`) drives
  react-virtuoso's `followOutput` through the `useStickyBottom` hook, which
  previously gated auto-follow on its own stick/free latch. The latch flips to
  "free" on *any* `atBottomStateChange(false)`, and under a ~200 line/s append
  burst Virtuoso emits a *transient* `atBottom=false` (the freshly-appended row
  sits below the viewport for a frame before the follow-scroll lands — and the
  old `"smooth"` scroll could not animate fast enough to keep pace). That
  transient permanently disabled auto-follow, forcing the operator to click
  "Jump to latest" over and over. `followOutput` is now always the instant
  `"auto"` behavior, decoupled from the latch: react-virtuoso self-gates to
  only auto-scroll when the list is already at the bottom, so the tail stays
  pinned through bursts yet never yanks an operator who has scrolled up to read.
  The `mode` latch survives but now only drives "Jump to latest" visibility. A
  defensive `atBottomThreshold={64}` on the `<Virtuoso>` swallows sub-row
  transients. Pinned by `tests/component/use-sticky-bottom.test.ts` (4 tests).

## [v2026.06.16.1] — 2026-06-16

### Changed
- **`open_db` now sets an explicit `PRAGMA busy_timeout = 30000` (was the
  implicit aiosqlite default of 5000 ms).** During a user-triggered HuggingFace
  model pull the process can saturate CPU/IO and briefly starve the asyncio
  event loop; the short-lived connections opened by background writers (stats
  sampler, pull poller) could then collide with the foreground writer and fail
  fast with `database is locked`. A generous 30 s busy_timeout makes them wait
  out the contention window instead. (`app/db/database.py`; covered by
  `tests/unit/db/test_database_pragmas.py`.) Note: this hardens against lock
  contention but is not the master cause of probe-kills during a pull — that is
  whole-container event-loop starvation, addressed at the deployment-probe layer
  in the PodWarden Hub stack template (lenient liveness + a `startupProbe`, and
  the Caddy front-door health probe decoupled from the api via a Caddy-local
  `/caddy-healthz` route).

## [v2026.06.15.1] — 2026-06-15

### Fixed
- **Engine now writes downloaded weights to the HF cache volume, not `/data`
  (ENOSPC crash fix).** The vLLM engine subprocess resolved its model cache via
  `HF_HOME`, whose on-disk layout (`$HF_HOME/hub/models--…`) did not match where
  the pull agent had placed the snapshot, so the engine re-downloaded the full
  model into the small `vllm-warden-data` volume (`/data`, 10 GiB), filling it
  to 0 bytes and taking the SQLite database down with it (`Fatal Python error:
  _PyImport_Init`). `app/runtime/env_builder.py` now sets `HF_HUB_CACHE` (whose
  layout `$HF_HUB_CACHE/models--…` matches `snapshot_download(cache_dir=…)`)
  instead of `HF_HOME`, so the engine and the pull agent agree on one path —
  the 100 GiB `vllm-warden-hfcache` volume. `HF_HOME` is no longer exported.

### Removed
- **The non-functional "HF cache directory" setting has been removed entirely.**
  The `hf_cache_dir` key was seeded into the settings KV table and exposed as an
  editable field in the General settings tab and the `/api/settings/runtime`
  surface, but its stored value was never read — the real cache path is
  env-driven only (`VW_HF_CACHE_DIR` → `Settings.hf_cache_dir`), so editing it in
  the UI silently did nothing. Removed the seed (migration `0010`), the runtime
  API key + coercer (`/api/settings/runtime` now rejects `hf_cache_dir` as an
  unknown key with `400`), and the General-tab field. New migration `0023`
  deletes any `hf_cache_dir` row left in existing databases (idempotent).

### Changed
- **`VW_FRONTEND_ORIGIN` is now optional in the PodWarden Hub stack template.**
  The hub compose previously hard-required `VW_FRONTEND_ORIGIN` via Compose's
  `${…:?}` operator, so a fresh install refused to start until the operator set
  it. The API already treats an explicitly-empty value as the localhost-default
  fallback (so a blank var never locks admins out), so the compose now uses
  `${WARDEN_FRONTEND_ORIGIN:-}` and the stack boots without it. Set it to your
  public UI origin to harden the CSRF/Origin check on `/api/auth/*`.

## [v2026.05.28.1] — 2026-05-28

### Fixed
- **Engine bind-host default flipped from `127.0.0.1` to `0.0.0.0`** so loads
  succeed on the docker-socket driver out of the box (#180). The docker driver
  (production default) runs each engine as a separate container; binding to
  loopback inside that container made `/health` and `/v1/*` unreachable from
  the api container and from docker-proxy's published port, so the supervisor
  health probe timed out and the row flipped to `failed` despite the engine
  emitting `Application startup complete.` Operators on the legacy in-process
  subprocess driver (engine shares the api container's netns) can opt back
  into loopback by exporting `VW_ENGINE_BIND_HOST=127.0.0.1` at deploy time.

## [v2026.05.27.6] — 2026-05-27

### Changed
- **Try-stack: "Try combo" button replaced by one-click "Load" that pins the
  combo and starts the engine in a single action.** Picking a channel + vLLM
  version on the per-model Try-stack panel and clicking Load now POSTs
  `/api/models/{id}/try-stack` (pin + record pending attempt) and then
  `/api/models/{id}/load` (start the engine) back-to-back, instead of forcing
  the operator to bounce to the model card's Load button after picking a
  combo. The button is now disabled while the model is `loading`, `loaded`,
  or `unloading`, with an inline note instructing the operator to unload
  first to try a different combo. Backend endpoints are unchanged.

### Added
- **Try-stack: warn before re-submitting a (channel, vLLM version) combo that
  previously failed.** When the picker's selection matches a historical
  attempt with `result === "failed"`, a yellow banner above the Load button
  surfaces the past error + timestamp; the first click on Load shows a
  "Click Load again to confirm" prompt instead of submitting, and the second
  click submits. Changing channel or version clears the banner and the
  confirm gate.
- **Try-stack: Retry link on each history row pre-fills the picker** with
  that row's `channel` + `vllm_version`. The Retry link is hidden on the
  most recent attempt while it is still pending (no signal in re-pinning a
  not-yet-reported attempt). When the prefilled combo matches a failed row,
  the duplicate-failed-attempt banner above fires naturally on the next Load
  click.

## [v2026.05.27.5] — 2026-05-27

### Added
- **Fit-preview now warns when the detected GPUs can't run the chosen
  quant/dtype well (#176).** The live GPU probe (`GET /api/system/gpus` and the
  fit-preview internals) now captures each card's CUDA compute capability
  (`compute_cap`, e.g. `8.6`), degrading gracefully to `null` when the driver
  reports `[Not Supported]`/`[N/A]`. `POST /api/models/fit-preview` crosses the
  weakest selected GPU's `compute_cap` with the candidate's quant/dtype (from
  `config.json` `torch_dtype` + `quantization_config.quant_method`; GGUF is
  treated as bf16-compute since it dequantizes at load) and appends actionable,
  human-readable strings to the existing `warnings` list — e.g. FP8 on an
  Ampere card (cc &lt; 8.9) is flagged as emulated (slow/inaccurate), bf16 below
  cc 8.0, NVFP4/FP4 below Blackwell (cc 10.0), and AWQ/GPTQ INT4 below the
  Marlin floor (cc 7.5). These are warnings only — fit-preview never blocks. The
  rule matrix lives in the pure module `app/models/gpu_capability.py`; when no
  GPU reports a compute capability, no capability warnings are produced.
- **The try-stack vLLM-version field is now a typeable dropdown backed by the
  published image-resolving versions (#177).** The free-text "vLLM version"
  input on the per-model Try-stack panel becomes a combobox populated from a new
  `GET /api/templates/engine-versions?channel=` endpoint, which lists the
  `vllm/vllm-openai` semver tags that actually resolve to an image for the
  selected channel (resolvable CUDA channels only; rocm/cpu/xpu/unknown return
  an empty list). Versions come from a 6h in-process cache keyed by image family
  so Docker Hub's anonymous rate limit is never hit per request, and a Docker
  Hub hiccup serves stale-or-empty rather than failing the page. The field stays
  free-text, so an operator can still enter an unpublished version or a pinned
  digest the catalog won't list.
- **Origin guard now supports a comma-separated allow-list and an opt-in
  reverse-proxy trust mode.** `VW_FRONTEND_ORIGIN` accepts a comma-separated
  list of exact origins (whitespace trimmed, trailing slashes stripped) instead
  of a single value, so more than one front-door origin can be accepted by the
  `/api/auth/refresh` and `/api/auth/logout` origin check. A new
  `VW_TRUST_PROXY_ORIGIN` env var (default off) additionally accepts a request
  whose `Origin` equals the proxy-derived `X-Forwarded-Proto://X-Forwarded-Host`
  — only enable it behind a trusted reverse proxy that sets those headers. This
  fixes reloads logging users out when the served origin isn't pinned ahead of
  time. The check remains fail-closed: an origin matching neither the allow-list
  nor (when enabled) the proxy-derived origin is rejected with `403`.
- **GPU selection is now checkbox-driven from live system inventory (#175).**
  The default GPU indices field on the Settings → General tab and the per-model
  settings page replaces the comma-text field with a `GpuChecklist` component
  populated from a live `GET /api/system/gpus` probe. Each GPU renders as a
  labelled checkbox (index, name, VRAM). Configured GPU indices that are no
  longer present in the live probe render as a removable amber "not present" row
  with a warning banner so operators can identify and clean up stale assignments
  without losing the value before they decide what to do. The per-model
  `gpu_indices` field now enforces a client-side minimum of one selection:
  emptying it disables Save and renders an inline "Select at least one GPU."
  validation message next to the field (the Settings → General
  `default_gpu_indices` field may still be empty, meaning "no preset").
- **Model load fails fast with `422 gpu_index_missing` when a configured GPU
  is absent (#175).** The load pre-flight now cross-checks the model's
  `gpu_indices` against the current `nvidia-smi` probe and rejects immediately
  with a structured error (`detail.error_code = "gpu_index_missing"`, a
  human-readable `detail.message` naming the absent indices, and
  `detail.available` listing the GPUs the probe did see) rather than letting
  vLLM start and crash with a cryptic NVML error several seconds later. The
  pre-flight now **fails closed**: a probe error (e.g. `nvidia-smi`
  unavailable) leaves no ground truth confirming any configured GPU, so the
  load also 422s `gpu_index_missing` (with `detail.available = []` and the
  probe's error in `detail.probe_error`) instead of optimistically proceeding.
  The descriptive message is surfaced in the UI's load-error toast so the
  operator knows which index to fix.
- **Proxy throughput: per-engine concurrency + native vLLM priority (#173).**
  The `/v1` proxy's `PriorityScheduler` no longer serializes the whole engine
  behind a single global in-flight slot — the historic cause of the product
  path flat-lining at ~40 tok/s while the engines themselves sustained 1000+
  tok/s aggregate. It now admits up to `VW_PROXY_MAX_INFLIGHT` (default 16)
  concurrent requests **per engine** (keyed on the model id), so a busy engine
  never blocks a request bound for an idle one, and continuous batching inside
  vLLM is actually exercised. Token priority (0..9, 9 = highest) is preserved
  as the admission ordering under contention **and** pushed into the engine:
  every engine launches with `--scheduling-policy priority` and the proxy
  injects a per-request `priority = -token.priority` into the forwarded
  chat/completions body (vLLM orders its waiting queue ascending, so the
  negation puts high-priority traffic ahead of default-0 traffic). Warden
  priority 0 maps to vLLM's inert default and is not injected; a client that
  sets `priority` explicitly is left untouched.
- **Docs: `docs/tested-stacks.md` — validated model + engine combinations.**
  A living record of model / vLLM-engine stacks driven end-to-end through the
  Chat Playground UI and human-confirmed coherent on sm_86 Ampere hardware
  (4× RTX A4000 on host d5). Documents six validated combos (gpt-oss-20b,
  Qwen2.5-3B, Nemotron-Nano-8B, Mistral-7B, Llama-3.1-8B-AWQ, Qwen2.5-14B-AWQ)
  plus the Ampere quantization rules that drove the choices: bf16 8B OOMs on a
  single 16 GiB card, `--quantization fp8` is emulated and produces gibberish,
  AWQ INT4 is the working single-GPU path, and `--enforce-eager` avoids the
  inductor compile-time OOM. Also records the "Save working combo as template"
  gap (it drops `extra_args` and the `gpu_memory_utilization` override).
- **Open-source release: Apache-2.0 LICENSE + GitHub publish flow.** vLLM
  Warden is now open source under Apache-2.0. A manual `publish:github` CI
  job (on `main` push) mirrors the tracked tree to the public
  [github.com/Podwarden/vllm-warden](https://github.com/Podwarden/vllm-warden)
  repo as a single squashed "Initial public release" commit via
  `scripts/publish-github.sh`. Ships with a public-facing README (install,
  usage, build-from-source) and UI screenshots.

### Fixed
- **Live engine logs now stream under the docker engine driver (#177
  follow-up).** Under `DockerSocketDriver` the engine runs as a sibling
  container whose vLLM output goes only to `docker logs <engine-container>`;
  no per-model log file was ever written. But the SSE log endpoint
  (`app/models/routes_logs.py`) tail-follows `<data_dir>/logs/<model_id>.log`
  exclusively, so the UI "Live logs" panel showed stale pre-switch content
  (or nothing) — on one host this made a freshly-loaded `v0.21.0` engine look
  like it was still running `v0.20.0`. `DockerSocketDriver` now mirrors the
  engine container's combined stdout+stderr into that same per-model log file:
  on `spawn()` it truncates the file (clearing any stale subprocess-driver
  content) and pumps `container.logs(stream=True, follow=True, ...)` — a
  blocking generator — on a dedicated daemon thread, so `routes_logs.py` and
  the SSE pipeline stream live output unchanged. The no-log-file behavior is
  preserved when no log dir is configured.
- **Engine-version pins are no longer silently discarded under the in-container
  subprocess driver (#177).** The Try-stack version selector pins a vLLM engine
  image onto a model, but under the default `LocalSubprocessDriver` the warden
  runs the vLLM binary baked into its own image and never reads `EngineSpec.image`
  — so a pinned version (e.g. `v0.21.0`) was quietly ignored and the in-image
  version launched instead. Drivers now declare a `supports_engine_image`
  capability (`False` for subprocess, `True` for the docker driver), and
  `Supervisor.load` refuses an unsupported pin up front (raising
  `EnginePinUnsupported` before claiming any GPU) so it surfaces as a failed load
  with an actionable message instead of a wrong-version launch. A new
  JWT-protected `GET /api/system/engine` reports the active driver
  (`subprocess`/`docker`), whether version selection is supported, and the
  baked-in vLLM version; the Try-stack panel reads it to disable the channel /
  version / Try controls and show an amber note explaining that version selection
  requires the docker engine driver when the deployment can't honor a pin.
- **Settings inputs for GPU indices, extra args, and extra env no longer wipe
  in-progress text (#175).** The `int-list`, `string-list`, and `kv-map`
  `SettingField` kinds were using uncontrolled-to-controlled input patterns that
  reset the cursor whenever the parent re-rendered with a round-tripped API
  value — making it impossible to type a trailing comma or newline before the
  next value. These fields are now fully controlled: the local draft state is the
  single source of truth during editing and only flushes to the server on Save,
  so more than one GPU index (or extra arg, or env entry) can be entered without
  the field wiping itself mid-keystroke.
- **Docker driver: engines pinned to a non-zero GPU index no longer crash at
  startup (#172).** When the `DockerSocketDriver` spawns a sibling engine
  container it pins the model's physical GPUs via `device_requests`; the NVIDIA
  container runtime then renumbers those passed-through devices to `0..N-1`
  *inside* the container. `env_builder` sets `CUDA_VISIBLE_DEVICES` to the
  model's **host** indices (correct for the in-process `LocalSubprocessDriver`,
  whose parent container sees every GPU), so a model on host GPU 1/2/3 received
  `CUDA_VISIBLE_DEVICES=1/2/3` against a container whose NVML view only had
  index 0 — vLLM's `nvmlDeviceGetHandleByIndex(host_idx)` raised
  `NVMLError_InvalidArgument` and the engine exited `rc=1`. Only GPU 0 happened
  to work (host index 0 == relative 0), silently masking the bug on
  single-engine hosts. The driver now remaps `CUDA_VISIBLE_DEVICES` to the
  container-relative range (`0..N-1`) while `device_requests` still pins the
  correct physical GPUs — unblocking multi-engine / full-cluster utilization.
- **"Save working combo as template" now captures `extra_args` +
  `gpu_memory_utilization` (#170).** The try-stack save-as-template flow only
  persisted the engine axis, repo, `max_model_len` and `tensor_parallel_size`,
  silently dropping the live model's `extra_args` (came back `[]`) and its
  `gpu_memory_utilization` override (came back as the `0.9` default). An AWQ
  single-GPU template saved with `--enforce-eager` +
  `gpu_memory_utilization=0.92` therefore re-instantiated without them and
  OOM'd. `TemplateCreate` now accepts the live `model_id`; the backend sources
  `extra_args` + `gpu_memory_utilization` from that model row (the same source
  of truth the create form uses, with explicit body fields still winning), and
  the try-stack panel passes `model_id` so the saved template reproduces the
  combo that actually worked.
- **`terminate()` reaps a cancelled engine wait instead of 500-ing (#171,
  root cause behind #166).** `DockerHandle.wait()` memoizes one
  `_wait_task` shared between the exit-watcher and `terminate()`. When
  `Supervisor.unload()` cancels the watcher it cancels that shared task; a
  later `terminate()` awaiting the same task re-raised
  `asyncio.CancelledError` — a `BaseException` since py3.8, so it escaped
  `terminate()`'s `except Exception` reap guard, propagated through
  `unload()`, and collapsed in the CSRF middleware to
  `RuntimeError("No response returned.")` → HTTP 500. Fixed in two places for
  robustness: the reap guard now also catches `asyncio.CancelledError` (treats
  the container as exited, returncode 0), and `wait()` no longer reuses a
  poisoned (already-cancelled) memoized task — it recreates the reap against
  the live container. This addresses the underlying defect that #166's
  background-teardown change worked around.
- **Unload no longer 500s / strands the row in `unloading` (#166).** The
  unload route used to `await` the full engine teardown inline. A large
  multi-GPU engine takes many seconds to terminate, so the client/proxy
  disconnected mid-request and the CSRF `BaseHTTPMiddleware` raised
  Starlette's `RuntimeError("No response returned.")` → HTTP 500 — leaving the
  model stuck in the transient `unloading` state with the only recovery being
  a control-plane restart. Teardown now runs in a background task (mirroring
  the load path): the fast refusal check is still surfaced synchronously as
  409, the route returns 202 immediately, and the row always reaches a
  terminal state — even if teardown raises — so recovery never requires a
  restart. New `Supervisor.ensure_unloadable()` pre-flight separates the
  instant state check from the slow teardown.
- **Supervisor.unload: never leak the in-memory GPU claim when teardown
  raises.** `unload()` released GPU ownership only as its trailing statement,
  so if `driver.terminate()` raised — or the await was cancelled by a
  client/proxy disconnect mid-teardown — `self.gpus.release()` was skipped and
  the GPUs stayed permanently `already claimed` by a model that no longer
  exists. Every subsequent load onto those GPUs then failed with
  `GPUs [...] already claimed`, recoverable only by restarting the control
  plane. Teardown now runs in a `try/finally` so the GPU claim and lifecycle
  bookkeeping are always released past the refusal gate; a refused unload
  (transient state, no `force`) still correctly retains the claim. Observed on
  d5 after a `#166` client-disconnect unload stranded a multi-GPU engine's
  claim. Complements `#166` (which decouples teardown from the request so the
  primary cancellation path no longer triggers) with defense-in-depth for any
  other teardown exception.
- **DockerSocketDriver: give engine siblings adequate shared memory for
  tensor-parallel.** The sibling engine container inherited docker's default
  64MB `/dev/shm`, so vLLM `tensor_parallel_size > 1` hung indefinitely at
  the distributed-init barrier (workers communicate over POSIX shared memory
  via the shm_broadcast message queue + custom-all-reduce CUDA-IPC handles) —
  the model sat in `loading` forever with no crash and no `last_error`. The
  driver now spawns engines with `ipc_mode=host` (matching the control-plane
  compose override) plus an explicit `shm_size` fallback. Both are
  env-overridable (`VLLM_ENGINE_IPC_MODE`, `VLLM_ENGINE_SHM_SIZE`); set
  `VLLM_ENGINE_IPC_MODE=""` for a private but enlarged `/dev/shm`. Surfaced
  loading gpt-oss-20b at TP=2 on the docker driver.
- **try-stack (#162): wire the missing report (ok/failed) control.** The
  model-detail try-stack panel recorded a "pending" attempt and offered
  "Save working combo as template" (gated on `result === "ok"`), but had no
  control to report whether the model actually came up — so a pending attempt
  could never become "ok" through the UI and save-as-template was unreachable.
  The panel now renders "Mark working" / "Mark failed" (with an optional
  failure-detail field) that POST to `/api/models/{id}/try-stack/{attempt_id}`,
  completing the try → report → save round-trip the panel already documented.
- **publish:github: normalize the SSH deploy-key file to a trailing
  newline before use.** GitLab stores File-variable values without a
  trailing newline, so the mounted key tripped OpenSSH's
  `Load key "...": error in libcrypto` and the GitHub mirror push failed
  with `Permission denied (publickey)` despite a valid, write-enabled
  deploy key. `scripts/publish-github.sh` now rewrites the key to exactly
  one trailing newline (`printf '%s\n' "$(cat ...)"`) regardless of how
  the variable was stored.
- **chat: send `served_model_name` in completion requests instead of
  internal model id.** Fixes `model '<hash>' is not loaded` 404 surfaced
  2026-05-22 by Qwen3.6-27B's distinct served name on
  `https://vllm.protrener.com/ui/chat`. The bug was latent for the
  entire history of the chat playground because every prior model on
  the fleet had `served_model_name == id`; Qwen3.6-27B is the first
  deployment whose served name differs from its internal id, and every
  chat send started 404'ing the moment it landed. The picker still
  keys on the row's `id` — only the wire field changes. Both
  `handleSubmit` and `handleRegenerate` (the two chat-completion call
  sites in `frontend/src/app/chat/page.tsx`) now resolve
  `modelId → served_model_name` via the already-loaded `loadedModels`
  list before placing it on the request. Regression test
  (`frontend/tests/component/chat-page-served-model-name.test.tsx`)
  pins the wire contract.

### Added
- **Caddy: 308-redirect v1-era bare paths to /ui/\*.** Old bookmarks /
  browser-typed paths (`/login`, `/models`, `/settings`, `/setup`,
  `/stats`, `/tokens`, `/cache`, `/chat`) now 308-redirect to their
  basePath-prefixed equivalent under `/ui/*` instead of hitting Caddy's
  404 fallthrough. 308 preserves request method. Also: `/favicon.ico`
  is aliased (301) to `/ui/icon.svg` so browser-auto requests for a
  favicon resolve against the icon the Next.js UI already ships.

- **New `public_url` runtime setting (#154, subsumes #151).** Sets the
  external base URL used in client-facing snippets (curl examples,
  OpenAI client configs) when the warden sits behind a reverse proxy
  whose external URL differs from the browser's address bar. Leave
  blank to use `window.location.origin` — that's the right default
  for direct-access deployments. Backend validates http/https scheme,
  non-empty netloc, ≤2048 chars, and strips any trailing slash before
  persisting. New frontend helper `getPublicBaseUrl()` (and its
  pure-function inner `_resolvePublicBaseUrl()`) is the single source
  of truth for snippet rendering; defends in depth by falling back to
  origin if a malformed value somehow lands in the row. Migration
  0021 is doc-only (no seed — the key is intentionally unset by
  default).

- **Per-tab key membership contract test (#154).** The frontend
  contract test `settings-tab-membership.test.ts` fails CI if a new
  runtime key is added to `RUNTIME_HINTS` without being placed in
  exactly one of the four tab arrays — guards against fields silently
  disappearing from the UI on future additions.

### Changed
- **Landing page (/_landing) redesigned to podwarden.com visual identity.**
  Retro-dark palette (slate-950 background, emerald-400 accent,
  slate-100 text), DM Sans typography via Google Fonts CDN, Lucide
  shield + Simple Icons GitHub marks inlined as SVG (no external JS).
  Copy rewritten around operator-console positioning ("Self-host vLLM,
  with the operator surface you actually wanted") with a four-card
  feature grid. Load-bearing entry-points (`/ui/`, GitHub repo,
  podwarden.com) preserved; existing contract test still passes,
  plus a new redesign-surface test pins the new copy and visual
  identity tokens.

- **/settings page reorganised into five purpose-grouped tabs (#154).**
  The previous 13-field "Runtime" flat list (one 581-line component) is
  replaced with five purpose-grouped tabs — General, Networking,
  Sessions & Tokens, Maintenance, Model — each composed of titled
  section cards. Each runtime sub-tab owns its own Edit/Save lifecycle
  scoped to that tab's keys, so editing one tab no longer marks the
  entire form dirty. The underlying `GET /api/settings/runtime` fetch
  is deduplicated by SWR across tabs, so the page still issues exactly
  one read per visit. Spec lives at
  `docs/superpowers/specs/2026-05-24-settings-redesign-design.md`.

- **Token rotation renames old row, mints new with original name (#150).**
  Operators previously had to rename-then-mint by hand (three steps,
  collision risk on the typed-back name). Now a single `Rotate` click:
  - Renames the existing row to `"{name} (old N)"` (N = next free slot;
    starts at 1, cascades to `(old 2)`, `(old 3)` … on subsequent
    rotations — gaps in N are NOT reused, so a deleted `(old 1)` does not
    make slot 1 available again).
  - Mints a brand-new row that keeps the ORIGINAL name, inheriting
    `rate_limit_tps` and `priority` from the predecessor.
  - Returns both `name` (active) and `renamed_to` (predecessor's new
    name) so the UI's success modal can tell the operator exactly where
    each token landed.
  - Rejects with **409 Conflict** if the requested row was already
    rotated — the UI disables the button on those rows, but the
    server-side guard defends against direct API callers and racing
    double-clicks. Single SQLite transaction wraps SELECT + UPDATE
    (rename) + INSERT (successor) + UPDATE (rotated_at / revoked_at), so
    a crash mid-flight cannot leave rotation half-applied.
  - **`app/db/repos/tokens.py`** — `rotate()` signature changed: dropped
    the `new_name` parameter (repo now derives names internally) and now
    returns `(new_id, new_plaintext, renamed_to)`. Added private
    `_next_old_suffix()` helper with strict regex parsing and LIKE-
    escape handling so token names containing `_` or `%` cannot
    false-match into someone else's `(old N)` ladder.
  - **`app/tokens/routes_api.py`** — rotate endpoint returns `name` +
    `renamed_to` fields and raises 409 on already-rotated rows. The
    public response shape gained two fields; existing callers reading
    `id` / `plaintext` / `prefix` / `rotated_from` are unaffected.
  - **`frontend/src/components/tokens/rotate-token-dialog.tsx`** —
    success-state footer now reads "New active token: `<name>`. The
    previous token was renamed to `<renamed_to>` …" instead of the
    stale "Rotated from `<old_id>`" copy.
  - **Tests** — repo-layer (`tests/unit/db/test_token_rotate_naming.py`,
    6 cases covering happy-path naming, cascading `(old 2)`, MAX+1 with
    gaps, 409-on-already-rotated, predecessor secret intact during
    grace, and SQL-wildcard escaping). Endpoint-layer additions in
    `tests/unit/tokens/test_rotate_endpoint.py` (3 cases) pin the
    response shape and the 409. Existing rotate tests (4 files)
    updated to drop the `new_name=` argument and assert the new
    naming contract.
  - **Docs** — `docs/operating.md` rotate-token section rewritten to
    show the new response fields and document the 409 case.

### Added
- **Unified-port architecture: single host port `:8080` fronted by Caddy (#155).** A new `caddy:2-alpine` service joins the compose stack as the only host-published port. It fans out `/` → FastAPI `/_landing`, `/ui/*` + `/_next/*` → ui:3000, `/api/*` + `/v1/*` + `/healthz` → api:8080. The `api` and `ui` containers move to `expose:` only (no host mappings) so operators no longer have to remember "did I open 8080 or 3000?" — there's only one port. SSE routes (`/api/chat/*`, `/api/models/{id}/logs/stream`, `/api/stats/stream`) use `flush_interval -1` so chunks flush immediately. Caddy admin API is `admin off` — no port 2019 to lock down. See `deploy/caddy/Caddyfile` for the routing map and `docs/operating.md#unified-port-topology-155` for the operator notes.
- **Public landing page at `/_landing` with opt-out (#155).** New JWT-exempt route returns a single-file HTML page with the project title, a deep link to `/ui`, the source repo, and `podwarden.com`. Caddy rewrites `/` to `/_landing` so an anonymous browser hitting the unified-port root sees something useful. New runtime setting `landing_page_enabled` (boolean, default `true`, restart kind `none`) lets operators disable the page for private deployments; with the toggle off both `/` and `/_landing` return `404 landing page disabled`. UI surfaces the toggle as a new field on `/settings/runtime`. Migration `0020_landing_page_setting.sql` seeds the default.
- **`make smoke` target (#155).** Five-curl liveness check against the live unified-port front-door: `/`, `/_landing`, `/ui/`, `/api/csrf`, `/healthz` — all must return 200. Intended for post-`docker compose up -d` sanity in operator runbooks and CI smoke jobs.
- **System Configuration panel on `/stats` (#148).** New section at the bottom of the stats page surfaces host inventory the operator wants when interpreting the live metrics above it: CPU model + physical/thread counts, total RAM, OS release + kernel, Docker server version + default runtime, and one card per GPU (name, VRAM, driver version, CUDA version). Backed by a new JWT-gated `GET /api/system/info` endpoint that composes from `/proc/cpuinfo`, `/proc/meminfo`, `/etc/os-release`, `uname -r`, `nvidia-smi`, and `docker info`; each source degrades independently (no NVIDIA → `gpus: []`, no docker socket → `available: false`). 60s in-process cache absorbs the page's 30s SWR poll so shell-outs happen at most once per minute.
- **Favicon (#152).** Browser tabs and iOS home-screen now render a
  branded icon instead of the default globe. Ships two assets in
  `frontend/src/app/`: a static `icon.svg` (filled emerald Shield
  matching the navbar wordmark, white "W" centred — Next App Router
  auto-emits `<link rel="icon" type="image/svg+xml">`) and a
  programmatic `apple-icon.tsx` that uses `next/og`'s `ImageResponse`
  to render the same composition as a 180×180 PNG at build time. No
  raster asset is shipped in the repo — the apple-icon is statically
  optimized and cached by Next. Colour pinned to Tailwind `emerald-500`
  (`#10b981`) for consistency with the navbar Shield.

### Changed
- **Next.js UI now lives under `basePath: '/ui'` (#155).** Required so a single Caddy listener can fan out `/` → landing, `/ui/*` → Next, `/api`/`/v1` → FastAPI without path conflicts. `basePath` is baked into the standalone build at image-build time. SSR `BACKEND_URL` for the Next.js server still points at `api:8080` over the compose network; client-side fetches now use relative paths because the browser is same-origin with Caddy. `VW_FRONTEND_ORIGIN` default changes from `http://localhost:3000` to `http://localhost:8080` to match the new front-door.

### Removed
- **Next.js `/api/:path*` and `/v1/:path*` rewrites (#155).** Caddy owns these routes in the unified-port topology and proxies them directly to FastAPI; double-proxying through Next would (a) re-engage the gzip path that #52 had to disable for SSE and (b) bake a hardcoded backend URL into the standalone build. The `/healthz` → `/api/health` Next-internal alias (#48) is retained.
- **Stale "Storage moved to Cache" toast on `/stats` (#153).** The migration banner (carried over from S6 #105) was retired now that the cache surface has been on its own `/cache` route for multiple releases. The toast, its `vw.storage.toast.dismissed` localStorage key, and the lingering `next/link` + `lucide-react` imports it pulled in have all been removed.

### Fixed
- **Token-modal "Copy" button works on non-secure (HTTP) origins (#149).**
  Operator on d5 (production, served as raw HTTP over Tailscale at
  `http://10.10.0.187:8080`) hit *"Copy failed — select and copy the
  token manually"* every time on freshly-minted tokens, because
  `navigator.clipboard.writeText()` is undefined outside HTTPS/localhost.
  - **`copyToClipboard` (`frontend/src/lib/utils.ts`)** — now tries the
    async Clipboard API only when `window.isSecureContext` is true, then
    falls back to a hidden-`<textarea>` + `document.execCommand("copy")`.
    `execCommand`'s boolean return value is now checked (it was
    previously swallowed), and synchronous throws from hardened browsers
    are caught — the helper rejects only when BOTH paths fail, so the
    "select manually" hint is reserved for actual double-failure.
  - **All copy buttons refactored to use the shared helper** —
    `create-token-dialog.tsx`, `rotate-token-dialog.tsx`,
    `chat/message-list.tsx`, and the Effective-argv panel
    (`app/models/[id]/settings/page.tsx`). Previously each call site
    inlined its own `navigator.clipboard?.writeText(…)` with no
    fallback; now all four benefit from the secure-context check + the
    execCommand fallback in one place.
  - **Tests (`frontend/tests/component/copy-to-clipboard.test.ts`,
    additions in `tokens.test.tsx`)** — pin three branches called out by
    the issue AC: (a) clipboard available + `writeText` succeeds, (b)
    clipboard undefined + `execCommand` returns true, (c) clipboard
    undefined + `execCommand` returns false. Plus dialog-level tests
    that drive the success/failure toast paths end-to-end through both
    Create and Rotate modals to prevent a regression that strips the
    fallback.

## [v2026.05.23.1] — 2026-05-23

**Highlights — `2026-05 overhaul` epic close-out (S1-S9).** Stats v2
replaces the v1 GPU table with an operator cockpit (four current-value
tiles + 2×2 chart grid for VRAM / GPU util / power / tokens) plus a
per-key tokens table, fed by a new power-samples pipeline (5s sampler,
minute-bucket accumulator). New `/chat` playground proxies streaming
completions through the warden's JWT-authed SSE path; supervisor gains
a deterministic pre-spawn health-wait so cold loads no longer race a
half-up vLLM. API token rotation gains a configurable grace window so
callers can swap credentials without downtime. Bench v2 removed (CTO
decision; #117). First release tag of the epic — successor of
v2026.05.20.1.

### Added
- **Stats v2 frontend — rebuilt `/stats` page (epic/overhaul S7, #124).**
  Frontend half of the stats redesign. Consumes the new `/api/stats/v2/*`
  contract documented below.
  - **Page rebuild (`frontend/src/app/stats/page.tsx`)** — replaces the
    legacy v1 GPU table + ad-hoc cards with one operator-cockpit layout:
    a current-row of four tiles (VRAM used / total, GPU util max, summed
    power, TPS over the last full minute) sitting above a 2×2 chart grid
    (VRAM, GPU util, power, prompt+completion tokens) and a sortable
    per-key tokens table. Range selector (1h / 6h / 24h / 7d) drives both
    `/overview` and `/tokens-per-key` in one render pass; selection
    persists to `localStorage` under `vw.stats.range` and survives a
    reload. Active-model strip renders as emerald chips when at least one
    model is `loaded`; hidden entirely otherwise. SWR polls at 30s and
    pauses when the tab is hidden. The S6 "Storage moved to Cache" toast
    is preserved verbatim.
  - **`StatCard` (`frontend/src/components/stat-card.tsx`)** — reusable
    tile (label + value + optional unit + optional hint + optional
    `title` tooltip). Single source of truth so the current-row stays
    visually consistent with future surfaces (next slice's overview
    rail). Test IDs `stat-card`, `stat-card-unit`, `stat-card-hint`.
  - **`usePersistedRange` (`frontend/src/lib/use-persisted-range.ts`)** —
    SSR-safe `localStorage`-backed range hook. Returns the fallback
    until first effect-tick so server and first-paint markup agree
    (no hydration mismatch), then upgrades to the stored value on
    mount. Validates the stored string against the `StatsRange` union
    and silently swallows `localStorage`-unavailable errors.
  - **`v2-charts.tsx`** — four Recharts panels (`VramChart`, `UtilChart`,
    `PowerChart`, `TokensChart`) sharing a `ChartShell` empty-state
    wrapper. Time-scaled XAxis pinned to the range window so a sparse
    series doesn't collapse to a fake-wide chart. Animations disabled
    everywhere — a 30s poll on a flickering chart is exhausting. The
    `PowerChart` `supported` prop distinguishes "no samples yet" from
    "GPU can't report" so the operator stops waiting on a virtualised
    host.
  - **`lib/stats-v2.ts`** — frontend types mirroring the v2 docstrings
    plus formatters (`mibToGib`, `formatWatts`, `formatTps`) and the
    `withTs` helper that attaches an epoch-ms column for Recharts'
    time-scaled axis.
  - **API types regenerated** — `frontend/src/lib/api-types.generated.ts`
    picks up the two new v2 paths (response bodies arrive as bare
    objects since the backend routes return raw dicts; the page locks
    the inner shape via component tests).
- **Stats v2 backend — power telemetry + per-key tokens (epic/overhaul S7, #124).**
  Backend half of the stats redesign. The frontend half lands separately in
  dev-2's slice (same epic) and consumes the contract documented below.
  - **New migration `0019_power_samples.sql`** — per-(gpu, minute) watts
    accumulator. Schema is a write-path aggregator (`watts_sum` + `samples`
    counter) keyed on `(gpu_idx, minute)`; the read side recovers a true
    minute-average via `watts_sum / NULLIF(samples, 0)` rather than
    last-write-wins, so a 12-tick minute yields a representative number
    rather than "whichever 5s tick landed last". Retention follows
    `gpu_samples` (7d) and is handled by the existing stats pruner.
    Manual rollback recipe documented in the migration header.
  - **GPU sampler — single-pass util + mem + power** (`app/system/gpu.py`,
    `app/runtime/stats_sampler.py`). The existing `nvidia-smi` query
    now also fetches `power.draw`; one subprocess invocation per tick
    produces util, mem, and power together (no second pass). Sampler
    cadence drops from 60s to 5s (override via `VW_STATS_SAMPLER_INTERVAL_S`
    env var) so each minute bucket sees ~12 samples, then writes
    `gpu_samples` (last-write-wins) and `power_samples` (accumulator).
    Cards that report `[Not Supported]` / `[N/A]` for `power.draw` flow
    through as `power_w=None` and the sampler skips the power write for
    that tick — util/mem still land normally; no zero-row corruption.
  - **`GET /api/stats/v2/overview?range=1h|6h|24h|7d`** — JWT-gated.
    Returns the dashboard payload in one round-trip: a `current` snapshot
    (most-recent-minute VRAM%, util%, summed power W, TPS over the last
    full minute), `active_models` (only `status='loaded'`), and `series`
    with per-minute `vram` / `util` / `power` / `tokens` arrays.
    Power series sums per-GPU averages across the box; degenerate
    `samples=0` rows are filtered.
  - **`GET /api/stats/v2/tokens-per-key?range=...`** — JWT-gated.
    LEFT JOINs `token_usage_minute` onto `api_tokens` so the response
    carries the human-readable `name` + `prefix` alongside `token_id`.
    Tokens with no usage in the window are omitted; orphan rows
    (usage exists but `api_tokens` row was deleted) surface as
    `"(unknown)"` rather than being dropped, preserving historical
    visibility. Rows sorted by `total_tokens DESC`.
  - **CTO decision #7 honoured** — v1 endpoints (`/api/stats/models`,
    `/api/stats/gpus`) coexist with v2 untouched; both response shapes
    are stable.

### Fixed
- **Tokenizer cache no longer leaks on unload (epic/overhaul S7, #124).**
  `TokenizerCache` (in `app/proxy/tokenizers.py`) held one fully-loaded
  `AutoTokenizer` per `(hf_repo, trust_remote_code)` tuple with no
  eviction path — a long-lived warden that cycled models would accumulate
  one entry per ever-seen repo and eventually OOM. Now exposes
  `evict(hf_repo)` (drops both trust-remote-code variants) and `size()`;
  the model unload route (`POST /api/models/{id}/unload`) wires the
  eviction into its success tail, so unload reliably frees the entry.

- **`/cache` page + chained model delete (#105, #117).**
  Frontend-only re-home of the Storage section.
  - New top-level `/cache` route lifts the HF cache surface out of `/stats`
    so it sits alongside Models / Tokens. Reuses `CacheTable` and
    `CacheGcButton` verbatim; the page adds a header, summary tiles
    (`Total cached` / `Repos` / `Orphans`), and a first-class empty-state
    hero card. SWR polls `GET /api/cache/models` every 30s and pauses when
    the tab is hidden. No new backend endpoints.
  - **Delete-model confirm modal** with an opt-in "Also free cache"
    checkbox. When ticked, the modal chains
    `DELETE /api/models/{id}` → `DELETE /api/cache/models/{repo}?force=true`
    in that order. Step 2 is best-effort; a 404 is treated as success
    (cache was already gone), and any other failure surfaces an inline
    notice ("Model removed; cache delete failed — visit /cache to retry.")
    while still notifying the parent that the row was deleted. If step 1
    fails, step 2 never runs. Closes #105.
  - Nav-bar gains a single Cache entry (HardDrive icon) between Stats
    and Settings.
  - One-time toast on `/stats` ("Storage moved to Cache") with a Dismiss
    button — persisted in `localStorage` as `vw.storage.toast.dismissed`
    so operators see the signpost exactly once per browser.

- **Chat playground at `/chat` (epic/overhaul S8, #117, closes #125).** New
  session-only chat surface for probing loaded models:
  - Model picker (filtered to `status === 'loaded'`, auto-selects first
    available and resets when the current selection unloads),
    temperature (0..2) and max_tokens (1..8192) sliders, Send/Stop
    button that flips role based on phase. Enter sends, Shift+Enter
    inserts a newline, Esc aborts an in-flight stream.
  - Streaming via a new server-side proxy at `POST /api/chat/completions`
    (JWT- and CSRF-gated by the existing middleware) which forwards to
    the local `/v1/chat/completions` over httpx + SSE; the proxy
    forges the `Authorization: Bearer …` header server-side from a
    plaintext-cached `vw-playground` system token so the bearer never
    enters browser memory.
  - `POST /api/chat/playground/ensure` mints and caches the
    `vw-playground` token on first visit (idempotent — handles
    fresh-DB, stale-cache, and DB-row-without-plaintext-cache
    recovery paths by sweeping stale rows and re-minting).
  - Playwright happy-path polls the new admin counter (below) to
    verify abort cleanup — the server-side finally block in the
    streaming generator decrements the counter on socket close,
    fencing against the SSE-leak risk called out in the slice plan.
  - Operator-cockpit aesthetic per the `frontend-design` skill output:
    monospace body, emerald-400 accents on slate-900, role rails
    instead of bubbles, blinking emerald cursor block while streaming,
    hover-only Copy + Regenerate-last on assistant messages.
  - Conversation history is React state only — refresh discards it.
    Neither localStorage nor the DB are touched. (Explicit S8
    contract — playground is a probe, not a journal.)
- **`GET /api/admin/active-requests` endpoint (epic/overhaul S8, #117, closes
  #125).** JWT-gated live counter of in-flight chat-completion streams —
  returns `{count: int}`. Operator-facing diagnostic for triaging stuck SSE
  proxies in production, and the polling target the Playwright happy-path
  uses to assert abort cleanup converges back to 0.
- **Add-model + Tune UX overhaul (epic/overhaul S4, #117).** Frontend-led
  rework of the two highest-traffic pages, plus thin backend support:
  - Settings page rebuilt as a sectioned instrument panel (Identity /
    Memory / Compute / Advanced) with collapsible discrete sections —
    each section's subtitle shows a live "N unsaved" count, surfacing
    dirty state at a glance instead of buried per-field.
  - Empty-state hero on `/models`. When no models are registered the
    page now centres a dashed-border card with a short explanation of
    what adding a model does, replacing the one-line placeholder. Part
    of #117 (S4 design principle #2).
  - New **Presets** strip: four curated chips
    (`a4000-tight-awq`, `h100-single-shot`, `dev-tiny`, `moe-balanced`)
    served from `GET /api/presets` (auth-required, JSON-backed by
    `app/presets/builtin.json`). Clicking a chip opens a confirm
    popover showing a per-key diff against the current draft. Apply
    merges the sparse preset into the draft — never auto-saves.
  - New **Suggest values** inline panel in the Memory section. Reuses
    the existing `GET /api/models/{id}/suggest-config` (S3) and renders
    the suggestion as a diff list. Apply / Dismiss are explicit; the
    `disclaimer` field is shown above the diff as rationale. Null
    fields are filtered out so they don't render as bogus "→ null"
    diff rows.
  - New **Effective argv** faux-terminal panel at the bottom of the
    settings page, backed by a new `GET /api/models/{id}/effective-argv`
    endpoint that returns the exact `vllm serve` argv that would be
    used for the *persisted* settings (preview port 10000). Re-fetched
    after every Save so the operator gets a one-second feedback loop:
    edit → save → see argv update. Changed tokens flash emerald via
    the existing `row-flash-emerald` keyframe; honours
    `prefers-reduced-motion`.
  - Add-model modal **shard grouping (#112).** Sharded weights
    (`model-00001-of-NNNNN.safetensors`) collapse into a single
    disclosure-triangle family row showing the total shard count.
    Single-shard and loose single-file weights stay as regular rows
    (no disclosure noise). Members appear when the row is expanded.
  - Add-model modal **per-file GGUF arch warnings (#101).**
    `discover_repo_files` now emits `gguf_arch_unknown` /
    `gguf_arch_unsupported` warnings against the new `KNOWN_GGUF_ARCHES`
    allowlist (21 families — see operating.md
    `#supported-gguf-architectures`). Arch is inferred from
    `config.general.architecture` first, then `model_type`, then
    `architectures[0]`, then a filename heuristic with longest-match-
    first ordering. The modal renders the warning inline against the
    offending file row with the inferred arch surfaced for
    debuggability.
  - Closes #101 (GGUF arch warnings) and #112 (shard grouping). Part
    of #117 (overhaul epic S4).
- **Live header-metrics widget (epic/overhaul S2).** Compact instrument
  cluster in the nav bar — VRAM%, GPU%, and the currently-loaded
  model name — refreshed via a single shared SSE stream
  (`GET /api/header/metrics/stream`, default 2s cadence, configurable
  through `VW_HEADER_METRICS_INTERVAL_S`, floored at 0.5s). The widget
  uses one `EventSource` per browser tab regardless of how many times
  `<NavBar />` re-mounts (React StrictMode, hot reload, route change),
  and shares `app.state.gpu_probe_cache` with `/api/system/gpus` so
  multi-tab operators don't double `nvidia-smi` load. Accent colour
  encodes status: slate=idle, emerald=model loaded, amber=probe error
  or reconnecting, red=terminal/401. Hidden on `/login` and `/setup`
  (server-side, via the same exact-match guard NavBar uses) plus on
  viewports below md (`hidden md:inline-flex`). Closes #67 (UI
  build-args), part of #117 (overhaul epic S2).
- **`build:ui` CI job and Dockerfile `ARG VW_BUILD_VERSION` /
  `ARG VW_BUILD_SHA`.** The frontend image is now built with the same
  identity build-args as the backend so the nav footer never falls
  back to "dev" / "unknown" on CI builds. Both images carry matching
  tags for a given commit. Closes #67.
- **`GET /api/models/{id}/suggest-config` (epic/overhaul S3, #113).** New
  endpoint returns a STARTING-POINT configuration blob for a registered
  model — `gpu_memory_utilization`, `max_model_len`, and `kv_cache_dtype`
  — explicitly NEVER auto-applied. Heuristics in `app/models/suggest.py`:
  `gpu_memory_utilization=0.92` (per-process util fraction);
  `max_model_len` from `config.max_position_embeddings` (None when
  absent — no guessing); `kv_cache_dtype='fp8'` iff the model looks
  AWQ-quantized (any of `config.quantization_config.quant_method=='awq'`,
  hf_repo name marker, or filename marker). Payload always carries a
  `disclaimer` field with the phrasing "starting points only and are
  never auto-applied" so a downstream consumer that auto-applies the
  values ignores a contract red flag. Closes #113.

### Changed
- **`/stats` Storage section removed (#105).** The HF
  cache table and GC button no longer render on `/stats`; visit `/cache`
  instead. A one-time `localStorage`-keyed toast on `/stats` is the
  bridge for operators who land on the old location out of muscle
  memory. Closes #105 (frontend half — the backend endpoints were
  already in place).
- **`/stats` page rebuilt on the v2 contract (epic/overhaul S7, #124).**
  Replaces the v1 GPU table and ad-hoc cards with persistent range, four
  time-series charts, four current-value tiles, active-models strip, and
  tokens-per-key table. See the corresponding `### Added` bullet above
  for the full feature breakdown.
- **Settings PATCH allowlist is now derived from `ModelRow` (epic/overhaul
  S3, #110).** `app/settings/routes_api.py` no longer hand-maintains the
  `_PATCHABLE_MODEL_FIELDS` set. The allowlist is computed once at
  import time as `dataclasses.fields(ModelRow) - _NEVER_PATCH`, where
  `_NEVER_PATCH` is the explicit blocklist of lifecycle-owned columns
  (`id`, `status`, `pulled_bytes`, `pulled_total`, `last_error`,
  `updated_at`, `created_at`). A startup-time drift guard raises if any
  name in `_NEVER_PATCH` is not present on `ModelRow` (modulo the
  documented `created_at` SQL-only exception). Side effect, intended:
  `filename`, `parallelism_strategy`, `max_batch_size`, `hf_config_repo`,
  and `tokenizer_repo` (added by #85 / #106) are now patchable; the
  hand-maintained list never picked them up. Plan called for
  `ModelRow.__fields__` (Pydantic accessor) — actual dataclass shape
  used `dataclasses.fields()`. Closes #110.
- **`wait_for_health` timeout injectable via module constant (epic/overhaul
  S3, #99).** `app/runtime/supervisor.py` now exports
  `DEFAULT_HEALTH_TIMEOUT_S: float = 600.0` and `wait_for_health` takes
  `timeout_s: float = DEFAULT_HEALTH_TIMEOUT_S` so callers can override
  without parroting the magic number. The production load runner already
  passes `settings.load_timeout_s` explicitly; this only changes
  ad-hoc callers (integration tests, scripts) and centralizes the
  single source of truth.

### Fixed
- Quarantined flaky `test_on_exit_callback_flips_status_to_failed_and_releases_port` until supervisor state-leak fix lands. Develop pipeline #6621 blocker. Related: #127.
- Quarantined 2 sibling `test_supervisor_real_subprocess.py` flakes (`test_supervisor_spawns_real_fake_process` + `test_supervisor_extra_env_reaches_subprocess`) — same `UnloadRefused: state is LOADING, not READY` family as the sibling above. Pre-emptive on the same MR per the 2026-05-23 "don't ship known-failing siblings" lesson. Tracked under #143; restore once the supervisor state-leak fix lands.
- **Nav-bar double "v" version prefix (#83).** When the backend
  returned a CalVer tag that already included a leading `v` (e.g.
  `v2026.05.19.1`), the nav footer prefixed its own `v`, producing
  `vv2026.05.19.1`. `formatVersion` now emits the `version` field
  verbatim — the fallback string retains the literal `v? · ?` for the
  "could not load" case so the shape stays unambiguous. Regression
  pinned in `frontend/tests/component/nav-bar.test.tsx`.
- **Exact-match unauthenticated-route guard (#39).** NavBar previously
  used `path.startsWith('/login')` to suppress the nav chrome on
  unauthenticated routes, which silently swallowed any path beginning
  with `/login` (a hypothetical `/login-help` marketing page would
  have rendered as unauthenticated). The guard now exact-matches
  `/login`, `/login/`, `/setup`, `/setup/` and retains the
  `/setup/*` wizard subtree carve-out. `auth-fetch.ts` got the
  matching exact-match check on its 401 redirect path so it doesn't
  loop-redirect from a hypothetical `/login-help` either. Pinned in
  `nav-bar.test.tsx` (5 new cases).
- **Flaky `401 invalid credentials` in JWT login fixtures (#55).**
  `tests/conftest.py` now ships `seed_admin_user` and `jwt_login`
  shared helpers that close the WAL frame propagation race between
  the sync `sqlite3` admin seed and the async `aiosqlite` route
  handler. Primary fix is S5's `seed_admin_user()` (adopted verbatim
  during the S2→develop reconcile): `PRAGMA journal_mode = WAL`
  matching the app pool, an explicit `BEGIN IMMEDIATE` for the INSERT,
  and a `PRAGMA wal_checkpoint(FULL)` before close so the frame is
  visible to every subsequent reader. Defence in depth is `jwt_login`'s
  bounded retry-on-401 (`LOGIN_RETRY_MAX = 5`, exponential backoff,
  returns immediately on any non-401). 18 unit tests + 1 integration
  test migrated to the shared helpers (4 token-auth and seed-only
  tests skipped — no race).
- **`mark_runtime_dead_on_startup` now wipes `pulling` rows + zeros pull
  progress (epic/overhaul S3, #11).** Before this change, a process
  killed mid-pull would leave the row stuck at `status='pulling'`
  forever with stale `pulled_bytes` / `pulled_total` — the only recovery
  was a manual SQL update. The startup sweeper now flips `pulling` rows
  to `failed` alongside the existing `loaded`/`loading`/`unloading`
  sweep, and uses `CASE WHEN status = 'pulling' THEN 0 ELSE pulled_*
  END` so progress counters are zeroed for `pulling` rows only — the
  `pulled_total` on a `loaded` row is the persisted weights size and
  must NOT be wiped. Closes #11.
- **`health_ok` column now written after warmup probe succeeds
  (epic/overhaul S3, #29).** The load runner in
  `app/models/routes_api.py` calls `RuntimeRepo.update_health(model_id,
  True, <utcnow>)` immediately after `Supervisor.mark_ready()` so the
  `runtime` row reflects the actual health state. Before, `health_ok`
  stayed at its previous value (frequently `0`) even when the model
  was successfully serving; the column is read by the live status
  badge on `/models/[id]`, so the UI was silently degraded.
- **Operator guidance: `VW_CONTAINER_GPU_COUNT` (epic/overhaul S3, #46).**
  New 'Container GPU isolation' chapter in `docs/operating.md` covers
  what the variable does (the container's stated GPU-slice size that
  the wizard, validators, and probes index against), how to set it for
  whole-host / split / single-GPU deployments, a worked docker compose
  example, and common misconfig symptoms (count > exposed → silent
  until pull/load CUDA-error; count < exposed → wizard hides GPUs).
  Explicit decoupling note: warden never reads
  `NVIDIA_VISIBLE_DEVICES`; operators reconcile the two manually.

### Removed
- **Bench v2 subsystem fully excised (epic/overhaul S1).** Every
  Bench-v2 source file, persistence table, HTTP route, UI surface,
  and test is gone. Specifically: `app/bench/` (supervisor, runner,
  cli, harness, events, internal_routes, best, repos.bench), the
  `/api/bench/*` and `/api/models/{id}/benchmark` endpoints, the
  `/benchmarks` UI page + `/models/[id]` Benchmark tab,
  `FromBenchmarkChip` and its consumers on the settings page, the
  `BENCH_HEALTH_WAIT_S` env var + `bench_health_wait_s` setting,
  `_is_bench_internal_path` in CSRF middleware, the
  `integration-tests` CI job, `scripts/publish_to_hub.py` (dead
  publisher), `scripts/build_long_corpus.py` (bench fixture
  generator), and the bench-v2 design spec at
  `docs/superpowers/specs/2026-05-13-vllm-warden-benchmark-v2-design.md`.
  Migration `0017_drop_bench.sql` drops the `bench_run`,
  `bench_load_config_attempt`, and `bench_cell` tables (children
  first; idempotent via `DROP TABLE IF EXISTS`). The 5xx envelope
  enricher (`app/proxy/envelope_hint.py`) was rewritten as a slim
  ~80-LOC stub that attaches `models.last_error` to OpenAI-style
  error envelopes when the upstream returns 5xx — replacing the
  bench-coupled hint engine. Closes #14, #15, #16, #17, #20, #24,
  #25, #31, #47. (~23k source / ~11.6k test LOC removed; rationale:
  external benchmarking tool, see overhaul plan §S1)
- **JWT re-login required after this rollout.** No auth-cookie
  schema change, but the bench WS-event channel previously held
  long-lived browser sessions open; operators with stale tabs
  pointing at `/benchmarks` will see 404 on next click and should
  refresh. New nav menu drops the Benchmarks entry (Models /
  Tokens / Stats / Settings remain).
- **Migration 0017 caveat.** The migration uses
  `DROP TABLE IF EXISTS` so it is idempotent and safe against
  pre-overhaul DBs that may have only some of the three tables
  (e.g. a partially-failed prior install). Downgrading past 0017
  is not supported — the bench code is gone, so there is nothing
  to recreate the tables for. Migration 0016 is intentionally
  reserved (the overhaul plan slot it lived in moved to a later
  slice; numbering stays monotonic).

### Changed
- **Pin `transformers <5.7` and `huggingface_hub <1.16`** in
  `requirements-dev.txt`. Transformers 5.7.0 + hub 1.16.x broke
  pytest-asyncio fixture collection (AttributeError on
  `tests/unit/system/test_hf.py`: `'function' object has no attribute
  '__func__'`). Locked to versions matching the vllm/vllm-openai 0.20.0
  base image (5.6.2 / 1.12.0). Unrelated to bench removal — surfaced by
  fresh pip install on this branch's CI re-run.
- **`app/proxy/envelope_hint.py` replaced with `models.last_error` stub
  (~80 LOC).** The bench-coupled 5xx hint engine was rewritten as a slim
  stub that attaches `models.last_error` from the model row to OpenAI-style
  error envelopes when the upstream vLLM returns 5xx. No regression to bare
  500s — operators see the last known error from the model row.
- **Supervisor refuses unload during model load/warmup.** The vLLM
  subprocess lifecycle now goes `LOADING → WARMING → READY → UNLOADING`
  inside the supervisor. Calling `POST /api/models/{id}/unload` while
  the model is not yet `READY` returns 409 with the current state in
  the body. To force-terminate a stuck or still-warming model, use
  `POST /api/models/{id}/unload?force=true`. A warmup verification
  probe (`POST /v1/completions max_tokens=1`) runs after vLLM's
  `/health` returns 200 — DB status only flips to `loaded` once that
  probe succeeds, closing the race window where Qwen3-VL's
  `_warmup_mm_processor` was still running but the row looked
  serviceable. When the probe or health-wait fails, the row is marked
  `failed` and the subprocess is **left running** holding its GPUs
  until an explicit `?force=true` unload — operators must check
  `last_error` (contains `subprocess still holding`) and decide
  whether to retry or release. The `warmup_probe_timeout_s` setting
  (default 60s, env `VW_WARMUP_PROBE_TIMEOUT_S`) controls the probe
  budget. See spec
  `docs/superpowers/specs/2026-05-20-vllm-warden-unload-race-design.md`.

### Fixed
- **#104 — Flaky `tmp_data_dir` fixture race in test infra.** `tests/conftest.py:seed_admin_user()`
  pre-creates the `data/` parent directory before yielding the bind-mount path, so concurrent
  test-worker fixture instantiation can no longer race the `mkdir`. Pinned by a 10-iteration
  parametrized stability test (`tests/unit/tokens/test_rotate_endpoint_104_stability.py`)
  so the flake cannot silently return. (S5 scope — surfaced while extending the token test suite.)
- **Qwen3-VL crash loop during multimodal warmup.** Three independent bugs
  surfaced loading `Qwen3.6-35B-A3B-AWQ-4bit` on a fresh warden host: (1)
  `extra_args` from the model row was defined in the schema but never
  forwarded to `vllm serve` argv, so per-model flags like
  `--enable-auto-tool-choice --tool-call-parser qwen3_xml` were silently
  dropped (`app/runtime/cmd_builder.py`); (2) `_extract_prompt` in the
  proxy assumed `message.content` was always a string, raising on the
  OpenAI multimodal shape where `content` is a list of
  `{"type":"text"|"image_url", ...}` blocks (`app/proxy/routes.py`); (3)
  HuggingFace Hub's revalidation HEAD requests were unauthenticated
  because the new `HF_TOKEN` env var (the `huggingface_hub` >=0.23 name)
  was not set in the subprocess env — only the legacy
  `HUGGING_FACE_HUB_TOKEN` was — making cold multimodal warmup slow
  enough to expand the SIGTERM race window (`app/runtime/env_builder.py`,
  also locked against extra_env override).

### Added
- **Per-token rate limit, priority, and 24h usage rollup (S5, closes #104).**
  Migration `0018_tokens_rate_priority.sql` adds two columns to `api_tokens`:
  `rate_limit_tps INTEGER NULL` (NULL = unlimited; positive integer = sliding
  10-second budget in tokens/sec) and `priority INTEGER NOT NULL DEFAULT 5`
  bounded 0..9 via BEFORE INSERT/UPDATE triggers. A new `token_usage_minute`
  table holds per-token, per-minute request/prompt/completion counters, and
  the proxy success path now writes one row per request alongside the
  existing `/counters` + `/model_samples` writes. New endpoints:
  `PATCH /api/tokens/{id}` (update rate/priority), `GET /api/tokens/{id}/usage?range={1h|24h|7d}`
  (minute-bucketed rollup + totals), and `POST /api/tokens/{id}/test` (lookup
  + scope projection + proxy liveness check for the UI "Test token" button).
  The proxy adds a sliding-window `TokenRateLimiter` (rejected requests do
  NOT consume budget — otherwise oversized prompts lock the token out
  indefinitely) and a STRICT `PriorityScheduler` (priority-9 always served
  before priority-0). Rate-limited requests get **HTTP 429** to match
  OpenAI client retry semantics. The window length is operator-tunable via
  `VW_RATE_LIMIT_WINDOW_S`. **Starvation is by design under STRICT
  scheduling** — priority-0 tokens can wait indefinitely under sustained
  priority-9 pressure; documented in `docs/operating.md` and surfaced in
  the create-dialog form helper text + the priority column tooltip. The
  `/tokens` UI page grew three columns (Rate / Priority / Last 24h) and
  a Test button per row.
- **HF cache management endpoints + /stats Storage section (#114).** New
  Storage section on `/stats` lists every cached HuggingFace repo under the
  configured cache root with size, last-used age, and the model rows that
  own it (or "orphan" if none). Per-row Delete refuses if the repo backs an
  active model (`loaded`/`loading`/`unloading`/`pulling`) and requires
  `?force=true` when the row is `pulled`/`idle` but not active. A
  "Garbage-collect" button shows a dry-run preview (orphans + failed rows
  older than 24h by default) before the destructive call, so the operator
  confirms the exact bytes/repos that will be freed. Three new endpoints:
  `GET /api/cache/models`, `DELETE /api/cache/models/{repo:path}`,
  `POST /api/cache/models/gc?dry_run={bool}&failed_older_than_hours={int}`.
  All gated by `require_jwt`; destructive ones additionally validate
  `repo` against `^[\w.-]+/[\w.-]+$` (no `..`, single slash) at the
  boundary and 400 on traversal-shaped input. Filesystem walks run on
  `asyncio.to_thread` so a multi-GiB rmtree never blocks the event loop.

### Fixed
- **#115** Qwen3.5 / Qwen3.6 MoE GGUFs (e.g. `unsloth/Qwen3.6-35B-A3B-GGUF`) now
  load. Companion to #107: that earlier patch handled the dense `qwen3_5 → qwen35`
  rename, but the MoE variant declares `model_type: "qwen3_5_moe"` /
  `architectures: ["Qwen3_5MoeForConditionalGeneration"]` and hit the same
  `Unknown gguf model_type: qwen3_5_moe` failure at
  `vllm/model_executor/model_loader/gguf_loader.py:208`. vLLM v0.20.0 already
  ships the `Qwen3_5MoeForConditionalGeneration` model class and the bundled
  gguf-py already has `MODEL_ARCH.QWEN35MOE='qwen35moe'` with the correct fused
  `experts.gate_up_proj` / `experts.down_proj` tensor map that the model class
  expects — only the rename was missing. Dockerfile now runs a third `sed -i`
  (right after the #107 RUN) inserting `if model_type == "qwen3_5_moe":
  model_type = "qwen35moe"`. We intentionally do NOT install a manual
  `gguf_to_hf_name_map` block (the existing `qwen2_moe`/`qwen3_moe` branch
  targets the older split-expert layout `experts.X.gate_proj/up_proj/down_proj`
  which the new fused qwen3_5 MoE doesn't use; `gguf.get_tensor_name_map(
  MODEL_ARCH.QWEN35MOE, ...)` covers the new layout natively). The RUN is
  guarded by `grep -q` on the post-patch marker and `py_compile`s the file so a
  broken sed fails the image build loudly. No upstream PR yet — vLLM PR #38140
  covers `qwen3_5` dense only and is still OPEN. Remove this RUN + #107 + #108
  RUNs when we bump the vLLM base past a release that adds the MoE rename.
- **#111** Sharded HF safetensors pull only fetched the first shard. Picking
  a `model-NNNNN-of-NNNNN.safetensors` member in the Add Model wizard now
  expands the HF `allow_patterns` to cover all sibling shards plus
  `model.safetensors.index.json` (the vLLM weight_map). The shard-set regex
  that fit-preview already used (`app/models/routes_api.py` `_shard_glob_for`)
  is lifted to a shared module `app/models/sharding.py` and reused by
  `allow_patterns_for` so both code paths agree on what "the whole shard
  family" means. Affects every sharded repo — AWQ/GPTQ/AutoRound/FP8/BF16/GGUF
  — broken since v2026.05.19.3 (#85). Fixes #111.
- **#108** Multimodal Qwen3.5 / Qwen3.6 GGUFs (e.g. `unsloth/Qwen3.6-27B-GGUF`) now load fully without crashing.
  After #107 renamed the architecture so `Qwen3_5ForConditionalGeneration`
  resolves, the next line of vLLM's `gguf_loader.py` (`vision_num_layers =
  config.vision_config.num_hidden_layers`, line 214 in vLLM 0.20.0) crashed
  with `AttributeError: 'Qwen3_5VisionConfig' object has no attribute
  'num_hidden_layers'`. The new `Qwen3_5VisionConfig` (transformers 5.6.2,
  `transformers/models/qwen3_5/configuration_qwen3_5.py:118-140`) follows the
  Qwen2.5-VL precedent and stores the vision layer count as `depth: int = 27`
  with no `num_hidden_layers` shim. The Dockerfile now runs a second `sed -i`
  (right after the #107 RUN) that rewrites that one line to
  `vision_num_layers = getattr(config.vision_config, "num_hidden_layers",
  None) or config.vision_config.depth` so legacy vision configs that DO
  expose `num_hidden_layers` are unaffected and the new Qwen3.5 family
  picks up `depth` as the fallback. The RUN is guarded by a `grep -q` on
  the post-patch marker so rebuilds re-run the step idempotently, and
  `py_compile` runs at the end so a broken sed fails the image build
  loudly. Same upstream PR ([vllm#38140](https://github.com/vllm-project/vllm/pull/38140))
  is the long-term fix — drop both RUN steps when we bump the vLLM base
  image past that merge.
- **#107** Qwen3.5 / Qwen3.6 GGUFs (e.g. `unsloth/Qwen3.6-27B-GGUF`) now load.
  vLLM 0.20.0's `gguf_loader.py` ships a model_type translation map for
  `qwen2_moe` / `qwen3_moe` / `gemma3_text` / `cohere` / `deepseek_v3` etc. but
  is missing the `qwen3_5` → `qwen35` rename, so any GGUF whose HF config
  declares `model_type: "qwen3_5"` died at load with `Unknown gguf model_type:
  qwen3_5`. Upstream fix lives in
  [vllm#38140](https://github.com/vllm-project/vllm/pull/38140) but the PR is
  OPEN — no released vLLM version contains it. As a workaround the Dockerfile
  now runs a one-line `sed -i` against the installed
  `vllm/model_executor/model_loader/gguf_loader.py` at image build time,
  inserting an `if model_type == "qwen3_5": model_type = "qwen35"` block right
  after the existing `gemma3_text` rename. The RUN step also greps the result
  and `py_compile`s the patched file so a broken sed fails the image build
  loudly. Stale `v0.6.3` comment on the `FROM` line was corrected to `v0.20.0`
  to match the pinned digest. Remove the patch + RUN once upstream PR is
  merged and we bump the vLLM base image to a release that contains it.
- **#106** GGUF deployments whose quantized repo omits `config.json` (the
  common unsloth republish shape) now boot. `app/runtime/cmd_builder.py` emits
  `--hf-config-path <hf_config_repo>` and `--tokenizer <tokenizer_repo>` when
  the new columns are set; migration `app/db/sql/0015_models_hf_config_tokenizer.sql`
  adds nullable `hf_config_repo` and `tokenizer_repo` to `models`; the
  Pydantic schema (`app/models/schemas.py`) validates them with the same
  `^[\w.-]+/[\w.-]+$` owner/name pattern as `hf_repo` and a
  `mode="before"` field validator coerces empty strings to `None` so a
  cleared FE field round-trips. The Add Model wizard (`frontend/src/components/models/add-model-modal.tsx`)
  grows a "Base repo (config + tokenizer)" Input in the Advanced section
  that fans out to both fields by default, plus a nested `<details>`
  "Override tokenizer separately" for the rare split case. Legacy / non-GGUF
  rows are unaffected — both columns stay `NULL` and the cmd builder skips
  the flags. This closes the `--tokenizer` / `--hf-config-path` follow-up
  explicitly deferred from #100.
- **#100** GGUF deployments now boot. `app/runtime/cmd_builder.py` derives the
  vLLM-required `:quant_type` suffix from `model.filename` (regex
  `-(Q\d[_A-Za-z0-9]*)\.gguf$`) and emits `--model <hf_repo>:<quant>`. Without
  this every GGUF model added since #85 (v17.13) failed at vllm subprocess
  startup with rc=1 because vLLM 0.20.0 requires the colon form. Safetensors /
  non-quantized rows are unchanged — a missing filename or non-matching
  pattern falls back to the bare `hf_repo`. The `--tokenizer` /
  `--hf-config-path` follow-up landed in #106.

### Added
- **Frontend: Add Model wizard advanced section + tooltip math + parallelism
  wiring (#87).** `frontend/src/components/models/add-model-modal.tsx` grows
  a collapsible "Advanced" disclosure (native `<details>`, collapsed by
  default) housing three new inputs: `parallelism_strategy` (auto/tp/pp),
  `max_batch_size` (1–64), and an explicit `max_model_len` override. All
  three flow through to `POST /api/models` so the backend's #88 wiring sees
  them. Single-host pipeline-parallel is explicitly allowed — no client-side
  guard against `pp` when only one host is configured, mirroring the
  cmd_builder's behaviour. The fit-badge tooltip already surfaced the four
  primitives (`bytes_per_token`, `kv_reserve`, `weights_budget`, `ratio`);
  `orange` ("tight") rows now also surface a "Recommended max_model_len"
  hint below the badge, preferring the backend's `recommended_max_model_len`
  field and falling back to a client-side solver in `frontend/src/lib/fit.ts`
  (`recommendMaxModelLen`) when the backend value is null. New
  `RECOMMENDATION_TARGET_RATIO=0.70` constant mirrors `app/models/fit.py`
  so the two solvers agree. Vitest component tests pin the collapse
  behaviour, the input wiring, the no-PP-block contract, the four-primitive
  tooltip, and the verdict-gated recommendation hint. **CR fix-up:**
  `fetchFitPreview()` now threads `max_batch_size` and `max_model_len`
  through the `POST /api/models/fit-preview` body whenever the operator
  has set them in the Advanced section, and a 300 ms-debounced effect
  refetches on each numeric edit so the tooltip's `kv_reserve` / verdict /
  recommended-L stay live as the operator tunes batch & context. A new
  per-request `fitSeqRef` short-circuits stale writes when a slow earlier
  fetch lands after a fresh override. Unit coverage for
  `recommendMaxModelLen()` lives in
  `frontend/tests/contract/fit-classifier.test.ts` (happy path against the
  backend formula + every null-return branch); component coverage in
  `frontend/tests/component/add-model-modal.test.tsx` pins the FE-fallback
  hint on orange rows when the BE returns `recommended_max_model_len: null`,
  the debounced refetch carrying overrides, and a consequence check that
  asserts `kv_reserve` scales ~16× with `max_batch_size=16` vs the baseline.
- **Backend: parallelism strategy + gated-repo polish + shard aggregation
  finalise (#88).** `app/runtime/cmd_builder.py` now consults
  `ModelRow.parallelism_strategy` (from migration 0014): `auto`/`tp` keep
  emitting `--tensor-parallel-size=N` (legacy behaviour, default), `pp`
  swaps in `--pipeline-parallel-size=N` instead. Single-host PP is not
  blocked at the builder layer — vLLM accepts it and operators benchmark
  both strategies in the wizard. Benchmark-v2's `tensor_parallel_size`
  override still drives the parallelism *degree* regardless of which
  flag is emitted, so existing sweep semantics keep working when an
  operator flips strategy mid-experiment. `app/models/pull_task.py`
  surfaces gated/private HF repos as a typed `PullAuthRequired` sentinel
  (mirrors `DiscoveryAuthRequired` from #84): both the metadata fetch
  (`estimate_repo_bytes`) and the actual `snapshot_download` route 401/403
  refusals through `_classify_hf_auth_error`, and `run_pull` writes
  `last_error = "auth_required: <hint with /setup/hf-token> (<exc>)"` —
  same prefix the FE keys off in the discovery stage. 5xx HF outages
  still fall through to the generic `pull error:` envelope so operators
  don't chase a token rotation for a server-side problem. Fit-preview
  shard aggregation in `app/models/routes_api.py` now covers GGUF splits
  (`*-NNNNN-of-NNNNN.gguf`) alongside the safetensors set shipped in #85;
  picking part 1-of-3 of a Llama 70B GGUF classifies against the full
  ~40 GB, not 13 GB.
- **Frontend: Add Model wizard rebuilt as a 4-state machine (#86).**
  `frontend/src/components/models/add-model-modal.tsx` now drives a
  `enter-repo → discovering → select-file → submitting` flow against the new
  `GET /api/models/discover` and `POST /api/models/fit-preview` endpoints.
  The select-file stage renders the repo's files (filename, size, kind
  badge, quant, fit verdict) with a per-row colored fit badge; ticking the
  GPU checkboxes recomputes the verdict client-side via the mirrored
  classifier in `frontend/src/lib/fit.ts` (constants locked to
  `app/models/fit.py:21-23`). The badge tooltip exposes the underlying
  math (`bytes_per_token`, `kv_reserve`, `weights_budget`, `ratio`).
  GGUF rows surface a soft "vLLM GGUF serving not yet supported" warning
  banner but still allow submit (operator override). Gated/private repos
  return `401 auth_required` → the modal swaps the file table for a CTA
  linking to `/setup/hf-token`. Vitest component + contract tests plus a
  route-mocked Playwright spec lock the new contract.
- **Backend: VRAM-fit math + per-file download + schema bump (#85).**
  New `app/models/fit.py` module with pure functions
  (`dtype_bytes_from_torch_dtype`, `kv_reserve_bytes`, `weights_budget_bytes`,
  `classify_fit`, `recommend_max_model_len`) — the source-of-truth for the
  4-way fit verdict (green < 0.55 ≤ yellow < 0.80 ≤ orange < 1.0 ≤ red).
  New `POST /api/models/fit-preview` route — accepts the discover-time
  config + the operator's GPU pick + max_model_len, and returns
  `verdict`, `ratio`, `kv_reserve`, `weights_budget`,
  `file_size`, plus a `recommended_max_model_len` for orange/red.
  Multi-part safetensors shards (`prefix-NNNNN-of-NNNNN.safetensors`)
  are aggregated to their full set before classification, so picking
  shard 1-of-4 of a 30 GB model classifies against 30 GB, not 7.5 GB.
  `ModelCreate` schema gained three persisted fields: `filename`
  (optional pinned weights file), `parallelism_strategy`
  (`tp`/`pp`/`auto`), `max_batch_size` (1-64). Migration 0014 extends
  the `models` table with matching columns + defaults so legacy rows
  decode unchanged. `app/models/pull_task.py` plumbs an
  `allow_patterns = [filename, config.json, tokenizer*, *.txt, *.md]`
  through both `estimate_repo_bytes` and `snapshot_download` when a
  filename is pinned — the disk-shortage pre-check now sizes against
  the filtered set, fixing a false shortage on 19.8 GB single-file
  pulls of repos with 200 GB total weights.

### Changed
- **SSE wire alignment.** Renamed/added fields on request_started,
  request_completed, request_summary, bench_progress to match the FE
  bench-event contract (resolves Phase 3 CR + QA blockers).
- **FE wire alignment with BE Bundle 2.** Reducer + types updated to
  consume the canonical BE wire shape (`status` string,
  `elapsed_ms`, `tokens_in`/`tokens_out`, `dropped_*`, `*_count`,
  `bench_progress.phase`). Dropped four event-type constants the BE
  does not emit (`RESUMED`, `LOAD_DONE`, `PHASE1_STEP`,
  `CELL_STARTED`) and their reducer branches. Regenerated
  `api-types.generated.ts` to pick up the new
  `MatrixOverrides.phase1_request_timeout_s` field. Added
  `tests/contract/bench-sse-wire.test.ts` — a 5-test suite that
  replays frozen BE-shipped JSON fixtures through `benchReducer` and
  fails loudly if the wire shape drifts from what the FE consumes.

### Added
- **Bench FE — virtualized Events tab with filter chips + per-request rows
  (#77, #75).** Replaces the prior `EventLog` single-feed component with a
  Virtuoso-backed list that interleaves request lifecycle rows
  (`request_started` → `request_completed` upsert keyed by `request_id`)
  with the raw event tail. Five-way filter (`system` / `cell_result` /
  `phase1_envelope` / `request` / `other`); empty selection = "show
  everything" (Gmail-pill UX). Backpressure summary banner appears when
  `request_summary.dropped_started + dropped_completed > 0`. Status chip
  variants: streaming → info, ok → success, fail / http_error → error,
  truncated / timeout → warning. Request grid: `req_id | status |
  elapsed | ttft | tok_out | fail_reason`. FE ring-buffer cap is
  `MAX_REQUEST_ROWS = 2000` (mirrors BE `REQUEST_EVENT_BUDGET_PER_S × 40
  s` design horizon — a regression test in
  `tests/component/virtualization-smoke.test.tsx` pins the FE constant
  so silent drift trips at test time). Shared `useStickyBottom` hook in
  `src/components/shared/use-sticky-bottom.ts` keeps log-stream and
  events-tab in lockstep on the "scroll up → free; jumpToLatest → stick"
  behavior so the two streaming surfaces can't drift on the first bug
  fix. Mounted in both `/benchmarks/[runId]` and the model-detail
  Benchmark tab.
- **Bench FE — virtualized model log tail with elided-count marker
  (#75).** `LogStream` switched from `<pre>` + `AnsiLog` to Virtuoso
  rendering with stable monotonic ids per line so FIFO eviction
  (`MAX_LINES = 5000`) doesn't re-mount surviving rows. An elided-count
  banner ("… N older line(s) elided") renders above the scroll region
  when the buffer has wrapped, so a long-running run doesn't silently
  appear "stuck" once it crosses 5k log lines. Vitest setup gained a
  `vi.mock('react-virtuoso')` shim that renders every item as a plain
  div: jsdom can't drive ResizeObserver, so the real library renders
  zero rows and every assertion downstream of it fails. The shim
  passes through `data` / `itemContent` / `computeItemKey` / `style` /
  `components.List` so role="log" semantics survive virtualization.
- **Bench FE — live Phase 1 probe view in `EnvelopePanel` (#76).** Until
  `phase1_envelope` lands, the panel renders the running probe state
  from `phase1_progress` instead of a 30–60 s "measuring…" skeleton.
  Shows probes_done/total + progress bar, current concurrency × max_new,
  pass-rate %, p50/p95 latency, truncated/failed/timeout counts.
  Selection order is `final` → `running` → `empty`/skeleton so the
  settled answer always wins when both slices are populated. The
  `EnvelopeLimitedBy` union widened to include `"timeout"`,
  `"ceiling"`, `"truncation"` (badge variants: timeout/oom → warning,
  rest → info).
- **Bench FE — sticky run-progress banner with phase chip + ETA (#78).**
  New `ProgressHeader` mounted at the top of both bench detail surfaces.
  Self-hides until the cli emits the first `bench_progress`; resets
  cleanly on `load_attempt` because the reducer wipes the slice when a
  new lc starts (so a fresh attempt's bar restarts at 0% rather than
  inheriting the previous attempt's tail). Shows phase chip (phase1 =
  info, phase2 = success), progress bar, %-complete, cells_done/total,
  elapsed, ETA. ETA renders "—" until the cli has ≥3 samples.
- **Bench FE — operator-tunable Phase 1 request timeout in the start-run
  form (#79).** `ModelBenchCard` on `/benchmarks` gained a numeric
  input (range 1..600, empty ⇒ 30 s server default) wired to
  `matrix_overrides.phase1_request_timeout_s` in the POST body. Helper
  text under the field surfaces the range + default so operators
  don't have to dig into the matrix-overrides schema to find the knob.
- **Bench FE reducer + types — Bundle 2 surface (#76, #77, #78, #80).**
  `useBenchSSE` grew six new state slices — `envelope_running`,
  `requests`, `requests_summary`, `progress`, `system_warning` — fed
  by the new event types `phase1_progress`, `request_started`,
  `request_completed`, `request_summary`, `bench_progress`,
  `system_warning`. `load_attempt` now resets every lc-scoped slice
  (envelope, envelope_running, requests, requests_summary, progress)
  so a new load-config never displays the previous one's tail. New
  reducer tests in `tests/component/bench-reducer-bundle2.test.tsx`
  pin the dispatch behavior for every Bundle 2 event type + the
  load_attempt reset. Event-type strings live in
  `src/lib/bench-event-types.ts` as `EVENT_TYPES_ALL` so no caller
  hand-rolls a literal.
- **Bench Phase 1 per-request wall-clock timeout (`phase1_request_timeout_s`).**
  Before this change a stalled vLLM server (queue saturated past the
  per-request budget) let the bench loop hang on a single
  `client.stream(...)` call until httpx's pool-level read timeout (120 s)
  fired, *then* surfaced as `http_error=True` (a 5xx-style ceiling). The
  envelope finder couldn't distinguish "server stalled" from "model
  produced garbage" because both ended up in the same `fail_count` /
  `pass_rate=0` bucket. `_stream_one` now wraps the entire stream in
  `asyncio.wait_for(..., timeout=request_timeout_s)`; on expiry the
  returned `_ReqResult` carries `timed_out=True` and
  `latency_ms == request_timeout_s * 1000`. The aggregator buckets those
  into a new `timeout_count` (excluded from p50/p95/ttft/pass_rate) and
  emits `limited_by="timeout"` when timeouts dominate the failure modes
  for the cell. `matrix_overrides.phase1_request_timeout_s` (1..600,
  default 30) flows through `_drive_run` to every Phase 1 probe and
  Phase 2 load cell, replacing the implicit p95-latency cap that
  `_classify` used to apply. Closes #79.
- **Bench `/internal/event` passthrough enabler.** The existing
  `/internal/cell` and `/internal/status` routes are tied to specific
  `bench_cell` / `bench_run` side-effects; any *additional* event type
  the cli wants the SSE relay to see (fine-grained `cell_progress`
  ticks, the `populated_buckets` warning, future hooks) flows through
  the new `POST /api/bench/runs/{run_id}/internal/event`. Body cap is
  4 KB (matches `MAX_EVENT_BYTES`; checked on `content-length` header
  THEN raw body so an unannounced upload can't OOM the parser); per-run
  rate cap is 1000 eps via a token bucket (`BENCH_EVENT_RATE_LIMIT_EPS`,
  refilled on monotonic clock). Structural fields `ts` / `run_id` /
  `seq` are stripped silently — the supervisor + EventWriter are the
  sole source of truth. Auth: same bench-internal token scheme as the
  other internal routes. Status codes: 200 ok, 400 bad payload (missing
  `type`, non-object, malformed JSON), 413 oversize, 429
  rate-limit-exceeded (with `Retry-After: 1`). Bucket lifecycle is
  reset on run cancel and run-failed paths so a fixture-reused
  `run_id` starts with a full bucket. Sync 1 of v17.11 Bundle 2 —
  unblocks Task C and dev-2's UI surfacing.
- **Bench corpus stub detection (`populated_buckets`).** v1 ships
  `app/bench/corpus/128k` and `1m_plus` as README-only stubs; before
  this change the matrix sweep would still enumerate cells against
  them and every request would fail "no prompts for size" — N empty
  cells per load-config with no operator-visible reason. The new
  `matrix.populated_buckets(corpus_root) -> (buckets, skipped)` scans
  `<corpus_root>/<bucket>/*.json` non-recursively and splits the
  canonical bucket set into populated (a `dict[str, int]` that drops
  straight into `cells_for_envelope(buckets=...)`) and skipped (the
  unpopulated names in canonical order). `_drive_run` calls the
  scanner once at the top and emits a single `system_warning` event
  (`type=system_warning`, `kind=populated_buckets`, `skipped=[...]`,
  `populated=[...]`) via the new `post_event` passthrough when the
  skipped list is non-empty. The populated subset feeds
  `cells_for_envelope` so Phase 2 only iterates real content. Failure
  to emit the warning is non-fatal (log-and-continue) — the event is
  informational. Closes #80.
- **Bench corpus: long-context tiers populated (64k / 128k / 256k /
  512k / 1m_plus).** v1 shipped `128k` and `1m_plus` as README-only
  stubs (the trigger for the #80 `system_warning`); this change adds
  real prompt files for those two and introduces three new tiers —
  `64k`, `256k`, `512k` — bringing the bench length axis to nine
  buckets. `matrix.DEFAULT_BUCKET_TOKEN_BUDGETS` extends from six to
  nine entries (`1k`, `4k`, `16k`, `31k`, `64k`, `128k`, `256k`,
  `512k`, `1m_plus`) so the sweep enumerates the new tiers without
  any matrix-override. Per-tier prompt counts: 6 for
  `64k`/`128k`/`256k`/`512k`, 4 for `1m_plus` (million-token prompts
  already cost real GPU minutes per request — four is enough to spot
  a regression without making a sweep prohibitive). Each prompt is
  one of three task families wired to the existing bench graders:
  `needle` (literal-string recall), `keyword_in` (case-insensitive
  substring presence), `json_extract` (single-field JSON output);
  this is the same grader contract `populated_buckets` and the
  Phase 2 cell renderer already speak. Token sizes are measured with
  tiktoken `cl100k_base` and held to a ±10% band per tier
  (`test_long_buckets_are_sized_appropriately` extended for all five
  new tiers). The corpus is generated by a new deterministic CLI —
  `python -m app.bench.corpus._generate <tier>` — which materialises
  one tier from a fixed manifest plus
  `app/bench/corpus/_haystack.SEEDS` (28 license-clean synthetic
  technical paragraphs written for this benchmark, no third-party
  content); two runs with the same args produce byte-identical JSON,
  so the files are checked in rather than rebuilt at sweep time. Per-tier
  `README.md` in each new directory documents prompt mix, tokenizer
  choice, content source, and regeneration command; the top-level
  `app/bench/corpus/README.md` gains a "Regenerating long tiers"
  section pointing at `_generate.py`. `tiktoken>=0.7` is added to
  `requirements-dev.txt` only — the bench runtime path never imports
  it. Closes #81.
- **Bench SSE event-type wire constants.** `app/bench/events.py` now
  publishes one `EVENT_TYPE_*` constant per canonical event ``type``
  the supervisor or cli stamps onto the SSE stream (`run_started`,
  `run_done`, `cancelled`, `paused`, `error`, `load_attempt`,
  `load_result`, `phase1_envelope`, `cell_result`, `cell_progress`,
  `system_warning`) plus a `EVENT_TYPES_ALL` frozen set covering them.
  Producers can now import the symbol instead of hard-coding the
  string; a regex-over-source integrity test guards against rogue
  type literals leaking into the supervisor or cli. Sync 1 of v17.11
  Bundle 2 — wire-name compat with dev-2's SSE relay.
- **Bench `load_config_grid` Cartesian shorthand.**
  `matrix_overrides.load_config_grid` accepts per-axis arrays
  (e.g. `{"gpu_memory_utilization": [0.7, 0.8, 0.9], "max_model_len": [4096, 8192]}`)
  and `resolve_matrix` expands them into the Cartesian product as N
  `LoadConfig` rows (first axis varies slowest — matches
  `itertools.product` default order). Cap of 25 products; rejects
  oversize grids, empty grids, empty axes, unknown axes (typo guard),
  and mixing with the explicit `load_configs` list with a 400 at the
  schema layer. Sets up (#62) 3D sweep chart input — the natural
  shape for "sweep across each axis of a 3D plot". Closes #59.
- **Bench: real per-LoadConfig vLLM reload (`apply_load_config`).** The
  bench `Supervisor.apply_load_config()` was previously a logging-only
  stub — supplying multiple LoadConfigs in `matrix_overrides.load_configs`
  did nothing, so the 3D sweep chart (gpu_memory_utilization × concurrency
  × max_model_length) was unreachable. The supervisor now performs the
  real model swap for every LoadConfig the cli requests: it unloads the
  current vLLM process (SIGTERM → 30s → SIGKILL), releases the port,
  allocates a fresh port, calls `RuntimeSupervisor.load(model, port,
  overrides=lc.to_overrides())`, then waits up to
  `settings.bench_health_wait_s` for `/health` to go green. On success
  it returns `(load_ok=True, load_ms, load_error=None)` and updates the
  `model_runtime` row with the new pid/port. On failure it classifies
  the cause: `"oom"` (CUDA OOM marker found by scanning the last 64 KB
  of the vllm log), `"crashed"` (subprocess died without an OOM marker),
  `"health_timeout"` (subprocess alive but `/health` never green), or
  `"<ExcType>:<truncated msg>"` for anything else. A failed reload on
  LoadConfig A does NOT abort the run — the cli records the failure on
  the attempt row (`load_ok=0`, `load_error`, `ended_at`) and proceeds
  to LoadConfig B. Wired via a new internal endpoint
  `POST /api/bench/runs/{run_id}/internal/apply_load_config` (bench-scope
  token, same CSRF bypass as the rest of the internal callback
  perimeter); the cli's `_drive_run` calls it once per LoadConfig before
  `find_envelope`. On run end (`_reap_on_exit` or `cancel`) the
  supervisor restores the model to its row-default overrides via
  `restore_original_config`, so a cancelled or completed bench run
  leaves the model in the same shape the operator booted it in. Tests:
  7 unit tests cover every error-classification branch + missing-wiring
  guard + restore-noop; 3 cli-side unit tests pin the multi-LoadConfig
  isolation contract (apply-failure on A → run continues on B; RPC
  exceptions treated as `load_failed`; full 5-field LoadConfig dict on
  the wire); 1 integration test posts two LoadConfigs through the
  internal-callback perimeter and asserts 2 attempt rows, 2
  `phase1_envelope` events, 2 `phase1_probe` cells, and the run row
  stays `running` after LC-A's quality-floor failure. Closes #58.
  - **Fix-up from code review (`4d2dc3c` → next).** Addresses 7
    review findings without changing the public contract: (C1) every
    bench-spawned vLLM now ships with an `on_exit` callback that marks
    the model row `failed`, clears the runtime row, and releases the
    port — mirrors the production `/load` route so a vLLM that dies
    mid-bench no longer leaves the runtime in a zombie state. (C2)
    `apply_load_config` snapshots the LIVE overrides via the new
    public `RuntimeSupervisor.get_overrides(model_id)` accessor (no
    more reading `_overrides` from the runtime supervisor's
    private state); operator-tuned configs (e.g. a pre-bench load
    with `gpu_memory_utilization=0.85`) now survive a bench run
    exactly as the operator left them, instead of silently regressing
    to row defaults. (I1) replaced the private `sup._processes[id].pid`
    read with a public `RuntimeSupervisor.get_pid(model_id)` accessor
    that returns `None` when the process is gone, classified as
    `crashed` (with OOM-scan fallback) instead of raising `KeyError`.
    (I2) `bench/cli.py` derives the `apply_load_config` HTTP timeout
    from `BENCH_HEALTH_WAIT_S` (`max(180s, wait_s + 60s)`) so slow
    loaders no longer trip a hard 180s RPC ceiling. (I6) `cancel`'s
    no-active-run branch now also invokes `restore_original_config`,
    matching the with-active-run branch — an operator-cancelled
    bench run no longer leaves the model swapped to the last bench
    LoadConfig. (M1) OOM scan now reads from a per-model offset
    bookmark recorded at the start of `apply_load_config`, so LC #1's
    OOM marker can never mis-classify LC #3's plain crash on the
    append-only vLLM log. (M3) timestamps in `RuntimeRepo.upsert`
    are ISO-8601 via `datetime.now(UTC).isoformat()`, matching the
    production `/load` route. Tests: 4 new unit tests pin the C1/C2/I6
    contracts (snapshot-captures-live-overrides, on_exit-is-wired,
    cancel-invokes-restore, reap-on-exit-invokes-restore); the
    pre-existing 8 tests continue to pass. Full suite: 639 passed.
- **Bench: Phase-1 envelope probe steps now persist as `bench_cell`
  rows.** Each per-step probe the envelope finder runs (`c=1, 2, 4, …`
  at `4k` / `max_new=256`, then the `max_new` sweep at the discovered
  `max_concurrency`) lands in `bench_cell` with `status="phase1_probe"`
  and the full signal set (agg_tps, p50/p95 ttft, p50/p95 latency,
  pass_rate, ok/fail counts, error). Pre-issue-#57 every probe was
  discarded on the floor — operators could see the final
  `Envelope(max_concurrency, max_new, limited_by)` but not *why* that
  decision was made, so a zero envelope from a bad-quality grader was
  indistinguishable from one caused by an actual model failure.
  `GET /api/bench/runs/{id}/cells` returns the probe rows alongside
  Phase-2 cells; the chart endpoint (issue #60) can filter on
  `status="phase1_probe"` to compute per-row `limited_by`.
  Schema-shape choice: the sentinel rides on the existing
  `bench_cell.status` column rather than introducing a new `phase`
  column — `BenchCellRepo.list_ok_for_resume` and `best_by_metric` both
  filter on `status='ok'`, so probe rows are automatically excluded from
  resume and best-of queries without a migration. Also: probe-drain
  failures from the internal callback are now logged-and-continued
  rather than fatal (a single bad `post_cell` no longer sinks the
  Phase-2 status emit); the cells table renders probe rows with a
  neutral `info` badge labelled "probe" instead of the red error
  variant. Closes #57.
- **Live GPU/VRAM probe endpoint (`GET /api/system/gpus`).** Returns a
  fresh nvidia-smi snapshot — per-GPU memory total / used / free / utilisation,
  GPU model name, and a list of `holders` (PIDs currently using GPU memory).
  Holders are attributed to vLLM Warden models when the supervisor recognises
  the PID (or its process group leader, so tensor-parallel workers map back
  to their parent model). Unknown PIDs are labelled `kind: external` with
  the bare process name. Response includes a `probed_at` ISO timestamp and a
  `probe_error` field so the UI can render a clean empty state on dev boxes
  without an NVIDIA driver. Burst polling is absorbed by an in-process 2 s
  cache to keep the call cheap for 5–10 s fast polling. Backend-only — the
  frontend gauges are Phase 2.

  **Deployment requirement.** Holder attribution requires the api container
  to share the host PID namespace — `pid: host` in docker-compose,
  `hostPID: true` in K8s pod spec. nvidia-smi reports host PIDs while the
  supervisor tracks in-container PIDs, so without this flag the namespaces
  diverge and every model holder silently degrades to `kind: external`
  (no crash, no error — just attribution loss). The bundled
  `docker-compose.yml` and `deploy/hub/compose.yaml` already set
  `pid: host`. Caveat: PodWarden's compose-stack → K8s translation does
  not currently honour `pid: host` (a follow-up bug is filed against
  PodWarden); K8s deployments via the PodWarden Hub catalog stay in the
  degraded-but-functional state until that lands.
- **GPU model name on `gpu_samples`.** Migration `0013` adds a nullable
  `name` column to `gpu_samples`; the per-minute stats sampler now writes the
  card model (e.g. "NVIDIA RTX A4000") on each insert and `GET /api/stats/gpus`
  returns it alongside the existing memory/utilisation fields so the
  historical chart can label gauges without a second round-trip. Pre-0013 rows
  keep `name = NULL` and the API surfaces it as such; UI must fall back to
  "GPU N".

### Fixed
- **CI: pin transformers below 5.8 to dodge upstream pip metadata bug.**
  `transformers-5.8.1` shipped on PyPI with a `Version` object (not a
  string) in its `requires-python` metadata, which crashes pip's
  specifier parser with `TypeError: expected string or bytes-like
  object, got 'Version'` during `pip install` in the integration-tests
  job. Capped `requirements-dev.txt` to `transformers>=4.55,<5.8` to
  exclude 5.8.x until upstream republishes; production unaffected
  (transformers comes from the vllm/vllm-openai base image, not this
  pin).
- #91 — Disable vLLM async output processor on `gpt_oss` family models to
  work around concurrent `RefCell::borrow_mut` panic in the
  `openai_harmony` Rust extension. The model template registry entry for
  `gpt-oss-20b` now defaults `VLLM_USE_ASYNC_OUTPUT_PROC=0` in
  `extra_env`; the synchronous output path serializes harmony access
  per-engine so two overlapping `POST /v1/chat/completions` calls no
  longer race and the second caller no longer surfaces
  `RuntimeError: Already borrowed` as HTTP 500. Root cause is a shared
  module-level `_harmony_encoding` singleton in
  `vllm/entrypoints/openai/harmony_utils.py` — lazy-initialized once and
  returned by reference, with no lock or per-request isolation. The
  singleton is still present in upstream v0.21.0 (verified by reading
  the file on the published tag); PRs #41181 and #40059 touch
  `vllm/tokenizers/hf.py` / `vllm/renderers/hf.py` /
  `tool_parsers/*.py` but do not modify `harmony_utils.py`, so a base-image
  bump alone is not sufficient. Scope is intentionally narrow: the flip
  applies only to newly-created models from the `gpt-oss-20b` template
  via the FE wizard, leaving other model families on the async output
  path. **Follow-up TODO:** remove `VLLM_USE_ASYNC_OUTPUT_PROC=0` from the
  template once upstream vLLM lands per-request isolation in
  `harmony_utils.py` (tracking upstream issue). Operators with an
  already-deployed gpt-oss model must edit the model row's `extra_env`
  via `PUT /api/models/{id}` (or re-create from the updated template) to
  pick up the workaround. Closes #91.
- **Add Model modal crash on file-picker stage.** `/api/system/gpus` returns `{ gpus: [...] }` but the modal treated the response as a bare array, throwing `TypeError: gpus.filter is not a function` the moment the user clicked Discover. Unwrap `.gpus` and add defensive `Array.isArray` guard. (#98)
- **Frontend: silent refresh + transient/terminal classification — no more
  "logged out every few minutes" (#97).** Two narrowly-scoped changes in
  `frontend/src/lib/auth-fetch.ts`. (1) Every successful login or refresh
  now schedules a proactive refresh at 80% of `expires_in` (12 min for the
  default 15 min access TTL), so the in-memory access token is rotated
  *before* it expires — operators no longer get punted to /login at the
  hard 15-minute boundary. (2) `refresh()` now distinguishes "the backend
  rejected the refresh cookie" (HTTP 401/403 → terminal, redirect to
  /login under the existing single-shot guard) from "the backend
  hiccupped" (HTTP 5xx, 429, network error, malformed body → transient,
  surface the original 401 to the caller, do NOT redirect). Pre-fix a
  single 502 from /api/auth/refresh during a backend bounce evicted the
  user; now the next user action retries. Login page reads `expires_in`
  from the /api/auth/login body and passes it through `setAccessToken`.
- **Bench: `BENCH_HEALTH_WAIT_S` default bumped 30s → 180s for 20B+
  cold-loads.** Cold-starting a `gpt-oss-20b` (or any 20B+ model) under
  `apply_load_config` reliably exceeds the prior 30s health-wait budget
  off NVMe — the supervisor would abort the load, the cli would then
  mis-attribute the supervisor's health-wait timeout as
  `apply_load_config_rpc:ReadTimeout` on the wire, and operators got a
  misleading "RPC timed out" status even though the model was still
  loading correctly. New default (`Settings.bench_health_wait_s = 180.0`,
  and the `BENCH_HEALTH_WAIT_S` env fallback in `load_settings` now
  matches) gives the model 3 minutes to register healthy; the cli's
  `_apply_load_config_timeout_s` likewise mirrors the new 180s floor so
  it can't ever undershoot the supervisor. Operators with small / fast
  models can still shrink the wait via `BENCH_HEALTH_WAIT_S=<seconds>`.
  3 unit tests pin the dataclass default + the env override + the
  unset-env path. Closes #69.
- **Bench Phase-1: probes honour the corpus's `expected_max_new`
  floor.** The envelope finder hard-codes `max_new=256` on its sweep,
  but the bundled `4k` corpus has prompts that ask for 384–512 tokens
  of structured output. At 256 those prompts were truncated mid-answer,
  the grader scored every one of them as a fail, and Phase 1 saw an
  artificial "concurrency=1 fails quality" wall it couldn't get past.
  `run_cell_http` now accepts `respect_prompt_max_new=True` (set
  automatically by `envelope_adapter`) and promotes the per-cell
  `max_new` to `max(p.expected_max_new for p in prompts)` (capped at
  2048 to bound any single bad corpus tag — unannotated prompts keep
  the legacy 256 default so existing corpora are not penalised). The
  envelope finder's own `max_new` sweep (256 → 4096) still stacks on
  top of the floor: the requested value wins when it's higher than
  the prompt-derived ask. Phase 2 (matrix sweep) leaves the flag
  False — the matrix runner already chose `max_new` deliberately and
  must not be silently promoted. 7 unit tests pin the helper's
  per-axis behaviour (largest-wins, requested-above-floor-wins,
  ceiling-caps, missing-annotation-default, mixed annotated/un, zero/
  null defensive) plus 2 end-to-end tests on `run_cell_http` that
  verify outbound request bodies carry the floored `max_tokens` only
  when the flag is on. Closes #70.
- **Bench: truncated responses (`finish_reason="length"`) no longer
  drag pass_rate down.** A request that vLLM cuts off via the
  `max_tokens` budget is a *capacity* outcome — the model didn't fail,
  the budget did — but the grader saw an incomplete answer and counted
  it as a quality fail, so cells whose `max_new` was even slightly
  short showed a "the model is terrible" pass_rate when really the
  bench's own knob was wrong. `_ReqResult.truncated` is now set when
  the final stream chunk carries `finish_reason="length"`; the cell
  aggregator buckets these under `truncated_count` and EXCLUDES them
  from the `pass_rate` denominator (`pass_rate = ok / (ok + fail)`,
  not `ok / total`). If *every* completed request was truncated,
  `limited_by="truncation"` rides alongside on the `cell_result` SSE
  event payload so the supervisor / UI can show "this cell's max_new
  was the bottleneck, not the model" rather than "this cell scored
  zero". Both fields are additive on the event payload only — no
  `bench_cell` schema migration, the supervisor reads them off the
  raw cli POST and includes them in the emit. 3 unit tests cover
  the three regimes (mixed ok/fail/truncated → pass_rate honours the
  exclusion; all-truncated → `limited_by="truncation"`; no
  truncation → legacy behaviour unchanged). Closes #71.
- **Bench Phase-1: probe cells now pin sampling for reproducibility.**
  Probe runs used to feed the model with whatever temperature / top_p
  the runtime defaulted to and an unseeded RNG, so two reruns of the
  same probe against the same model build could produce different
  text — making the grader's pass_rate a function of "what RNG state
  did vLLM pick this morning" rather than "is this concurrency
  workable". `run_cell_http` now accepts `pinned_sampling=True`
  (set automatically by `envelope_adapter`); when on, every outbound
  /v1/completions body carries `temperature=0.0`, `top_p=1.0`, and a
  deterministic 31-bit `seed = blake2b(cell_id|attempt_id|prompt_idx)`
  so two probe runs against the same model with the same
  coordinates yield identical text. `attempt_id` is forwarded by the
  cli's `_drive_run` so the seed is also stable across re-attempts of
  the same load-config; when not supplied the seed falls back to
  coordinates-only (still deterministic within a run). Phase 2 load
  cells deliberately leave the flag False — they're meant to measure
  real-workload variance. 8 unit tests pin the seed function
  (same-inputs-same-seed, prompt-idx / cell-id / attempt-id all
  perturb it, 31-bit mask is honoured) plus the
  `run_cell_http`-level integration (body fields when pinned, body
  fields absent when not pinned, two runs same coords → same seed,
  envelope_adapter wires all three flags). Closes #72.
- **Bench `json_path` grader: leading preambles no longer shadow the
  real structured answer.** `_extract_json_blob` used to find the
  FIRST balanced `{...}` / `[...]` substring and return it — so an
  output like `{plan: lots of reasoning... {"category":"auth"}}`
  (unclosed pseudo-JSON preamble + real answer) tripped the depth
  scanner: depth stayed >0 past the inner object, the scanner fell
  through to array search, and the grader returned the inner
  `key_phrases` list or a `no JSON` failure instead of the real
  answer. The scanner now starts a fresh nesting frame at every
  top-level opener (so a never-closed preamble doesn't block the rest
  of the string) and `_extract_json_candidates` collects EVERY
  successful `json.loads` candidate in source order. `_grade_json_path`
  walks the candidate list in reverse and accepts the first one
  whose path resolves — i.e. the LAST candidate where the requested
  key exists, matching operator intent ("the final structured output
  is the real answer; preambles are commentary"). `_extract_json_blob`
  itself stays as the public "give me the last parseable JSON" thin
  wrapper for non-path callers. 7 new unit tests cover the
  preamble-doesn't-block-inner-json bug, last-candidate-wins,
  last-resolving-candidate-wins, missing-key surfaces a path
  reason (not a JSON-extraction error), and the truly-no-JSON
  legacy reason is preserved. Closes #73.
- **Bench: phase-1 envelope failures no longer surface as silent
  `load_result load_ok=false load_error=null`.** When `find_envelope`
  returns a zero envelope (e.g. concurrency=1 probe fails the quality
  grader), the SSE `load_result` event now carries an explanatory
  `load_error="phase1_no_envelope: limited_by=<reason>"` so operators can
  tell envelope-failed-quality apart from a real vLLM load failure. Also
  re-ordered the supervisor's SSE emits so `load_result` precedes
  `phase1_envelope` in the per-POST emit sequence. Closes #56.
- **CI `integration-tests` no longer wedges for 25+ minutes after pytest
  passes.** Pipelines 5867 / 5871 all showed `47 passed in ~25s` followed
  by a multi-minute hang the runner couldn't even cancel cleanly.
  Root cause (proven via `faulthandler.dump_traceback_later` inside the
  CI image): `aiosqlite.Connection` extends `threading.Thread` and is
  constructed with the default `daemon=False`. Its worker `run()` loop
  parks on `self._tx.get()` waiting for the stop sentinel that
  `Connection.close()` enqueues. When pytest-asyncio tears down the
  per-test event loop, occasional close paths fail mid-flight
  (`RuntimeError: Event loop is closed` surfacing inside the worker's
  `loop.call_soon_threadsafe`), the sentinel is never delivered, the
  non-daemon thread stays parked, and `threading._shutdown` blocks
  interpreter exit indefinitely — so container PID 1 never exits,
  `docker run --rm` never returns, and the runner waits on docker
  forever. Fix: monkey-patch `aiosqlite.core.Connection.__init__` in
  `tests/integration/conftest.py` to flip `self.daemon = True` after the
  base `Thread.__init__` runs. Daemon worker threads do not block
  interpreter shutdown — Python reaps them on `main` exit. Test
  semantics are unchanged (close still runs on the happy path); only
  the wedge mode is reachable. Conftest also leaves an armed
  `faulthandler.dump_traceback_later(30)` in `pytest_unconfigure` as a
  regression canary: any future re-emergence will print the live-thread
  dump 30s after pytest finishes. Production code is untouched.
  Closes #43.
- **Bench CLI: `--corpus-root` default now points at the bundled corpus
  directory.** The CLI fell back to `/data/bench/corpus` when the supervisor
  spawned it without `VW_BENCH_CORPUS_ROOT` set (which is the steady-state
  case — the supervisor never injected it), producing
  `corpus bucket missing: /data/bench/corpus/4k` and failing every bench run
  within ~10ms before any model load. Default is now
  `str(Path(__file__).parent / "corpus")`, which resolves to
  `/app/app/bench/corpus` in the production image and to the repo-relative
  corpus directory in dev/tests. Closes #54.
- **Live logs panel now streams in real time.** Next.js standalone's
  default `compression` middleware was silently gzipping the
  `text/event-stream` SSE response; because the stream is sparse,
  the gzip encoder buffered indefinitely waiting for a flush block
  and bytes never reached the browser. The panel connected and
  stayed blank. Disabled compression at the Next.js layer
  (`compress: false` in `next.config.ts`) — SSE must be uncompressed,
  and Caddy/Traefik already handles ingress-side compression for
  bulk HTML/JSON. As a secondary cleanup we also removed the v17.2
  status-keyed `<LogStream>` remount that amplified the problem via
  proxy churn (it now keys on model id alone and consumes `status`
  as a prop), retuned the SSE backoff from 1s/30s to 250ms/5s so
  transient blips resolve sub-second, and skip opening an
  EventSource entirely while the model is in the `registered` state
  (no log file content exists yet — a placeholder explains the
  wait). Closes #52, closes #53.
- **Live logs panel no longer goes dead on model status transitions.**
  Same-second SSE ticket re-mints for the same `(sub, path)` produced
  byte-identical HMAC strings; the single-use deny-list treated the
  second consume as a replay and rejected it. A per-mint 8-byte `jti`
  nonce in the payload ensures distinct strings across same-second
  mints. Closes #51.
- **First-load no longer logs three `401 Unauthorized` errors in the
  console.** `authFetch` now eager-refreshes the access token before
  the first request when its in-memory copy is null (hard reload, new
  tab, deep link); the post-401 replay path remains for mid-session
  expiry. Part of #50.
- **Live-logs panel: re-subscribes when the model status changes.**
  `<LogStream>` now keys on status so it fully remounts on every
  `unloaded → pulled → loading → loaded` transition, opening a fresh
  SSE connection (with backend backfill) instead of holding the same
  stream across all four states. Part of #50.
- **Release procedure: `docker buildx --push` now passes `VW_BUILD_VERSION` and `VW_BUILD_SHA` build-args** to both the backend and UI buildx commands so the version banner shows the real tag instead of `vdev · unknown`. Also added a runtime override path via env on the `api` service in `deploy/hub/compose.yaml`. Closes #45.
- **`/healthz` now returns 200 instead of 404 for Kubernetes liveness probes and uptime monitors.** Added a Next.js rewrite that aliases `/healthz → /api/health` so the operator-convention path works without reconfiguring probe manifests. Closes #48.
- **Live-logs panel: "(no log lines yet)" placeholder for connected
  empty streams.** `LogStream` previously rendered an empty
  `<div role="log">` when the SSE handshake had succeeded but the
  subprocess hadn't produced any stdout — operators saw a blank panel
  and couldn't distinguish a healthy-but-quiet stream from a wedged
  one. The component now shows an explicit `role="status"` placeholder
  in that window; once the first line arrives it falls back to the
  normal log container. Part of v2026.05.15.5 hotfix bundle.
- **`GET /api/models/{id}/logs/stream` now opens-or-creates the log
  file.** Previously returned HTTP 404 when the log path didn't exist
  yet, which surfaced through `useEventSource`'s ticket-mint preflight
  as a terminal-error — the UI stuck on "stream unavailable" even
  though the supervisor would create the file moments later. Matches
  the supervisor's `O_CREAT | O_APPEND` behaviour. A missing parent
  directory still 500s (real deployment bug, not a race).
- **SSE log stream emits keepalive comment lines every 15s during
  idle.** Without `: keepalive\n\n` comments, intermediate proxies
  (nginx/Caddy default 60s idle timeout) and EventSource's own
  heuristics could declare the connection dead during long pre-first-
  line silences — e.g. vLLM loading a 20GB weight set before
  producing any stdout. Comment lines are silently ignored by the
  EventSource client but traverse proxies, so the connection stays
  healthy.
- **Subprocess env: `PYTHONUNBUFFERED=1` set on every vLLM child.**
  CPython block-buffers stdout when stdout is a file fd, so a fast-
  crashing subprocess (rc=1 within <1s) could exit before its
  traceback was flushed — operators saw nothing in the log file even
  though Python printed the error. Defence-in-depth alongside the
  open-or-create fix above.
- **`/setup` root URL no longer 404s.** Added a server-side redirect
  page so `https://vllm.protrener.com/setup` lands on
  `/setup/welcome` instead of returning a Next.js 404. The /setup
  segment had a layout.tsx and five child pages but was missing the
  root `page.tsx`. Closes #41.
- **`docs/releasing.md`: UI image publish step documented.** The
  release runbook only described the backend `vllm-warden` image
  buildx command; operators were silently dropping the
  `vllm-warden-ui` push, leading to deploy-host pulls of stale UI
  tags. Added a dedicated UI section with digest-verification.
  Closes #40.
- **Login page no longer enters an infinite reload loop.** On
  v2026.05.15.3, opening the login page caused a continuous page
  reload: the NavBar's version-check request returned 401 (no session
  yet), which triggered a redirect back to /login, resetting all
  in-flight state and starting the cycle again. The version check is
  now suppressed on /login and /setup, and the redirect logic has a
  guard that prevents /login → /login navigation under any
  circumstances. Closes #38.
- **Frontend live-log panel: renders blank when other authenticated
  endpoints simultaneously 401.** The login-redirect fallback in
  `auth-fetch` was firing for every concurrent 401 (e.g. parallel
  `/api/version` + `/api/models` failures during session-token expiry), and the
  overlapping `window.location.replace('/login')` calls stranded the
  SSE preflight (`POST /api/auth/sse-ticket`) before it could mint a
  ticket — `useEventSource` saw `authFetch` resolve to a value it
  didn't classify and `LogStream` stayed on the "Connecting…"
  placeholder. The redirect is now de-duplicated via a module-level
  `loginRedirectInFlight` flag, the 401 Response is still returned to
  the caller so its error handling runs, and `useEventSource` checks
  the flag (before and after preflight) — transitioning to terminal
  `"session expired"` instead of looping into reconnects that would
  all fail the same way during the unload window. Closes #37.
- **Bench v2: load step no longer crashes with
  `asyncio.run() cannot be called from a running event loop`.** The cli's
  `_drive_run` is itself an async coroutine, and the supervisor invokes
  `find_envelope` with a sync `run_cell` closure that called
  `asyncio.run(run_cell_http(...))` directly. Switching that to
  `await asyncio.to_thread(find_envelope, envelope_adapter(http_args), …)`
  hands the synchronous probe off to a worker thread so the inner
  `asyncio.run` is no longer nested inside the cli's own loop. The bench
  wizard now reaches phase-2 grid without the loader silently aborting.
  Closes #28.
- **Frontend /benchmarks: completed-run timestamps are no longer off by
  a factor of 1 000.** `fmtTs` was multiplying its input by `1000`, but the
  API already returns epoch-ms (`finished_at`), so a run finished at
  16:42 UTC rendered as a year-58341 date. Removed the multiplication;
  added a regression test pinning the contract. Closes #33.
- **Frontend /stats: chart X-axis now actually honours the range selector.**
  The throughput and GPU-util charts relied on recharts' default
  `["dataMin","dataMax"]` domain, so switching from 1h → 7d resized the
  chart container but kept the axis pinned to the data window (which is
  often only a few minutes when the rollup table is sparse). New
  `rangeBounds(range, now)` helper derives `[startMs, endMs]` from the
  selector; both charts now pass it into `<XAxis domain>` with
  `allowDataOverflow` so the axis covers the requested window even when
  data is sparse, and the tick formatter switches to "MMM dd" for the
  7d view so the label row doesn't smear into illegibility. Closes #34.
- **Frontend SSE: terminal stream failures no longer hide behind an
  unbounded "Connecting…" placeholder.** `useEventSource` previously
  exp-backoff'd forever on every error class — a 404 / 401 ticket-mint
  looked identical to a transient network blip, so operators saw a stuck
  spinner with no actionable signal. The hook now mints the SSE ticket
  via `POST /api/auth/sse-ticket` before each connect and uses that
  request's HTTP status to classify failures: 4xx (except 429) is
  terminal and stops the loop; 5xx / 429 / network / EventSource onerror
  is transient with capped exp-backoff (MAX_RECONNECT = 5). The hook
  surfaces `{status, errorCode, attempts}` so `LogStream` can render
  distinct UI for 401 ("session expired"), 404 ("Log stream not found"),
  502+/HTTP-coded ("Stream unavailable (HTTP …)"), and post-retry
  exhaustion ("after N retries"). Terminal banners use `role="alert"`;
  transient retry banners use `role="status"`. Closes #35.
- **CI: integration tests** — set `VW_HF_CACHE_DIR` in 5 tests that previously
  crashed on `PermissionError: /root/.cache/huggingface` under non-root CI
  runner. Closes #30.
- **Bench v2: SSE event stream now populated; UI no longer stuck on
  "Events failed".** The bench supervisor was driving DB rows directly but
  never opening the per-run jsonl event log that `app/bench/sse.py` tails,
  so every wizard session that started a run saw the SSE relay give up
  after its 10 s wait-for-file window and surface "Events failed" with no
  way to recover except a hard refresh. Supervisor now opens an
  `EventWriter` at run start and emits `run_started`, `load_attempt`,
  `phase1_envelope`, `load_result`, `cell_result`, `paused`, `cancelled`,
  `error`, and `run_done` events at the correct lifecycle points (all
  types match `frontend/src/components/bench/use-bench-sse.ts` VALID_TYPES).
  The first event fires *before* the health-wait so the SSE relay's
  10 s discovery window always succeeds. Issue #26.
- **Bench v2: cli no longer races the model loader.** A bench POST that
  arrived before `model_runtime` reported healthy made the cli's first
  `/v1/chat/completions` call against a not-yet-listening port; the cli
  exited rc≠0, the supervisor caught the error but never finalised the
  run (no `ended_at`, no summary), and the UI sat on "running" until
  manual refresh. Supervisor now polls `runtime.supervisor.wait_for_health`
  with a configurable budget (`BENCH_HEALTH_WAIT_S`, default 30 s) before
  spawning the cli; if the port is missing it marks the run failed with
  `runtime_not_loaded`, and if the health check times out it marks it
  failed with `health_timeout`. The cli's fast-fail status payload was
  also extended: it now includes `load_error`, `load_ok=False`,
  `ended_at`, and either `load_config_id` (for failures inside an attempt)
  or `pre_load: True` (for failures before the first attempt), so the
  supervisor's `write_status` handler can finalise both the attempt row
  and the run row in one POST. Issue #27.

### Notes
- **After upgrading on the bonus deployment: re-run the initial setup
  wizard.** The v2026.05.15.3 deadlock recovery on 2026-05-17 required
  recreating the data PVC, which deleted all user accounts. No backup
  existed. Navigate to /setup to create the admin account again before
  using the UI.

### Changed
- New `BENCH_HEALTH_WAIT_S` env var (default `30.0`) controls the
  supervisor's pre-spawn health-wait budget. Bump for slow loaders.

## [v2026.05.19.2] — 2026-05-19

### Added
- **HF discovery service + `GET /api/models/discover` endpoint (#84).**
  New JWT-required route that lists a HuggingFace repo's files (with
  per-file `kind` / `quant` / `params` hints derived from filename)
  alongside the six `config.json` keys the Add Model wizard needs
  (`hidden_size`, `num_hidden_layers`, `num_attention_heads`,
  `num_key_value_heads`, `max_position_embeddings`, `torch_dtype`).
  Pure-helper triplet (`_classify` / `_parse_quant` / `_parse_params`)
  lives in `app/models/discovery.py` so the same classification logic
  can be reused by the FE wizard without recomputing buckets. Gated /
  private repos silently reuse `data_dir/hf-token` (no inline
  re-confirmation prompt — per CTO decision in #84) and surface as a
  typed `auth_required` envelope when the stored token is insufficient.
  HF 404 maps to `repo_not_found`; transport errors to `discovery_failed`
  (sanitised — no raw HF stack trace crosses the wire). In-process
  `(repo_id, revision)`-keyed cache with 60 s TTL mirrors
  `_ProbeCache` in `app/system/routes_gpus.py` so the wizard can
  refetch on selection changes without thrashing huggingface.co.
  Wire shape pinned by `tests/fixtures/discovery/qwen-gguf.json` —
  dev-2 mirrors this fixture in #86 for FE mocking; if the wire shape
  drifts the contract snapshot test fails loudly. Blocks #82.2 and
  #82.3.

## [2026.05.15.0] — 2026-05-15

### Fixed
- Non-streaming `/v1/chat/completions` with large `max_tokens` (e.g. 32768)
  no longer returns HTTP 500 / ECONNRESET after exactly 30 s. Root cause:
  the request path is `client → Caddy → ui (Next.js rewrites) → api → vLLM`,
  and Next.js's `experimental.proxyTimeout` defaults to **30 000 ms** for
  rewrites. When the upstream vLLM generation took longer than 30 s, Next.js
  cut the socket and returned a stock 500, while the api saw the socket
  close mid-request (no uvicorn access-log entry for the failure). Three-
  layer bypass test (api podIP 114 ms ✓, api Service 100 ms ✓, ui Service
  with 32k repro hangs >28 s ✗) localised the cut to Next.js. Set
  `experimental.proxyTimeout: 600_000` in `frontend/next.config.ts` so
  long completions get a 10-minute budget through the UI server. Issue #13.
- Removed unused `IDX_STRIDE` import from `tests/unit/bench/test_events_writer.py`
  (ruff F401). Unblocks the `develop` lint job (pipeline 5581).

## [2026.05.12.1] — 2026-05-12

### Fixed
- HuggingFace model cache now lives on the dedicated `vllm-warden-hfcache`
  PVC (mounted at `/root/.cache/huggingface`) instead of the small data
  PVC subdirectory `/data/hf-cache`. The old path was a `@property` on
  `Settings`, so even though `deploy/hub/compose.yaml` already mounted the
  dedicated PVC at `/root/.cache/huggingface`, the pull task wrote to a
  directory living on the data PVC — which `shutil.disk_usage(...).free`
  reported as 0 GiB for any non-trivial model. `Settings.hf_cache_dir` is
  now a real dataclass field populated from `VW_HF_CACHE_DIR` (default
  `/root/.cache/huggingface`). The Hub template now requires
  `VW_COOKIE_SECRET` (the api refused to boot without it) and surfaces
  `VW_CONTAINER_GPU_COUNT`. The api service also publishes container
  port 8080 so the K8s compose-bridge emits a ClusterIP Service for
  the ui→api hop.
- vLLM subprocess no longer dies at import time with
  `ImportError: cannot import name 'Gemma3Config' from 'transformers'`.
  Root cause: `requirements.txt` pinned `transformers==4.47.1`, which
  downgraded the version that ships with the `vllm/vllm-openai` base
  image. The base image's `vllm==0.20.0` imports `Gemma3Config` (added
  in transformers 4.50), so every model load failed with `rc=1` before
  the model config was even parsed. Pin removed; the base image's vLLM
  dependency now constrains transformers transitively. Our only direct
  use is `from transformers import AutoTokenizer` in
  `app/proxy/tokenizers.py`, which is API-stable across versions.

## [2026.05.12.0] — 2026-05-12

### Fixed
- **Auth: duplicate `Authorization` header on the wire.**
  `auth-fetch.ts:headersToObject` emitted the header twice — once
  lowercase (from `Headers.forEach`, per WHATWG spec) and once
  canonical-cased. fetch's case-insensitive header merge concatenated
  them into `Authorization: Bearer xxx, Bearer xxx`; the API parsed the
  comma-suffixed string as a malformed token and returned 401, putting
  the entire v2 UI into an infinite refresh loop the moment a user
  logged in. Same treatment applied to `X-CSRF-Token`.
  **Production-breaking.**
- **vLLM subprocess: `--dtype` / `--max-model-len` emitted as None.**
  `cmd_builder.build_vllm_args` appended both flags unconditionally. When
  the model row had `dtype IS NULL` or `max_model_len IS NULL`,
  `asyncio.create_subprocess_exec` raised `expected str, bytes or
  os.PathLike object, not NoneType` — the subprocess never started, and
  the failure surfaced only as a cryptic `last_error` on the row. Both
  flags now omitted when unset (vLLM picks safe defaults).
- **Supervisor: missing `/data/hf-token` killed every load.** Previously
  `Path(hf_token_path).read_text()` raised `FileNotFoundError` whenever
  the operator had not seeded the token file. Now treated as an empty
  string, mirroring the graceful handling already present in
  `pull_task._read_hf_token`. Public HF repos load without a token;
  private repos still need it to be present.
- **Auto-pull on register.** Backend exposed `POST /api/models/{id}/pull`
  but the v2 UI had no caller anywhere, leaving freshly-registered models
  stuck at `status="registered"` forever. `AddModelModal` now
  fire-and-forgets the pull request immediately after a successful POST
  /api/models; pull failures surface on the detail page via the row's
  `last_error`.
- **Frontend: `BACKEND_URL` baked into the standalone build.** Next 15
  freezes `rewrites()` into `routes-manifest.json` at build time, so
  setting `BACKEND_URL` only at runtime was a no-op — every `/api/*`
  request 404'd at the UI tier before reaching the API. Wired in the
  Dockerfile build stage; works for both the local compose stack and
  `deploy/hub/compose.yaml` because both expose the API as `api:8080`.
- **Playwright selector drift.** Cleaned up login, create-token,
  token-row, and the modal Add/Create button scoping to match the v2
  redesign so the happy-path spec can drive the UI without
  pointer-events fights.

### Tests
- `tests/e2e/happy-path.spec.ts` rewritten with stale-state cleanup,
  CSRF cookie warmup, dialog-scoped clicks, `waitForResponse(201)` on
  register, and a 600s test budget that accommodates the 120s pull +
  60s load + vLLM warmup windows. Full path verified GREEN against
  the local compose stack: login → add opt-125m → auto-pull (753 MB)
  → load → /v1/completions → mint+rotate token → unload → delete,
  ~32s warm.

## [2026.05.11.2] — 2026-05-11

### Fixed
- HuggingFace model cache now lives on the dedicated `vllm-warden-hfcache`
  PVC (mounted at `/root/.cache/huggingface`) instead of the small data
  PVC subdirectory `/data/hf-cache`. The old path was a `@property` on
  `Settings`, so even though `deploy/hub/compose.yaml` already mounted the
  dedicated PVC at `/root/.cache/huggingface`, the pull task wrote to a
  directory living on the data PVC — which `shutil.disk_usage(...).free`
  reported as 0 GiB for any non-trivial model. `Settings.hf_cache_dir` is
  now a real dataclass field populated from `VW_HF_CACHE_DIR` (default
  `/root/.cache/huggingface`). The Hub template now requires
  `VW_COOKIE_SECRET` (the api refused to boot without it) and surfaces
  `VW_CONTAINER_GPU_COUNT`. The api service also publishes container
  port 8080 so the K8s compose-bridge emits a ClusterIP Service for
  the ui→api hop.
- vLLM subprocess no longer dies at import time with
  `ImportError: cannot import name 'Gemma3Config' from 'transformers'`.
  Root cause: `requirements.txt` pinned `transformers==4.47.1`, which
  downgraded the version that ships with the `vllm/vllm-openai` base
  image. The base image's `vllm==0.20.0` imports `Gemma3Config` (added
  in transformers 4.50), so every model load failed with `rc=1` before
  the model config was even parsed. Pin removed; the base image's vLLM
  dependency now constrains transformers transitively. Our only direct
  use is `from transformers import AutoTokenizer` in
  `app/proxy/tokenizers.py`, which is API-stable across versions.

## [2026.05.11.0] — 2026-05-11

### Added
- `CUDA_DEVICE_ORDER=PCI_BUS_ID` is now baked into the per-model
  subprocess env baseline (`app/runtime/env_builder.py`) and added to
  the hard-locked keys list. Fixes a latent footgun on heterogeneous
  GPU hosts (e.g. pw_prod `bonus` mixes Quadro RTX 4000 on slot 0
  with A4000s on 1–2): without `PCI_BUS_ID`, NVML may reorder devices
  by compute capability, so `gpu_indices=[1,2]` could land on the
  wrong physical GPUs. vLLM warned about this at load time.
- gpt-oss-20b preset template + `extra_env` plumbing + cmd_builder
  typo cure (bundle landed via `feat/model-templates`).

## [2026.05.10.2] — 2026-05-10

### Fixed
- Pull progress on `/models` now actually advances. `run_pull` was
  inserting models with `pulled_bytes=0, pulled_total=NULL` and never
  writing to either column, so the static `<progress>` element rendered
  nothing useful for the entire download. The new `run_pull` (1) calls
  `HfApi.repo_info(files_metadata=True)` upfront to persist
  `pulled_total` (the bar's denominator), (2) spawns a background poller
  that walks `{hf_cache_dir}/models--{org}--{repo}/blobs/` once per
  second and writes the summed file size to `pulled_bytes`, and (3)
  writes a final `pulled_bytes` after `snapshot_download` returns
  (`app/models/pull_task.py`). Regression tests in
  `tests/unit/models/test_pull_progress.py` pin total/final-bytes
  persistence and the `_snapshot_dir_size` walk.

### Added
- `GET /api/models/{id}/pull/progress` streams pull state as SSE.
  Emits one JSON event per second with `{status, bytes, total,
  last_error}` and terminates when the row leaves
  `pulling`/`registered` (or when the row is missing — emits
  `{"status": "missing"}` and closes, since EventSource doesn't
  surface HTTP errors usefully). `app/web/models/_card.html` now
  consumes this stream with `EventSource`, computing a rolling
  download rate and ETA from successive byte deltas, and reloads
  the page when the row reaches a terminal status. Auth required
  via `require_session_json` (401 not redirect, so the FE stream
  doesn't silently swallow a session expiry).

## [2026.05.10.1] — 2026-05-10

### Fixed
- Pull/Load/Unload buttons on `/models` now reload the page on success
  and surface the server's error body via `alert()` on failure
  (`hx-on::after-request` handlers in `app/web/models/_card.html`) —
  previously `hx-swap="none"` ate the response and gave the operator
  no visible feedback.
- `GET /models/{id}/logs` now resolves: route was missing (only the
  card linked to it). New page streams stderr via `EventSource` against
  the existing `/api/models/{id}/logs/stream` endpoint
  (`app/web/models/logs.html`, route added in `app/models/routes_web.py`).
- Welcome page no longer flashes raw JSON into the DOM after admin
  bootstrap. `app/web/setup/welcome.html` was using `hx-swap="innerHTML"`
  on `<body>` against a JSON endpoint; replaced with `fetch` +
  `window.location.href` redirect.

### Added
- Disk-space safeguard is now overridable. `POST /api/models/{id}/pull`
  accepts `?force=true` which threads through to `run_pull(force=True)`
  and skips the cache-free-bytes precheck. `_card.html` exposes this as
  a "Force pull" button (with `hx-confirm`) for `registered` and
  `failed` models. Regression test
  `tests/unit/models/test_pull_endpoint.py::test_pull_endpoint_passes_force_flag`
  pins the wiring.
- `/models` now renders an "Available capacity" banner showing combined
  VRAM across allowed GPUs and free space in the HF cache directory.
  Per-model card adds a hint with `Weights: X GiB · Assigned VRAM: Y GiB
  (GPUs [...])` and a red warning when weights exceed assigned VRAM
  (`app/models/routes_web.py` builds the rows; `_card.html` and
  `index.html` render them).

### Added
- `internal/web/`: wired the setup wizard's HTTP surface on top of
  the Phase-1 server. New options `WithWizard(launcher, drafts, bus)`
  and `WithAuth(mw, sessions, csrf, handlers, hasAdminPassword)` are
  applied independently; both panic on nil collaborators because
  `cmd/sidekick` constructs them together at boot. Routes land in
  sibling files: `auth_routes.go` mounts `GET/POST /login` and
  `POST /logout` (login skips CSRF on first POST since no session
  exists yet); `wizard_routes.go` mounts `GET/POST /setup/step/{n}`,
  `POST /api/v1/wizard/launch`, `GET /api/v1/wizard/progress` (SSE
  with snapshot-then-delta replay, drop-oldest ring, 25s
  `:keepalive` heartbeat), and `POST /api/v1/wizard/reset` behind
  `LoadSession → RequireAuth → RequireCSRF` (the only wizard
  endpoint that lives on the protected surface — for an admin
  re-running the wizard on a fresh box). `middleware.go` adds
  `wizardOnlyWhenUnsetup` which 303s `/setup/*` to `/login` once
  the admin password is set. `handleRoot` now branches on
  `hasAdminPassword`: 303 to `/setup/step/1` when unset, to
  `/login` when set, falling back to the Phase-1 placeholder when
  no auth is wired (preserves existing tests). Templates: 9 new
  files under `web/templates/{auth,setup,partials}/` — full HTML
  page for login, plus a step chrome (`step.html`) that branches
  on `StepName` to dispatch to the five step bodies (GPU pick,
  HF token, model repo+revision, runtime args, review+launch).
  Step 5 wires Alpine inline against the SSE endpoint to walk
  through phase events and follow the launcher's `redirect` on
  `done`. New dep: `github.com/google/shlex` for `extra_args`
  parsing on step 4. New integration test
  (`wizard_integration_test.go`) walks a real `Server` (with
  real `DraftStore` + `Bus` + auth middleware) end-to-end:
  GET `/` → POST steps 1-4 → POST `/launch` (202) → subscribe
  to `/progress` until `done` → GET `/` now redirects to
  `/login` (the unsetup gate flipped) → anonymous reset is
  rejected. Only HF fetch, vLLM subprocess+prober, and
  `ConfigSaver` are faked. Closes #771.

- `internal/wizard/`: new package implementing the five-step setup
  wizard's domain layer. Contents: `Step`/`Draft`/`FieldError`
  state types with `Validate` cross-field rules; `DraftStore` —
  schema-versioned (`DraftSchemaVersion=1`) JSON persistence at
  `<dataDir>/wizard-draft.json` with mode 0o600 atomic writes via
  temp-file rename; `Bus` — bounded ring buffer (cap 64) +
  drop-oldest pub/sub for SSE progress events with monotonic
  sequence numbers and replay-from-cursor semantics; `Promote` /
  `BuildConfig` — converts a healthy draft to `config.File` with
  bcrypt-hashed admin password and canonical-order vLLM args;
  `Launcher` — single-flight orchestration that calls the
  injected `Fetcher` (HF model fetch), `Process` (vLLM start +
  stop) and `Prober` (HTTP health) under a goroutine bound to
  `context.Background()` (per #9 regression guard) with
  10-minute health timeout, 2-second poll interval, panic
  recovery, LIFO cleanup deferral, and benign-tolerant draft
  delete on success. `launchertest/` exposes `FakeFetcher` /
  `FakeVLLM` fakes plus `NewHTTPHealthServer` for downstream
  tests in #771. 38 tests cover happy-path promote, save error
  propagation, draft round-trip, schema-version rejection,
  bus replay/seq/time-override/drop-oldest semantics, and the
  full launcher state machine including health timeout
  (deterministic via injected clock), single-flight rejection,
  panic recovery, kill-window enforcement, request-context
  isolation, and benign delete-failure. No HTTP routes yet —
  those land in #771. Closes #770.

- `internal/auth/`: new package providing the auth primitives
  consumed by the setup wizard and protected routes. Contents:
  bcrypt password hashing (`Hash`/`Verify`, cost 12); in-memory
  session store with 7-day absolute and 24-hour idle TTLs and a
  background sweeper; per-session double-submit CSRF tokens with
  constant-time comparison; per-IP sliding-window login rate limit
  (5/minute); chi-compatible middleware bundle (`LoadSession`,
  `RequireAuth`, `RequireCSRF`); login/logout HTTP handlers with
  `next=` open-redirect defence (`SafeNextParam`). 26 unit tests
  cover happy-path, TTL eviction, header/form CSRF acceptance,
  rate-limit window slide and per-IP isolation, redirect
  sanitization (8 cases), and middleware context propagation. No
  routes are wired yet — that lands with the wizard handlers
  (#771). Closes #769.

### Docs
- `docs/superpowers/specs/2026-05-02-441-setup-wizard-design.md`:
  added the design spec for the 5-step first-run setup wizard
  (#441). Covers state model, auth (bcrypt + session + CSRF +
  rate-limit), wizard backend (draft store, progress.Bus,
  single-flight launcher), HTMX/Alpine/Tailwind frontend, per-step
  contracts, and a 5-phase implementation plan that splits into
  children #769–#773. Refs #441.

### Test
- `internal/gpu/detect_test.go`: added
  `TestAutoDetector_Poll_NoBackendReportsGPU` mirroring the existing
  `TestAutoDetector_NoBackendReportsGPU` so `AutoDetector.Poll`'s
  final `noGPUInfo` return is exercised. Closes the 1/7-statements
  (14.3%) coverage gap flagged in QA of MR !7. Closes #6.

### Removed
- `internal/vllm/fakes_test.go`: dropped three never-called test
  helpers — `(*fakeRunner).lastChild`, `(*fakeChild).writeStdout`,
  and `(*captureLogger).all`. Flagged by golangci-lint v1.61
  `unused` linter during QA of MR !10. No callers in the repo;
  receiver fields they touched (`created`, `stdout`, `lines`) are
  retained because other helpers/methods still use them. Closes #7.

### Test fixtures
- `internal/gpu/testdata/bonus-3gpu.csv`: replaced the synthesized
  three-GPU fixture (Quadro RTX 4000 + 2× RTX A4000) with a live
  capture from the `bonus` host. Column count, format, and static
  fields (name, memory.total, power.limit, compute_cap) all match
  the synthesized values exactly; only runtime-state columns
  (memory.free/used, power.draw, temperature.gpu) refresh to the
  observed values. The live capture exercises a sub-1W
  `power.draw` reading (`0.53`) and a 1 MiB `memory.used` reading
  that the synthesized fixture didn't hit. Closes #4.

### Polish
- `internal/config/config.go`: `Manager.Save` now fsyncs the parent
  directory after the temp-file rename so the directory-entry
  update is durable, not just the file contents. Best-effort —
  errors are swallowed because the user-visible save has already
  succeeded. Closes a durability gap on NFS / overlayfs under
  power loss; ext4 and Docker bind-mounts (the common deployment
  target) were already safe. Item 2 of #5.
- `internal/db/sqlite.go` + `internal/db/sqlite_test.go`: introduced
  a tiny `nowFunc` clock seam for `CreateAPIKey` and rewrote
  `TestListAPIKeys_OrderingAndRevoked` to drive it from a
  fixed-timestamp pair instead of `time.Sleep(1100ms)`. Drops
  ~1.1s from the `internal/db` package suite. Item 3 of #5.
- CI polish from MR !4 review (#2): pinned `golangci/golangci-lint`
  image from `latest-alpine` to `v2.12.1-alpine` so an upstream
  default-linter change can no longer break the pipeline
  unannounced; added a one-line comment in the build job's CalVer
  block explaining that `TAGS` is *reset* (not appended) on a
  CalVer tag push to keep `:production` / `:latest` from a prior
  main build from bleeding onto the immutable tag.
- `internal/web/server.go`: corrected the `NewServer` docstring to
  match reality — Options are applied BEFORE `s.routes()`, not
  after. The inline comment and the code already agreed; only the
  docstring lagged. No behavioural change. Spotted in post-merge
  code-review of MR !10.
- `internal/web/vllm_routes.go`: extracted the SSE keepalive
  interval to a package-level `sseKeepaliveInterval` var (default
  unchanged at 25s) so a unit test can shorten it. Added
  `TestLogsStreamEmitsKeepaliveComments` in `vllm_routes_test.go`
  — overrides the interval to 50ms, asserts >=2 `:keepalive`
  comment frames arrive on an idle SSE stream within ~250ms, and
  cancels the request context before the test exits to avoid a
  goroutine leak. Closes #8.

### Added
- Phase 2 vLLM process manager (#440):
  - `internal/vllm`: child-process supervisor for the vLLM server.
    `Manager` (constructed via `NewManager(Options{...})`) owns a
    state machine — `stopped → starting → running → stopping →
    stopped`, plus `crashed` / `failed` for unexpected exit and
    failed-startup respectively. `Start(ctx, LaunchSpec)` /
    `Stop(ctx)` / `Restart(ctx)` / `Status()` are the public API,
    safe for concurrent use. `LaunchSpec` decouples per-launch
    settings (model id, binary path, argv, env, pip extras, health
    URL) from the long-lived Manager so a single Manager can serve
    successive different models.
  - Lifecycle: SIGTERM-then-SIGKILL escalation on Stop with a 30s
    grace window (configurable via `Options.ShutdownTimeout`); HTTP
    `/health` probe every 5s while in `starting` or `running`;
    `Options.StartupTimeout` (default 5min) caps the wait for the
    first 200 before declaring the launch a `failed` and tearing
    the child down. Crash detection auto-restarts with exponential
    backoff (1s → 60s by default, doubling per crash, reset on
    successful `running`).
  - `Runner`/`Child`/`HealthChecker`/`PipRunner` interfaces let
    tests swap in deterministic fakes; production
    implementations (`execRunner`, `httpHealthChecker`,
    `PipExecRunner`) live in `runner.go` and `health.go`. The
    production `execRunner` sets `Setpgid: true` and signals via
    `syscall.Kill(-pid, sig)` so SIGTERM propagates to vLLM's
    tensor-parallel worker children when the warden is PID 1.
  - `internal/vllm/logs.go`: `LogBuffer` is the in-memory + on-disk
    + SSE fan-out for the child's stdout/stderr. 5000-line ring
    buffer (per #440 spec), optional on-disk file at
    `{LogDir}/vllm-{unix-nano}.log` (raw bytes persisted before
    line-splitting so a partial trailing line survives a crash),
    bounded per-subscriber channels with detach-on-slow-consumer
    so a stalled SSE listener never blocks the writer.
  - `internal/web`: `NewServer` now accepts functional `Option`s
    so independent feature packages can attach routes without
    conflicting on the constructor signature. `WithVLLM(sup,
    specProvider)` registers `/api/v1/vllm/{status,start,stop,
    restart,logs/tail,logs/stream}`. `/start` returns 409 when the
    supervisor is already running; `/stop` is idempotent (200 even
    when already stopped); `/logs/stream` is SSE with a 25s
    keepalive comment.
  - `cmd/sidekick/main.go`: constructs the supervisor at boot,
    auto-starts vLLM if `config.json` has a model AND the model is
    `ready` in the SQLite store, stops the supervisor (with up to
    35s of slack) before HTTP shutdown on SIGTERM/SIGINT.
- Phase 2 curated model catalog + GPU fit verdicts (#439):
  - `internal/models/curated_models.json`: 18 editorial entries
    covering the families called out in the issue — gpt-oss-20b
    (featured) and gpt-oss-120b, Llama 3 / 3.1 / 3.2, Mistral 7B,
    Mixtral 8x7B, Qwen2.5 (7B/14B/32B/Coder), DeepSeek V2 Lite
    (Chat/Coder), Phi-3.5-mini and Phi-3-medium, Gemma 2 9B/27B.
    Each row carries the full schema from the issue
    (`id`, `name`, `arch`, `params_total`, `params_active`,
    `min_vram_gb`, `recommended_tp`, `quantizations`,
    `default_max_len`, `disk_gb`, `vllm_extra_args`,
    `vllm_pip_extra`, `hf_gated`, `description`, `tags`).
  - `internal/models/curated.go`: embeds the JSON via `go:embed`,
    decodes it once behind `sync.Once`, and validates every row
    against the architecture invariants (dense ⇒ active==total,
    moe ⇒ active<total, all required fields populated). Public
    surface: `LoadCurated()`, `LookupCurated(id)`, plus the
    `Architecture` enum with a guarded `UnmarshalJSON` so a typo'd
    `arch` value fails the build instead of silently misclassifying.
  - `internal/models/compatibility.go`: GPU fit calculator. Adds
    `FitStatus` (green/yellow/red), `FitReport`, and two entry
    points: `FitCurated(model, gpu.GPUInfo)` for catalog rows and
    `EstimateFit(paramCount, precision, info)` for the
    paste-arbitrary-repo path. Aggregates per-GPU memory across a
    multi-GPU host, honours `recommended_tp`, applies a 20%
    headroom for KV-cache + activations, degrades gated repos
    from green to yellow with a "request access" note, and
    surfaces a yellow-with-quantization hint when the model only
    fits at sub-FP16 precision. `CuratedWithFit(info)` pairs every
    catalog row with its verdict in one call for the API.
  - `internal/web/catalog_routes.go`: new HTTP surface.
    `GET /api/v1/catalog` returns the curated list with per-row
    fit verdicts and a host-GPU summary; `GET /api/v1/catalog/{id}`
    returns one row by exact id (URL-encoded slashes accepted),
    404 on miss, 405 on non-GET, 400 on empty id. The handler
    swallows GPU-detection errors so the wizard remains usable
    when `nvidia-smi` is hung or absent — no GPU collapses every
    verdict to red rather than 500-ing the request.
  - `internal/web/server.go`: `NewServer` now takes functional
    options. `WithGPUDetector(gpu.Detector)` wires the detector
    used by the catalog route. `cmd/sidekick/main.go` passes
    `gpu.NewAutoDetector()` so the live binary serves real
    verdicts.
  - Tests: 22 new test functions across `internal/models` and
    `internal/web`. Coverage: `internal/models` 83.6%,
    `internal/web` 87.3%. Acceptance criterion from the issue
    (gpt-oss-20b is curated with the expected MoE shape) is
    pinned by `TestCuratedIncludesGPTOSS20B`.

- Phase 2 HuggingFace client + model downloader (#438):
  - `internal/hfclient`: pure-Go HTTP client for the Hub. No third-party
    HF SDK, no CGO. `Client` (constructed via `New(Options{...})`)
    exposes `Whoami` (`/api/whoami-v2`), `Search` (`/api/models`),
    `ModelInfo` (`/api/models/{repo}`), `ListFiles`
    (`/api/models/{repo}/tree/{rev}?recursive=true`) and `Download`
    (`/{repo}/resolve/{rev}/{path}`). Sentinel errors `ErrUnauthorized`
    (folds 401+403), `ErrNotFound`, `ErrRateLimited` for typed
    branching in callers. `User-Agent: vllm-warden/0.1
    (+https://podwarden.com)` and a 30s per-call HTTP timeout (vs.
    Go's no-timeout default) on every request.
  - `Download` semantics: HEAD probe reads `Content-Length` and
    `Accept-Ranges`, GET sends `Range: bytes=N-` when a `<file>.part`
    is present and the server advertises range support. Server
    returning `200 OK` to a Range request triggers a truncate-and-
    restart. `.part` is only deleted when the server cannot resume,
    fsync'd before rename, atomically renamed onto the final path.
    Size mismatch leaves the `.part` in place for retry. Throttled
    progress callback at 200ms cadence; final 100% callback fired
    unconditionally.
  - LFS handling: `ListFiles` flattens the LFS sub-object so
    `RepoFile.Size` is the canonical weight size (not the
    `.gitattributes` pointer's ~130 bytes) and `RepoFile.SHA256` is
    the LFS oid for non-empty LFS rows.
  - `ModelInfoResponse.IsGated()` normalises HF's quirky
    boolean-or-string `gated` field.
  - `internal/models`: `Manager` orchestrator composes `hfclient` +
    `db.Store` + on-disk layout. `LocalPathFor(repo, rev)` returns
    `{baseDir}/{owner}/{name}/{rev}`. `EnsureDiskSpace` runs
    `syscall.Statfs` against `baseDir` and rejects allocations that
    would leave less than 2 GiB headroom (`ErrInsufficientDiskSpace`).
    `EstimateVRAMBytes(params, precision)` returns the FP16 / INT8 /
    INT4 weight footprint for the wizard's GPU-fit check.
    `DownloadModel` ties it together: `ModelInfo` → resolve sha →
    create row → `ListFiles` → space check → per-file
    `UpsertModelFile` + `Download` + aggregate `UpdateModelProgress`
    → `UpdateModelStatus(ready)`. Gated repos surface `ErrGatedRepo`
    up front.
  - `internal/db/migrations/0002_models.sql`: `models` table
    (`id`, `repo`, `revision`, `status` enum
    `pending|downloading|ready|failed`, `local_path`,
    `total_size_bytes`, `downloaded_size_bytes`, `created_at`,
    `updated_at`, `error_message`; `UNIQUE(repo, revision)`,
    `idx_models_status`) and `model_files` table with
    `FOREIGN KEY(model_id) REFERENCES models(id) ON DELETE CASCADE`,
    `UNIQUE(model_id, path)`, `idx_model_files_model`. All
    timestamps unix seconds matching the existing convention.
  - `Store` extended with eight model-state methods (`CreateModel`
    is a `INSERT ... ON CONFLICT DO UPDATE ... RETURNING` upsert so
    re-entry from the wizard returns the existing row rather than
    failing; `UpdateModelStatus` clears `error_message` on every
    non-failed transition; `DeleteModel` is idempotent;
    `UpsertModelFile` keys on `(model_id, path)`). Tests cover
    insert, conflict-on-repo-revision, status transitions, progress
    updates preserving `total` when zero is passed, FK cascade, and
    the SQLite `CHECK` constraint on illegal status strings.
- Phase 2 config & SQLite data layer (#437):
  - `internal/config`: env-driven `Runtime` (`VLLM_WARDEN_DATA_DIR`,
    `VLLM_WARDEN_LISTEN_ADDR`, `VLLM_WARDEN_LOG_LEVEL`) plus a JSON
    `Manager` for `/data/config.json` (`model`, `vllm_args`,
    `admin_password_hash`, `hf_token`, `trust_proxy_auth`). Atomic
    write (temp-file + rename + fsync), 0o600 perms, RWMutex-guarded
    cache, race-clean tests.
  - `internal/db`: pure-Go SQLite store on
    `modernc.org/sqlite v1.38.2` (`CGO_ENABLED=0` build still
    static), embedded `migrations/*.sql` via `go:embed`, idempotent
    migration runner with `schema_version` bookkeeping.
  - Initial schema (migration `0001_init.sql`): `api_keys`
    (id, name, key_hash, key_prefix, rate_limit, created_at,
    last_used_at, revoked + idx_api_keys_hash), `usage_log`
    (id, key_id, timestamp, endpoint, prompt_tokens,
    completion_tokens, model, latency_ms + idx by ts and (key, ts)),
    `usage_hourly` (key_id, hour, request_count, prompt_tokens,
    completion_tokens; PK (key_id, hour)).
  - `Store` interface + `*SQLiteStore` impl: `CreateAPIKey` (mints
    `pwk_<64hex>` with 256-bit entropy, SHA-256 hash stored,
    plaintext returned once), `ListAPIKeys`, `GetAPIKeyByPlaintext`
    (rejects revoked keys as `ErrKeyNotFound`, bumps last_used_at),
    `RevokeAPIKey` (idempotent), `RecordUsage` (single-tx
    log+rollup upsert), `HourlyForKey`, `PruneUsageLogOlderThan`.
  - `cmd/sidekick/main.go` wires the new pieces in minimally:
    `LoadRuntime` → `db.Open(rt.DataDir)` → `config.Manager.Load`
    (missing file is the expected first-boot state); `--addr`
    flag retained as an explicit override above
    `VLLM_WARDEN_LISTEN_ADDR`.

### Changed
- GPU detect contract refinements (#3): clarified `GPU.MemoryUsedBytes` docstring (populated by both `Detect` and `Poll`); normalised `ComputeCapability` parser-boundary sentinels (`[N/A]`, `N/A`, `-`, `[Not Supported]`, case-insensitive) to empty string via new `normalizeOptional` helper; added `gpu.DefaultDetectTimeout = 10 * time.Second` constant with `Detector` interface guidance for callers; new `[N/A]` testdata fixture (`old-driver-na-compute-cap.csv`) and unit tests covering the normaliser, the `ComputeCapability` round-trip, and `AutoDetector.Poll` dispatch / fall-through / error-propagation paths.
- Replaced Phase 0 CI echo-skeleton with real lint+test+build+push pipeline (F-3, issue #1).
  - **lint** stage: `golangci-lint` (staticcheck, govet, errcheck, ineffassign, gofmt) via `golangci/golangci-lint:latest-alpine`.
  - **test** stage: `CGO_ENABLED=1 go test -race -coverprofile=coverage.out ./...` with `coverage.out` artifact (7-day TTL); gcc installed on Alpine to enable the race detector.
  - **build** stage: `docker buildx` → `registry.podwarden.com/podwarden/apps/vllm-warden`; tag scheme aligned with workspace standard (`sha-{commit}` always, `{branch-slug}` always, `:staging` on develop, `:production`+`:latest` on main (manual gate), `v{YYYY.MM.DD.N}` on CalVer tag push).
  - Added `.golangci.yml` (staticcheck, govet, errcheck, ineffassign, gofmt; no stylistic noise).
- Fixed data race in `internal/web.Server`: `httpSrv` field now guarded by `sync.Mutex` so concurrent `ListenAndServe` and `Shutdown` calls do not race. Detected by `-race` when wiring up the CI test stage.
- Hardened HTTP server with `ReadHeaderTimeout` (gosec G112).
- Added graceful shutdown on SIGINT/SIGTERM with a 10s drain budget so
  `docker stop` (and systemd / k8s) exit cleanly with status 0 instead
  of being SIGKILL'd after the orchestrator's grace window.

### Added
- GPU detection module (#436): pure-Go `internal/gpu` package built around a
  vendor-neutral `Detector` interface and a `GPUInfo`/`GPU` data shape consumed
  by `internal/models/fit.go` (issue #439).
  - `NvidiaDetector` shells out to `nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu,power.draw,power.limit,temperature.gpu,compute_cap --format=csv,noheader,nounits`
    and a second host-wide `driver_version,cuda_version` query; CSV parser is
    tolerant of `[N/A]` / `[Not Supported]` (coerce to zero) and short rows
    (degrades to per-field zero, never aborts the batch).
  - VRAM is normalised to bytes at the parser boundary; `GPUInfo.TotalVRAMBytes`
    is pre-summed for the fit logic.
  - "No GPU" is a non-error result (`Count: 0`, `Vendor: VendorNone`), not an
    error: matches both nvidia-smi missing and nvidia-smi exiting with
    "No devices were found" (exit 9). `Detect` only errors on a real failure.
  - `Detect` (one-time static facts) and `Poll` (live utilisation/power/
    temperature/used-VRAM) accept a `context.Context` honoured by
    `exec.CommandContext`.
  - `AutoDetector` composes per-vendor backends (NVIDIA only in v1) and routes
    to whichever reports a GPU; structured for ROCm/Intel additions later.
  - Test seam: `NvidiaDetector.runner` accepts a fake exec function. Tests
    drive committed `testdata/` fixtures (3-GPU bonus shape, single RTX 3090,
    Tesla M40 with `[Not Supported]` power readings) plus a real `sh -c` exec
    to verify the no-devices stderr-matching path.
- Phase 1 scaffolding (#435): `go mod init`, full directory tree per design
  spec §Project Structure, `go:embed` for `web/templates/` and `web/static/`,
  pinned Dockerfile (`vllm/vllm-openai@sha256:04563c302537a91aa49ebdfbceda96111c5712275999b7e8804fa598f0b5641d`),
  Makefile (`build`, `test`, `vet`, `fmt-check`, `docker-build`, `docker-run`),
  stub `cmd/sidekick/main.go` serving a Tailwind-CDN placeholder on `:8080`,
  `internal/web` server with `httptest` smoke coverage.
- HTMX 2.0.4 and Alpine.js 3.14.8 wired into `web/templates/layout.html` via
  pinned CDN scripts (foundation for #441 wizard interactivity); regression
  tripwire test in `internal/web/layout_libs_test.go` asserts both pinned
  URLs survive future edits.
- Phase 0 kickoff: repo created, CI skeleton green, bonus-node GPU prerequisite verified.
