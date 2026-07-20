# vllm-warden — Universal Engine: multi-accelerator builds + swappable engine drivers

**Status:** Draft (design anchor for follow-up issues — no MR open yet)
**Date:** 2026-05-25
**Author:** ip + Claude (Opus 4.7)
**Motivation:** vLLM Warden is now open source (github.com/Podwarden/vllm-warden, Apache-2.0). To be a *universal* self-hoster tool — not just a PodWarden component — it must run on hardware we don't control (older + newer NVIDIA, AMD ROCm, Intel XPU, …) and let users find a working vLLM build/version per model through trial and error, from the Warden UI.

> This spec defines the target architecture and the seams to build toward. It is intentionally broader than one MR. Lock decisions here; carve implementation into the issues listed under "Phasing". Subsequent refinements land as follow-up specs, not in-place edits.

## Goal

1. **Decouple the Warden control plane from the vLLM engine.** Stop baking the Warden app *into* the vLLM image. Ship a slim, accelerator-agnostic control-plane image; run the vLLM engine as a **separate, swappable container/pod** that Warden supervises.
2. **Support multiple accelerators and multiple vLLM builds** by selecting a different *engine image* — never by rebuilding Warden.
3. **Model-first, trial-and-error stack resolution from the UI:** the user picks a model, Warden recommends a starting engine image + config from detected hardware, and the user can "Try stack" (pull engine, attempt load, get a structured outcome) and iterate.
4. **One supervision contract, multiple drivers:** the same `vllm serve` argv runs under a Docker-socket driver (self-host) or a K8s-API driver (PodWarden / any cluster).

## Non-goals

- **Mac/Metal and Windows.** Explicitly dropped (user decision 2026-05-25). vLLM has no first-class Metal backend and Windows is out of scope. Linux + container runtime only.
- **Patching vLLM internals for hardware/quant support.** We select the right *upstream* engine image; we do not fork vLLM's kernels. (The existing GGUF `sed` patches in `Dockerfile` are a separate, temporary concern — see "Relationship to current patches".)
- **A Kubernetes operator / CRDs.** See "Decision 4". Warden's DB is already the source of truth; we do not need declarative reconciliation.
- **Auto-discovery of *every* accelerator's quirks in v1.** We ship a seed compat table + the trial-and-error loop; a shared community compat DB is future work.

## Current state (pre-change)

- **One image.** `Dockerfile` is `FROM vllm/vllm-openai@sha256:…` (vLLM v0.20.0) with the Warden FastAPI app layered on top. To support a new accelerator or vLLM version we would have to rebuild this combined image N times.
- **In-container child process.** `app/runtime/supervisor.py` spawns `vllm serve` as a **child process inside the Warden container** (`asyncio.subprocess`). This is why the PodWarden-rendered deployment needs `pid: host` for GPU-holder attribution (see [[project_compose_bridge_pid_mode_gap]]) — the engine has no identity of its own.
- **argv builder already isolated.** `app/runtime/cmd_builder.py:build_vllm_args(model, *, port, overrides)` already produces the full `vllm serve` argv from a model row + overrides. This is the contract both future drivers consume unchanged.
- **Lifecycle state machine already exists.** `Supervisor` has `ModelState {LOADING, WARMING, READY, UNLOADING}`, `UnloadRefused`, warmup probe (`app/runtime/warmup_probe.py`), GPU ownership (`app/runtime/gpu_ownership.py`). The state machine is driver-agnostic; only the *spawn/kill/observe* primitives are Docker-process-specific.
- **Hardware-aware config already exists.** `app/models/suggest.py` + `fit.py` + `sharding.py` recommend quantization / TP size / `gpu_memory_utilization` from detected VRAM. They do **not** yet reason about compute capability → engine image.

## Three orthogonal axes (the thing we're actually selecting)

"Different builds/versions" is three independent dimensions. The engine image encodes the first two; Warden's config encodes nothing new for the third beyond what `cmd_builder` already does.

| Axis | What it determines | Example |
|---|---|---|
| **A. Accelerator family** | Which upstream image *lineage* | `vllm/vllm-openai` (CUDA), `rocm/vllm` (ROCm), `Dockerfile.xpu` (Intel), `intel/vllm-gaudi` (HPU), `Dockerfile.cpu` |
| **B. vLLM version + bundled toolkit** | Which archs / quants / compute-caps are supported | newer GPU (Blackwell sm_100/120, nvfp4/mxfp4) needs vLLM ≥ X + CUDA 12.8; an *old* GPU (Volta sm_70) may need an *older* vLLM |
| **C. Our GGUF patches** | Loader fixes applied at *our* build time | the `qwen3_5→qwen35` seds (see below) |

Key consequence: **"older and newer" cuts both ways.** Blackwell needs an edge build; Volta/Turing need fp16-only and possibly a pinned older build. A single "latest" tag cannot serve both ends, which is why we need **channels**, not one image.

### Hardware ↔ quant reality (seeds the recommender)

| Arch | Compute cap | fp16 | fp8 | nvfp4/mxfp4 | Notes |
|---|---|---|---|---|---|
| Volta | sm_70 | ✅ | ✗ | ✗ | fp16 only; may need pinned older vLLM |
| Turing | sm_75 | ✅ | ✗ | ✗ | fp16 only |
| Ampere | sm_80/86 | ✅ | emulated | ✗ | fp8 KV is sm_86 emulated → `nvidia-smi %util` undercounts (see [[project_vllm_gpt_oss_20b_a4000_ceiling]]) |
| Ada | sm_89 | ✅ | native | ✗ | |
| Hopper | sm_90 | ✅ | native | ✗ | native FP8 KV |
| Blackwell | sm_100/120 | ✅ | native | ✅ | needs newer vLLM + CUDA 12.8 (`:cuda-edge`) |
| AMD CDNA3 | — | ✅ | ✅ | ✗ | `rocm/vllm` lineage |
| Intel XPU | — | bf16 | limited int8 | ✗ | `Dockerfile.xpu` lineage |

## Decisions (locked)

### Decision 1 — Engine-sidecar split

The Warden control-plane image contains **only** the FastAPI app, the UI, and the driver code. It carries **no CUDA, no vLLM, no torch.** It is small and accelerator-agnostic. The vLLM engine runs as a **separate container/pod** ("engine") that Warden creates, observes, and tears down. Warden talks to the engine over HTTP (`/health`, `/v1/*`) exactly as the OpenAI proxy does today.

### Decision 2 — Engine driver interface

Introduce `app/runtime/engine/` with an abstract `EngineDriver`. The supervisor's lifecycle FSM stays; the *spawn/observe/kill* primitives move behind the driver. All drivers consume the **same argv** from `build_vllm_args(...)`.

```python
class EngineSpec:
    model_id: str
    image: str                 # resolved engine image (channel → digest)
    argv: list[str]            # from build_vllm_args(...)
    env: dict[str, str]        # from env_builder
    gpus: list[int] | "all"    # device indices
    port: int
    volumes: dict[str, str]    # named volumes only (see DooD gotcha)

class EngineHandle:
    id: str                    # container id / pod name
    base_url: str              # http://<addr>:<port>

class EngineDriver(Protocol):
    async def ensure_image(self, image: str) -> None: ...      # pull + progress
    async def start(self, spec: EngineSpec) -> EngineHandle: ...
    async def stop(self, handle: EngineHandle, *, grace_s: float) -> None: ...
    def logs(self, handle: EngineHandle) -> AsyncIterator[str]: ...
    async def health(self, handle: EngineHandle) -> bool: ...
```

### Decision 3 — `DockerSocketDriver` (self-host) = host-socket-sibling (DooD)

The default driver for `docker`-install self-hosters. **Plain language:** Warden and the engine are two containers *side by side* (siblings) on the same machine. Warden is given the host's Docker control channel (`/var/run/docker.sock`); when a user loads a model, Warden asks the **host** Docker daemon to pull the chosen engine image and run it as a sibling container, and the host's NVIDIA Container Toolkit maps the GPUs into that engine container the same proven way it does today. No nested Docker, no `--privileged`.

Locked details:
- **GPU mapping:** host daemon via `--gpus '"device=0,1"'` (NVIDIA), or `--device /dev/kfd --device /dev/dri` (ROCm), `/dev/dri` (XPU).
- **Volumes must be named, not bind paths.** Bind paths resolve on the *host*, not inside Warden. Reuse the existing named volumes `vllm-warden-hfcache`, `vllm-warden-data`. (DooD gotcha — documented prominently.)
- **Observability still works:** engine is a host PID, so the GPU-ownership and stats samplers see it normally (and the `pid: host` workaround becomes unnecessary in this topology).
- **Security:** `docker.sock` is root-equivalent. Default-document the risk; ship an *optional* `tecnativa/docker-socket-proxy` sidecar scoped to `IMAGES=1 CONTAINERS=1 POST=1` (pull/create/start/stop/logs only). Not mandatory for a single-owner box; recommended for shared hosts.
- **Rejected alternative — Docker-in-Docker (nested):** requires `--privileged`, GPUs injected into the outer container *and* an inner daemon with its own NVIDIA toolkit (double driver injection), plus duplicated image storage. Fragile, no upside. Rejected.
- **Rejected alternative — sysbox:** Nestybox/`sysbox-runc` would let us run a *nested* unprivileged daemon, but it must be installed on the host (root, kernel ≥5.12 / shiftfs) and its GPU passthrough is experimental. DooD already runs the engine unprivileged without any of that. Not needed.

### Decision 4 — `K8sApiDriver` (generic K8s / PodWarden) — NOT an operator

In a pod there is no Docker socket. The K8s equivalent of "host daemon attaches the GPU" is **the API server + NVIDIA device plugin**: create a Deployment whose pod spec carries `resources.limits."nvidia.com/gpu": N` + nodeSelector/affinity, and the device plugin injects the GPUs. The driver is **imperative** create/patch/delete against `apps/v1` Deployments + `core/v1` Service/pods/logs — *no CRDs, no controller-runtime, no Helm operator.* Warden's DB + FSM are already the source of truth, so there is nothing for a reconcile loop to own.

RBAC footprint: a **namespaced Role** (deployments, services, pods, pods/log: get/list/watch/create/patch/delete) bound to Warden's ServiceAccount, in Warden's own namespace. That is the entire infra delta.

### Decision 5 — `PodWardenDriver` (optional, deferred)

Because vLLM Warden is itself a catalog app *deployed by* PodWarden, engine pods created by the raw `K8sApiDriver` are **off the books** for PodWarden's inventory, capacity accounting (`find_gpu_capacity`), and doctor checks. An optional third driver delegates engine lifecycle to PodWarden's existing API (`create_deployment` / `update_deployment` / `delete_deployment`), so engine pods show up as first-class managed workloads and inherit PodWarden's placement, pull secrets, and GPU scheduling. Cost: Warden must know it runs under PodWarden and hold a PodWarden token.

**Shipping order:** `DockerSocketDriver` + `K8sApiDriver` first (both fully self-contained, no PodWarden dependency). `PodWardenDriver` only if/when we want engine pods first-class in PodWarden's tracking. This keeps the open-source product standalone while leaving a clean seam for the PodWarden-native integration.

### Decision 6 — Engine image channels + tagging

Warden resolves a **channel** (stable contract) to an **immutable digest** (what actually runs), recording the digest on the model's engine profile.

| Channel tag | Lineage | For |
|---|---|---|
| `:cuda-stable` | `vllm/vllm-openai` pinned | Ampere/Ada/Hopper, mainstream |
| `:cuda-edge` | newest vLLM + CUDA 12.8 | Blackwell, newest quants |
| `:cuda-legacy` | pinned older vLLM | Volta/Turing fp16-only |
| `:rocm` | `rocm/vllm` | AMD CDNA |
| `:xpu` | built from `Dockerfile.xpu` | Intel GPU |
| `:cpu` | built from `Dockerfile.cpu` | no-GPU smoke / tiny models |

Underlying immutable tags: `…-vllm<ver>-<accel>-<YYYYMMDD>`. Our GGUF patches (axis C) are applied in *our* engine-image build, layered on the upstream base — so patched engine images live under our registry, unpatched ones can point straight at upstream.

### Decision 7 — Model-first trial-and-error UI

Per-model **engine profile** persisted on the model row: `engine_channel`, `resolved_engine_image` (digest), plus the existing load-config overrides. UI flow:

1. User adds/selects a model.
2. Warden detects accelerator family + compute cap and **recommends** a starting channel + quant (seeded by the table above via `suggest.py`/`fit.py`).
3. **"Try stack"** action: `ensure_image` (with pull progress) → `start` → warmup probe → **structured outcome**.
4. On failure, a **classifier** (regex over vLLM startup stderr) buckets the cause: `unknown-arch` / `unsupported-quant` / `compute-cap-too-old` / `missing-kernel` / `OOM` / `hf-gated` — each mapped to a concrete next suggestion ("try `:cuda-edge`", "drop to fp16 on `:cuda-legacy`", "lower `gpu_memory_utilization`", "accept HF terms").
5. User iterates; the winning profile is saved on the model.

## Relationship to current GGUF patches

The three idempotent `sed` patches in `Dockerfile` (`qwen3_5`→`qwen35`, `qwen3_5_moe`→`qwen35moe`, vision-config `num_hidden_layers` fallback; issues #107/#115/#108, upstream vllm#38140 OPEN) are **axis C** — loader fixes, not hardware/quant fixes. Under this design they move into the *engine-image* build (our patched `:cuda-*` images), out of the control-plane image entirely. They get deleted when #38140 merges and we bump the engine base.

## File-by-file change list (target; carved across issues — see Phasing)

**New:**
- `app/runtime/engine/__init__.py` — `EngineDriver` protocol, `EngineSpec`, `EngineHandle`
- `app/runtime/engine/docker_socket.py` — `DockerSocketDriver` (DooD)
- `app/runtime/engine/k8s_api.py` — `K8sApiDriver`
- `app/runtime/engine/podwarden.py` — `PodWardenDriver` (deferred / Phase 4)
- `app/runtime/engine/images.py` — channel→digest resolution + pin table
- `app/models/stack_resolver.py` — accelerator/compute-cap detection → recommended channel + quant
- `app/models/failure_classifier.py` — regex buckets over vLLM startup stderr
- `docs/superpowers/specs/2026-05-25-universal-engine-driver-design.md` — this spec
- engine-image build context(s): `engine/cuda/`, `engine/rocm/`, `engine/xpu/`, `engine/cpu/` (Dockerfiles layering our GGUF patches on the upstream lineage)
- tests under `tests/unit/engine/`, `tests/unit/models/test_stack_resolver.py`, `tests/unit/models/test_failure_classifier.py`

**Modified:**
- `app/runtime/supervisor.py` — keep FSM; route spawn/observe/kill through the selected `EngineDriver`; drop the in-container `asyncio.subprocess` path (becomes the DooD driver's job)
- `app/runtime/cmd_builder.py` — unchanged contract; verify argv is engine-image-agnostic
- `app/runtime/gpu_ownership.py` / `stats_sampler.py` — read engine identity from `EngineHandle` instead of assuming a child PID
- `app/models/schemas.py` + DB migration — add `engine_channel`, `resolved_engine_image` columns
- `app/models/suggest.py` / `fit.py` — consult `stack_resolver`
- `app/models/routes_api.py` — `POST /api/models/{id}/try-stack`; surface classifier outcome
- frontend: model detail — engine profile panel, "Try stack" + pull progress, classifier outcome + next-step chips
- `Dockerfile` — slim control-plane image (remove vLLM/CUDA base + the GGUF seds)
- `docker-compose.yml` — mount `docker.sock` (+ optional socket-proxy); named volumes only
- `.gitlab-ci.yml` — engine-image **publish matrix** (lineage × channel)
- `docs/operating.md`, `docs/releasing.md`, `README.md`, `changelog.md`, K8s RBAC manifest

## Phasing (proposed issues)

1. **Engine-sidecar split + `DockerSocketDriver`** — slim control-plane image, driver interface, DooD driver, named-volume migration, socket-proxy doc. *Unblocks self-host universality.*
2. **Engine-image channels + CI publish matrix** — `:cuda-stable/-edge/-legacy/:rocm/:xpu/:cpu`; move GGUF patches into engine build.
3. **`stack_resolver` + `failure_classifier` + "Try stack" UI** — the model-first trial-and-error loop.
4. **`K8sApiDriver`** (+ RBAC manifest) — generic-cluster + PodWarden parity.
5. **`PodWardenDriver`** (optional) — first-class engine pods in PodWarden inventory.
6. **Upstream vllm#38140 follow-through** — delete GGUF seds when merged + base bumped.

## Risks

- **`docker.sock` exposure** is root-equivalent on the host. Mitigation: default-document it, ship optional scoped socket-proxy, recommend it for shared hosts. This is the central security tradeoff of the self-host path and must be stated plainly in `operating.md` and the catalog/news copy.
- **Engine images are large (5–15 GB each).** Multiple channels multiply disk. Mitigation: pull-on-demand, pull-progress UI, GC of unused engine images, disk-pressure surfacing (ties to [[reference_docker29_containerd_image_store]] — snapshot tree is 3–5× content tree; channel sprawl accumulates fast).
- **Classifier false-buckets.** Regex over vLLM stderr drifts as upstream changes wording. Mitigation: classifier returns the raw tail alongside the bucket; never hide the real error; treat buckets as *suggestions*, not gates.
- **Two (then three) drivers diverge.** Mitigation: the `EngineDriver` protocol + a shared conformance test suite (same `EngineSpec` → same observable lifecycle) run against a Docker fixture and a kind/k3d fixture.
- **`pid: host` removal regressions.** The DooD topology makes the engine a real host PID, so GPU attribution should *improve*; verify the stats sampler before deleting the `pid: host` assumption for the in-container path.

## What is explicitly NOT changing

- `build_vllm_args` contract and the override keys — unchanged.
- The `ModelState` FSM, `UnloadRefused`, warmup-probe semantics — unchanged; only the spawn/kill primitives move behind a driver.
- vLLM upstream — not forked for hardware/quant.
- Mac/Metal, Windows — out of scope, not revisited here.
