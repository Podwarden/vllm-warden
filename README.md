# LLM Warden

**Run your own OpenAI-compatible LLM API on your own GPUs — with a UI, not a config file.**

An inference engine such as [vLLM](https://github.com/vllm-project/vllm) is fast, but it
ships as a single-model process: no UI, no auth, no model switching, and no view of what
your GPUs are doing. LLM Warden wraps the engine in a control plane so you can pull a model
from HuggingFace, load it, and hand your team an API key — from a browser, in minutes.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Engine](https://img.shields.io/badge/engine-vLLM%20v0.26.0-4b8bbe.svg)](https://github.com/vllm-project/vllm)
[![Deploy](https://img.shields.io/badge/deploy-Docker%20Compose-2496ed.svg)](#install)

![Models list](assets/screenshots/01-models-list.png)

---

## Quick start

A Linux host with Docker, Docker Compose v2.24+, an NVIDIA GPU, the NVIDIA Container
Toolkit, and 40 GB of free disk where Docker keeps its images:

```bash
git clone https://github.com/Podwarden/vllm-warden.git
cd vllm-warden
./install.sh
```

The installer checks the host, lets you pick GPUs, generates the secrets, pulls the release
images and offers to start the stack (details under [Install](#install)).
Open `http://YOUR-HOST:8080/ui/`. A first-run wizard walks you through picking
which GPUs to use, adding a HuggingFace token, and creating your admin account.
Then go to **Models → Add model** and pull your first model.

Your OpenAI-compatible endpoint is live at:

```
http://YOUR-HOST:8080/v1/chat/completions
```

Point any OpenAI client at it — LangChain, OpenWebUI, the `openai` SDK, your agents.
Only the `base_url` and the key change.

## Why you might want it

| Without | With LLM Warden |
|---|---|
| One model per container, restart to switch | Hot-swap models from the browser |
| A single shared API key, or none | Per-key tokens, rate limits, priority lanes, rotation |
| `nvidia-smi` in a second terminal | VRAM, utilisation and power graphs, per-GPU |
| No idea why a request is slow | Live per-request table, KV-cache pressure, TTFT/latency percentiles |
| Hand-edited `--tensor-parallel-size` flags | Guided model setup with a config UI |
| An abandoned request pins a GPU slot | Request reaper reclaims it automatically |

## Features

- **Browser UI** — model management, live vLLM logs, chat playground, stats dashboards
- **OpenAI-compatible gateway** at `/v1/*` — drop-in for any existing client
- **Model lifecycle** — pull from HuggingFace, hot-swap without restarting the container,
  per-model settings, GGUF support via the out-of-tree
  [`vllm-gguf-plugin`](https://pypi.org/project/vllm-gguf-plugin/)
- **Multi-token auth** — per-key rate limits, priority lanes, rotation grace windows, usage stats
- **Realtime dashboard** — per-second engine load, KV-cache pressure, throughput and latency
  percentiles, live request table
- **HuggingFace cache manager** — see what's on disk, garbage-collect orphans
- **GPU observability** — per-GPU VRAM/utilisation/power, with vLLM process attribution
- **Automatic crash recovery** — a dead engine is detected, evidence is captured, and the
  model is reloaded without human intervention
- **Single-port topology** — one Caddy front door on `:8080` serves UI, control API and the
  OpenAI shim, so one reverse-proxy rule covers everything

## Screenshots

<table>
  <tr>
    <td><a href="assets/screenshots/02-chat-playground.png"><img src="assets/screenshots/02-chat-playground.png" width="260" alt="Chat playground"/></a><br/><sub><b>Chat playground</b> — try a model without wiring up a client</sub></td>
    <td><a href="assets/screenshots/03-stats-dashboard.png"><img src="assets/screenshots/03-stats-dashboard.png" width="260" alt="Stats dashboard"/></a><br/><sub><b>Stats</b> — throughput, latency, token usage per key</sub></td>
  </tr>
  <tr>
    <td><a href="assets/screenshots/12-model-configuration-detail.png"><img src="assets/screenshots/12-model-configuration-detail.png" width="260" alt="Model configuration"/></a><br/><sub><b>Model config</b> — parallelism, quantization, context length</sub></td>
    <td><a href="assets/screenshots/13-model-detail-live-logs.png"><img src="assets/screenshots/13-model-detail-live-logs.png" width="260" alt="Live logs"/></a><br/><sub><b>Live engine logs</b> — stream vLLM output while it loads</sub></td>
  </tr>
  <tr>
    <td><a href="assets/screenshots/05-api-tokens-list.png"><img src="assets/screenshots/05-api-tokens-list.png" width="260" alt="API tokens"/></a><br/><sub><b>API tokens</b> — per-key limits and usage</sub></td>
    <td><a href="assets/screenshots/04-cache-manager.png"><img src="assets/screenshots/04-cache-manager.png" width="260" alt="Cache manager"/></a><br/><sub><b>HF cache</b> — reclaim disk from orphaned weights</sub></td>
  </tr>
</table>

<details>
<summary>More screenshots — settings tabs</summary>

| General | Networking | Sessions &amp; Tokens | Maintenance |
|---|---|---|---|
| <a href="assets/screenshots/07-settings-general.png"><img src="assets/screenshots/07-settings-general.png" width="180"/></a> | <a href="assets/screenshots/08-settings-networking.png"><img src="assets/screenshots/08-settings-networking.png" width="180"/></a> | <a href="assets/screenshots/09-settings-sessions-tokens.png"><img src="assets/screenshots/09-settings-sessions-tokens.png" width="180"/></a> | <a href="assets/screenshots/10-settings-maintenance.png"><img src="assets/screenshots/10-settings-maintenance.png" width="180"/></a> |

</details>

## Install

Everything needed to run LLM Warden is in this repository: `docker-compose.yml` is the
stack, `.env.example` is the configuration contract, `install.sh` turns them into a running
install, and the `Makefile` runs it day to day. No account, no catalog, nothing phones
home. The only network access is pulling the release images from
`registry.podwarden.com` (anonymous pull), and even that can be replaced by `docker load`
— see [Offline / air-gapped install](#offline--air-gapped-install).

**Requirements**

- Linux x86_64 with Docker Engine and the Docker Compose v2 plugin, **2.24 or newer**
  (`docker compose version`). The generated override uses Compose's `!override` tag.
- One or more NVIDIA GPUs with the driver installed (`nvidia-smi` lists them).
- The NVIDIA Container Toolkit, registered with Docker (`docker info` lists an `nvidia`
  runtime). A working `nvidia-smi` is **not** enough — the driver can be fine while
  Docker still cannot hand a GPU to a container. The installer checks this specifically
  and offers to install and register the toolkit.
- **40 GB free on the Docker data root** before any model is pulled — the filesystem
  under `docker info --format '{{.DockerRootDir}}'` (usually `/var/lib/docker`), which is
  often not the one under `/`. The api image is 9.2 GB compressed on the wire but **~29 GB
  as Docker stores it**: 19.7 GB unpacked, and the containerd image store that a fresh
  Docker Engine 29 uses keeps the compressed layers alongside. The ui and caddy images add
  ~0.4 GB. Models come on top and are not small: a single 7B AWQ checkpoint is ~5 GB more,
  and the HuggingFace cache (the `vw-hfcache` volume) grows with every model you pull,
  without bound — size the disk for the models you mean to keep. The installer measures
  free space there, warns under 40 GB, and refuses a first pull under 20 GB, where the
  image cannot even be unpacked (`--check` reports the number without installing).

### Interactive

```bash
git clone https://github.com/Podwarden/vllm-warden.git
cd vllm-warden
./install.sh
```

The installer, in order: verifies Docker, Compose, free disk space and the NVIDIA runtime; lists the GPUs and
asks which to pass to the engine (all, by default); creates `.env` from `.env.example` and
generates `VW_COOKIE_SECRET`; pins `VERSION` to the release this tree documents; writes
`docker-compose.override.yml`; validates the merged config; pulls the images; and asks
whether to start. Re-running it is safe: `.env` is kept, only blank secrets are filled in,
and the override is regenerated.

| File | Written by | What it holds |
|---|---|---|
| `docker-compose.yml` | git | the stack: services, wiring, volumes. Never edited by the installer. |
| `.env` | `install.sh`, once | secrets, `VERSION`, `WARDEN_PORT`, `VW_*` knobs — every one documented in `.env.example` |
| `docker-compose.override.yml` | `install.sh`, every run | the release images, the GPU selection, health checks, the front-door port |

### Unattended

```bash
./install.sh --gpus all --yes --start                                   # all GPUs, start now
./install.sh --gpus 0,1 --origin https://llm.example.com --port 8080 --yes
GPU_TOOLKIT_INSTALL=yes ./install.sh --gpus all --yes                   # also install the toolkit
./install.sh --gpus none --yes                                          # CPU-only control plane (evaluation, CI)
./install.sh --check                                                    # preflight only, writes nothing
```

| Flag | Meaning |
|---|---|
| `--dir PATH` | install directory (default: the checkout; `/opt/vllm-warden` or `~/vllm-warden` when downloaded) |
| `--version TAG` | release to run, e.g. `v2026.09.03.5` or `latest` (default: the newest release in `CHANGELOG.md`) |
| `--gpus all\|none\|0,1` | GPUs passed to the engine, by `nvidia-smi` index |
| `--origin URL[,URL]` | `VW_FRONTEND_ORIGIN`: the public URL(s) of the UI, enforced by the CSRF check once set |
| `--port N` | `WARDEN_PORT`: host port of the single published front door (8080) |
| `--no-generate-secrets` | leave `VW_COOKIE_SECRET` blank for you to fill in |
| `--no-pull` | do not pull images (air-gapped: `make load-images` first) |
| `--start` / `--no-start` | start when done / never (default: ask on a terminal) |
| `-y`, `--yes` | never prompt |
| `--check` | run the host preflight and stop |
| `GPU_TOOLKIT_INSTALL=yes\|no` | install the NVIDIA Container Toolkit without asking / never. **`yes` restarts the Docker daemon** — see the warning below |

Exit status: `0` installed and startable; `1` a preflight or argument problem; `2` files
written but the stack cannot start yet — the message says what is missing (typically the
NVIDIA runtime).

> **Installing the toolkit restarts Docker.** Registering the `nvidia` runtime
> writes `/etc/docker/daemon.json` and then restarts `dockerd`, which restarts
> **every container on the host**, not only this stack's. Containers with a
> restart policy come back; anything started without one does not. On a host
> already running other services, install the toolkit yourself at a time you
> choose and re-run `./install.sh` — the preflight (`./install.sh --check`)
> tells you whether it is needed without changing anything.

### Without a clone

```bash
curl -fsSL https://raw.githubusercontent.com/Podwarden/vllm-warden/main/install.sh | sh -s -- --dir /opt/vllm-warden
```

Downloads the source tree into `--dir`, then proceeds exactly as above. Prompts still work
when piped (the script reads them from the terminal, not stdin); add `--yes` for automation.

The one-liner on the [PodWarden Hub catalog page](https://podwarden.com/catalog/vllm-warden)
also still works. It is a convenience wrapper around the same stack; this repository is the
reference and needs nothing from podwarden.com.

### Did it work?

Whichever route you took, `make smoke` is the one-line check that the front
door is really serving:

```bash
make smoke      # asserts 200s across / /_landing /ui/ /api/csrf /healthz
```

### First run without a browser

The wizard is a browser flow, but it is only a client of `/api/setup/*` — so a
first run can be scripted end to end. Everything below runs against a freshly
started stack on `http://127.0.0.1:8080` and needs nothing but `curl` and `jq`.

`/api/setup/*` is exempt from the CSRF check, so these six calls need no token
and no cookie jar. They are a strict state machine: posting out of order returns
`400 not at <step> step (current: <where you are>)`, and `GET /api/setup/state`
always tells you where that is — so an interrupted run resumes rather than
restarts.

```bash
W=http://127.0.0.1:8080

# 1. Where are we? A fresh install answers {"step":"welcome","done":false}.
curl -s $W/api/setup/state

# 2. Acknowledge the welcome step.                    -> {"step":"gpus"}
curl -s -X POST $W/api/setup/welcome

# 3. List the GPUs the container can see, with their indices.
curl -s $W/api/setup/gpus
# [{"index":0,"name":"NVIDIA RTX A4000","memory_total_mib":16376,
#   "memory_used_mib":0,"utilization_pct":0}, ...]

# 4. Choose which of those indices the engine may use.  -> {"step":"hf_token"}
curl -s -X POST $W/api/setup/gpus -H 'Content-Type: application/json' \
     -d '{"allowed_gpu_indices":[0,1]}'

# 5. HuggingFace token. JSON null is valid and is the right answer unless you
#    need gated models (Llama, Mistral, gpt-oss).      -> {"step":"admin"}
curl -s -X POST $W/api/setup/hf_token -H 'Content-Type: application/json' \
     -d '{"hf_token":null}'

# 6. Create the admin account.                          -> {"step":"done"}
#    Password rules: at least 6 characters and at most 72 BYTES. The upper
#    bound is bcrypt's, and it is rejected rather than silently truncated —
#    worth knowing before you generate a long passphrase.
curl -s -X POST $W/api/setup/admin -H 'Content-Type: application/json' \
     -d '{"username":"admin","password":"CHANGE-ME"}'
```

`GET /api/setup/gpus` returns `404` once setup is done — deliberate, so a
completed install does not report its hardware to anonymous callers.

**Then mint an API key.** Unlike `/api/setup`, the rest of the control API does
enforce CSRF, so this is a two-header call: the CSRF token from `/api/csrf`
plus the JWT from the login.

```bash
# The response key is `csrf`, NOT `csrf_token`. Reading the wrong field sends
# an empty header and the server answers a flat
# `403 {"detail":"csrf token invalid"}` that says nothing about which field
# was wrong — so this one line is worth copying exactly.
CSRF=$(curl -s -c jar $W/api/csrf | jq -r .csrf)

JWT=$(curl -s -b jar -c jar -X POST $W/api/auth/login \
        -H 'Content-Type: application/json' \
        -d '{"username":"admin","password":"CHANGE-ME"}' | jq -r .access_token)

curl -s -b jar -X POST $W/api/tokens \
     -H 'Content-Type: application/json' \
     -H "Authorization: Bearer $JWT" -H "X-CSRF-Token: $CSRF" \
     -d '{"name":"my-first-key"}'
# {"id":"53e7011c…","name":"my-first-key","plaintext":"vw_su3yl…",
#  "prefix":"vw_su3yl","expires_at":"2027-09-06 17:01:32", …}
```

**`plaintext` is shown exactly once.** Only a SHA-256 hash is stored, so the
token list can never show it again — save it now, or mint another key. It is
the bearer token for `/v1/*`:

```bash
curl -s $W/v1/models -H "Authorization: Bearer vw_su3yl…"
```

One more thing a script needs to know: `/v1/*` is bearer-gated and CSRF-exempt
— it is an API for programs, not for browsers.

### Adding a model from the API

**Models → Add model** in the UI drives exactly these calls. Every one is
JWT-gated and CSRF-checked, so reuse `$JWT` and `$CSRF` from above; `AUTH`
below is just those two headers.

Register, pull, load are **three separate, asynchronous steps**. Each returns
`202` immediately and reports progress somewhere else — the row's `status`
walks `registered → pulling → pulled → loading → loaded`, and you poll
`GET /api/models/{id}` for it.

```bash
AUTH=(-H "Authorization: Bearer $JWT" -H "X-CSRF-Token: $CSRF"
      -H 'Content-Type: application/json')

# 1. Register. `gpu_indices` is required and must be a subset of what you
#    allowed in the wizard. Everything else has a default.
curl -s -X POST $W/api/models "${AUTH[@]}" -d '{
      "served_model_name": "qwen2.5-1.5b",
      "hf_repo": "Qwen/Qwen2.5-1.5B-Instruct",
      "gpu_indices": [0],
      "max_model_len": 8192
    }'
# 201 {"id":"9895a1829559533c","served_model_name":"qwen2.5-1.5b",
#      "status":"registered"}

M=9895a1829559533c

# 2. Pull the weights from HuggingFace.        202 {"status":"pulling","force":false}
curl -s -X POST $W/api/models/$M/pull "${AUTH[@]}"

# Progress is a SERVER-SENT EVENT stream — `data: {…}` lines, one per second,
# not a JSON document. Read it line by line; piping it to `jq` will not work.
curl -sN $W/api/models/$M/pull/progress -H "Authorization: Bearer $JWT"
# data: {"status": "pulling", "bytes": 2126013055, "total": 3098973447, ...}
# data: {"status": "pulled",  "bytes": 3098973447, "total": 3098973447, ...}

# 3. Load it onto the GPU.                     202 {"status":"loading","port":10000}
curl -s -X POST $W/api/models/$M/load "${AUTH[@]}"

# 4. Watch it become `loaded` (or `failed`, with `last_error` saying why).
curl -s $W/api/models/$M -H "Authorization: Bearer $JWT" | jq '.status, .last_error'

# 5. Free the GPU again.                       202
curl -s -X POST $W/api/models/$M/unload "${AUTH[@]}"
```

Loading is not instant: the weights go to the GPU, CUDA graphs are captured
and a warmup request is served before the model is reported `loaded`. Tens of
seconds is normal for a small model, minutes for a large one.

`POST /api/models/fit-preview` answers "will this fit?" before you pull
anything. It takes the same `hf_repo` and (optional) `filename` as the create
call and returns a `green`/`yellow`/`orange`/`red` verdict with the arithmetic
behind it.

### Two things about GPUs that will surprise you

**One loaded model per GPU.** A GPU is claimed exclusively by the model loaded
on it. A second load onto a busy card is refused outright:

```
GPU 0 is already serving 'qwen2.5-1.5b' — unload it first, or load this model
on a free GPU. LLM Warden runs one loaded model per GPU.
```

That is an ownership rule, not a capacity check — it is decided before the
engine starts, so no amount of tuning gets past it. Switching models means
unload, then load.

**`gpu_memory_utilization` is a fraction of the whole card, not of the model.**
This is the number that surprises people. vLLM reserves that share of the card
up front and fills the part not holding weights with KV cache. At the default
`0.9`, a 1.5B model whose weights are ~3 GB occupies **~15 GB of a 16 GB card**
— about 10.5 GB of it KV cache. Nothing is wrong: a big KV cache is what lets
the server run many concurrent requests, and the engine log says so plainly
(`Available KV cache memory: 10.55 GiB`).

Two consequences worth internalising:

- **Sizing a card by model weights is wrong.** Size it by weights *plus* the
  context you intend to serve, or lower `gpu_memory_utilization` and accept
  fewer concurrent requests.
- **A card that looks 94 % full is not a leak.** It is the reservation. The
  free-VRAM figure `nvidia-smi` shows for a serving card tells you almost
  nothing about how much more work it could take.

### Day-to-day

| Command | What it does |
|---|---|
| `make start` | start the stack (detached) |
| `make stop` | stop and remove the containers; volumes and data stay |
| `make restart` | stop + start, re-reading `.env` and the override |
| `make logs` | follow live logs (`make logs S=api` for one service) |
| `make status` | container state and health |
| `make pull` | pull the release named by `VERSION` in `.env` and restart on it |
| `make config` | print the fully merged compose config — what actually runs |
| `make preflight` | re-run the installer's host checks |
| `make smoke` | check the front door end to end (`/`, `/_landing`, `/ui/`, `/api/csrf`, `/healthz`) |
| `make uninstall` | stop and delete the data volumes (asks first) |
| `make help` | list every target |

**Upgrading:** set `VERSION` in `.env` to the new release (or `./install.sh --version vX`)
and `make pull`. When the stack files themselves changed, `git pull && ./install.sh`
refreshes `docker-compose.yml` and the override and re-pins `VERSION`; `.env` is kept.

Once running:

- **UI** — `http://YOUR-HOST:8080/ui/`
- **OpenAI API** — `http://YOUR-HOST:8080/v1/chat/completions`
- **Control API** — `http://YOUR-HOST:8080/api/` (JWT-gated)
- **Health** — `http://YOUR-HOST:8080/healthz`

For gated models (Llama, Mistral, gpt-oss) you need a HuggingFace token. The
first-run wizard asks for one, and you can change it later under
**Settings → General**.

**On HTTP vs HTTPS.** Plain `http://` works — the session cookies follow the
scheme the browser actually used, so a LAN install stays signed in. It is
still an evaluation posture: the API key and the session both cross the
network in the clear. For anything shared, terminate TLS in front of the
`:8080` listener and tell the warden its public URL with
`./install.sh --origin https://llm.example.com`; the cookies then carry
`Secure`. If your proxy sets `X-Forwarded-Proto`, set `VW_TRUST_PROXY_ORIGIN=1`
in `.env` so the warden believes it.

### Offline / air-gapped install

Nothing in the stack needs the internet at run time except model pulls from HuggingFace,
and those can be pre-seeded. The transport is three image tarballs plus, optionally, a
tarball of the model cache; the `make` targets address the same image names and volume the
stack uses, so nothing is typed twice.

On a machine **with** internet access:

```bash
VERSION=v2026.09.03.5                                   # pick a release from CHANGELOG.md
git clone https://github.com/Podwarden/vllm-warden.git && cd vllm-warden

# 1. Stage an install (no GPU needed here) and save its images:
#    vllm-warden, vllm-warden-ui and caddy:2-alpine, all at $VERSION.
./install.sh --dir /tmp/vw-stage --version "$VERSION" --gpus none --yes
make -C /tmp/vw-stage save-images IMAGES_FILE=/tmp/llm-warden-$VERSION.tar

# 2. (Optional) pre-seed the model cache. The api pulls with
#    snapshot_download(cache_dir=<volume root>), so download with the same
#    layout: models--org--name directories at the top of the tarball.
pip install -U huggingface_hub
huggingface-cli download --cache-dir /tmp/hf-seed Qwen/Qwen2.5-7B-Instruct
tar -C /tmp/hf-seed -cf /tmp/hf-cache.tar .
```

Copy the source tree (this checkout or the GitHub tarball), `llm-warden-$VERSION.tar` and
`hf-cache.tar` to the isolated host. There:

```bash
# Docker, Compose 2.24+, the NVIDIA driver and the NVIDIA Container Toolkit
# come from your own OS mirrors -- the installer cannot download them here.
cd vllm-warden
make load-images IMAGES_FILE=/path/llm-warden-$VERSION.tar
./install.sh --version "$VERSION" --no-pull --gpus all --yes
make import-hf-cache CACHE_FILE=/path/hf-cache.tar        # optional
echo 'HF_HUB_OFFLINE=1' >> .env                           # never contact huggingface.co
make start
```

Then add the model in the UI by its HuggingFace name (`Qwen/Qwen2.5-7B-Instruct`); with
`HF_HUB_OFFLINE=1` the pull resolves from the seeded cache. `make export-hf-cache` does the
reverse on a running install, so a cache warmed on one host can seed the next.

## Architecture

One published port. Caddy fans out to internal-only `api` and `ui` containers:

| Path | Backend | Notes |
|---|---|---|
| `/` | FastAPI `/_landing` | Public landing page (can be disabled) |
| `/ui/*` | Next.js | Browser UI |
| `/api/*` | FastAPI | JWT-gated control plane |
| `/v1/*` | FastAPI | OpenAI-compatible proxy (token-gated) |
| `/healthz` | Next.js | Liveness probe |

The `api` container shares the host PID namespace so GPU process attribution can map host
PIDs back to supervisor-tracked vLLM workers. Routing lives in `deploy/caddy/Caddyfile`.

## Build from source

Both images build from this repository with nothing but Docker. There is no token, no
account, and no private registry anywhere in the build: every input is public npm, public
PyPI, Docker Hub, or a pinned commit on GitHub.

The operator files `./install.sh` writes double as the dev loop. `docker-compose.yml`
carries `build:` and the generated override carries `image:`, so `docker compose build`
tags what it builds with the release image name: `make restart` then runs your build, and
`make pull` puts the published image back.

```bash
git clone https://github.com/Podwarden/vllm-warden.git
cd vllm-warden

# .env + docker-compose.override.yml. --no-pull/--no-start keep this step
# registry-free: nothing is downloaded, nothing is started.
./install.sh --gpus all --no-pull --no-start -y

docker compose build            # both images; ~15 min cold, mostly the CUDA base
make restart
make smoke      # asserts 200s across / /_landing /ui/ /api/csrf /healthz
```

Or one image at a time:

```bash
docker compose build api        # engine image: CUDA base + llama.cpp compile
docker compose build ui         # UI image: chat-ui from source + Next.js build
```

**The `api` image** starts from the digest-pinned `vllm/vllm-openai` base (currently
v0.26.0), installs `requirements.txt` and `vllm-gguf-plugin` from PyPI, and compiles
llama.cpp from `github.com/ggml-org/llama.cpp` at the tag pinned in `ARG LLAMACPP_BUILD`.
It needs an x86_64 host: the llama.cpp step compiles SASS for `sm_75;sm_86` only, and the
base image's CUDA toolchain is what the pin was chosen against. No GPU is needed to
*build* it — only to run it. Expect a large download (the CUDA base is tens of GB
unpacked) and 40+ GB of free disk. Every patch it applies is commented in `Dockerfile`,
with the failure each one exists to prevent; read it before bumping the base digest.

**The `ui` image** compiles [`@podwarden/chat-ui`](https://github.com/Podwarden/chat-ui)
from source in its own build stage — cloned from GitHub at the exact commit pinned in
`frontend/Dockerfile`:

```dockerfile
ARG CHATUI_REPO=https://github.com/Podwarden/chat-ui.git
ARG CHATUI_REF=<40-char commit SHA>
ARG CHATUI_VERSION=<the version that commit publishes>
```

That stage runs chat-ui's own `npm ci && npm run build && npm pack`, and the packed
result replaces the copy `npm ci` installed for this project. The lockfile still pins
chat-ui's ~20 runtime dependencies with integrity hashes — that closure is what
`package-lock.json` is for — while the component's own code is the one compiled here.
The build asserts the pinned version against both the lockfile and the freshly built
tarball, so a `CHATUI_REF`/`CHATUI_VERSION` mismatch fails loudly instead of shipping a
component the dependency graph was not resolved for.

To build against a fork or an unreleased chat-ui, override the args — no repo edit needed:

```bash
docker build -t vllm-warden-ui \
  --build-arg CHATUI_REPO=https://github.com/you/chat-ui.git \
  --build-arg CHATUI_REF=<sha> --build-arg CHATUI_VERSION=<version> \
  frontend/
```

Note that `CHATUI_VERSION` must still match what `frontend/package.json` depends on,
because the lockfile supplies that release's dependency closure.

`@podwarden/chat-ui` is also on public npm, published from that same GitHub repository by
OIDC trusted publishing with a provenance attestation, so a plain `npm ci` in `frontend/`
(for `npm test`, `npm run typecheck`, or `next dev`) works without any of the above and
without a token. The Docker build compiles from source anyway, so that what ships in the
image is something you built rather than a tarball you downloaded.

Every dev target runs in Docker — no host Python or Node required:

| Command | What it does |
|---|---|
| `make test` | full pytest suite in `python:3.11-slim` |
| `make test-unit` / `make test-integration` | one suite only |
| `make lint` / `make format` | `ruff check` / `ruff format` |
| `make typecheck` | `mypy app/` gated on `mypy-baseline.txt` — fails on new errors and on stale baseline entries |
| `make typecheck-baseline` | regenerate `mypy-baseline.txt` after fixing (or deliberately accepting) mypy errors |
| `make docker-build` | build the api image as `vllm-warden:dev` |
| `make generate-api-types` | regenerate frontend types from the FastAPI OpenAPI schema |

The image is built on a digest-pinned `vllm/vllm-openai` base (currently v0.26.0). GGUF
support moved out of vLLM core in the 0.25.x line, so the build installs and patches
`vllm-gguf-plugin`. The `Dockerfile` documents each patch and why it exists — worth reading
before bumping the base image.

## Contributing

Issues and pull requests are welcome. Please run `make lint`, `make typecheck` and `make test`
before opening a PR; all three run in containers, so a working Docker install is the only
prerequisite. `make typecheck` compares `mypy --strict` against the committed
`mypy-baseline.txt`, so it is green on a clean checkout and red only for errors you introduced
(or baseline entries you fixed — run `make typecheck-baseline` and commit the shrunken file).

## License

[Apache License 2.0](LICENSE).

## Trademarks

vLLM is a project of the [vLLM team](https://github.com/vllm-project/vllm). PodWarden is a
trademark of its operators. LLM Warden is not affiliated with or endorsed by either project.
