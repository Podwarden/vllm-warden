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
| `GPU_TOOLKIT_INSTALL=yes\|no` | install the NVIDIA Container Toolkit without asking / never |

Exit status: `0` installed and startable; `1` a preflight or argument problem; `2` files
written but the stack cannot start yet — the message says what is missing (typically the
NVIDIA runtime).

### Without a clone

```bash
curl -fsSL https://raw.githubusercontent.com/Podwarden/vllm-warden/main/install.sh | sh -s -- --dir /opt/vllm-warden
```

Downloads the source tree into `--dir`, then proceeds exactly as above. Prompts still work
when piped (the script reads them from the terminal, not stdin); add `--yes` for automation.

The one-liner on the [PodWarden Hub catalog page](https://podwarden.com/catalog/vllm-warden)
also still works. It is a convenience wrapper around the same stack; this repository is the
reference and needs nothing from podwarden.com.

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

The operator files `./install.sh` writes double as the dev loop. `docker-compose.yml`
carries `build:` and the generated override carries `image:`, so `docker compose build`
tags what it builds with the release image name: `make restart` then runs your build, and
`make pull` puts the published image back.

```bash
git clone https://github.com/Podwarden/vllm-warden.git
cd vllm-warden
./install.sh --gpus all          # .env + override, pulls the published images
docker compose build api         # the engine image: CUDA base + llama.cpp compile
make restart
make smoke      # asserts 200s across / /_landing /ui/ /api/csrf /healthz
```

The UI image installs `@podwarden/chat-ui` from a private npm registry, so
`docker compose build ui` needs a read token passed as a BuildKit secret
(`--secret id=npm,env=NPM_TOKEN`; see `frontend/Dockerfile`). Without one, keep running the
published `vllm-warden-ui` image — the release pipeline builds it from this same tree.

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
