# Operating vllm-warden during the MR-1 → MR-2 gap

After MR-1 (`feat/jwt-and-jinja-cleanup`) ships, vllm-warden has no UI until MR-2 lands. Use curl for everything until the new Next.js UI is deployed.

All examples assume `jq` is on `PATH`. Replace `ORIGIN` with the externally-served origin of your warden, and the placeholder passwords / IDs as instructed.

## Prerequisites

- The container is up and the first-run setup wizard at `/api/setup/*` has completed (a user exists in the `users` table). Setup endpoints don't require auth and are out of scope for this runbook.
- `VW_COOKIE_SECRET` is set (>=32 chars) in the container env.
- `VW_FRONTEND_ORIGIN` lists the origin(s) allowed on `/api/auth/*` calls. As of the origin-allow-list change it accepts a **comma-separated allow-list** of exact origins (e.g. `https://vllm.example.com,https://alt.example.com`); whitespace is trimmed and trailing slashes are stripped. Default `http://localhost:8080` in the unified-port topology shipped in #155 (was `:3000` before). If your `Origin` header matches none of the listed origins, `/api/auth/refresh` and `/api/auth/logout` reject with `403 origin mismatch`. Behind a trusted reverse proxy you can instead set `VW_TRUST_PROXY_ORIGIN=1` (see below) to accept the proxy-derived origin without pinning the exact public URL.
- `VW_JWT_SECRET` is optional. If unset, the secret is auto-generated and persisted at `<VW_DATA_DIR>/jwt_secret` (default data dir is `/data`).

## Shared variables

```bash
set -euo pipefail
ORIGIN=https://vllm.protrener.com
COOKIES=/tmp/vw-cookies.txt
rm -f "$COOKIES"
```

## CSRF token (required for `/api/tokens`, `/api/models`, `/api/settings`)

The CSRF middleware bypasses `/api/auth/*` and `/v1/*` but enforces a valid `X-CSRF-Token` on every other state-changing request. Mint one once per cookie jar — the server also sets a `vw_csrf_id` cookie that binds it.

```bash
CSRF=$(curl -fsS -b "$COOKIES" -c "$COOKIES" "$ORIGIN/api/csrf" | jq -r .csrf)
```

Re-run the above if you ever rotate the cookie jar.

## Login

Login is CSRF-exempt and origin-unchecked — only `/api/auth/refresh` and `/api/auth/logout` require the `Origin` header to match one of the origins in `VW_FRONTEND_ORIGIN` (or, when `VW_TRUST_PROXY_ORIGIN=1`, the proxy-derived `X-Forwarded-Proto://X-Forwarded-Host`). Save the refresh cookie into `$COOKIES`.

```bash
ACCESS=$(curl -fsS -c "$COOKIES" \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"YOUR-PASSWORD"}' \
  "$ORIGIN/api/auth/login" | jq -r .access_token)
```

The refresh cookie (`vw_refresh`) is scoped to `path=/api/auth`; `$COOKIES` will only send it back on `/api/auth/*` URLs.

## Refresh access token

```bash
ACCESS=$(curl -fsS -b "$COOKIES" -c "$COOKIES" \
  -H "Origin: $ORIGIN" -X POST \
  "$ORIGIN/api/auth/refresh" | jq -r .access_token)
```

## Mint an API token

`/api/tokens` is CSRF-gated. Token plaintext is returned in the `plaintext` field — **save it now**, it is never shown again.

```bash
TOKEN=$(curl -fsS -b "$COOKIES" -c "$COOKIES" \
  -H "Authorization: Bearer $ACCESS" \
  -H "X-CSRF-Token: $CSRF" \
  -H "Content-Type: application/json" \
  -d '{"name":"ci-bot","expires_in_days":90}' \
  -X POST "$ORIGIN/api/tokens" | jq -r .plaintext)
```

### Optional per-token rate limit and priority (S5, #104)

`POST /api/tokens` (and `PATCH /api/tokens/{id}`) accept two extra fields:

- `rate_limit_tps` (integer, 1..1_000_000, or `null`) — sliding **10-second** window measured in tokens per second. `null` means unlimited (default). Over-budget requests return **HTTP 429** (matching OpenAI client retry semantics — clients with built-in exponential backoff will recover on their own). The window length is operator-tunable via `VW_RATE_LIMIT_WINDOW_S`.
- `priority` (integer, 0..9, default 5) — STRICT scheduler priority. The proxy maintains one global queue ordered by priority; **the highest priority is always served first** regardless of arrival order. FIFO is preserved only within a single priority tier.

> **Starvation is by design.** Under sustained pressure from priority-9 traffic, a priority-0 token can wait **indefinitely** for service. This is the explicit semantics of STRICT scheduling — choose it when you want a hard "VIP first" guarantee and you're willing to accept that low-priority work may never run during peak load. If you need fair sharing instead, set every token to the same priority and the queue degrades to FIFO inside a single tier.
>
> **Concrete example:** if 90% of your traffic flows through priority-9 tokens (e.g. a production inference API), any priority-0 tokens (e.g. a nightly batch job) will never get a slot during peak hours. To avoid this, either limit the number of priority-9 tokens or give batch work a mid-range priority (e.g. 3–4) and accept that it completes more slowly rather than not at all. Monitor the `Last 24h` column on the `/tokens` page — a zero count for a low-priority token during a busy period is the first sign of starvation.
>
> The scheduler is implemented in `app/proxy/scheduler.py` (`PriorityScheduler`). The queue is a single global `asyncio.PriorityQueue` ordered by `(-priority, arrival_sequence)` — highest priority wins, FIFO within a tier. There is no time-slicing, preemption, or starvation recovery mechanism.

```bash
# Token rate-limited to 50 tokens/sec with elevated priority 7.
curl -fsS -b "$COOKIES" -c "$COOKIES" \
  -H "Authorization: Bearer $ACCESS" \
  -H "X-CSRF-Token: $CSRF" \
  -H "Content-Type: application/json" \
  -d '{"name":"batch-worker","rate_limit_tps":50,"priority":7}' \
  -X POST "$ORIGIN/api/tokens"

# Patch an existing token — clear its rate limit (back to unlimited) and
# drop its priority to 2.
curl -fsS -b "$COOKIES" -c "$COOKIES" \
  -H "Authorization: Bearer $ACCESS" \
  -H "X-CSRF-Token: $CSRF" \
  -H "Content-Type: application/json" \
  -d '{"rate_limit_tps":null,"priority":2}' \
  -X PATCH "$ORIGIN/api/tokens/TOKEN_ID"
```

### Inspect per-token usage

`GET /api/tokens/{id}/usage?range=24h` returns minute-bucketed usage (request count + prompt/completion tokens). Supported ranges: `1h`, `24h`, `7d`. The 24h totals also appear in the `Last 24h` column of the `/tokens` UI page.

```bash
curl -fsS -b "$COOKIES" -c "$COOKIES" \
  -H "Authorization: Bearer $ACCESS" \
  "$ORIGIN/api/tokens/TOKEN_ID/usage?range=24h" | jq .totals
```

## Register, pull, and load a model

Routes after creation use the server-generated **model id**, not `served_model_name`. Capture `MODEL_ID` from the create response.

```bash
MODEL_ID=$(curl -fsS -b "$COOKIES" -c "$COOKIES" \
  -H "Authorization: Bearer $ACCESS" \
  -H "X-CSRF-Token: $CSRF" \
  -H "Content-Type: application/json" \
  -d '{"served_model_name":"opt-125m","hf_repo":"facebook/opt-125m","gpu_indices":[0]}' \
  -X POST "$ORIGIN/api/models" | jq -r .id)

curl -fsS -b "$COOKIES" -c "$COOKIES" \
  -H "Authorization: Bearer $ACCESS" \
  -H "X-CSRF-Token: $CSRF" \
  -X POST "$ORIGIN/api/models/$MODEL_ID/pull"

# Poll status until "pulled" (or watch the SSE stream at
# GET /api/models/$MODEL_ID/pull/progress with the same Bearer header).
curl -fsS -H "Authorization: Bearer $ACCESS" "$ORIGIN/api/models/$MODEL_ID" | jq .status

curl -fsS -b "$COOKIES" -c "$COOKIES" \
  -H "Authorization: Bearer $ACCESS" \
  -H "X-CSRF-Token: $CSRF" \
  -X POST "$ORIGIN/api/models/$MODEL_ID/load"
```

Wait until `GET /api/models/$MODEL_ID` reports `"status":"loaded"` before sending traffic.

### GGUF models — config/tokenizer in a sibling HF repo

Quantized GGUF repacks (e.g. unsloth republishes) often omit `config.json` and the
tokenizer files; vLLM requires them to boot. Pass the two optional Advanced fields at
registration time:

| Field | vLLM flag emitted | Example |
|---|---|---|
| `hf_config_repo` | `--hf-config-path <value>` | `"unsloth/Llama-3.2-1B-Instruct-GGUF"` |
| `tokenizer_repo` | `--tokenizer <value>` | `"meta-llama/Llama-3.2-1B-Instruct"` |

Both fields accept the same `owner/name` format as `hf_repo`. Leave them null (the
default) for safetensors or any GGUF repo that ships its own `config.json`.

```bash
MODEL_ID=$(curl -fsS -b "$COOKIES" -c "$COOKIES" \
  -H "Authorization: Bearer $ACCESS" \
  -H "X-CSRF-Token: $CSRF" \
  -H "Content-Type: application/json" \
  -d '{"served_model_name":"llama3-q4","hf_repo":"unsloth/Llama-3.2-1B-Instruct-GGUF",
       "hf_config_repo":"unsloth/Llama-3.2-1B-Instruct-GGUF",
       "tokenizer_repo":"meta-llama/Llama-3.2-1B-Instruct",
       "filename":"Llama-3.2-1B-Instruct-Q4_K_M.gguf","gpu_indices":[0]}' \
  -X POST "$ORIGIN/api/models" | jq -r .id)
```

## Run a completion

The OpenAI-compat proxy at `/v1/*` uses Bearer **API tokens** (the `$TOKEN` from the mint step), not the JWT `$ACCESS`. `model` is the `served_model_name`, not the id.

```bash
curl -fsS -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"opt-125m","prompt":"Hello","max_tokens":16}' \
  "$ORIGIN/v1/completions"
```

`/v1/chat/completions` and `/v1/models` work identically.

## HF cache management

Three endpoints under `/api/cache/*` let an operator inspect and reclaim disk under the configured HuggingFace cache directory (`VW_HF_CACHE_DIR`). All three are JWT-gated; the destructive ones (`DELETE`, `POST /gc`) also require `X-CSRF-Token`. Same `$ACCESS`/`$CSRF` from the sections above.

### List cached repos

`GET /api/cache/models` returns every `models--<org>--<name>` directory the scanner finds under the cache root, with its on-disk size, last-modified timestamp, and the `models` table rows that own it (empty list ⇒ orphan).

```bash
curl -fsS -H "Authorization: Bearer $ACCESS" \
  "$ORIGIN/api/cache/models" | jq '.[] | {repo, size_bytes, matched_models}'
```

### Delete one cached repo

`DELETE /api/cache/models/{repo}` drops a single `<org>/<name>` cache directory. The path captures the slash (`{repo:path}`), so a literal `/` between org and name is fine.

Safety ladder:

| Matching row status | Response | `?force=true` override |
|---|---|---|
| Any of `loaded` / `loading` / `unloading` / `pulling` | **409 active** | NO — never overridable; deleting under a running vLLM crashes the worker |
| Any of `pulled` / `idle` (not active) | **409 force-required** | YES — pass `?force=true` |
| Only `failed`, or no matching rows (orphan) | 204 No Content | n/a |
| No cache dir on disk | 404 Not Found | n/a (idempotent — second delete is a no-op) |

```bash
# Plain attempt; the server will 409 + a "pass force=true" detail if the
# repo backs a benign-but-alive (pulled/idle) row.
curl -fsS -b "$COOKIES" -c "$COOKIES" \
  -H "Authorization: Bearer $ACCESS" \
  -H "X-CSRF-Token: $CSRF" \
  -X DELETE "$ORIGIN/api/cache/models/Qwen/Qwen3-9B"

# Forced delete — only escapes the pulled/idle 409, never the active 409.
curl -fsS -b "$COOKIES" -c "$COOKIES" \
  -H "Authorization: Bearer $ACCESS" \
  -H "X-CSRF-Token: $CSRF" \
  -X DELETE "$ORIGIN/api/cache/models/Qwen/Qwen3-9B?force=true"
```

Invalid repo ids (multi-slash, `..` traversal, whitespace, non-alnum-`_-.`) reject with **400** before any DB or filesystem touch.

### Garbage-collect orphans + stale failures

`POST /api/cache/models/gc` sweeps everything that's clearly disposable in one call:

- **orphan** — the cache dir exists but no `models` row references its `hf_repo`;
- **failed_stale** — EVERY matching `models` row is `status=failed` AND `updated_at` is older than `failed_older_than_hours` (default 24).

Repos with any `pulled`/`idle`/active row are excluded. `dry_run=true` (the default) returns the candidate list with no side effects — always preview before the destructive call.

```bash
# Preview: see what GC would free, no deletes.
curl -fsS -b "$COOKIES" -c "$COOKIES" \
  -H "Authorization: Bearer $ACCESS" \
  -H "X-CSRF-Token: $CSRF" \
  -X POST "$ORIGIN/api/cache/models/gc?dry_run=true" \
  | jq '{total_bytes_freed, candidates: [.candidates[] | {repo, reason, size_bytes}]}'

# Real run: confirm the preview first, then drop dry_run.
curl -fsS -b "$COOKIES" -c "$COOKIES" \
  -H "Authorization: Bearer $ACCESS" \
  -H "X-CSRF-Token: $CSRF" \
  -X POST "$ORIGIN/api/cache/models/gc?dry_run=false" \
  | jq '{total_bytes_freed, deleted_paths}'

# Aggressive: include failures younger than 24h (operator override; useful
# right after a known-bad pull when you don't want to wait out the window).
curl -fsS -b "$COOKIES" -c "$COOKIES" \
  -H "Authorization: Bearer $ACCESS" \
  -H "X-CSRF-Token: $CSRF" \
  -X POST "$ORIGIN/api/cache/models/gc?dry_run=false&failed_older_than_hours=0"
```

The endpoint never deletes the underlying `models` row — only the on-disk cache. Re-pulling a deleted row from its `hf_repo` will recreate the cache dir.

## Rotate an API token

Rotation renames the OLD row to `"{name} (old N)"` (N = next free slot, starts at 1; cascades to `(old 2)`, `(old 3)` … on subsequent rotations) and mints a brand-new row that keeps the ORIGINAL name. The old row keeps working for `grace_hours` so callers can swap without downtime.

The response includes the new plaintext token — capture it now, the API will never return it again — plus `name` (the active token's name, unchanged) and `renamed_to` (the predecessor's new `"{name} (old N)"` name so you can spot it in the list).

Rotating an already-rotated row returns **409 Conflict**; rotate the successor instead.

```bash
ROTATE_JSON=$(curl -fsS -b "$COOKIES" -c "$COOKIES" \
  -H "Authorization: Bearer $ACCESS" \
  -H "X-CSRF-Token: $CSRF" \
  -H "Content-Type: application/json" \
  -d '{"grace_hours":24}' \
  -X POST "$ORIGIN/api/tokens/OLD_TOKEN_ID/rotate")
echo "New token:        $(echo "$ROTATE_JSON" | jq -r .plaintext)"
echo "Active name:      $(echo "$ROTATE_JSON" | jq -r .name)"
echo "Old row renamed:  $(echo "$ROTATE_JSON" | jq -r .renamed_to)"
```

## Revoke an API token

```bash
curl -fsS -b "$COOKIES" -c "$COOKIES" \
  -H "Authorization: Bearer $ACCESS" \
  -H "X-CSRF-Token: $CSRF" \
  -X DELETE "$ORIGIN/api/tokens/TOKEN_ID"
```

## JWT secret rotation

Invalidates every outstanding access token and refresh cookie instantly. Operators will all have to log in again.

1. Stop the warden container.
2. Either: delete `<VW_DATA_DIR>/jwt_secret` (default `/data/jwt_secret`) on the persistent volume, **or** set a new `VW_JWT_SECRET` env var on the container (env override wins over the on-disk file).
3. Start the warden container.

## Admin password reset (lockout recovery)

If you still have a valid JWT, `PATCH /api/settings/runtime` with `{"admin_password":"new-pw"}` rotates it in-band (Bearer + CSRF required, same as any other mutating call). If you are fully locked out, there is no CLI yet — patch the row directly. Run from the host that has the warden's data volume mounted:

Find the volume name with `docker volume ls | grep warden` (or `docker inspect <container> --format '{{ range .Mounts }}{{ .Name }}{{"\n"}}{{ end }}'` if the container is still up). For bind mounts, point `-v` at the host path holding `vllm-warden.db`.

```bash
docker run --rm -it \
  -v vllm-warden-data:/data \
  python:3.11-slim bash -c '
    pip install -q bcrypt >/dev/null
    python - <<EOF
import bcrypt, sqlite3, getpass
pw = getpass.getpass("new admin password: ").encode()
h  = bcrypt.hashpw(pw, bcrypt.gensalt()).decode()
con = sqlite3.connect("/data/vllm-warden.db")
con.execute("UPDATE users SET password_hash = ? WHERE id = (SELECT MIN(id) FROM users)", (h,))  # single-admin model: MIN(id) is the bootstrap admin
con.commit(); con.close()
print("ok")
EOF'
```

Adjust the `-v <volume>:/data` mount to match your deployment (compose volume name, bind mount, etc.). A dedicated CLI is tracked as a follow-up; until then this is the supported escape hatch.

## Container GPU isolation (`VW_CONTAINER_GPU_COUNT`) — #46

When vllm-warden runs in a container, it sees only the GPUs the runtime
exposes to it, not the full host inventory. The setup wizard's GPU table,
the `allowed_gpu_indices` validator on `POST /api/models`, and the live
nvidia-smi probe at `GET /api/system/gpus` all index from `0` against
the container's slice — NOT the host.

`VW_CONTAINER_GPU_COUNT` is the operator's contract with the container:
it tells warden "you have exactly this many GPUs, indexed `0..N-1`".
Setting it correctly is what keeps the wizard from rejecting otherwise
valid `gpu_indices=[3]` selections, and what keeps the
`POST /api/models/fit-preview` endpoint from reporting `gpu_index_missing`
on a GPU the container CAN see but the probe missed momentarily.

**Setting it right depends on how you split GPUs across containers:**

| Deployment shape | Set `VW_CONTAINER_GPU_COUNT` to | Set `NVIDIA_VISIBLE_DEVICES` to |
|---|---|---|
| Whole host, all 4 GPUs to one warden | `4` | `all` (or `0,1,2,3`) |
| Two wardens, GPUs `0,1` vs `2,3` | `2` on each | `0,1` and `2,3` respectively |
| Single warden, single GPU (laptop dev) | `1` | `0` |

Worked example for a 2-GPU container (docker compose):

```yaml
services:
  warden:
    image: registry.podwarden.com/vllm-warden:sha-...
    environment:
      VW_CONTAINER_GPU_COUNT: "2"
      NVIDIA_VISIBLE_DEVICES: "0,1"   # host indices
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0", "1"]
              capabilities: [gpu]
```

Inside that container, warden sees the GPUs as indices `0` and `1`
(NOT `0` and `1` from the host's POV — those are the same numbers
only because we asked for the first two; if you'd asked for `device_ids: ["2","3"]`
they'd still appear as `0` and `1` inside).

**Common misconfigurations:**

- **`VW_CONTAINER_GPU_COUNT` higher than actually exposed.** The setup
  wizard will let you tick GPU index `3` even though `NVIDIA_VISIBLE_DEVICES=0,1`
  exposes only two. The first `POST /api/models` with `gpu_indices=[3]`
  is registered fine, but `pull`/`load` fails downstream when vLLM tries
  to claim a device the container can't see. Symptom: `CUDA error: invalid device ordinal`
  in the model log.
- **`VW_CONTAINER_GPU_COUNT` lower than exposed.** The wizard hides
  the extra GPUs and operators can't select them. No model can be created
  on them until the env var is fixed and the warden restarts.
- **Mismatched with `NVIDIA_VISIBLE_DEVICES`.** Warden never reads
  `NVIDIA_VISIBLE_DEVICES` — only the container runtime does. The two are
  decoupled; `VW_CONTAINER_GPU_COUNT` is warden's stated belief about
  the slice, and you (the operator) are responsible for keeping it in
  sync with the runtime's view. Note: warden internally sets
  `CUDA_VISIBLE_DEVICES` on each vLLM subprocess from the model's
  `gpu_indices` DB column — that variable is hardcoded and cannot be
  overridden via container env.

Restart the warden container after changing the variable; it's read at
process startup, not on each request.

## Stats sampler cadence (`VW_STATS_SAMPLER_INTERVAL_S`) — #124

How often the runtime stats sampler ticks GPU/model/power samples into the local DB; defaults to `5.0` (seconds). Values `<= 0` or unparseable fall back to the default. Read at process startup — restart warden after changing.

## System Configuration panel on `/stats` — #148

The bottom of `/stats` shows a static-ish "System Configuration" panel that helps interpret the live metrics above it: CPU model + physical/thread counts, total RAM, OS release + kernel, Docker server version + default runtime, and one card per GPU (name, VRAM, driver version, CUDA version).

Data source — `GET /api/system/info` (JWT-gated). The endpoint composes its payload from `/proc/cpuinfo`, `/proc/meminfo`, `/etc/os-release`, `uname -r`, `nvidia-smi --query-gpu=...`, `nvidia-smi --version`, and `docker info --format '{{json .}}'`. Each source degrades independently:

- `/proc/cpuinfo` or `/proc/meminfo` unreadable → that field is `null`; the UI renders an em-dash.
- `nvidia-smi` not on `PATH` (dev workstations, sandboxed builds) → `gpus: []`; the UI renders a "No NVIDIA GPUs detected." empty state instead of erroring the page.
- `docker info` unavailable (the common case in production — the API container has no docker socket mounted) → `docker.available: false` and the card shows a "Docker not available" placeholder.

The endpoint caches its full payload for 60 seconds in-process. The `/stats` page polls every 30 seconds, so a typical request path is a single dict lookup — `nvidia-smi` and `docker info` shell-outs happen at most once per minute, regardless of how many tabs are open.

## Tunable env vars

Every runtime knob the warden reads at process startup. All are optional unless marked; restart the container after changing.

| Variable | Default | Required | Purpose |
|----------|---------|----------|---------|
| `VW_DATA_DIR` | `/data` | no | Where the SQLite DB, JWT secret, HF token, and log files live. |
| `VW_HF_CACHE_DIR` | `/root/.cache/huggingface` | no | HF weights cache. Production typically mounts a dedicated PVC here so the data PVC can stay small. |
| `VW_COOKIE_SECRET` | — | **yes** | Cookie-signing secret. Must be at least 32 characters; warden refuses to start otherwise. |
| `VW_CONTAINER_GPU_COUNT` | `0` | no | Warden's stated belief about how many GPUs are exposed by the container runtime. See §"Container GPU isolation" above for the operator contract. |
| `VW_FRONTEND_ORIGIN` | `http://localhost:8080` | no | Comma-separated **allow-list** of exact origins accepted by the `Origin` header check on `/api/auth/refresh` and `/api/auth/logout` (whitespace trimmed, trailing slashes stripped). Default was `:3000` before #155 — the unified-port Caddy front-door publishes `:8080` now. Set to the externally-served origin(s) in production. An explicitly-empty value falls back to the localhost default by design (so a blank env var doesn't lock admins out); to fail closed, set a real allow-list and keep `VW_TRUST_PROXY_ORIGIN` off. |
| `VW_TRUST_PROXY_ORIGIN` | `0` (off) | no | When truthy (`1`/`true`/`yes`/`on`), the origin check also accepts a request whose `Origin` equals the proxy-derived origin `X-Forwarded-Proto://X-Forwarded-Host` (first value of each, for proxy chains; `Host` is used if `X-Forwarded-Host` is absent). Only enable this **behind a trusted reverse proxy that sets those headers** — it lets you avoid pinning the exact public URL in `VW_FRONTEND_ORIGIN`. Off by default; the allow-list is still checked first. |
| `VW_WARMUP_PROBE_TIMEOUT_S` | `60.0` | no | Max seconds the warmup verification probe waits for a successful `POST /v1/completions` before marking the load failed. Bump for multimodal models with slow processor warmup (Qwen3-VL etc.). |
| `VW_JWT_SECRET` | auto-generated | no | JWT signing secret. When unset, warden generates one and persists it at `<VW_DATA_DIR>/jwt_secret`; provide an explicit value only when rotating across nodes. |
| `VW_STATS_SAMPLER_INTERVAL_S` | `5.0` | no | Stats sampler tick cadence. See section above. |
| `VW_RATE_LIMIT_WINDOW_S` | `10.0` | no | Width (seconds) of the per-token sliding-window rate-limiter charge. The S5 token `rate_limit_tps` field is interpreted against this window. |
| `VW_HEADER_METRICS_INTERVAL_S` | `5.0` | no | How often the header-metrics SSE singleton pushes a fresh per-tab snapshot. Lower = snappier UI, more SSE traffic. |
| `VW_BUILD_VERSION` | `dev` | no | Set by CI to the git tag (or short SHA for branch builds); surfaced through `GET /api/version` for the nav footer. Operators rarely set this manually. |
| `VW_BUILD_SHA` | `unknown` | no | Set by CI to the short SHA; sibling of `VW_BUILD_VERSION`. |

## Release process

vllm-warden uses CalVer (`v{YYYY.MM.DD.N}`) — same scheme as PodWarden core and Hub. Tags are immutable; the `:staging` / `:production` Docker tags move with develop / main heads.

Release procedure (operator runs from a clean develop):

1. Open a develop → main MR. Confirm CI is green on develop tip.
2. Bump `app/__init__.__version__` and add the highlights paragraph + `## [v{YYYY.MM.DD.N}]` heading in `changelog.md` (rename the existing `[Unreleased]` block; insert a fresh empty `[Unreleased]` above it). PM approves the MR.
3. Merge with the GitLab squash button. The squash commit on main triggers the `build-image` + `build:ui` CI jobs that publish `sha-<commit>` + `production` + `latest` tags to `registry.podwarden.com`.
4. Push the CalVer tag: `git tag v<YYYY.MM.DD.N> <merge-commit-sha> && git push origin v<YYYY.MM.DD.N>`. The tag pipeline re-uses the cached image and adds the immutable `v...` tag. (Today's tag is `v2026.05.23.1`.)
5. Create the GitLab release page from the new tag, pasting the changelog section as the release notes body.
6. Open a chore/sync-main-into-develop MR to back-merge the squash commit; see `docs/releasing.md` for the rationale (squash commits are otherwise invisible to develop and break the next dev → main MR's rebase).

Deployment happens out-of-band — the site-agent or k8s ImagePullPolicy pulls the new `:production` tag; this codebase does not own the deploy step.

## Supported GGUF architectures

The Add Model wizard warns inline when a GGUF repo's `general.architecture` field (or the filename heuristic, when `config.json` is absent) is outside the vLLM-known allowlist. Most failures on unsupported arches are silent vLLM-side init crashes that are very hard to diagnose post-pull — the warning catches them at file-pick time.

Canonical allowlist (kept in sync with `app/models/discovery.py::KNOWN_GGUF_ARCHES`):

| Family | Architectures |
|--------|---------------|
| Llama family | `llama`, `llama2`, `llama3` |
| Mistral family | `mistral`, `mixtral` |
| Qwen family | `qwen`, `qwen2`, `qwen2_moe`, `qwen3`, `qwen3_moe` |
| Phi family | `phi3`, `phi3_5` |
| Gemma family | `gemma`, `gemma2`, `gemma3` |
| DeepSeek family | `deepseek`, `deepseek_v2`, `deepseek_v3` |
| Other | `yi`, `command_r`, `starcoder2` |

If your target arch isn't on the list, check the vLLM release notes for the deployed `vllm_version` (visible at `/settings`) and file an issue against this repo so we can add it.

## Built-in tuning presets

Four presets ship with the warden as the "Apply preset" chips strip on each model's `/settings` page. They write `gpu_memory_utilization`, `max_model_len`, `dtype`, `trust_remote_code`, and (for `dev-tiny`) `extra_args`; the supervisor flow (model reload) is the same as a manual settings edit. Source of truth is `app/presets/builtin.json`.

| Preset id | Target archetype | Notes |
|-----------|------------------|-------|
| `a4000-tight-awq` | 1× RTX A4000 (16 GB) | Conservative VRAM budget; AWQ-quantized 7B fits with room for batching. FP16 weights, GPU memory utilization 0.82, max_model_len 8192. |
| `h100-single-shot` | 1× H100 (80 GB) | Latency-sensitive single-conversation workloads. BF16, GPU memory utilization 0.92, max_model_len 32768. Bump `max_num_seqs` manually for batched serving. |
| `dev-tiny` | Any GPU ≥ 6 GB | Minimal settings for dev/test on Qwen-0.5B, opt-125m. GPU memory utilization 0.5 so other workloads coexist, max_model_len 4096, `--enforce-eager` so failures surface fast. |
| `moe-balanced` | 2× A4000 (16 GB) | Tensor-parallel split for Mixtral 8x7B. After applying, set `tensor_parallel_size=2` and list both GPUs in `gpu_indices` manually — the preset doesn't touch those columns. |

## Unified-port topology (#155)

The deployment publishes a single host port — `:8080` — fronted by a Caddy
container that fans out to two internal services on the compose network:

```
host :8080 ──> caddy ──┬─> ui  :3000    (Next.js, basePath '/ui')
                       └─> api :8080    (FastAPI)
```

Caddy's routing map (`deploy/caddy/Caddyfile`):

| Path        | Backend            | Notes                                  |
|-------------|--------------------|----------------------------------------|
| `/`         | api `/_landing`    | Public HTML landing page (opt-out)    |
| `/_landing` | api `/_landing`    | Same content, direct path             |
| `/ui/*`     | ui                 | Browser-facing UI                      |
| `/_next/*`  | ui                 | Build assets                           |
| `/api/*`    | api                | JWT-gated control plane                |
| `/v1/*`     | api                | OpenAI-compatible proxy (token-gated)  |
| `/healthz`  | ui (alias)         | Liveness probe; ui rewrites → /api/health |

A few operational consequences worth knowing:

- **No `:3000` exposure.** The ui container is `expose: ["3000"]` (compose
  network only); curl-from-host smoke checks must hit `:8080/ui/...`
  through Caddy.
- **No `:8080` direct-api exposure.** Same — the api container is
  `expose: ["8080"]` only.
- **CSRF / origin checks.** `VW_FRONTEND_ORIGIN` must include the
  externally served origin of `:8080` (e.g. `https://vllm.protrener.com`),
  because the browser is now same-origin with Caddy. It accepts a
  comma-separated allow-list, so multiple front-door origins can be listed.
  The default development value is `http://localhost:8080` (was `:3000`
  pre-#155). Alternatively, behind a trusted reverse proxy that sets
  `X-Forwarded-Proto`/`X-Forwarded-Host`, set `VW_TRUST_PROXY_ORIGIN=1` to
  accept the proxy-derived origin without pinning the exact public URL —
  this is what stops a page reload from logging the user out when the
  served origin isn't known ahead of time. The check fails closed: an
  origin that matches neither the allow-list nor (when enabled) the
  proxy-derived origin is rejected with `403 origin mismatch`.
- **SSE routes** (`/api/chat/*`, `/api/models/{id}/logs/stream`,
  `/api/stats/stream`) are fronted by Caddy with `flush_interval -1` so
  chunks flush immediately. If you front the warden with another reverse
  proxy (Traefik, nginx, Cloudflare), make sure that proxy ALSO has
  buffering disabled — otherwise logs and chat will appear "stuck".
- **Caddy admin API is OFF** (`admin off` in the Caddyfile). There is no
  port 2019 to lock down.

### Disabling the public landing page

For a private deployment that should 404 at the root, toggle the
`landing_page_enabled` runtime setting off. PATCH it from `/ui/settings`
or via curl:

```bash
curl -fsS -b "$COOKIES" \
  -H "Authorization: Bearer $ACCESS" \
  -H "X-CSRF-Token: $CSRF" \
  -H "Content-Type: application/json" \
  -X PATCH \
  -d '{"landing_page_enabled": false}' \
  "$ORIGIN/api/settings/runtime"
```

The change takes effect immediately (`requires_restart_kinds: []`). With
the toggle off, `GET /_landing` returns `404 landing page disabled` and
Caddy serves the same 404 at `/`. The `/ui` UI and `/api`/`/v1` paths
remain unaffected.
