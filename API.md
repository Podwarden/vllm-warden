# Driving LLM Warden from the API

The browser UI is a client of the control API and nothing more. Everything it
does — the first-run wizard, minting a key, registering, pulling and loading a
model — is a handful of `curl` calls, and this document is those calls, for a
script, a CI job or an agent that will never open `/ui/`.

[INSTALL.md](INSTALL.md) walks the same first run and the same model add as a
recorded transcript, with the failures it hit on the way and what they meant.

## First run without a browser

<!-- The `shared:` marker pairs in this file fence generated regions. Their text
     is shared with the PodWarden Hub catalogue listing for this app, and
     `scripts/sync-shared-docs.py` rewrites them -- a hand edit inside a pair is
     reverted on the next sync and fails CI in the meantime. Everything outside
     those markers is hand-written: edit it freely. -->

<!-- shared:headless-first-run -->
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
<!-- /shared:headless-first-run -->

---

## Adding a model from the API

**Models → Add model** in the UI drives exactly these calls. Every one is
JWT-gated and CSRF-checked; `$JWT` and `$CSRF` are the two values produced by
the login sequence in *First run without a browser*, and `AUTH` below is just
those two headers.

Register, pull, load are **three separate, asynchronous steps**. Each returns
`202` immediately and reports progress somewhere else — the row's `status`
walks `registered → pulling → pulled → loading → loaded`, and you poll
`GET /api/models/{id}` for it.

```bash
# -b jar is REQUIRED, not optional: the CSRF token is bound to the cookie the
# jar holds, and the header alone gets you 403 {"detail":"csrf token invalid"}.
AUTH=(-b jar -H "Authorization: Bearer $JWT" -H "X-CSRF-Token: $CSRF"
      -H 'Content-Type: application/json')

# 1. Register. `gpu_indices` is required and must be a subset of what you
#    allowed in the wizard. `backend` defaults to "vllm"; the other value is
#    "llamacpp", which wants a GGUF `filename`.
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

Loading is not instant: the weights go to the GPU, CUDA graphs are captured and
a warmup request is served before the model is reported `loaded`. Tens of
seconds is normal for a small model, minutes for a large one.

`POST /api/models/fit-preview` answers "will this fit?" before you pull
anything, returning a `green`/`yellow`/`orange`/`red` verdict with the
arithmetic behind it.

**On a llama.cpp row**, four fields have no vLLM equivalent and are worth
knowing:

- `filename` — point it at **shard one** of a split set
  (`…-00001-of-000NN.gguf`) and llama.cpp finds the rest itself.
- `mmproj_filename` — the vision projector, a separate GGUF beside the weights.
  A vision model started without it loads, serves, and silently ignores every
  image, so a set-but-missing projector is a hard error before any process
  starts.
- `tokenizer_repo` — the safetensors sibling, for exact token accounting.
- `n_gpu_layers` — leave it NULL and llama.cpp sizes itself to the card. An
  explicit integer is a deliberate partial CPU offload: it works, it is
  llama.cpp's real differentiator, and it is a performance cliff. It is never
  chosen for you.

A load refused because the card is already serving another model is refused
**asynchronously** — the `POST` still answers `202`. Why, and what to do about
it, is under [One loaded model per GPU](HAZARDS.md#one-loaded-model-per-gpu).
`POST /api/models/{id}/stress`, which measures the context length a loaded
model can actually serve rather than the one it starts with, is described under
[Measuring instead of guessing](HAZARDS.md#measuring-instead-of-guessing).
