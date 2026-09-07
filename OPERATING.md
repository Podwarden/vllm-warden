# Operating LLM Warden day to day

Everything here assumes an install that already passes `make smoke`.
Getting there is [INSTALL.md](INSTALL.md); the things to read before the
first model is loaded are [HAZARDS.md](HAZARDS.md).

## Day-to-day

| Command | What it does |
|---|---|
| `make start` | start the stack (detached) |
| `make stop` | stop and remove the containers; volumes and data stay |
| `make restart` | stop + start, re-reading `.env` and the override |
| `make logs` | follow live logs (`make logs S=api` for one service) |
| `make status` | container state and health |
| `make smoke` | check the front door end to end (`/`, `/_landing`, `/ui/`, `/api/csrf`, `/healthz`) |
| `make pull` | pull the release named by `VERSION` in `.env` and restart on it |
| `make config` | print the fully merged compose config — what actually runs |
| `make preflight` | re-run the installer's host checks |
| `make save-images` / `make load-images` | move a release between hosts as a tarball |
| `make export-hf-cache` / `make import-hf-cache` | move the model cache the same way |
| `make uninstall` | stop and delete the data volumes (asks first) |
| `make help` | list every target |

**Upgrading:** set `VERSION` in `.env` to the new release (or
`./install.sh --version vX`) and `make pull`. When the stack files themselves
changed, `git pull && ./install.sh` refreshes `docker-compose.yml` and the
override and re-pins `VERSION`; `.env` is kept.

Once running:

- **UI** — `http://YOUR-HOST:8080/ui/`
- **OpenAI API** — `http://YOUR-HOST:8080/v1/chat/completions`
- **Control API** — `http://YOUR-HOST:8080/api/` (JWT-gated)
- **Health** — `http://YOUR-HOST:8080/healthz`

For gated models (Llama, Mistral, gpt-oss) you need a HuggingFace token. The
first-run wizard asks for one, and you can change it later under
**Settings → General**.

**On HTTP vs HTTPS.** Plain `http://` works — the session cookies follow the
scheme the browser actually used, so a LAN install stays signed in. It is still
an evaluation posture: the API key and the session both cross the network in the
clear. For anything shared, terminate TLS in front of the `:8080` listener and
tell the warden its public URL with
`./install.sh --origin https://llm.example.com`; the cookies then carry
`Secure`. If your proxy sets `X-Forwarded-Proto`, set `VW_TRUST_PROXY_ORIGIN=1`
in `.env` so the warden believes it.

---

## Where request content can end up

By default no prompt or completion text is stored anywhere. `request_history`
— the table behind the requests chart and the per-key rollups — has no column
that holds it; it carries identifiers, token counts, timings and finish
reasons, and that is the whole of the metadata path. Two diagnostic features
can capture content. Both are off by default, and with both off the proxy's
forward path is the code it would be in a build that never had them.

**God Mode** (`VW_GODMODE_ENABLED`) streams prompts, completions and any
inline images to one privileged viewer — the way to settle "the model said X"
when the client is somebody else's code. What it keeps lives in a bounded
in-memory ring: 2000 events and ~4 million characters by default
(`VW_GODMODE_RING_EVENTS`, `VW_GODMODE_RING_CHARS`), oldest evicted first,
with inline images in a separate bounded store beside it. Nothing reaches the
database or the disk, and all of it is gone when the container restarts. A
prompt longer than `VW_GODMODE_MAX_PROMPT_CHARS` (16000) is captured as its
head plus a `VW_GODMODE_PROMPT_TAIL_CHARS` tail, so the newest turn survives a
large repeated system prompt — display-only, and never a change to what is
forwarded. The bearer token never reaches the viewer; only the key's name
does. While it is on, that text sits in the api container's memory and any
holder of the admin session can read it: there is no separate role that
cannot, because the project has no roles.

**The content log** (`VW_CONTENT_LOG_ENABLED`) is the one that writes to disk.
Each logged request appends a JSON line — prompt, completion, token counts,
finish reason — to `VW_CONTENT_LOG_PATH`, which defaults to
`/data/logs/content.jsonl`. That is the persistent data volume, the same one
that holds the SQLite database, so the file is inside whatever backs that
volume up. It is scoped as well as gated: a request is logged only when the
flag is on **and** its token id appears in `VW_CONTENT_LOG_TOKENS`, an
explicit comma-separated allowlist that is empty by default. There is no "log
everything" mode. `VW_CONTENT_LOG_MAX_CHARS` (40000) caps the prompt and the
completion within each record.

Three properties of that file to know before enabling it:

- **It has a size ceiling, not rotation.** `VW_CONTENT_LOG_MAX_CHARS` bounds
  each record, not the file; `VW_CONTENT_LOG_MAX_BYTES` (512 MiB) bounds the
  file. On reaching it the writer stops and logs one warning naming the file
  and the limit — deliberately, so the incidents the log was armed to capture
  are not deleted to make room for newer ones. Nothing prunes, truncates or
  rotates it; archive or remove the file, or turn the flag off, to resume.
  `0` or a negative value switches the ceiling off, the same convention
  `VW_ENGINE_LOG_MAX_BYTES` uses.
- **It shares a volume with the database.** The ceiling exists because
  letting it grow until that volume is full is an outage, not merely a large
  file.
- **It is created owner-only.** The `logs` directory is created `0700` and
  the file `0600`, both applied at creation rather than left to the process
  umask. An existing file keeps whatever mode it has — the product never
  `chmod`s it — so a content log written by a build before v2026.09.07.1 keeps
  its old, umask-derived mode until you `chmod 600` it once.

Retention past the ceiling, and shipping the file anywhere, are yours to
arrange.

`VW_RUNAWAY_MODE=log` has no sink of its own. It attaches its trip signal to a
record the content log was already going to write, so it captures nothing
unless content logging is enabled *and* the request's token is on that
allowlist.

`VW_GODMODE_ENABLED` and all five content-log variables are in `.env.example`,
each with what its default does; set them in `.env`, or in the environment of
whatever runs the api container.
