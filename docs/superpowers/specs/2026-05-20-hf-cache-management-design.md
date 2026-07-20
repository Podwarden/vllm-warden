# HF cache management — API + UI for orphaned / failed / unused model cache cleanup

**Date:** 2026-05-20
**Issue:** vllm-warden#114
**Author:** CTO (brainstorm), implementation TBD (dev-1 unless redirected)

## Problem

vllm-warden writes downloaded HuggingFace model weights into a long-lived
on-disk cache rooted at `Settings.hf_cache_dir` (env `VW_HF_CACHE_DIR`,
defaults to `/root/.cache/huggingface`). Operators have no first-class way to
see what's on disk and no safe way to clean it up:

- **Orphaned cache**: a row in `models` was deleted, but its
  `models--<org>--<name>/` tree still consumes disk. Confirmed by backlog
  issue #105 ("orphan model files on delete") — `delete_model` does not
  touch the cache. On the bonus host this is already the dominant disk-fill
  vector during model experimentation.
- **Failed-pull cache**: a pull that died after writing some blobs leaves
  partial state under `blobs/` indefinitely. Row stays at `status=failed`,
  cache stays.
- **Cold-and-unused cache**: a model the operator hasn't loaded in weeks
  still pins ≥10 GiB.
- **No visibility**: operator can only learn what's cached by SSH-ing into
  the api container and running `du -sh /root/.cache/huggingface/*`.

The bonus host is currently the live example: 2× A4000 + the disk on the
api PVC has been filled multiple times during the v17.18–v17.20 GGUF
experiments, requiring DevOps to manually nuke caches (task #329) to
unblock the operator.

## Goals

1. Operator can see every `models--*` directory under the HF cache, its
   size, last-modified, and whether it maps to a live `models` row.
2. Operator can delete a single repo's cache, with safety: refuse if the
   matched row is currently in-use (`loaded` / `loading` / `pulling` /
   `unloading`); warn otherwise.
3. Operator can sweep all clearly-disposable cache in one shot — orphans
   (no row) and failed pulls (`status=failed` older than N hours) — with a
   mandatory dry-run preview before the real delete.

## Non-goals

- Auto-GC on a timer. Operator-triggered only this round; a scheduled
  policy lives in a separate spec if/when needed.
- Per-snapshot or per-blob accounting. Repo-level granularity is enough.
- Touching HF cache for other purposes (datasets, spaces) — vllm-warden's
  cache only stores model repos.
- Quota enforcement / disk-pressure throttling.
- Cleaning up the `models` row when its cache is deleted (the row may
  still be valid metadata for a re-pull). Symmetric "cascade row → cache"
  cleanup belongs in #105, not here.

## Design

### Bundle scope

Single MR. Three BE endpoints + one FE surface + tests. Sequenced after
v2026.05.20.1 (the #1038+#1025 bundle now live on podwarden.h).

### Backend — new module `app/cache/`

New top-level package mirroring `app/stats/`, `app/models/`, etc.

```
app/cache/
  __init__.py
  routes_api.py    # FastAPI router under /api/cache
  scanner.py       # pure-sync HF cache walker (testable in isolation)
```

Registered in `app/main.py` next to the existing routers:
```python
from app.cache import routes_api as cache_routes_api
app.include_router(cache_routes_api.router)
```

#### `scanner.py` — pure helper

```python
@dataclass(frozen=True)
class CachedRepo:
    repo: str            # "Qwen/Qwen3.6-27B-GGUF"   (decoded from dir name)
    path: Path           # absolute path to models--Qwen--Qwen3.6-27B-GGUF
    size_bytes: int      # recursive total: blobs/ + snapshots/ + refs/
    last_modified: float # max mtime under the dir

def scan_hf_cache(cache_root: Path) -> list[CachedRepo]:
    """Walk cache_root for models--<org>--<name> directories.

    cache_root is settings.hf_cache_dir. The HF library may store these
    either directly under cache_root OR under cache_root/hub/ depending on
    HF_HOME vs HUGGINGFACE_HUB_CACHE layout. Scanner probes both and
    returns the union, deduping by absolute path.

    Repo name is decoded by stripping the "models--" prefix and replacing
    "--" with "/" — same convention `pull_task._snapshot_dir_size` already
    uses in reverse.

    Tolerates: missing cache_root (returns []), unreadable entries
    (logged + skipped), entries that don't match the prefix (skipped).
    Never raises.
    """
```

Decoupling the walker from FastAPI keeps tests trivial (tmpdir + create
some dirs + assert). The route handlers call `await asyncio.to_thread(
scan_hf_cache, ...)` so a slow walk doesn't block the event loop.

#### Endpoint 1 — `GET /api/cache/models`

```python
@router.get("/models", response_model=list[CachedRepoView])
async def list_cached_models(
    request: Request,
    _user: str = Depends(require_jwt),
):
    settings = request.app.state.settings
    repos = await asyncio.to_thread(scan_hf_cache, settings.hf_cache_dir)
    async with open_db(settings.db_path) as db:
        rows = await ModelRepo(db).list()
    by_repo = _index_rows_by_repo(rows)  # dict[str, list[ModelRow]]
    return [_to_view(r, by_repo.get(r.repo, [])) for r in repos]
```

`CachedRepoView` (Pydantic):
```python
class CachedRepoView(BaseModel):
    repo: str
    path: str
    size_bytes: int
    last_modified: float
    matched_models: list[MatchedModelRef]  # zero or more rows that point at this repo

class MatchedModelRef(BaseModel):
    id: str
    served_model_name: str
    status: str  # idle | pulled | loaded | failed | ...
```

`matched_models` is a list (not a single optional) because the same
`hf_repo` can legitimately back multiple rows (different GPU mappings,
different `served_model_name`s). Important for the safety check below.

#### Endpoint 2 — `DELETE /api/cache/models/{repo:path}`

`{repo}` uses `:path` so `org/name` and `org/name-GGUF` arrive intact
(otherwise FastAPI would split on `/`). Query param: `?force=true` to
override the soft `idle`-status warning.

```python
@router.delete("/models/{repo:path}", status_code=204)
async def delete_cached_model(
    repo: str,
    force: bool = False,
    request: Request = ...,
    _user: str = Depends(require_jwt),
):
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        rows = await ModelRepo(db).list_by_repo(repo)  # new repo method

    in_use = [r for r in rows if r.status in ACTIVE_STATUSES]
    if in_use:
        raise HTTPException(
            409,
            f"refusing: repo {repo!r} backs {len(in_use)} active model(s) "
            f"({', '.join(r.id for r in in_use)})",
        )
    benign_but_alive = [r for r in rows if r.status in ("pulled", "idle")]
    if benign_but_alive and not force:
        raise HTTPException(
            409,
            f"repo {repo!r} backs {len(benign_but_alive)} pulled-but-unloaded "
            f"row(s); pass ?force=true to delete cache anyway "
            f"(the row(s) will move to 'failed' on next load attempt)",
        )

    safe = "models--" + repo.replace("/", "--")
    deleted_paths = await asyncio.to_thread(
        _delete_repo_dirs, settings.hf_cache_dir, safe
    )
    if not deleted_paths:
        raise HTTPException(404, f"no cache dir found for repo {repo!r}")
    return Response(status_code=204)
```

`ACTIVE_STATUSES = ("loaded", "loading", "unloading", "pulling")` — same set
guarded by `delete_model` in `models/routes_api.py:654`. We re-use it
verbatim and lift it to a shared constant (`app/db/constants.py` or
`app/models/__init__.py`).

`_delete_repo_dirs` walks both `cache_root/{safe}` and
`cache_root/hub/{safe}`, deletes each that exists, returns the list of
removed paths. Pure-sync, `asyncio.to_thread`-wrapped.

New `ModelRepo.list_by_repo(hf_repo: str) -> list[ModelRow]` — trivial
SELECT addition. Tested with two-row case.

#### Endpoint 3 — `POST /api/cache/models/gc?dry_run=true|false`

```python
@router.post("/models/gc")
async def gc_cached_models(
    dry_run: bool = True,
    failed_older_than_hours: int = 24,
    request: Request = ...,
    _user: str = Depends(require_jwt),
) -> GcResult:
    """Sweep:
      • repos with no matching models row at all (orphans), AND
      • repos whose ONLY matching rows are status=failed and were
        last touched > failed_older_than_hours hours ago.

    Refuses to touch a repo if any matching row is in ACTIVE_STATUSES or
    in ("pulled", "idle") — those are user-owned even if unused.
    """
```

Response shape:
```python
class GcCandidate(BaseModel):
    repo: str
    reason: Literal["orphan", "failed_stale"]
    size_bytes: int
    matched_rows: list[MatchedModelRef]  # always [] for orphan; >=1 for failed_stale

class GcResult(BaseModel):
    dry_run: bool
    total_bytes_freed: int  # sum of candidate sizes
    candidates: list[GcCandidate]
    deleted_paths: list[str]  # empty if dry_run=true
```

`failed_older_than_hours` defaults to 24 — long enough to be sure the
operator has seen the failure and chosen not to retry, short enough that
sweep on the bonus host actually frees disk. Operator can override per
call.

`updated_at` on the `models` row is the freshness signal (already touched
on every status transition — verified in `repos/models.py:142`).

### Backend — registration + auth pattern

Match existing routers: `require_jwt` dependency on every route, prefix
`/api/cache` set on the `APIRouter` constructor, no admin-vs-operator
split (vllm-warden does not currently distinguish; if it grows that
distinction, every route gets `require_admin` in one sweep).

### Frontend — Storage tab on `/stats`

`frontend/src/app/stats/page.tsx` is the natural home: the page already
exists, is gated by auth, and is the operator's "what is this box doing"
dashboard. Add a third top-level section *below* Throughput / GPU Util,
or — if the page is starting to feel crowded — convert the page into a
Tabs container with `Performance` and `Storage` tabs. Default to the
**section approach** (simpler, no router change) and only escalate to
tabs if the implementer finds it cramped.

New components:
- `components/stats/cache-table.tsx` — sortable table: repo, size, last
  used, status badge, actions (`Delete`).
- `components/stats/cache-gc-button.tsx` — opens a confirm dialog that
  shows the dry-run preview, then submits the real GC.

Data flow mirrors existing Stats panels:
```tsx
const { data: cache, mutate } = useSWR<CachedRepoView[]>(
  "/api/cache/models", authFetchJSON,
  { refreshInterval: 30_000 },  // disk doesn't change that often
);
```

- Delete button per row → `authFetchJSON("/api/cache/models/<repo>", {method:"DELETE"})`
  → on 409 with "pulled-but-unloaded" message, show a second confirm that
  re-submits with `?force=true`.
- GC button → first calls `?dry_run=true`, renders the candidates
  table in a modal with total bytes to be freed, operator confirms,
  re-submits with `?dry_run=false`.
- Both actions call `mutate()` on success to refresh the list.

Bytes formatted via the existing `formatBytes` helper (used in fit-preview
modal — confirmed in `frontend/src/lib/`).

### Testing

**Backend unit (`tests/cache/test_scanner.py`):**
- Empty cache dir → returns `[]`.
- Three `models--A--B/`, `models--C/D-GGUF/`, plus a stray non-matching dir → returns only the two, name-decoded correctly.
- Both `<root>/` and `<root>/hub/` populated → returns union, deduped.
- Permission-denied subdir → logs + skips, doesn't raise.

**Backend integration (`tests/cache/test_routes.py`, FastAPI TestClient):**
- `GET /api/cache/models` returns scanner output joined with DB rows; row with `hf_repo` matching a cached dir gets non-empty `matched_models`.
- `DELETE /api/cache/models/{repo}` against `status=loaded` row → 409, dir intact.
- `DELETE` against `status=idle` without `force` → 409 with "pulled-but-unloaded" message; dir intact.
- `DELETE` with `?force=true` against `status=idle` → 204, dir gone, row still exists in DB.
- `DELETE` against orphan repo (no matching row) → 204, dir gone.
- `DELETE` against repo with no cache dir → 404.
- `POST /api/cache/models/gc?dry_run=true` → returns candidates, deletes nothing.
- `POST /api/cache/models/gc?dry_run=false` → deletes orphan + stale-failed dirs, leaves `pulled`/`idle`/`loaded` alone.
- `failed_older_than_hours=0` → freshly-failed rows are included.

**Frontend (`frontend/__tests__/cache-table.test.tsx`):**
- Renders rows from SWR fixture.
- Delete button → 409 force-required → second confirm path.
- GC button → dry-run modal → confirm → real call fires.

**Manual smoke (post-deploy on bonus):**
- `GET /api/cache/models` returns the half-dozen cached repos on the bonus host with sizes matching `du -sh`.
- Delete a known-orphan repo (e.g. left over from a #320–#330 experiment) — frees disk visible in `df`.
- GC dry-run shows the orphan/failed candidates, confirm sweep, `df` reflects the freed bytes.

### Risks

- **Race**: operator clicks Delete just as a Load is starting → DB shows `idle` at request time, supervisor flips it to `loading` mid-delete, deletion succeeds and load fails noisily. Acceptable for v1 (load will fail clean and re-pull on retry). Worth a follow-up issue if it surfaces in practice.
- **HF cache layout drift**: future HF versions may change the
  `models--<org>--<name>` convention. The scanner is the only place that
  encodes it; one helper to update.
- **Scanner walk cost** on very large caches: the bonus host's
  current cache is < 200 GiB, walk completes in seconds. If real-world
  caches grow into terabytes, add a 30-second result cache in the route
  handler. Not needed in v1.

## Open follow-ups (file as separate issues, NOT in this MR)

- **#105 cascade**: when this lands, fold its scope into "delete model
  row also deletes cache" — but only if the operator opted in to it via a
  query flag on `DELETE /api/models/{id}?delete_cache=true`. The two
  features ship cleanly together; #105 piggy-backs on the
  `_delete_repo_dirs` helper introduced here.
- Scheduled GC policy (cron-style, configurable in settings).
- Per-snapshot cleanup (keep latest, drop older snapshots of the same
  repo) — only if disk pressure becomes acute.

## Changelog wording (proposed)

> **Added** — Storage tab on the Stats page lists every HuggingFace
> cache directory with size and which model row (if any) owns it. New
> endpoints `GET /api/cache/models`, `DELETE /api/cache/models/{repo}`,
> `POST /api/cache/models/gc` let operators reclaim disk from
> orphaned downloads and stale failed pulls. (#114)
