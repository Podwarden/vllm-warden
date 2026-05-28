"""Pure-sync HF cache walker.

Decoupled from FastAPI so unit tests can drop a few directories into a
tmpdir and assert on the returned list without standing up an app.
Route handlers wrap calls in ``asyncio.to_thread`` so a slow walk on a
large cache doesn't block the event loop.

The HF library writes its cache as ``models--<org>--<name>/`` directories
under either ``HF_HOME`` (``$HF_HOME/hub/``) or ``HUGGINGFACE_HUB_CACHE``
(directly under the configured root). vllm-warden's ``VW_HF_CACHE_DIR``
maps to whichever the runtime container ends up exporting, so the
scanner probes BOTH ``<root>/`` and ``<root>/hub/`` and returns the
union deduplicated by absolute path.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# The HF library encodes ``<org>/<name>`` as ``models--<org>--<name>``.
# Same convention ``pull_task._snapshot_dir_size`` walks in reverse.
_PREFIX = "models--"


@dataclass(frozen=True)
class CachedRepo:
    """A single decoded ``models--<org>--<name>`` directory on disk.

    ``repo`` is the decoded ``<org>/<name>`` HF repo id (the inverse of
    the on-disk encoding). ``path`` is the absolute path so callers
    that need to delete the dir don't have to reconstruct it. ``size_bytes``
    sums ``blobs/ + snapshots/ + refs/`` (all of HF's per-repo state).
    ``last_modified`` is the max mtime seen anywhere under the dir —
    proxies "when did vllm-warden last touch this".
    """

    repo: str
    path: Path
    size_bytes: int
    last_modified: float


def _decode_repo_name(dirname: str) -> str | None:
    """Convert ``models--Qwen--Qwen3.6-27B-GGUF`` to ``Qwen/Qwen3.6-27B-GGUF``.

    Returns None if ``dirname`` doesn't match the HF cache layout — the
    caller skips it. Note "--" inside a repo name (e.g.
    ``models--org--has--double--dashes``) is ambiguous; HF treats the FIRST
    ``--`` after the prefix as the org/name split and leaves the rest
    verbatim. We mirror that with a single ``replace("--", "/", 1)`` —
    consistent with how HF's own ``_get_repo_id_from_cache_dir`` decodes,
    and the same convention vllm-warden's pull task uses on the way in.
    """
    if not dirname.startswith(_PREFIX):
        return None
    body = dirname[len(_PREFIX):]
    if not body:
        return None
    # First "--" separates org from repo; rest of the name (which may
    # contain dashes or even "--" inside a repo slug) is preserved.
    return body.replace("--", "/", 1)


def _walk_size_and_mtime(root: Path) -> tuple[int, float]:
    """Recursive ``du`` + max mtime for one repo directory.

    Tolerates partially-unreadable subtrees: a single ``PermissionError``
    on a blob is logged and skipped, the rest of the walk continues. The
    HF cache uses symlinks from ``snapshots/<rev>/file`` to
    ``blobs/<sha>`` — ``os.walk`` follows the symlink target's stat for
    size, which is what we want (the actual blob bytes are what's on
    disk; the symlink itself is ~0 bytes).
    """
    total = 0
    latest = 0.0
    try:
        # ``stat_result.st_mtime`` on the root itself is the floor for
        # an empty repo dir — captures the directory creation timestamp.
        latest = root.stat().st_mtime
    except OSError as exc:
        logger.debug("scan: stat root failed %s: %s", root, exc)

    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        try:
            d_stat = os.stat(dirpath)
            if d_stat.st_mtime > latest:
                latest = d_stat.st_mtime
        except OSError as exc:
            logger.debug("scan: stat dir %s failed: %s", dirpath, exc)
        for name in filenames:
            full = os.path.join(dirpath, name)
            try:
                # follow_symlinks=True: HF stores actual bytes under
                # blobs/<sha> and symlinks them from snapshots/<rev>/file.
                # Stating the symlink target counts the bytes exactly once
                # across the union of symlinks pointing at it (each blob
                # is unique under blobs/, so the count doesn't double).
                st = os.stat(full)
            except OSError as exc:
                logger.debug("scan: stat %s failed: %s", full, exc)
                continue
            total += st.st_size
            if st.st_mtime > latest:
                latest = st.st_mtime
    return total, latest


def scan_hf_cache(cache_root: Path) -> list[CachedRepo]:
    """Walk ``cache_root`` for ``models--<org>--<name>`` directories.

    Probes both ``cache_root/`` and ``cache_root/hub/`` (HF layout drift)
    and returns the union deduplicated by absolute path. Tolerates:

    - missing ``cache_root`` (returns ``[]``);
    - unreadable individual subtrees (logged + skipped);
    - directories that don't match the prefix (skipped);
    - exotic encodings (skipped, logged).

    Never raises. The route handler calls this via
    ``asyncio.to_thread`` so a slow walk on a multi-100-GiB cache does
    not block the event loop.
    """
    if not cache_root.exists():
        return []

    by_path: dict[str, CachedRepo] = {}
    candidate_parents = [cache_root, cache_root / "hub"]
    for parent in candidate_parents:
        if not parent.exists() or not parent.is_dir():
            continue
        try:
            entries = list(parent.iterdir())
        except OSError as exc:
            logger.warning("scan: iterdir %s failed: %s", parent, exc)
            continue
        for entry in entries:
            try:
                if not entry.is_dir():
                    continue
            except OSError:
                continue
            repo = _decode_repo_name(entry.name)
            if repo is None:
                continue
            abs_path = entry.resolve()
            key = str(abs_path)
            if key in by_path:
                continue
            try:
                size, mtime = _walk_size_and_mtime(abs_path)
            except Exception as exc:  # belt-and-suspenders: never raise
                logger.warning("scan: walk %s failed: %s", abs_path, exc)
                continue
            by_path[key] = CachedRepo(
                repo=repo,
                path=abs_path,
                size_bytes=size,
                last_modified=mtime,
            )
    return sorted(by_path.values(), key=lambda r: r.repo)
