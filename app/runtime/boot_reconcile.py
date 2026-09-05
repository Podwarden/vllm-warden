"""Boot reconciliation for rows stranded in a transient status (#236).

WHY THIS EXISTS
---------------
``loading``, ``unloading`` and ``pulling`` all describe work owned by an
in-process object: the supervisor's engine handle, or the pull task's
asyncio task. Neither survives a restart of the warden process, so after a
restart a row in one of those statuses describes something that no longer
exists anywhere.

That is not merely cosmetic — every route out of those statuses is closed:

    POST /api/models/{id}/load    -> 409 (only 'pulled'/'failed' may load)
    POST /api/models/{id}/unload  -> 409 (only 'loaded'/'failed' may unload)

Observed live on a client install on 2026-09-03: ``docker compose up -d``
recreated the ``api`` container mid-load, the GPUs came back free (the
engine subprocess died with the container) and the row read ``loading``
with a null ``last_error`` across two further restarts. The only escape was
``UPDATE models SET status='pulled'`` by hand, which an operator without a
shell on the container does not have.

So on boot we demote every such row to a status an operator can act on:
``pulled`` when the weights are in the HF cache, ``registered`` when they
are not, with ``last_error`` saying why the status moved.

WHAT IS DELIBERATELY LEFT ALONE
-------------------------------
1. Any model the supervisor actually holds a handle for. At boot there is
   never one — the supervisor is freshly constructed and nothing has been
   spawned yet — but the check is the honest predicate for "is this load
   real?", and it keeps the function safe to call from anywhere later.

2. A ``loading`` row carrying a was-serving ``prior_status``: that is the
   watchdog restoring an engine that WAS serving when it died
   (``app/runtime/watchdog.py::_restart``). Demoting it to ``pulled``
   would quietly disable the automatic restore that
   ``restore_after_warden_restart`` exists for — prod stayed down for
   11.5h on 2026-08-18 without it. Those rows fall through to
   ``ModelRepo.mark_runtime_dead_on_startup`` in the same boot, which
   marks them ``failed`` + ``prior_status`` so the watchdog picks them up.

3. Terminal rows (``loaded``, ``failed``, ``pulled``, ``registered``).
   ``loaded`` is ``mark_runtime_dead_on_startup``'s business, not ours:
   it must become ``failed`` WITH a ``prior_status`` so the model is
   restored automatically, and stealing it here would break that.

ORDERING: this runs BEFORE ``mark_runtime_dead_on_startup`` in the
lifespan. After that call every transient row is already ``failed`` and
there would be nothing left for us to see.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.db.database import open_db
from app.db.repos.models import ModelRepo
from app.runtime.backends.paths import _snapshot_dir

logger = logging.getLogger(__name__)

# Statuses whose truth lives in this process's memory, not on disk.
TRANSIENT_STATUSES = ("loading", "unloading", "pulling")

# ``prior_status`` values that mean "this model was SERVING" — the flag
# watchdog.wants_restart keys on. A 'loading' row carrying one is a restore
# in flight, not an interrupted first load.
_WAS_SERVING = ("loaded", "loading")


def weights_present(hf_cache_dir: Path | str, hf_repo: str) -> bool:
    """True when ``hf_repo`` looks like a COMPLETE download in the HF cache.

    Reuses the resolver the launch path uses (``backends.paths``) so that
    "we think the weights are there" cannot drift from "the engine can
    find them" — including its search of both ``<root>`` and
    ``<root>/hub``.

    A snapshot directory alone is not enough: an interrupted ``pulling``
    row leaves one behind with ``*.incomplete`` blobs in it, and calling
    that ``pulled`` would offer the operator a Load button that fails.
    An empty snapshot dir is treated the same way.
    """
    try:
        snap = _snapshot_dir(Path(hf_cache_dir), hf_repo)
        if snap is None:
            return False
        if not any(snap.iterdir()):
            return False
        blobs = snap.parent.parent / "blobs"
        if blobs.is_dir() and any(blobs.glob("*.incomplete")):
            return False
    except OSError:
        # An unreadable cache is not evidence that the weights are there.
        logger.warning("boot reconcile: cache probe failed for %s", hf_repo,
                       exc_info=True)
        return False
    return True


def _is_held(supervisor: Any, model_id: str) -> bool:
    """Does the supervisor actually hold a live engine for this model?

    Defensive ``getattr``: callers pass ``None`` at the very start of boot
    (nothing can be in flight yet) and tests pass stand-ins.
    """
    if supervisor is None:
        return False
    is_running = getattr(supervisor, "is_running", None)
    if is_running is not None and is_running(model_id):
        return True
    get_state = getattr(supervisor, "get_state", None)
    return get_state is not None and get_state(model_id) is not None


async def reconcile_stranded_models(
    settings: Any, supervisor: Any = None
) -> list[tuple[str, str]]:
    """Demote transient rows with no backing process. Returns [(id, new)].

    See the module docstring for the full rationale and the two
    deliberate exemptions.
    """
    async with open_db(settings.db_path) as db:
        repo = ModelRepo(db)
        rows = await repo.list_all()
        moved: list[tuple[str, str]] = []
        for row in rows:
            if row.status not in TRANSIENT_STATUSES:
                continue
            if _is_held(supervisor, row.id):
                continue
            if row.status == "loading" and (row.prior_status or "") in _WAS_SERVING:
                # Watchdog restore of a model that was serving — leave the
                # was-serving path to mark_runtime_dead_on_startup. Same set
                # watchdog.wants_restart keys on, deliberately.
                continue
            new_status = (
                "pulled"
                if weights_present(settings.hf_cache_dir, row.hf_repo)
                else "registered"
            )
            logger.warning(
                "boot reconcile: %s was stranded in '%s' with no backing "
                "process — demoting to '%s'",
                row.id,
                row.status,
                new_status,
            )
            await repo.update_status(
                row.id,
                new_status,
                last_error=(
                    f"recovered from an interrupted {row.status} after a "
                    f"restart: no engine was running for this model"
                ),
            )
            if row.status == "pulling":
                # Same reasoning as mark_runtime_dead_on_startup: the pull
                # does not resume, it re-fetches, so stale progress on a
                # task that no longer exists would lie to the UI.
                await repo.update_pull_progress(row.id, 0, 0)
            moved.append((row.id, new_status))
    return moved
