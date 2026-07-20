"""GET /_landing — public landing page served at the unified-port root (#155).

Lives behind Caddy's `handle /` block, which rewrites the request to
`/_landing` before forwarding to the warden. The route is intentionally
unauthenticated: it exists so an operator who points a browser at
`https://vllm.protrener.com/` sees a useful entry-point instead of a 404.

The landing page can be disabled per-deployment via the
`landing_page_enabled` setting (default `'true'`, seeded by migration
0020). When disabled the route returns a plain HTTP 404 — Caddy
propagates it verbatim, so the unified-port root behaves identically to
"no landing page" without removing the route or changing the Caddyfile.

The HTML file is read once at import time. It is a static asset shipped
in the wheel; reloading on every request would be wasted I/O and we
explicitly do NOT support hot-editing it in a running container (rebuild
the image instead).
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, Response

from app.db.database import open_db
from app.db.repos.settings import SettingsRepo

router = APIRouter()

# Read the static landing HTML once at import time. The file ships next to
# this module so the path is stable across editable + standalone installs.
_LANDING_HTML: str = (Path(__file__).parent / "landing.html").read_text(
    encoding="utf-8"
)


async def _is_enabled(request: Request) -> bool:
    """Return True iff the `landing_page_enabled` setting is truthy.

    Defaults to True if the row is missing (defensive — migration 0020
    seeds it, but a partially-bootstrapped DB shouldn't silently mask
    the landing page).
    """
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        raw = await SettingsRepo(db).get("landing_page_enabled")
    if raw is None:
        return True
    # Stored as canonical 'true' / 'false' by the settings coercer, but
    # tolerate any common truthy spelling for forward-compat with manual
    # edits to the settings table.
    return raw.strip().lower() in {"true", "1", "yes", "on"}


@router.get("/_landing", include_in_schema=False)
async def landing(request: Request) -> Response:
    if not await _is_enabled(request):
        # Return 404 (not 403) so the unified-port root behaves like a
        # disabled feature, not like a permission boundary.
        return Response(status_code=404)
    return HTMLResponse(_LANDING_HTML)
