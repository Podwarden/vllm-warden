"""Tests for the public /_landing route (#155).

The route is intentionally public (no JWT) — it serves the unified-port
root behind Caddy. The 4 cases below pin the contract called out in
``docs/superpowers/specs/2026-05-23-unified-port-architecture-design.md``:

  1. Default — fresh DB has the migration 0020 seed (`true`) and the
     route returns 200 + HTML content.
  2. Disabled — operator PATCH'd `landing_page_enabled=false` → 404.
  3. Missing row — defensive: if the seed row is absent (partially
     bootstrapped DB) the route still defaults to enabled (200 + HTML),
     so a half-installed warden doesn't appear "down" at the root.
  4. Content — the served HTML contains the three load-bearing links
     called out in the spec (operator console, source repo, PodWarden
     site) plus the title, so a future refactor of landing.html can't
     silently strip the entry-points.

No JWT is involved — these tests exercise the same anonymous request
shape a browser hitting `https://vllm.protrener.com/` would make after
Caddy's rewrite.
"""

import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient


def _delete_landing_setting(db_path: Path) -> None:
    """Drop the migration-0020 seed row to simulate a partially bootstrapped
    DB. Uses sync sqlite3 with matching WAL pragmas (same shape as
    seed_admin_user in conftest) so the write is visible to the next
    aiosqlite reader.
    """
    with sqlite3.connect(db_path, isolation_level=None) as db:
        db.execute("PRAGMA journal_mode = WAL")
        db.execute("BEGIN IMMEDIATE")
        db.execute("DELETE FROM settings WHERE key = 'landing_page_enabled'")
        db.execute("COMMIT")
        db.execute("PRAGMA wal_checkpoint(FULL)")


def _set_landing_setting(db_path: Path, value: str) -> None:
    """Force the setting to an explicit canonical string."""
    with sqlite3.connect(db_path, isolation_level=None) as db:
        db.execute("PRAGMA journal_mode = WAL")
        db.execute("BEGIN IMMEDIATE")
        db.execute(
            "INSERT INTO settings(key, value) VALUES ('landing_page_enabled', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (value,),
        )
        db.execute("COMMIT")
        db.execute("PRAGMA wal_checkpoint(FULL)")


def test_landing_enabled_by_default_returns_html(client: TestClient) -> None:
    """Fresh DB has the 0020 seed (`true`) → 200 + text/html body."""
    r = client.get("/_landing")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "<html" in r.text.lower()


def test_landing_disabled_returns_404(client: TestClient, tmp_data_dir: Path) -> None:
    """Operator opt-out → 404 (not 403, not 503) so Caddy propagates it
    verbatim as "no landing page configured".
    """
    db_path = tmp_data_dir / "vllm-warden.db"
    _set_landing_setting(db_path, "false")
    r = client.get("/_landing")
    assert r.status_code == 404


def test_landing_missing_row_defaults_to_enabled(
    client: TestClient, tmp_data_dir: Path
) -> None:
    """Defensive — a partially-bootstrapped DB (no 0020 seed) MUST still
    serve the landing page rather than 404'ing the unified-port root.
    """
    db_path = tmp_data_dir / "vllm-warden.db"
    _delete_landing_setting(db_path)
    r = client.get("/_landing")
    assert r.status_code == 200
    assert "<html" in r.text.lower()


def test_landing_html_contains_required_entry_points(client: TestClient) -> None:
    """Lock the load-bearing surface area called out in the spec — the
    page MUST link to /ui/ (operator console), the public source repo,
    and the PodWarden site, and MUST carry the product title. A future
    refactor that strips any of these silently breaks the only entry-
    point a curious visitor sees.
    """
    body = client.get("/_landing").text
    assert "vLLM Warden" in body
    assert 'href="/ui/"' in body
    assert "github.com/Podwarden/vllm-warden" in body
    assert "podwarden.com" in body


def test_landing_html_redesign_surface(client: TestClient) -> None:
    """Pin the podwarden.com-aligned redesign surface (visual identity +
    rewritten copy + icon-link a11y). Separate from the contract test
    above so a future copy/visual refactor only touches this case.
    """
    body = client.get("/_landing").text
    # New tagline anchor — pins rewritten copy so v1 "Self-service appliance" regression is caught.
    assert "operator console for serving" in body.lower()
    # Visual identity pins (no full snapshot — keep refactor-friendly).
    assert "DM+Sans" in body or "DM Sans" in body
    assert "#020617" in body or "slate-950" in body
    # Accessibility: icon-only links carry aria-label.
    assert "aria-label" in body and "GitHub" in body
