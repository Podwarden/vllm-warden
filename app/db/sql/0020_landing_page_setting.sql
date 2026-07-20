-- 0020_landing_page_setting.sql
--
-- Issue #155 — unified-port architecture.
--
-- Seed the `landing_page_enabled` runtime setting. Defaults to 'true' so a
-- fresh warden serves the public landing page at the unified-port root
-- (`GET /` → Caddy rewrites to `/_landing` → FastAPI returns the HTML).
-- Operators opt out by PATCHing `/api/settings/runtime` with
-- `{"landing_page_enabled": false}`, which the FastAPI route coerces
-- back to the canonical 'true'/'false' string before storing.
--
-- Stored as a string (not BOOLEAN) for consistency with every other row
-- in the `settings` table — SQLite doesn't have a native bool type and
-- the SettingsRepo + runtime route_api stack already encodes booleans
-- as 'true' / 'false' lowercase strings end-to-end.
--
-- Rollback: `DELETE FROM settings WHERE key = 'landing_page_enabled';`
-- (the route's `_is_enabled()` helper defaults to True on missing rows,
-- so deleting the seed restores the default behaviour).

INSERT OR IGNORE INTO settings(key, value)
VALUES ('landing_page_enabled', 'true');
