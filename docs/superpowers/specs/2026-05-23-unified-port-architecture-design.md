# vllm-warden Unified-Port Architecture (Caddy + /ui basePath + Landing Page)

> Spec doc for issue #155. Locked at MR open time; subsequent refinements
> should land as follow-up specs, not in-place edits.

## Goal
Serve all vllm-warden surfaces from a single host-published port (8080) by adding a Caddy reverse-proxy container in front of the existing api + ui containers. Simplifies external reverse-proxy setup (`vllm.protrener.com → 10.10.0.187:8080`) — operator no longer needs to expose two ports and remember which is which.

## Current state (pre-change)
- `api` container: FastAPI on :8080, host-published. Already serves `/api/*` AND `/v1/*` (OpenAI-compatible proxy at `app/proxy/routes.py`).
- `ui` container: Next.js standalone on :3000, host-published. `next.config.ts` rewrites `/api/:path*` and `/v1/:path*` back to `BACKEND_URL` so the browser sees one origin — but only if you publish :3000. Today both are published; external proxy points at :8080, so `/ui` isn't reachable via the public hostname.

## Routing map after change

```
EXTERNAL                          INTERNAL DOCKER NETWORK
vllm.protrener.com ─[upstream]─→ :8080 caddy
                                   │
                                   ├─ "/"           → api:8080/_landing  (landing HTML or 404)
                                   ├─ "/ui*"        → ui:3000            (Next.js basePath:'/ui')
                                   ├─ "/_next/*"    → ui:3000 (legacy — Next emits /ui/_next under basePath; keep as safety)
                                   ├─ "/api/*"      → api:8080
                                   ├─ "/v1/*"       → api:8080  (flush_interval -1 for SSE)
                                   └─ "/healthz"    → api:8080
                                                                         api:8080  (internal-only)
                                                                         ui:3000   (internal-only)
```

## Decisions (locked)
1. **Front-door**: Caddy 2-alpine in a new `caddy` service. Sole host-published port `8080:8080`. Config at `deploy/caddy/Caddyfile`.
2. **Landing page**: FastAPI `GET /_landing` (public, no JWT) returns `app/landing/landing.html` when `landing_page_enabled=true` (default), else HTTP 404. HTML content: project name + tagline, Apache 2.0 license line, link to https://podwarden.com, link to https://github.com/Podwarden/vllm-warden. Minimal CSS — single file, no JS, no external assets. (Spec originally said `app/templates/landing.html`; that path is taken by the model-template Python package — placed next to the route module instead. See commit message for rationale.)
3. **Opt-out**: new boolean setting `landing_page_enabled` (default `true`) wired through existing settings module. Expose via the settings GET/PATCH API. Add a simple checkbox to /settings UI in this MR; deeper UX deferred to #154 redesign.
4. **Next.js basePath**: `basePath: '/ui'` in `next.config.ts`. Delete `/api/:path*` and `/v1/:path*` rewrites — Caddy owns those now. Keep `BACKEND_URL` for any SSR fetches.
5. **Dev mode**: Same Caddy topology in `make dev`. HMR WebSocket at `/ui/_next/webpack-hmr` passes through Caddy natively (Caddy auto-detects WebSocket Upgrade).
6. **Internal-only api+ui**: drop `ports:` arrays from both services in `docker-compose.yml`. Keep `EXPOSE 8080` / `EXPOSE 3000` in Dockerfiles for clarity.
7. **External-proxy compatibility**: upstream `vllm.protrener.com → :8080` unchanged. `:3000` bookmarks break (acceptable transitional cost; mention in changelog).
8. **No new auth at Caddy layer**: FastAPI keeps JWT/Bearer auth on `/api/*` and `/v1/*`.

## Out of scope
- Caddy admin API exposure (default port 2019 NOT published)
- HTTP/3 (defer)
- TLS termination at Caddy (external upstream handles it)
- d5 deploy runbook update (separate follow-up after this lands and v2026.05.23.2 ships)

## File-by-file change list

**New files:**
- `deploy/caddy/Caddyfile`
- `app/landing/__init__.py` — empty
- `app/landing/routes.py` — `GET /_landing` route + settings read
- `app/landing/landing.html` — static HTML (single-file with inline CSS, dark mode via prefers-color-scheme)
- `tests/unit/landing/__init__.py` — empty
- `tests/unit/landing/test_routes.py` — 4 cases (enabled-returns-html, disabled-returns-404, enabled-default-true, content-includes-required-links)
- `tests/unit/settings/test_landing_setting.py` — round-trips `landing_page_enabled` through GET/PATCH
- `frontend/src/lib/__tests__/next-config-basePath.test.ts` — frontend smoke for `basePath: '/ui'` + rewrites deleted
- `docs/superpowers/specs/2026-05-23-unified-port-architecture-design.md` — this spec
- `app/db/sql/0020_landing_page_setting.sql` — seed `landing_page_enabled='true'`

**Modified files:**
- `docker-compose.yml` — add caddy service, drop `ports:` from api+ui
- `app/main.py` — register landing router
- `app/settings/routes_api.py` — add `landing_page_enabled` to `RUNTIME_KEYS` and `_COERCERS`; GET returns persisted value; PATCH writes it
- `frontend/src/lib/settings-hints.ts` — hint copy + RestartKind for the new key
- `frontend/src/components/settings/runtime-tab.tsx` — render boolean field for new key
- `frontend/src/components/settings/setting-field.tsx` — add `kind="boolean"` if not present
- `frontend/next.config.ts` — add `basePath: '/ui'`, delete `/api` and `/v1` rewrites
- `README.md` — section on unified-port topology with diagram
- `docs/operating.md` — operator section on the new layout, how to opt out of landing
- `changelog.md` — Added (Caddy front-door + landing), Changed (UI at /ui), Removed (Next.js /api+/v1 rewrites)
- `Makefile` — add `smoke` target

## Caddyfile (~30 lines)

```caddyfile
{
    admin off
    log {
        output stderr
        format console
    }
}

:8080 {
    encode gzip

    handle / {
        reverse_proxy api:8080 {
            rewrite /_landing
        }
    }

    handle /ui* {
        reverse_proxy ui:3000
    }

    handle /_next/* {
        reverse_proxy ui:3000
    }

    handle /api/* {
        reverse_proxy api:8080 {
            flush_interval -1
        }
    }

    handle /v1/* {
        reverse_proxy api:8080 {
            flush_interval -1
        }
    }

    handle /healthz {
        reverse_proxy api:8080
    }

    handle {
        respond "Not Found" 404
    }
}
```

## Smoke test

```bash
curl -sf http://localhost:8080/                          # landing HTML (default) or 404 (if opted out)
curl -sf http://localhost:8080/ui/                       # next.js root page
curl -sf http://localhost:8080/healthz                   # {"ok": true}
curl -sf http://localhost:8080/api/version               # version json (401 without bearer is also OK — server reachable)
curl -sf -H "Authorization: Bearer $TOK" http://localhost:8080/v1/models  # OpenAI models list (requires runtime token)
```

## Test plan
- Backend: 4 pytest cases in `tests/unit/landing/test_routes.py` (enabled/disabled/default/content)
- Backend: 1 pytest case in `tests/unit/settings/test_landing_setting.py` covering GET/PATCH round-trip
- Frontend: 1 vitest case verifying `basePath: '/ui'` is set + `/api`/`/v1` rewrites are deleted
- Existing vitest + pytest suites must stay green

## Rollback
Single MR, atomic. Revert MR + redeploy = back to two-port layout. The `landing_page_enabled` setting becomes inert after revert (no destructive schema change).
