# vLLM Warden E2E Tests

End-to-end tests using Playwright. **Not run in CI** — exercised manually against a live `docker compose` stack to validate the operator's happy-path flow before each release.

## Running

```bash
# 1. Bring the stack up
cd /path/to/vllm-warden
docker compose up -d

# 2. Wait for the UI to be reachable
curl -fsS http://localhost:3000/api/health  # should return {"ok": true}

# 3. Run Playwright (inside the frontend dir, in Docker)
cd frontend
docker run --rm --network host -v "$PWD:/work" -w /work \
  mcr.microsoft.com/playwright:v1.49.0-jammy \
  npx playwright test
```

## Required env

- `E2E_ADMIN_PW` — the admin password set during `/setup`. Defaults to `change-me` in the spec.

## Test scope

The single happy-path spec exercises: login → add tiny model (facebook/opt-125m) → pull → load → mint API token → /v1/completions call → rotate token → unload → delete model. Anything beyond this is unit/component test territory.
