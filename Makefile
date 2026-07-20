IMAGE := vllm-warden:dev
PY_IMAGE := python:3.11-slim
RUN_PY := docker run --rm -v $(PWD):/app -w /app $(PY_IMAGE)

.PHONY: install test lint typecheck format docker-build docker-run shell generate-api-types smoke

install:
	$(RUN_PY) sh -c "pip install -q -r requirements-dev.txt"

test:
	$(RUN_PY) sh -c "pip install -q -r requirements-dev.txt && pytest -v $(ARGS)"

test-unit:
	$(RUN_PY) sh -c "pip install -q -r requirements-dev.txt && pytest -v -m 'not integration' tests/unit"

test-integration:
	$(RUN_PY) sh -c "pip install -q -r requirements-dev.txt && pytest -v -m integration $(ARGS) tests/integration"

lint:
	# Use the pinned ruff from requirements-dev.txt rather than `pip install ruff`
	# (which always pulls the latest version). Without this pin, new ruff
	# releases routinely fire lints CI doesn't have (CI installs the pinned
	# requirements), causing local-vs-CI false-fail drift.
	$(RUN_PY) sh -c "pip install -q -r requirements-dev.txt && ruff check app/ tests/"

format:
	# Same rationale as lint above — pin to requirements-dev.txt so formatting
	# changes never depend on which ruff version the operator happened to have
	# locally.
	$(RUN_PY) sh -c "pip install -q -r requirements-dev.txt && ruff format app/ tests/"

typecheck:
	$(RUN_PY) sh -c "pip install -q -r requirements-dev.txt && mypy app/"

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	# --pid=host: required so /api/system/gpus can attribute nvidia-smi
	# compute holders (host PIDs) to supervisor-tracked PIDs. See #42.
	docker run --rm --gpus all --pid=host -p 8080:8080 \
	  -e VW_COOKIE_SECRET=$$(openssl rand -base64 32) \
	  -e VW_CONTAINER_GPU_COUNT=4 \
	  -v $(PWD)/.data:/data \
	  $(IMAGE)

shell:
	$(RUN_PY) bash

generate-api-types:
	docker run --rm -v $(PWD):/app -w /app python:3.11-slim sh -c \
	  "pip install -q -r requirements-dev.txt && python -c \"import json, sys; from app.main import app; sys.stdout.write(json.dumps(app.openapi()))\"" > openapi.json
	docker run --rm -u $(shell id -u):$(shell id -g) -e HOME=/tmp -v $(PWD):/work -w /work/frontend node:20-alpine \
	  npx -y openapi-typescript@7 ../openapi.json -o src/lib/api-types.generated.ts

smoke:
	# #155 unified-port: end-to-end smoke against the live Caddy front-door
	# on :8080. Assumes `docker compose up -d` has already brought the api,
	# ui, and caddy services to healthy state. Each curl asserts a STATUS
	# code, not a body, because the bodies differ across builds (build SHA
	# in /api/csrf, Next chunk hashes in /ui/) and would make this brittle.
	#
	# `-L` follows redirects — `/ui/` 308→`/ui` 307→`/ui/models` 200 is
	# the natural Next.js basePath landing flow; we care that the chain
	# terminates in a 200, not the intermediate hops.
	#
	# Expected (final-status, after redirect chain):
	#   /            → 200 (landing page HTML, public)
	#   /_landing    → 200 (same content, direct)
	#   /ui/         → 200 (Next.js root page, possibly /ui/models or /ui/login)
	#   /api/csrf    → 200 (CSRF bootstrap, no auth required)
	#   /healthz     → 200 (uptime probe — Caddy → Next /healthz alias)
	@set -e; \
	for path in / /_landing /ui/ /api/csrf /healthz; do \
	  code=$$(curl -sL -o /dev/null -w "%{http_code}" "http://localhost:8080$$path"); \
	  printf "GET %-15s -> %s\n" "$$path" "$$code"; \
	  if [ "$$code" != "200" ]; then \
	    echo "FAIL: $$path returned $$code (expected 200)" >&2; \
	    exit 1; \
	  fi; \
	done; \
	echo "smoke OK"
