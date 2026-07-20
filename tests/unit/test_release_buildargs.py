"""Regression net for vllm-warden#45 — `vdev · unknown` banner.

The runtime version surfacing is already covered by
`tests/unit/system/test_routes_version.py`. This file guards the *release
procedure* side — the two places where the build-time / deploy-time identity
flows in:

  1. `docs/releasing.md` — the manual `docker buildx --push` recipe must
     pass `--build-arg VW_BUILD_VERSION=…` so the backend image bakes the
     real tag instead of the Dockerfile default `dev`. Both the backend and
     UI commands carry the arg (the UI Dockerfile currently ignores it but
     the doc keeps the convention symmetric and forward-compatible).

  2. `deploy/hub/compose.yaml` — the `api` service must expose
     `VW_BUILD_VERSION` (and `VW_BUILD_SHA`) on its env so a release
     engineer can override a wrongly-baked image at deploy time without
     rebuilding.

These are file-content asserts, not behavioural tests — pytest is just a
convenient harness because vllm-warden already runs unit tests via pytest
on every push. Failing tests here mean the gap that produced the
v2026.05.17.1 `vdev · unknown` banner has reopened.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASING_MD = REPO_ROOT / "docs" / "releasing.md"
HUB_COMPOSE = REPO_ROOT / "deploy" / "hub" / "compose.yaml"


def test_releasing_doc_passes_vw_build_version_buildarg_twice():
    """Both `docker buildx build --push` commands (backend + UI) must pass
    `--build-arg VW_BUILD_VERSION=…`. Two occurrences = both commands covered."""
    text = RELEASING_MD.read_text()
    occurrences = text.count("--build-arg VW_BUILD_VERSION=")
    assert occurrences >= 2, (
        f"Expected --build-arg VW_BUILD_VERSION= at least twice in "
        f"{RELEASING_MD.relative_to(REPO_ROOT)} (once per buildx command), "
        f"found {occurrences}. Regressing #45 ships images with the "
        f"Dockerfile default `dev` baked in — the version banner reads "
        f"`vdev · unknown` instead of the release tag."
    )


def test_releasing_doc_passes_vw_build_sha_buildarg_twice():
    """Companion to VW_BUILD_VERSION — both buildx commands must also bake the
    commit SHA so the banner's second segment isn't `unknown`."""
    text = RELEASING_MD.read_text()
    occurrences = text.count("--build-arg VW_BUILD_SHA=")
    assert occurrences >= 2, (
        f"Expected --build-arg VW_BUILD_SHA= at least twice in "
        f"{RELEASING_MD.relative_to(REPO_ROOT)} (once per buildx command), "
        f"found {occurrences}."
    )


def test_hub_compose_api_service_exposes_vw_build_version_env():
    """The `api` service in the production-style compose file must expose
    VW_BUILD_VERSION on its environment so deploy-time override works even
    when the image was baked without --build-arg."""
    text = HUB_COMPOSE.read_text()
    assert "VW_BUILD_VERSION" in text, (
        f"Expected VW_BUILD_VERSION in {HUB_COMPOSE.relative_to(REPO_ROOT)} "
        f"api.environment so the release engineer can override a wrongly-"
        f"baked image at deploy time. Without it, a mis-built image is "
        f"only recoverable by rebuilding and re-pushing."
    )
    assert "VW_BUILD_SHA" in text, (
        f"Expected VW_BUILD_SHA in {HUB_COMPOSE.relative_to(REPO_ROOT)} "
        f"api.environment alongside VW_BUILD_VERSION."
    )
