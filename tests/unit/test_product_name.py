"""Guard: the product's user-visible name is "LLM Warden", never "vLLM Warden".

The product now runs two inference backends (vLLM and llama.cpp), so the
two-word product name was renamed to **LLM Warden**. This is a *display-only*
rename: identity strings are deliberately untouched.

What this guard forbids and what it deliberately does NOT forbid:

* FORBIDDEN — the two-word product name with a whitespace separator, in any
  casing: ``vLLM Warden``, ``vllm warden``, ``VLLM Warden``.
* ALLOWED — the bare engine name ``vLLM``. The product serves vLLM, so
  "vLLM version", "the vLLM engine", "live vLLM logs" and ``vllm/vllm-openai``
  are all *correct* prose that this guard must never break. A test that
  forbade the bare word would block true statements about the engine.
* ALLOWED — the identity slug ``vllm-warden`` / ``vllm_warden`` (image names,
  the Hub catalogue slug and ``app_family``, k8s namespaces, ``/data``
  paths, container names, module paths, repo URLs). Those are a separate,
  breaking piece of work; the hyphen/underscore forms are not matched here.

The separator is the entire discrimination rule: whitespace means somebody
wrote the product's *name*; a hyphen or underscore means an *identifier*.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Whitespace separator only -- see the module docstring. `vllm-warden` and
# `vllm_warden` are identity and must NOT match.
OLD_PRODUCT_NAME = re.compile(r"vllm\s+warden", re.IGNORECASE)

PRODUCT_NAME = "LLM Warden"

# The surfaces a human actually reads. An explicit allowlist rather than a
# repo walk: the CI runner's build directory keeps untracked node_modules and
# root-owned caches from sibling jobs, and this test must not wander into
# them (nor into changelog.md or docs/superpowers/, which are historical
# records of work done under the old name and are left as written).
SCANNED_TREES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("app", (".py", ".html")),
    ("frontend/src", (".ts", ".tsx", ".html", ".svg", ".css")),
    ("deploy/hub", (".md", ".json")),
    ("publish", (".md", ".txt", ".sh")),
)

SCANNED_FILES: tuple[str, ...] = ("README.md",)


def _iter_surfaces() -> list[Path]:
    out: list[Path] = []
    for rel, suffixes in SCANNED_TREES:
        root = REPO_ROOT / rel
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in suffixes:
                out.append(path)
    for rel in SCANNED_FILES:
        path = REPO_ROOT / rel
        if path.is_file():
            out.append(path)
    return out


def test_the_scan_actually_covers_something() -> None:
    """Self-check: an allowlist that silently matched nothing would make
    every assertion below vacuously true."""
    surfaces = _iter_surfaces()
    assert len(surfaces) > 50, f"surface scan found only {len(surfaces)} files"
    assert any(p.name == "landing.html" for p in surfaces)
    assert any(p.name == "nav-bar.tsx" for p in surfaces)
    assert any(p.name == "README.md" for p in surfaces)


def test_no_user_visible_surface_says_vllm_warden() -> None:
    """No page title, header, wordmark, nav label, log line, error message,
    catalogue field or README sentence may present the product as
    "vLLM Warden"."""
    offenders: list[str] = []
    for path in _iter_surfaces():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):  # pragma: no cover - CI hygiene
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if OLD_PRODUCT_NAME.search(line):
                rel = path.relative_to(REPO_ROOT)
                offenders.append(f"{rel}:{lineno}: {line.strip()}")

    assert not offenders, (
        "the product is called 'LLM Warden'; these surfaces still say "
        "'vLLM Warden':\n  " + "\n  ".join(offenders)
    )


def test_the_guard_still_permits_bare_engine_references() -> None:
    """The regex must not fire on the inference engine. If it ever does, the
    guard would force prose that is factually wrong now that llama.cpp ships
    alongside vLLM."""
    for legitimate in (
        "vLLM 0.26.0",
        "the vLLM engine",
        "vllm/vllm-openai:v0.26.0",
        "live vLLM logs",
        "with vLLM process attribution",
        "vLLM Python package version",
        "Selected MarlinFP8LinearKernel",
    ):
        assert not OLD_PRODUCT_NAME.search(legitimate), legitimate


def test_the_guard_still_permits_the_identity_slug() -> None:
    """The hyphen/underscore forms are identity and out of scope for the
    display rename."""
    for identity in (
        "vllm-warden",
        "pw-vllm-warden",
        "/data/vllm-warden.db",
        "vllm-warden-engine-abc123",
        "app.vllm_warden",
        "registry.podwarden.com/podwarden/apps/vllm-warden",
    ):
        assert not OLD_PRODUCT_NAME.search(identity), identity


@pytest.mark.parametrize(
    "spelling", ["vLLM Warden", "vllm warden", "VLLM Warden", "vLLM  Warden"]
)
def test_the_guard_catches_every_casing(spelling: str) -> None:
    assert OLD_PRODUCT_NAME.search(spelling), spelling


def test_the_wordmark_and_page_titles_carry_the_new_name() -> None:
    """Positive half: renaming by deletion would pass the guard above."""
    landing = (REPO_ROOT / "app/landing/landing.html").read_text(encoding="utf-8")
    assert f"<title>{PRODUCT_NAME}</title>" in landing
    assert f'<span class="wordmark">{PRODUCT_NAME}</span>' in landing

    nav = (REPO_ROOT / "frontend/src/components/nav-bar.tsx").read_text(encoding="utf-8")
    assert f"<span>{PRODUCT_NAME}</span>" in nav

    # The browser tab title and the FastAPI/OpenAPI title used to carry the
    # identity slug (`vllm-warden`) as if it were the product's name. Those
    # two are display surfaces, so they get the real name -- but the slug
    # form is invisible to the regex above, hence these explicit pins.
    layout = (REPO_ROOT / "frontend/src/app/layout.tsx").read_text(encoding="utf-8")
    assert f"title: '{PRODUCT_NAME}'" in layout

    main = (REPO_ROOT / "app/main.py").read_text(encoding="utf-8")
    assert f'FastAPI(title="{PRODUCT_NAME}"' in main

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.startswith(f"# {PRODUCT_NAME}\n")

    hub = (REPO_ROOT / "deploy/hub/README-hub.md").read_text(encoding="utf-8")
    assert hub.startswith(f"# {PRODUCT_NAME}\n")
