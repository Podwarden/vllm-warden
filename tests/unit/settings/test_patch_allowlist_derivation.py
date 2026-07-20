"""Tests for the derived PATCH allowlist on ``/api/models/{id}/settings``
(#110).

The hand-maintained ``_PATCHABLE_MODEL_FIELDS`` set drifted from
``ModelRow`` every time a new column was added (#85 / #106 added five
new patchable-looking fields that were never wired up). The mitigation
is to derive the allowlist from ``ModelRow``'s dataclass fields minus
an explicit ``_NEVER_PATCH`` blocklist for the lifecycle-owned
columns; the blocklist is validated at import time so a rename of any
blocklisted field on ``ModelRow`` fails the test suite (and app
startup) rather than silently letting the field through as patchable.
"""
import dataclasses

import pytest

from app.db.repos.models import ModelRow
from app.settings.routes_api import (
    _NEVER_PATCH,
    _PATCHABLE_MODEL_FIELDS,
)


def test_never_patch_contains_lifecycle_owned_fields():
    """The blocklist must include every lifecycle column. If a field
    moves out of this set the PATCH endpoint would start accepting it,
    which is exactly the failure mode #110 guards against.
    """
    expected = {"id", "status", "pulled_bytes", "pulled_total", "last_error"}
    assert expected.issubset(_NEVER_PATCH)


def test_never_patch_names_exist_on_modelrow_dataclass():
    """Drift guard — every name in ``_NEVER_PATCH`` MUST correspond to a
    real ``ModelRow`` field (or be a known SQL-only column kept here
    for safety, like ``created_at``). A rename of a blocklisted field
    on ``ModelRow`` without updating ``_NEVER_PATCH`` would silently
    promote it back into the patchable set; this test catches that.
    """
    row_fields = {f.name for f in dataclasses.fields(ModelRow)}
    # ``created_at`` is the documented exception — it lives in the SQL
    # schema only, never decoded into ``ModelRow``, and stays on the
    # blocklist as a belt-and-suspenders measure should the dataclass
    # ever grow it.
    SQL_ONLY_EXCEPTIONS = {"created_at"}
    bad = [n for n in _NEVER_PATCH if n not in row_fields and n not in SQL_ONLY_EXCEPTIONS]
    assert not bad, (
        f"_NEVER_PATCH names {bad} are not ModelRow fields and not in "
        f"SQL_ONLY_EXCEPTIONS — did a column get renamed?"
    )


def test_patchable_fields_are_derived_minus_blocklist():
    """The allowlist is exactly (ModelRow fields) - _NEVER_PATCH. If a
    new column is added to ModelRow it becomes patchable by default;
    operators wanting to lock it MUST add it to _NEVER_PATCH (with a
    comment explaining why).
    """
    row_fields = {f.name for f in dataclasses.fields(ModelRow)}
    expected = row_fields - _NEVER_PATCH
    assert _PATCHABLE_MODEL_FIELDS == expected


def test_no_lifecycle_field_in_patchable_set():
    """Belt-and-suspenders — the lifecycle-owned columns MUST NOT be in
    the patchable set. Catches the worst-case regression even if the
    derivation logic itself gets refactored.
    """
    lifecycle = {"id", "status", "pulled_bytes", "pulled_total", "last_error", "updated_at"}
    leaked = _PATCHABLE_MODEL_FIELDS & lifecycle
    assert not leaked, f"lifecycle fields leaked into patch allowlist: {sorted(leaked)}"


@pytest.mark.parametrize(
    "field",
    [
        # The five fields that #85 and #106 added but the hand-maintained
        # allowlist never picked up. The fix's value to operators is
        # exactly that these become patchable.
        "filename",
        "parallelism_strategy",
        "max_batch_size",
        "hf_config_repo",
        "tokenizer_repo",
    ],
)
def test_newly_patchable_fields_are_in_allowlist(field):
    assert field in _PATCHABLE_MODEL_FIELDS, (
        f"{field} should be patchable post-#110 derivation"
    )
