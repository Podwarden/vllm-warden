import pytest

from app.runtime.gpu_ownership import GpuConflict, GpuOwnership


def test_claim_grants_exclusive_ownership():
    g = GpuOwnership()
    g.claim("m1", [0, 1])
    assert g.owner_of(0) == "m1"
    assert g.owner_of(1) == "m1"
    assert g.owner_of(2) is None


def test_claim_conflict_raises():
    g = GpuOwnership()
    g.claim("m1", [0, 1])
    with pytest.raises(GpuConflict) as ei:
        g.claim("m2", [1, 2])
    assert "1" in str(ei.value)
    assert g.owner_of(2) is None


def test_release_frees_gpus():
    g = GpuOwnership()
    g.claim("m1", [0, 1])
    g.release("m1")
    assert g.owner_of(0) is None
    g.claim("m2", [0, 1])
    assert g.owner_of(0) == "m2"


def test_release_unknown_is_noop():
    g = GpuOwnership()
    g.release("does-not-exist")
