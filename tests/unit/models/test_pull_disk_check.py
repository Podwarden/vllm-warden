import pytest

from app.models.pull_task import (
    DiskShortage,
    insufficient_disk,
)


async def test_insufficient_disk_raises_when_estimate_exceeds_free(monkeypatch, tmp_path):
    from app.models import pull_task as pt
    monkeypatch.setattr(pt, "disk_free_bytes", lambda p: 10 * 1024 * 1024 * 1024)

    async def fake_estimate(*args, **kwargs):
        return 8 * 1024 * 1024 * 1024
    monkeypatch.setattr(pt, "estimate_repo_bytes", fake_estimate)

    with pytest.raises(DiskShortage) as exc:
        await insufficient_disk("Qwen/Qwen3.5-9B", "main", tmp_path, hf_token=None)
    assert "needs" in str(exc.value).lower()


async def test_insufficient_disk_passes_when_enough(monkeypatch, tmp_path):
    from app.models import pull_task as pt
    monkeypatch.setattr(pt, "disk_free_bytes", lambda p: 100 * 1024 * 1024 * 1024)

    async def fake_estimate(*args, **kwargs):
        return 8 * 1024 * 1024 * 1024
    monkeypatch.setattr(pt, "estimate_repo_bytes", fake_estimate)

    await insufficient_disk("o/r", "main", tmp_path, hf_token=None)
