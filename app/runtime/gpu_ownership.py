from threading import Lock


class GpuConflict(RuntimeError):
    pass


class GpuOwnership:
    """In-memory exclusive GPU ownership: gpu_idx -> model_id."""

    def __init__(self) -> None:
        self._owner: dict[int, str] = {}
        self._lock = Lock()

    def claim(self, model_id: str, gpu_indices: list[int]) -> None:
        with self._lock:
            conflicts = [g for g in gpu_indices if g in self._owner and self._owner[g] != model_id]
            if conflicts:
                raise GpuConflict(f"GPUs {conflicts} already claimed")
            for g in gpu_indices:
                self._owner[g] = model_id

    def release(self, model_id: str) -> None:
        with self._lock:
            self._owner = {g: m for g, m in self._owner.items() if m != model_id}

    def owner_of(self, gpu_idx: int) -> str | None:
        with self._lock:
            return self._owner.get(gpu_idx)

    def all_claims(self) -> dict[int, str]:
        with self._lock:
            return dict(self._owner)
