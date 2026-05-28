import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.runtime.env_builder import (
    ALLOWED_ENV_EXACT,
    ALLOWED_ENV_PREFIXES,
    HARD_LOCKED_ENV_KEYS,
)

SLUG_RE = re.compile(r"^[a-zA-Z0-9._-]+$")


class ModelCreate(BaseModel):
    served_model_name: str = Field(..., min_length=1, max_length=100)
    # #162 — ``hf_repo`` is optional when a template supplies it. The route
    # enforces presence after merging template + body (explicit body wins).
    hf_repo: str | None = Field(default=None, pattern=r"^[\w.-]+/[\w.-]+$")
    hf_revision: str = "main"
    # #162 — engine templates. ``template_id`` selects a builtin/user template
    # whose fields prefill the model; the ``engine_*`` trio overrides the
    # template's engine axis (channel, vLLM version, optional pinned image).
    template_id: str | None = None
    engine_channel: str | None = None
    engine_vllm_version: str | None = None
    engine_image: str | None = None
    gpu_indices: list[int] = Field(..., min_length=1)
    tensor_parallel_size: int | None = None
    dtype: str | None = None
    max_model_len: int | None = Field(None, gt=0)
    gpu_memory_utilization: float = Field(0.9, gt=0, le=1.0)
    trust_remote_code: bool = False
    extra_args: list[str] = Field(default_factory=list)
    extra_env: dict[str, str] = Field(default_factory=dict)
    # New for #85 — per-file download + parallelism strategy + KV batch sizing.
    # ``filename`` narrows the pull to one weights file + tokenizer/config;
    # unset (None) preserves the legacy "pull the whole repo" path.
    filename: str | None = Field(None, min_length=1, max_length=512)
    # ``parallelism_strategy`` is exposed so the wizard can record the user's
    # choice; the runtime wiring (tp vs pp flags on vLLM) is downstream in
    # #82.5. ``auto`` is the safe default — single GPU is no-parallelism,
    # multi-GPU defaults to TP.
    parallelism_strategy: Literal["tp", "pp", "auto"] = "auto"
    # ``max_batch_size`` feeds into the KV-reserve math (see app/models/fit.py).
    # 1 = single-request, the wizard's default; 64 is a sane upper bound that
    # already overruns most consumer cards' KV budget.
    max_batch_size: int = Field(default=1, ge=1, le=64)
    # New for #106 — see migration 0015. ``hf_config_repo`` populates the vLLM
    # ``--hf-config-path`` flag (required when a GGUF repo omits config.json,
    # e.g. unsloth republishes). ``tokenizer_repo`` populates ``--tokenizer``
    # for the same upstream-vs-quant split. Reuse the ``hf_repo`` regex.
    # Both are optional; the wizard defaults them from a single "Base repo"
    # Input, and empty strings normalise to None so a cleared field round-
    # trips correctly.
    hf_config_repo: str | None = Field(
        default=None, pattern=r"^[\w.-]+/[\w.-]+$"
    )
    tokenizer_repo: str | None = Field(
        default=None, pattern=r"^[\w.-]+/[\w.-]+$"
    )

    @field_validator("served_model_name")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not SLUG_RE.match(v):
            raise ValueError("served_model_name must be alphanumeric/dot/dash/underscore")
        return v

    @field_validator("hf_config_repo", "tokenizer_repo", mode="before")
    @classmethod
    def _empty_string_to_none(cls, v: str | None) -> str | None:
        """Treat empty strings as unset (#106).

        The wizard's "Base repo" Input ships an empty string when the operator
        clears the field — without this the ``pattern=`` regex would reject
        it. Normalising here means the API surface accepts both ``""`` (UI)
        and ``null`` (programmatic) for "no override".
        """
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @field_validator("gpu_indices")
    @classmethod
    def _unique_gpus(cls, v: list[int]) -> list[int]:
        if len(set(v)) != len(v):
            raise ValueError("gpu_indices must be unique")
        if any(i < 0 for i in v):
            raise ValueError("gpu_indices must be >= 0")
        return v

    @field_validator("extra_env")
    @classmethod
    def _extra_env_allowlist(cls, v: dict[str, str]) -> dict[str, str]:
        """Enforce the same allowlist that runs at subprocess-spawn time.

        Defense-in-depth: today _filter_extra_env raises on hard-locked keys and
        silently drops unknown keys only when load() runs. By enforcing here, the
        API rejects bad extra_env synchronously at write time so operators get an
        immediate, descriptive error instead of mysteriously-missing env at load.
        """
        for key in v:
            if key in HARD_LOCKED_ENV_KEYS:
                raise ValueError(
                    f"extra_env key '{key}' is hard-locked and cannot be set via API"
                )
            if not (
                any(key.startswith(p) for p in ALLOWED_ENV_PREFIXES)
                or key in ALLOWED_ENV_EXACT
            ):
                raise ValueError(
                    f"extra_env key '{key}' is not in the allowlist "
                    f"(prefix one of {sorted(ALLOWED_ENV_PREFIXES)} "
                    f"or exact match in {sorted(ALLOWED_ENV_EXACT)})"
                )
        return v

    @model_validator(mode="after")
    def _tp_consistent(self) -> "ModelCreate":
        if self.tensor_parallel_size is None:
            object.__setattr__(self, "tensor_parallel_size", len(self.gpu_indices))
        elif self.tensor_parallel_size != len(self.gpu_indices):
            raise ValueError(
                f"tensor_parallel_size={self.tensor_parallel_size} "
                f"must equal len(gpu_indices)={len(self.gpu_indices)}"
            )
        return self


class EngineSpecBody(BaseModel):
    channel: str
    vllm_version: str
    image: str | None = None


class TemplateCreate(BaseModel):
    # ``model_id`` (below) sits in pydantic's reserved ``model_`` namespace;
    # opt out of the protection so the field name stays meaningful and no
    # spurious UserWarning is emitted at import.
    model_config = ConfigDict(protected_namespaces=())

    id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9._-]+$")
    label: str = Field(..., min_length=1, max_length=200)
    hf_repo: str = Field(..., pattern=r"^[\w.-]+/[\w.-]+$")
    hf_revision: str = "main"
    dtype: str = "auto"
    # Defaults let a try-stack "save working combo" POST omit these — that flow
    # only captures the engine axis + repo. The UI sends the model's actual
    # values when known; these fallbacks apply when the model left them unset.
    max_model_len: int = Field(8192, gt=0)
    tensor_parallel_size: int = Field(1, ge=1)
    gpu_memory_utilization: float = Field(0.9, gt=0, le=1.0)
    trust_remote_code: bool = False
    extra_args: list[str] = Field(default_factory=list)
    extra_env: dict[str, str] = Field(default_factory=dict)
    engine: EngineSpecBody | None = None
    # #170 — the try-stack "save working combo" flow references the live model
    # whose combo was just validated. When set, the route sources the model's
    # ACTUAL ``extra_args`` + ``gpu_memory_utilization`` (and other tuning) from
    # the live model row, so a saved AWQ template keeps e.g. --enforce-eager +
    # gpu_memory_utilization=0.92 instead of silently falling back to defaults.
    # Explicit body fields still win over the live row.
    model_id: str | None = None


class TryStackRequest(BaseModel):
    channel: str = Field(..., min_length=1)
    vllm_version: str = Field(..., min_length=1)
    image: str | None = None


class TryStackResult(BaseModel):
    result: Literal["ok", "failed"]
    error: str | None = None
