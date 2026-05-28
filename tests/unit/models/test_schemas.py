import pytest
from pydantic import ValidationError

from app.models.schemas import ModelCreate


def test_model_create_minimal():
    m = ModelCreate(
        served_model_name="qwen3.5-9b",
        hf_repo="Qwen/Qwen3.5-9B",
        gpu_indices=[1, 2],
    )
    assert m.tensor_parallel_size == 2
    assert m.hf_revision == "main"


def test_model_create_tp_must_match_gpu_count_when_unset_default_to_len():
    m = ModelCreate(
        served_model_name="x",
        hf_repo="o/r",
        gpu_indices=[0, 1, 2],
    )
    assert m.tensor_parallel_size == 3


def test_model_create_explicit_tp_must_match():
    with pytest.raises(ValidationError):
        ModelCreate(
            served_model_name="x", hf_repo="o/r",
            gpu_indices=[0, 1, 2],
            tensor_parallel_size=2,
        )


def test_model_create_rejects_empty_gpus():
    with pytest.raises(ValidationError):
        ModelCreate(served_model_name="x", hf_repo="o/r", gpu_indices=[])


def test_model_create_served_name_slug():
    with pytest.raises(ValidationError):
        ModelCreate(served_model_name="bad name!", hf_repo="o/r", gpu_indices=[0])


# ---------------------------------------------------------------------------
# extra_env allowlist enforcement at the API boundary (defense-in-depth)
# ---------------------------------------------------------------------------

def test_extra_env_rejects_hard_locked_cuda_visible_devices():
    with pytest.raises(ValidationError, match="hard-locked"):
        ModelCreate(
            served_model_name="x", hf_repo="o/r", gpu_indices=[0],
            extra_env={"CUDA_VISIBLE_DEVICES": "9"},
        )


def test_extra_env_rejects_hard_locked_path():
    with pytest.raises(ValidationError, match="hard-locked"):
        ModelCreate(
            served_model_name="x", hf_repo="o/r", gpu_indices=[0],
            extra_env={"PATH": "/evil"},
        )


def test_extra_env_rejects_unknown_key():
    with pytest.raises(ValidationError, match="allowlist"):
        ModelCreate(
            served_model_name="x", hf_repo="o/r", gpu_indices=[0],
            extra_env={"FOO_BAR": "1"},
        )


def test_extra_env_accepts_allowed_prefix():
    m = ModelCreate(
        served_model_name="x", hf_repo="o/r", gpu_indices=[0],
        extra_env={"VLLM_USE_V1": "1"},
    )
    assert m.extra_env == {"VLLM_USE_V1": "1"}


def test_extra_env_accepts_exact_match_cuda_module_loading():
    m = ModelCreate(
        served_model_name="x", hf_repo="o/r", gpu_indices=[0],
        extra_env={"CUDA_MODULE_LOADING": "LAZY"},
    )
    assert m.extra_env == {"CUDA_MODULE_LOADING": "LAZY"}


def test_extra_env_empty_dict_accepted():
    m = ModelCreate(
        served_model_name="x", hf_repo="o/r", gpu_indices=[0],
    )
    assert m.extra_env == {}


# ---------------------------------------------------------------------------
# #106: hf_config_repo / tokenizer_repo plumbing for GGUF repos that omit
# config.json. Both share the hf_repo owner/name slug regex; empty strings
# normalise to None so a cleared FE field round-trips correctly.
# ---------------------------------------------------------------------------


def test_hf_config_and_tokenizer_repo_accepted():
    """Happy path — both fields accept owner/name slugs and round-trip."""
    m = ModelCreate(
        served_model_name="x", hf_repo="unsloth/Qwen3-GGUF", gpu_indices=[0],
        hf_config_repo="Qwen/Qwen3-30B-A3B",
        tokenizer_repo="Qwen/Qwen3-30B-A3B",
    )
    assert m.hf_config_repo == "Qwen/Qwen3-30B-A3B"
    assert m.tokenizer_repo == "Qwen/Qwen3-30B-A3B"


def test_hf_config_repo_empty_string_normalises_to_none():
    """The FE Input ships an empty string when the operator clears the field.
    The before-validator must coerce both empty and whitespace-only to None
    so the pattern regex doesn't reject a cleared field."""
    m = ModelCreate(
        served_model_name="x", hf_repo="o/r", gpu_indices=[0],
        hf_config_repo="",
        tokenizer_repo="   ",
    )
    assert m.hf_config_repo is None
    assert m.tokenizer_repo is None


def test_hf_config_repo_rejects_bad_slug():
    """Non-owner/name strings must still be rejected — slug validation is the
    whole point of the field over a free-text path."""
    with pytest.raises(ValidationError):
        ModelCreate(
            served_model_name="x", hf_repo="o/r", gpu_indices=[0],
            hf_config_repo="not-a-slug",
        )
