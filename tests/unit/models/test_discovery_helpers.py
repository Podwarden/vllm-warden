"""Pure-helper unit tests for the discovery module (#84).

These cover filename-driven classification, quant parsing and parameter-count
parsing. They run without any HF I/O — the helpers are pure string ops so the
fixtures here are hand-written to cover the file-naming conventions we
actually see in the wild on HuggingFace today.
"""
from __future__ import annotations

import pytest

from app.models.discovery import _classify, _parse_params, _parse_quant


@pytest.mark.parametrize(
    "filename,expected",
    [
        # Sharded safetensors — the most common large-model layout.
        ("model-00001-of-00004.safetensors", "safetensors_sharded"),
        ("model-00004-of-00004.safetensors", "safetensors_sharded"),
        # Single-file safetensors.
        ("model.safetensors", "safetensors_single"),
        ("Qwen3.6-27B.safetensors", "safetensors_single"),
        # GGUF quantizations.
        ("Qwen3.6-27B-Q5_K_M.gguf", "gguf"),
        ("llama-2-7b-chat.Q4_K_M.gguf", "gguf"),
        # Legacy PyTorch checkpoints.
        ("pytorch_model.bin", "pytorch_bin"),
        ("pytorch_model-00001-of-00002.bin", "pytorch_bin"),
        # Config / tokenizer metadata.
        ("config.json", "config"),
        ("tokenizer.model", "tokenizer"),
        ("tokenizer.json", "tokenizer"),
        ("tokenizer_config.json", "tokenizer"),
        ("special_tokens_map.json", "tokenizer"),
        # Fallthrough — README, LICENSE, .md, junk we don't understand.
        ("README.md", "other"),
        ("LICENSE", "other"),
        ("random.bin.tmp", "other"),
    ],
)
def test_classify(filename, expected):
    assert _classify(filename) == expected


@pytest.mark.parametrize(
    "filename,expected",
    [
        # GGUF quant tags — case-insensitive in the input, returned canonical.
        ("Qwen3.6-27B-Q5_K_M.gguf", "Q5_K_M"),
        ("llama-2-7b-chat.Q4_K_M.gguf", "Q4_K_M"),
        ("model.Q8_0.gguf", "Q8_0"),
        ("model.Q2_K.gguf", "Q2_K"),
        ("model.Q6_K.gguf", "Q6_K"),
        ("model.q3_k_m.gguf", "Q3_K_M"),  # lowercase normalised
        # safetensors variants — AWQ / GPTQ / FP16 markers in filename.
        ("model-awq.safetensors", "AWQ"),
        ("model.AWQ.safetensors", "AWQ"),
        ("model-gptq-4bit.safetensors", "GPTQ"),
        ("model.fp16.safetensors", "FP16"),
        # No quant signal — return None rather than guess.
        ("model.safetensors", None),
        ("pytorch_model.bin", None),
        ("config.json", None),
        ("README.md", None),
    ],
)
def test_parse_quant(filename, expected):
    assert _parse_quant(filename) == expected


@pytest.mark.parametrize(
    "filename,expected",
    [
        # Standard "{Name}-{N}B" layout.
        ("Llama-2-7B-chat.safetensors", 7_000_000_000),
        ("Qwen-72B-Chat.safetensors", 72_000_000_000),
        ("Qwen3.6-27B-Q5_K_M.gguf", 27_000_000_000),
        ("Mistral-7B-Instruct-v0.2.safetensors", 7_000_000_000),
        # Sub-billion ("M") tags.
        ("phi-1_5-350M.safetensors", 350_000_000),
        ("smol-125m.gguf", 125_000_000),
        # Fractional billion ("1.3B", "0.5B").
        ("model-1.3B.safetensors", 1_300_000_000),
        ("Qwen2-0.5B-Instruct.safetensors", 500_000_000),
        # No param signal.
        ("model.safetensors", None),
        ("config.json", None),
        ("tokenizer.json", None),
        ("pytorch_model.bin", None),
    ],
)
def test_parse_params(filename, expected):
    assert _parse_params(filename) == expected
