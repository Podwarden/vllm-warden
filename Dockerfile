# syntax=docker/dockerfile:1.7
# TODO(release): replace digest below by running `docker pull vllm/vllm-openai:v0.20.0`
# and copying the sha256 from `docker images --digests vllm/vllm-openai`.
FROM vllm/vllm-openai@sha256:04563c302537a91aa49ebdfbceda96111c5712275999b7e8804fa598f0b5641d

WORKDIR /app

# In-place patch for qwen3_5 GGUF model_type translation. vLLM 0.20.0 (pinned
# above) ships with model_type mappings for qwen2_moe / qwen3_moe / gemma3_text
# / cohere / deepseek_v3 etc. but is missing the qwen3_5 → qwen35 rename,
# which blocks loading unsloth/Qwen3.6-27B-GGUF and similar Qwen3.5/3.6 GGUFs.
# Upstream fix: https://github.com/vllm-project/vllm/pull/38140 (OPEN, not in any release).
# Remove this RUN once the fix is merged and we bump to a vLLM release that contains it.
# Mirrors the existing gemma3_text → gemma3 rename pattern in the same function.
RUN sed -i '/^            model_type = "gemma3"$/a\        if model_type == "qwen3_5":\n            # Qwen3.5 uses "qwen3_5" in HuggingFace but "qwen35" in GGUF arch naming.\n            # Upstream fix: https://github.com/vllm-project/vllm/pull/38140 (remove when merged + released).\n            model_type = "qwen35"' \
        /usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/gguf_loader.py && \
    grep -q 'model_type == "qwen3_5"' /usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/gguf_loader.py && \
    python3 -m py_compile /usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/gguf_loader.py

# In-place patch for qwen3_5_moe GGUF model_type translation (vllm-warden#115).
# Companion to the qwen3_5 patch above (#107). Qwen3.6-35B-A3B and other Qwen3.5/3.6
# MoE GGUFs declare `model_type: "qwen3_5_moe"` / `architectures:
# ["Qwen3_5MoeForConditionalGeneration"]` in config.json. vLLM v0.20.0:
#   - DOES ship the Qwen3_5MoeForConditionalGeneration model class
#     (vllm/model_executor/models/qwen3_5.py, registry.py line ~541).
#   - DOES ship gguf-py MODEL_ARCH.QWEN35MOE='qwen35moe' with the correct fused
#     experts.gate_up_proj / experts.down_proj tensor name map that qwen3_5.py
#     expects (different from qwen2_moe/qwen3_moe split-expert layout).
#   - Only LACKS the model_type rename "qwen3_5_moe" -> "qwen35moe" in
#     gguf_loader._get_gguf_weights_map, so the arch lookup at line 204 raises
#     `Unknown gguf model_type: qwen3_5_moe`.
# Because the fused tensor map is covered natively by gguf.get_tensor_name_map(
# MODEL_ARCH.QWEN35MOE, ...), we do NOT install a manual gguf_to_hf_name_map
# block here (unlike the qwen2_moe/qwen3_moe and deepseek_v3 branches in the
# same function, which target the older split-expert format).
# Anchor: the line `            model_type = "qwen35"` is the unique tail of
# the patch1 insertion above, so this sed only matches AFTER patch1 has applied.
# Remove this RUN once an upstream PR adds the rename and we bump the base.
RUN sed -i '/^            model_type = "qwen35"$/a\        if model_type == "qwen3_5_moe":\n            # Qwen3.5 MoE uses "qwen3_5_moe" in HuggingFace but "qwen35moe" in GGUF arch naming.\n            # Bundled gguf-py MODEL_ARCH.QWEN35MOE already maps the fused experts.gate_up_proj /\n            # experts.down_proj tensors that vllm/model_executor/models/qwen3_5.py expects, so we\n            # intentionally skip the per-expert gguf_to_hf_name_map block used for qwen2_moe / qwen3_moe.\n            # Closes vllm-warden#115. No upstream PR yet (PR #38140 covers qwen3_5 dense only).\n            model_type = "qwen35moe"' \
        /usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/gguf_loader.py && \
    grep -q 'model_type == "qwen3_5_moe"' /usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/gguf_loader.py && \
    python3 -m py_compile /usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/gguf_loader.py

# In-place patch for gguf_loader.py:214 vision_num_layers crash on Qwen3.5 GGUFs.
# After the v17.19 qwen3_5 → qwen35 rename above resolves the architecture, the
# next thing _get_gguf_weights_map does is read `config.vision_config.num_hidden_layers`.
# Qwen3_5VisionConfig (transformers 5.6.2,
# transformers/models/qwen3_5/configuration_qwen3_5.py:118-140) stores the vision
# layer count as `depth: int = 27` and does NOT expose `num_hidden_layers` — same
# convention as Qwen2.5-VL. Without this patch, multimodal Qwen3.5/3.6 GGUFs
# (e.g. unsloth/Qwen3.6-27B-GGUF) crash at load with:
#   AttributeError: 'Qwen3_5VisionConfig' object has no attribute 'num_hidden_layers'
# (closes vllm-warden#108). The same upstream PR (vllm#38140) is the long-term
# fix; remove this RUN once we bump the vLLM base image past that merge.
# Idempotent: the guard `grep -q` short-circuits if the patched line is already
# present, so repeated builds (e.g. layer-cache invalidation upstream) re-run
# this step safely.
RUN if ! grep -q 'getattr(config.vision_config, "num_hidden_layers"' /usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/gguf_loader.py ; then \
        sed -i 's|^            vision_num_layers = config\.vision_config\.num_hidden_layers$|            vision_num_layers = getattr(config.vision_config, "num_hidden_layers", None) or config.vision_config.depth|' \
            /usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/gguf_loader.py ; \
    fi && \
    grep -q 'config\.vision_config\.depth' /usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/gguf_loader.py && \
    python3 -m py_compile /usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/gguf_loader.py

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY app /app/app

# Build-time identity (spec 2026-05-13 §Version surfacing, P1-9). The CI
# build job passes these via --build-arg from $CI_COMMIT_TAG /
# $CI_COMMIT_SHORT_SHA; local builds may pass them too. When unset, GET
# /api/version falls back to "dev" / "unknown" (see app/system/routes_version.py).
ARG VW_BUILD_VERSION=dev
ARG VW_BUILD_SHA=unknown
ENV VW_BUILD_VERSION=${VW_BUILD_VERSION} \
    VW_BUILD_SHA=${VW_BUILD_SHA}

VOLUME ["/data"]

EXPOSE 8080

# Override vllm/vllm-openai's ENTRYPOINT — we run our own FastAPI app
ENTRYPOINT []
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
