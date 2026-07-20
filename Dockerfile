# TODO(release): replace digest below by running `docker pull vllm/vllm-openai:v0.25.1`
# and copying the sha256 from `docker images --digests vllm/vllm-openai`.
FROM vllm/vllm-openai@sha256:e4f88a835143cd22aee2397a26ec6bb80b3a4a6fe0c882bcbc63822904766089

WORKDIR /app

# GGUF support moved out of vLLM core into the OOT vllm-gguf-plugin as of the
# 0.25.x line (core vllm/model_executor/model_loader/gguf_loader.py no longer
# exists). Warden supports GGUF model rows, so the plugin is required. Pinned
# for reproducibility; bump together with the base image.
# NOTE: the plugin's adapter-based loader has no _get_gguf_weights_map rename
# table, so the qwen3_5/qwen3_5_moe/vision_num_layers patches we carried on the
# 0.20.0 base (#107/#108/#115, upstream vllm PR #38140 — still OPEN) are gone
# with it. Qwen3.5/3.6 GGUF loadability under the plugin is UNVERIFIED — if a
# Qwen3.x GGUF row regresses, that is the first place to look.
RUN pip install --no-cache-dir vllm-gguf-plugin==0.0.4

# vLLM 0.25.1's ModelConfig init calls override_quantization_method(quant_cfg,
# user_quant, hf_config=self.hf_config) for EVERY registered quantization method
# (vllm/config/model.py:1064 _verify_quantization). Core's base signature grew a
# third param `hf_config` (base_config.py) but the plugin's GGUFConfig classmethod
# is stuck on the old 2-arg signature, so the hf_config kwarg is a TypeError —
# raised during init for ALL models (even non-GGUF FP8 rows), breaking 100% of
# loads with rc=1. Plugin 0.0.4 is the latest on PyPI (no fixed release), so we
# patch the signature at build time to accept and ignore the extra kwarg. The
# body already discards hf_quant_cfg, so ignoring hf_config too is safe. The
# `s2 != s` guard fails the build loudly if the upstream line ever changes.
#
# The write and the verification run in SEPARATE python processes on purpose:
# resolving the module path with find_spec imports the plugin package, which
# imports this config module as a side effect at the OLD source, caching its
# compiled code object in sys.modules. An in-process re-import would then return
# the stale pre-patch object and the signature assert would spuriously fail even
# though the file on disk is patched correctly. A fresh interpreter compiles the
# patched source and sees the new signature — exactly what the runtime
# vllm-serve subprocess does. (Deleting *.pyc handles the on-disk bytecode cache;
# the separate process handles the in-memory one.)
RUN python3 - <<'PY'
import importlib.util as u
p = u.find_spec("vllm_gguf_plugin.quantization.config").origin
s = open(p).read()
s2 = s.replace(
    "cls, hf_quant_cfg: dict[str, Any], user_quant: str | None",
    "cls, hf_quant_cfg: dict[str, Any], user_quant: str | None, hf_config: Any = None",
)
assert s2 != s, "override_quantization_method signature not found to patch"
open(p, "w").write(s2)
print("gguf-plugin override_quantization_method source patched at", p)
PY
RUN find /usr/local/lib/python3.12/dist-packages/vllm_gguf_plugin -name '*.pyc' -delete
RUN python3 -c "import inspect; from vllm_gguf_plugin.quantization.config import GGUFConfig; sig = inspect.signature(GGUFConfig.override_quantization_method); assert 'hf_config' in sig.parameters, f'patch did not take: {sig}'; print('gguf-plugin override_quantization_method patched:', sig)"

# Strip the nixl_ep packages (NVIDIA cross-node expert-parallel all2all) that
# the v0.25.1 base image bundles. Their compiled CUDA extension dlopens an
# AVX-built UCX library (libucs/libucp) at *import* time; on a CPU without AVX
# the UCX load-time feature check aborts the process:
#   "FATAL: UCX library was compiled with avx but CPU does not support it."
# That is a C-level SIGSEGV, not a catchable Python exception. The pw_prod GPU
# node runs a QEMU vCPU with no AVX, so *every* model load segfaulted (rc=-11)
# on 0.25.1 until this. vLLM only guards the crashing import with
# has_nixl_ep() (a bare importlib.find_spec presence check in
# fused_moe/all2all_utils.py), so deleting the packages flips that guard to
# False and the AVX-UCX path is never touched. nixl_ep is cross-node
# expert-parallel — irrelevant to our single-node TP setups. Base `nixl`
# (no import-time UCX dlopen) is intentionally left in place.
RUN rm -rf /usr/local/lib/python3.12/dist-packages/nixl_ep* && \
    python3 -c "import importlib.util as u; assert u.find_spec('nixl_ep') is None, 'nixl_ep still importable after removal'"

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
