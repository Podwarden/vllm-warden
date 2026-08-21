# TODO(release): replace the digest in VW_BASE_DIGEST below by running
# `docker pull vllm/vllm-openai:v0.26.0` and copying the sha256 from
# `docker images --digests vllm/vllm-openai`.
#
# v0.26.0 (was v0.25.1, sha256:e4f88a83...). Upgraded 2026-08-17 to diagnose an
# EngineCore that dies mid-serving with NO traceback, NO CUDA error and no OOM
# kill (node-wide /proc/vmstat oom_kill stayed 0 across nine deaths). v0.26.0
# adds "log worker exit code when a process dies unexpectedly" (#38641), which
# names the fatal signal instead of leaving us to infer it. Also carries a
# DSv3.2 + MTP sequence-parallel accuracy fix; this deployment runs MTP.
#
# Declared as an ARG, not inlined into FROM, so this line is the SINGLE place
# the base digest lives. CI's post-build cleanup greps it out of this file to
# decide which vllm/vllm-openai layer sets are superseded and safe to evict
# (vllm-warden#208). Inlining it would let the Dockerfile and the cleanup drift,
# and a cleanup that reads a stale digest deletes the base you are still using.
ARG VW_BASE_DIGEST=sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52
FROM vllm/vllm-openai@${VW_BASE_DIGEST}

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

# Bump NCCL 2.28.9 -> 2.30.4 (the base image ships 2.28.9).
#
# NCCL 2.28.9 deadlocks on CUDA-graph replay containing NCCL collectives:
# the collective never returns, the TP worker stops answering, and vLLM's
# watchdog eventually fires `TimeoutError: RPC call to sample_tokens timed
# out` -> EngineDeadError. Workers stay ALIVE with no traceback, no CUDA
# error and oom_kill=0, so every memory theory reads clean. Upstream
# vllm#52504 verified deadlock on 2.28.9 and clean on 2.30.4 with nothing
# else changed; `--enforce-eager` also avoids it, which is what isolated the
# hang to captured-graph collectives rather than the collectives themselves.
#
# pw_prod ran exactly 2.28.9 with cudagraph_mode=FULL_AND_PIECEWISE and TP=4,
# and died with that signature three times (2026-08-02, and twice on
# 2026-08-18). Note the `sample_tokens` in the message is a red herring — the
# sampler is innocent; the hang is upstream of it in the collective.
#
# torch 2.11.0+cu130 hard-pins nvidia-nccl-cu13==2.28.9, so pip prints an
# incompatibility warning. It is metadata-only: NCCL keeps ABI compatibility
# within major 2, and torch dlopens libnccl.so.2 from THIS package directory
# (confirmed via /proc/self/maps) rather than the apt-installed
# /usr/lib/x86_64-linux-gnu copy, which is left untouched.
#
# The assertion checks the wheel version AND that torch still loads the lib
# from this package -- do NOT "verify" with torch.cuda.nccl.version(), which
# reports the version torch was COMPILED against (still 2.28.9) and happily
# reports success on a machine where nothing was upgraded.
RUN pip install --no-cache-dir "nvidia-nccl-cu13==2.30.4" && \
    python3 -c "import importlib.metadata as m, torch; v = m.version('nvidia-nccl-cu13'); assert v == '2.30.4', 'nccl wheel is ' + v; libs = [l.split()[-1] for l in open('/proc/self/maps') if 'libnccl.so' in l]; assert libs, 'torch mapped no libnccl'; assert 'dist-packages/nvidia/nccl' in libs[0], 'torch loads libnccl from ' + libs[0] + ' -- the upgraded wheel is NOT the runtime library'; print('nccl runtime lib:', libs[0], v)"

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
