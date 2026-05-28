# Tested model + engine stacks

This is a living record of model / vLLM-engine combinations that have been
**driven end-to-end through the UI** and confirmed to produce coherent output.
Each row was validated by loading the model, sending a reasoning prompt through
the **Chat Playground** (`/ui/chat`), and human-judging the answer — not by a
backend smoke probe. Every working combo is also saved as a reusable template
(visible in the "Try stack" panel and the template picker on model create).

## Validation hardware

All combos below were validated on host **d5** (`10.10.0.187`):

- 4× NVIDIA **RTX A4000**, 16 GiB each (~15.6 GiB usable), compute capability
  **sm_86** (Ampere).
- Driver = `DockerSocketDriver` (Docker-out-of-Docker): each engine runs as a
  sibling container `vllm-warden-engine-<model_id>`.
- Engine image channel `cuda-stable`, vLLM **0.20.0**
  (`vllm/vllm-openai:v0.20.0`) for every case.

Capability class matters: the quantization findings below are specific to
**sm_86 Ampere**. Hardware with native FP8 (sm_89 Ada / sm_90 Hopper) will
behave differently — re-validate before trusting an FP8 combo there.

## sm_86 quantization findings (read before picking a combo)

These are the rules that determined every model choice on A4000-class cards:

1. **A bf16 8B model OOMs on a single 16 GiB A4000.** Weights alone are
   ~15 GiB, leaving only a few hundred MiB — load dies during the
   `torch._inductor` autotuning allocation
   (`GPU 0 ... 359 MiB free ... tried to allocate 1002 MiB`). Either split
   across 2 GPUs (TP=2) or use a quantized checkpoint.

2. **`--quantization fp8` produces complete gibberish on sm_86.** FP8 weight
   quantization is *emulated* on Ampere and is numerically broken — the model
   loads and streams tokens, but the output is garbage
   (e.g. `presentforge Levine…HttpServlet…charCodeAt`). Do **not** ship an FP8
   combo for these cards. This also means the legacy `Qwen3.6-27B-FP8`
   checkpoint is not worth testing here — it would be gibberish regardless of
   engine version.

3. **AWQ INT4 is the working single-GPU path** for 7B–14B models. Coherent
   output, ~5.5–10 GiB of weights, fits one A4000 with headroom. Prefer a
   pre-quantized AWQ checkpoint over an unquantized one when you only have one
   card to give the model.

4. **`--enforce-eager` avoids the compile-time OOM and speeds load.** It skips
   `torch._inductor` autotuning, which is both where the bf16 OOM hits and a
   large chunk of load latency. Use it on memory-tight single-GPU loads.

> **Template limitation (file a follow-up if this bites you):** the "Save
> working combo as template" action captures `engine{channel,vllm_version,
> image}`, `max_model_len` and `tensor_parallel_size`, but **not**
> `extra_args` or the live `gpu_memory_utilization` override. A template saved
> from an `--enforce-eager` / `gpu_memory_utilization=0.92` run comes back with
> empty `extra_args` and `gpu_memory_utilization=0.9`. Re-apply those two
> overrides at model-create time until the save path is fixed.

## Validated stacks

All on `cuda-stable` / vLLM `0.20.0`, single host d5 (4× A4000 sm_86).

| Model (HF repo) | Quant | TP | GPUs | max_model_len | gpu_mem_util | extra_args | Coherence check |
|---|---|---|---|---|---|---|---|
| `openai/gpt-oss-20b` | mxfp4 (native) | 2 | 2× A4000 | 32000 | 0.7–0.9 | — | reasoning prompt, coherent |
| `Qwen/Qwen2.5-3B-Instruct` | bf16 | 1 | 1× A4000 | 8192 | 0.9 | — | reasoning prompt, coherent |
| `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | bf16 | 2 | 2× A4000 | 32768 | 0.9 | — | reasoning prompt, coherent |
| `mistralai/Mistral-7B-Instruct-v0.3` | bf16 | 2 | 2× A4000 | 8192 | 0.9 | — | reasoning prompt, coherent |
| `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` | AWQ INT4 | 1 | 1× A4000 | 8192 | 0.9 | `--enforce-eager` | Rayleigh-scattering Q, correct |
| `Qwen/Qwen2.5-14B-Instruct-AWQ` | AWQ INT4 | 1 | 1× A4000 | 8192 | 0.92 | `--enforce-eager` | sheep riddle ("all but 9"), answered **9** |

### Notes per family

- **gpt-oss-20b** — ships as a built-in template (`gpt-oss-20b`, TP=2, bf16,
  `gpu_memory_utilization=0.7`). The 20B MoE needs two A4000s; it does not fit
  one card. A user template at `gpu_memory_utilization=0.9` also validated.

- **Qwen2.5-3B** — small enough to run single-GPU at bf16 with no tricks; the
  baseline "does the single-GPU path work at all" case.

- **Nemotron-Nano-8B** and **Mistral-7B** — both bf16 at 8B/7B, so both were
  given **two** GPUs (TP=2) to clear the single-card weight ceiling. An AWQ
  checkpoint would let either run single-GPU (see Llama/Qwen below).

- **Llama-3.1-8B** — the decisive case for the sm_86 rules. bf16 single-GPU
  OOMed; FP8 single-GPU loaded but emitted gibberish; the AWQ-INT4 checkpoint
  (`hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`) ran single-GPU and
  answered correctly. Use the AWQ repo, not the base repo, on one A4000.

- **Qwen2.5-14B** — chosen as the "large Qwen" representative via its AWQ-INT4
  checkpoint so it fits a **single** A4000 (TP=1), avoiding a 2-GPU layout.
  Answered the "17 sheep, all but 9 run away" riddle with **9** and a correct
  one-sentence justification.

## How to reproduce a combo

1. Create the model (template picker, or paste the HF repo + set GPUs / TP /
   `max_model_len`). For AWQ single-GPU loads add `--enforce-eager` to
   `extra_args` and bump `gpu_memory_utilization` toward `0.92`.
2. **Pull**, then **Load**; wait for status `loaded`.
3. Open `/ui/chat`, pick the model, send a short reasoning prompt, and read the
   streamed answer. Judge coherence yourself — a model that streams fluent-
   looking tokens can still be numerically broken (see the FP8 finding).
4. If good, open the model page → **Try stack** → enter the channel + vLLM
   version → **Try combo** → **Mark working** → **Save working combo as
   template**. Remember to re-add `extra_args` / `gpu_memory_utilization` when
   you later instantiate from that template (see the limitation note above).
