export type RestartKind = 'none' | 'model-reload' | 'warden-restart';

export interface FieldHint {
  label: string;
  hint: string;
  restart: RestartKind;
}

export const RUNTIME_HINTS: Record<string, FieldHint> = {
  admin_username: {
    label: 'Admin username',
    hint: 'The single operator account. Used for the login page.',
    restart: 'none',
  },
  admin_password: {
    label: 'Admin password',
    hint: 'Updates the bcrypt hash. All sessions invalidated on change.',
    restart: 'none',
  },
  hf_token: {
    label: 'Hugging Face token',
    hint: 'Passed to vLLM via `HF_TOKEN` env. Required for gated repos (Llama, gpt-oss, Mistral, etc.). New value applies to next model load.',
    restart: 'model-reload',
  },
  default_gpu_indices: {
    label: 'Default GPU indices',
    hint: 'Pre-selected when adding a new model. Check the GPUs to include by default.',
    restart: 'none',
  },
  default_token_expiration_days: {
    label: 'Default token expiration',
    hint: 'Pre-fills the new-token dialog. Affects new tokens only.',
    restart: 'none',
  },
  rotation_grace_hours: {
    label: 'Rotation grace window',
    hint: 'When you rotate a token, the old one stays valid for this many hours. Lets you swap creds in your clients without downtime.',
    restart: 'none',
  },
  session_access_ttl_minutes: {
    label: 'Session access TTL',
    hint: 'How long a login JWT stays valid before refresh. Short = safer. Existing tokens keep their original TTL.',
    restart: 'warden-restart',
  },
  session_refresh_ttl_days: {
    label: 'Session refresh TTL',
    hint: 'How long until forced re-login. Existing refresh cookies keep their original TTL.',
    restart: 'warden-restart',
  },
  sse_ticket_ttl_seconds: {
    label: 'SSE ticket TTL',
    hint: 'How long a single SSE connect ticket is valid. SSE streams stay open after auth; only the initial connect needs the ticket.',
    restart: 'none',
  },
  vllm_version: {
    label: 'vLLM version',
    hint: 'vLLM Python package version. Warden container rebuild required (image baked at build time).',
    restart: 'warden-restart',
  },
  log_retention_lines: {
    label: 'Log retention',
    hint: 'Per-model log buffer size kept in memory + on disk. Applied to new lines after change.',
    restart: 'none',
  },
  landing_page_enabled: {
    label: 'Public landing page',
    hint: 'When enabled, the unified-port root (`https://your-warden/`) serves a public HTML landing page with links to /ui, the source repo, and podwarden.com. Disable for a private deployment that should 404 at the root.',
    restart: 'none',
  },
  public_url: {
    label: 'Public URL',
    hint: 'External base URL used in client-facing snippets (curl examples, OpenAI client configs). Leave unset to use the browser address bar — that\'s the right default unless this warden sits behind a reverse proxy whose external URL differs from what your browser sees.',
    restart: 'none',
  },
};

// ---------------------------------------------------------------------------
// Per-tab key sets for the #154 settings redesign.
//
// The /settings page is split into five tabs: General → Networking →
// Sessions & Tokens → Maintenance → Model. The first four are backed by
// these RUNTIME_HINTS keys; the fifth (Model) is a navigation pivot with
// no fields of its own.
//
// Each tab imports its own slice in declared render order. The
// `settings-tab-membership.test.ts` contract test pins that:
//   * every key in RUNTIME_HINTS appears in exactly one of these arrays
//     (no orphans, no duplicates);
//   * the four arrays together equal Object.keys(RUNTIME_HINTS).
// Adding a new RUNTIME_HINTS entry without placing it in one of these
// arrays fails CI — that's the point.
// ---------------------------------------------------------------------------

export const RUNTIME_GENERAL_KEYS = [
  // Identity
  'admin_username',
  'admin_password',
  // Hugging Face
  'hf_token',
  // Defaults for new models
  'default_gpu_indices',
] as const;

export const RUNTIME_NETWORKING_KEYS = [
  // Public access
  'public_url',
  'landing_page_enabled',
] as const;

export const RUNTIME_SESSIONS_KEYS = [
  // Browser session
  'session_access_ttl_minutes',
  'session_refresh_ttl_days',
  // Token defaults
  'default_token_expiration_days',
  'rotation_grace_hours',
  // Streaming
  'sse_ticket_ttl_seconds',
] as const;

export const RUNTIME_MAINTENANCE_KEYS = [
  // vLLM runtime
  'vllm_version',
  // Logs
  'log_retention_lines',
] as const;

export const MODEL_HINTS: Record<string, FieldHint> = {
  served_model_name: {
    label: 'Served model name',
    hint: 'The name clients pass in `model:` for `/v1/completions`. Slug only — current backend regex allows alphanumeric + `.`, `_`, `-`. Dots permitted because vLLM model names like `Mixtral-8x7B-v0.1` are common. Frontend mirrors that regex client-side for instant validation.',
    restart: 'model-reload',
  },
  hf_repo: {
    label: 'Hugging Face repo',
    hint: 'Hugging Face repo path, e.g. `facebook/opt-125m` or `openai/gpt-oss-20b`.',
    restart: 'model-reload',
  },
  hf_revision: {
    label: 'HF revision',
    hint: 'Branch, tag, or commit SHA. Default `main` — pin to a SHA for reproducibility.',
    restart: 'model-reload',
  },
  gpu_indices: {
    label: 'GPU indices',
    hint: 'Which GPUs this model loads on. At least one required. Determines `CUDA_VISIBLE_DEVICES` for the vLLM subprocess.',
    restart: 'model-reload',
  },
  tensor_parallel_size: {
    label: 'Tensor-parallel size',
    hint: 'Number of GPUs that shard each layer\'s weights. Auto-set to `len(gpu_indices)` because we only support **tensor-parallel** today. Data-parallel and pipeline-parallel are out of scope.',
    restart: 'model-reload',
  },
  gpu_memory_utilization: {
    label: 'GPU memory utilization',
    hint: 'Fraction of VRAM vLLM may consume per GPU (0.0–1.0, default 0.9). Lower if you hit OOM during paged-attention warmup.',
    restart: 'model-reload',
  },
  dtype: {
    label: 'dtype',
    hint: 'One of `auto`, `float16`, `bfloat16`, `float32`. `auto` follows the model\'s config.',
    restart: 'model-reload',
  },
  quantization: {
    label: 'Quantization',
    hint: 'One of `none`, `awq`, `gptq`, `fp8`, `bitsandbytes`. Most models ship with weights pre-quantized — leave `none` unless you know otherwise.',
    restart: 'model-reload',
  },
  kv_cache_dtype: {
    label: 'KV cache dtype',
    hint: 'One of `auto`, `fp8`, `fp8_e5m2`. `fp8` halves KV cache memory; tiny accuracy impact.',
    restart: 'model-reload',
  },
  max_model_len: {
    label: 'Max model length',
    hint: 'Max sequence length (prompt + generation). Lower = less KV cache memory; higher = fits longer contexts. Capped by model\'s training context.',
    restart: 'model-reload',
  },
  block_size: {
    label: 'Block size',
    hint: 'Affects GPU memory granularity; 16 is correct for nearly all deployments. Changing it requires a model reload and invalidates the KV cache.',
    restart: 'model-reload',
  },
  swap_space: {
    label: 'Swap space (GiB)',
    hint: 'GiB of CPU RAM to spill KV cache to under pressure. Default 4. 0 = disable.',
    restart: 'model-reload',
  },
  max_num_seqs: {
    label: 'Max concurrent seqs',
    hint: 'Max concurrent sequences. Higher = more throughput, more memory.',
    restart: 'model-reload',
  },
  max_num_batched_tokens: {
    label: 'Max batched tokens',
    hint: 'Per-step token budget across all sequences. Default = `max_model_len`. Lower to bound step latency.',
    restart: 'model-reload',
  },
  enforce_eager: {
    label: 'Enforce eager',
    hint: 'If true, disable CUDA graphs. Useful for debugging and tiny models; ~5% slower.',
    restart: 'model-reload',
  },
  trust_remote_code: {
    label: 'Trust remote code',
    hint: 'Required for models that ship Python in their HF repo (e.g. some custom architectures). Off by default for security.',
    restart: 'model-reload',
  },
  disable_log_requests: {
    label: 'Disable log requests',
    hint: 'Suppress per-request vLLM logs. Default off.',
    restart: 'model-reload',
  },
  extra_args: {
    label: 'Extra args',
    hint: 'Free-form list passed to `vllm serve` after the curated flags. One arg per row, e.g. `--worker-use-ray`, `--scheduler-delay-factor`, `0.3`.',
    restart: 'model-reload',
  },
  extra_env: {
    label: 'Extra env',
    hint: 'Free-form env vars passed to the vLLM subprocess. Allowlisted by `app.runtime.env_builder` (prefix `VLLM_`, `HF_`, `TRITON_`, etc.).',
    restart: 'model-reload',
  },
};
