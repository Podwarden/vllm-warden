// Frontend types for the /api/system/info contract (#148).
//
// The endpoint returns a static-ish inventory used by the "System
// Configuration" panel on /stats: CPU model + core/thread counts,
// total RAM, per-GPU name / VRAM / driver / CUDA, OS release + kernel,
// and Docker version + runtime. See app/system/system_info.py for the
// authoritative shape; this file mirrors it so a drift fails typecheck
// in CI before it ships.
//
// Why hand-typed: FastAPI's OpenAPI schema for `dict[str, Any]` is
// opaque (the generated type collapses to `{ [key: string]: unknown }`),
// so openapi-typescript can't reach the inner shape. The contract is
// pinned by the backend unit tests + the page component test instead.
//
// Optional/null fields and their semantics:
//   * `cpu` — `null` when /proc/cpuinfo couldn't be read or yielded
//     no usable processor entries (e.g. exotic kernels).
//   * `ram` — `null` when /proc/meminfo couldn't be read or had no
//     MemTotal line. UI shows "—".
//   * `gpus` — always an array; `[]` when nvidia-smi is unavailable.
//   * `os` — always present; missing fields default to "unknown" rather
//     than null so the card always has something to render.
//   * `docker` — always present with `available: boolean`; when false
//     the `version`/`runtime` slots are null and the card shows the
//     "not available" hint.
//   * Per-GPU `cuda_version` — `null` on the rare host where nvidia-smi
//     prints a card row but no version banner (corrupt install / WSL).

export interface SystemInfoCpu {
  model: string;
  /** Distinct (physical_id, core_id) pairs in /proc/cpuinfo, or the
   *  total processor count when that metadata is absent. */
  physical_cores: number;
  /** Total processor entries in /proc/cpuinfo (== logical cores). */
  threads: number;
}

export interface SystemInfoRam {
  total_mb: number;
}

export interface SystemInfoGpu {
  index: number;
  name: string;
  vram_total_mb: number;
  driver_version: string;
  /** `null` when nvidia-smi --version didn't print a CUDA banner. */
  cuda_version: string | null;
}

export interface SystemInfoOs {
  /** Pretty OS name e.g. "Ubuntu". Defaults to "unknown". */
  name: string;
  /** Release version e.g. "24.04". Defaults to "unknown". */
  version: string;
  /** `uname -r` output e.g. "6.8.0-1008-nvidia". Defaults to "unknown". */
  kernel: string;
}

export interface SystemInfoDocker {
  /** Server version from `docker info`. `null` when unavailable. */
  version: string | null;
  /** Default runtime (typically "runc" or "nvidia"). `null` when
   *  unavailable. */
  runtime: string | null;
  /** False when the `docker info` shell-out failed (no socket / not
   *  installed). The card renders a "not available" placeholder. */
  available: boolean;
}

export interface SystemInfo {
  cpu: SystemInfoCpu | null;
  ram: SystemInfoRam | null;
  gpus: SystemInfoGpu[];
  os: SystemInfoOs;
  docker: SystemInfoDocker;
}

// ---- Formatters ----------------------------------------------------------

/** Render a MB count as a human-friendly GiB string, e.g.
 *  `64277` → `"62.8"`. One decimal place — matches the VRAM tile. */
export function mbToGib(mb: number): string {
  if (!mb) return "0";
  return (mb / 1024).toFixed(1);
}

/** Compose the OS line e.g. "Ubuntu 24.04". Hides "unknown" tokens so a
 *  partial /etc/os-release doesn't print "unknown unknown". */
export function formatOsName(os: SystemInfoOs): string {
  const parts: string[] = [];
  if (os.name && os.name !== "unknown") parts.push(os.name);
  if (os.version && os.version !== "unknown") parts.push(os.version);
  return parts.length > 0 ? parts.join(" ") : "unknown";
}
