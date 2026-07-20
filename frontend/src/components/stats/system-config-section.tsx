// System Configuration panel for /stats (#148).
//
// Static-ish inventory the operator wants when they're interpreting the
// live numbers above it — CPU model + core counts, total RAM, OS +
// Docker context, and one card per GPU. Backed by the in-process 60s
// cache on the API so the 30s page poll never hits the underlying
// nvidia-smi / docker shell-outs more than once a minute.
//
// Layout (locked by issue #148 ACs):
//   * Top row: 2-column grid (1 column on narrow) — CPU, RAM, and a
//     combined OS + Docker card.
//   * Bottom: one card per GPU (one column on narrow, two on wide).
//
// All values come from a single SWR fetch — we render a placeholder
// skeleton until the payload arrives, and a `—` glyph for any null
// field (e.g. no nvidia-smi present, /proc unreadable in a constrained
// sandbox).
//
// Refresh cadence: this panel reuses the parent page's 30s SWR poll
// indirectly — it owns its own SWR key. The hardware inventory rarely
// changes mid-uptime, but we deliberately do NOT pin it as `revalidate:
// false`; a hot-swap or driver reload should show up within a poll
// cycle. The backend cache absorbs the duplication.

import useSWR from "swr";
import { authFetchJSON } from "@/lib/auth-fetch";
import {
  formatOsName,
  mbToGib,
  type SystemInfo,
  type SystemInfoGpu,
} from "@/lib/system-info";
import { StatCard } from "@/components/stat-card";
import { Skeleton } from "@/components/ui/skeleton";

// 30s cadence matches the rest of /stats so a single SWR refresh on
// page-focus revalidates everything together. Pause on hidden tabs.
const REFRESH_MS = 30_000;
const refreshInterval = () =>
  typeof document !== "undefined" && document.hidden ? 0 : REFRESH_MS;

const DASH = "—"; // em-dash — same glyph other tiles use for "missing"

export function SystemConfigSection() {
  const { data, error, isLoading } = useSWR<SystemInfo>(
    "/api/system/info",
    authFetchJSON,
    { refreshInterval },
  );

  return (
    <section
      aria-label="System configuration"
      data-testid="system-config-section"
      className="space-y-3"
    >
      <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
        System configuration
      </h2>

      {error && !data ? (
        <p
          data-testid="system-config-error"
          className="text-sm text-red-500"
        >
          Failed to load system info
          {error instanceof Error ? `: ${error.message}` : "."}
        </p>
      ) : isLoading || !data ? (
        <div className="space-y-3">
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3">
            <Skeleton className="h-24 w-full" />
            <Skeleton className="h-24 w-full" />
            <Skeleton className="h-24 w-full" />
          </div>
        </div>
      ) : (
        <SystemConfigCards data={data} />
      )}
    </section>
  );
}

// Split out so the test can render it with a known payload without
// needing to stub SWR — and so the loading/error branches above stay
// simple. Pure-presentational, no hooks.
function SystemConfigCards({ data }: { data: SystemInfo }) {
  const cpu = data.cpu;
  const ram = data.ram;

  // The CPU value is the model name itself — physical/threads go into
  // the `hint` line so the tile reads "Intel Xeon E5-2680v4 · 14C/28T"
  // at a glance.
  const cpuValue = cpu ? cpu.model : DASH;
  const cpuHint = cpu ? `${cpu.physical_cores}C / ${cpu.threads}T` : undefined;

  const ramValue = ram ? mbToGib(ram.total_mb) : DASH;
  const ramHint = ram ? `${ram.total_mb.toLocaleString()} MB total` : undefined;
  const ramUnit = ram ? "GiB" : undefined;

  return (
    <>
      <div
        data-testid="system-config-top-row"
        className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3"
      >
        <StatCard
          label="CPU"
          value={
            <span
              data-testid="system-cpu-value"
              className="text-base font-medium"
              title={cpu?.model}
            >
              {cpuValue}
            </span>
          }
          hint={cpuHint}
        />
        <StatCard
          label="RAM"
          value={
            <span data-testid="system-ram-value">{ramValue}</span>
          }
          unit={ramUnit}
          hint={ramHint}
        />
        <OsDockerCard data={data} />
      </div>

      {data.gpus.length === 0 ? (
        <div
          data-testid="system-gpus-empty"
          className="rounded-md border border-dashed border-slate-700 bg-slate-900/30 p-4 text-center text-sm text-slate-400"
        >
          No NVIDIA GPUs detected.
        </div>
      ) : (
        <div
          data-testid="system-gpus-grid"
          className="grid grid-cols-1 gap-3 md:grid-cols-2"
        >
          {data.gpus.map((gpu) => (
            <GpuCard key={gpu.index} gpu={gpu} />
          ))}
        </div>
      )}
    </>
  );
}

// Combined OS + Docker card — keeps the top row at exactly 3 tiles so
// the layout doesn't reflow when docker is unavailable.
function OsDockerCard({ data }: { data: SystemInfo }) {
  const os = data.os;
  const docker = data.docker;
  const osLine = formatOsName(os);
  const kernelLine =
    os.kernel && os.kernel !== "unknown" ? `kernel ${os.kernel}` : undefined;

  return (
    <div
      data-testid="system-os-docker-card"
      className="rounded-lg border border-slate-700 bg-slate-900/50 p-4"
    >
      <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
        OS &amp; Docker
      </p>
      <p
        data-testid="system-os-value"
        className="mt-2 text-base font-medium text-slate-100"
      >
        {osLine}
      </p>
      {kernelLine && (
        <p className="text-xs text-slate-500" data-testid="system-os-kernel">
          {kernelLine}
        </p>
      )}
      <div className="mt-3 border-t border-slate-800 pt-2 text-xs">
        {docker.available ? (
          <p data-testid="system-docker-available" className="text-slate-300">
            Docker{" "}
            <span className="font-mono text-slate-100">
              {docker.version ?? DASH}
            </span>
            <span className="text-slate-500"> · runtime </span>
            <span className="font-mono text-slate-100">
              {docker.runtime ?? DASH}
            </span>
          </p>
        ) : (
          <p
            data-testid="system-docker-unavailable"
            className="text-slate-500"
            title="docker info shell-out failed — no socket mounted or daemon unreachable"
          >
            Docker not available
          </p>
        )}
      </div>
    </div>
  );
}

// One card per GPU. Value is the model name; hint stacks VRAM, driver,
// and CUDA. Index is shown as the label so a two-card row reads
// "GPU 0 / GPU 1" at a glance.
function GpuCard({ gpu }: { gpu: SystemInfoGpu }) {
  return (
    <div
      data-testid="system-gpu-card"
      data-gpu-index={gpu.index}
      className="rounded-lg border border-slate-700 bg-slate-900/50 p-4"
    >
      <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
        GPU {gpu.index}
      </p>
      <p
        data-testid="system-gpu-name"
        className="mt-2 text-base font-medium text-slate-100"
        title={gpu.name}
      >
        {gpu.name}
      </p>
      <dl className="mt-2 grid grid-cols-[auto,1fr] gap-x-3 gap-y-0.5 text-xs">
        <dt className="text-slate-500">VRAM</dt>
        <dd
          data-testid="system-gpu-vram"
          className="font-mono tabular-nums text-slate-200"
        >
          {mbToGib(gpu.vram_total_mb)} GiB
          <span className="text-slate-500"> ({gpu.vram_total_mb} MB)</span>
        </dd>
        <dt className="text-slate-500">Driver</dt>
        <dd
          data-testid="system-gpu-driver"
          className="font-mono text-slate-200"
        >
          {gpu.driver_version}
        </dd>
        <dt className="text-slate-500">CUDA</dt>
        <dd data-testid="system-gpu-cuda" className="font-mono text-slate-200">
          {gpu.cuda_version ?? DASH}
        </dd>
      </dl>
    </div>
  );
}
