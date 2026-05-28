"use client";

// Try-stack panel (#162) — per-model trial-and-error engine loop.
//
// An operator picks an engine combo (CUDA channel + vLLM version), the backend
// pins it onto the model and records a "pending" attempt. The operator then
// reports whether the model came up (ok/failed); on failure the backend's
// classifier annotates the attempt with a category + a suggested next combo.
// Once an attempt is "ok", the combo can be saved as a reusable template.
//
// Two exports:
//   - TryStackHistory: presentational attempt list (unit-tested in isolation)
//   - TryStackPanel: container — SWR fetch + the try / report / save round-trips
import { useState } from "react";
import useSWR from "swr";
import { authFetch, authFetchJSON } from "@/lib/auth-fetch";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Combobox, type ComboboxSuggestion } from "@/components/ui/combobox";
import { Input } from "@/components/ui/input";

// Only the CUDA channels resolve to an image today (app/templates/resolver.py);
// the others 400 on try-stack, so we don't offer them in the picker.
const CHANNELS = ["cuda-stable", "cuda-edge", "cuda-legacy"] as const;

export interface StackAttempt {
  id: string;
  channel: string;
  vllm_version: string;
  image: string | null;
  result: "pending" | "ok" | "failed";
  error: string | null;
  category: string | null;
  suggested_next: { suggestion: string } | null;
  created_at: string | null;
}

function resultVariant(r: StackAttempt["result"]): "success" | "error" | "info" {
  if (r === "ok") return "success";
  if (r === "failed") return "error";
  return "info";
}

export function TryStackHistory({ attempts }: { attempts: StackAttempt[] }) {
  if (attempts.length === 0) {
    return <p className="text-sm text-slate-500">No attempts yet.</p>;
  }
  return (
    <ul className="space-y-2" data-testid="try-stack-history">
      {attempts.map((a) => (
        <li
          key={a.id}
          className="rounded-md border border-slate-700 bg-slate-900/40 p-3 text-sm"
        >
          <div className="flex items-center justify-between gap-2">
            <span className="font-mono text-slate-200">
              {a.channel} · vLLM {a.vllm_version}
            </span>
            <Badge variant={resultVariant(a.result)}>{a.result}</Badge>
          </div>
          {a.error && <p className="mt-1 text-red-300">{a.error}</p>}
          {a.suggested_next?.suggestion && (
            <p className="mt-1 text-amber-300">
              Suggested next: {a.suggested_next.suggestion}
            </p>
          )}
        </li>
      ))}
    </ul>
  );
}

interface TryStackResponse {
  attempts: StackAttempt[];
}

// #177: GET /api/templates/engine-versions?channel= — the published
// vllm/vllm-openai semver tags that resolve to a real image for a channel.
// Empty for non-resolvable channels or a Docker Hub hiccup (never 500s).
interface EngineVersionsResponse {
  channel: string;
  family: string | null;
  versions: string[];
  error: string | null;
}

// #177: the active engine driver's capability. Under the in-container
// subprocess driver the engine version is fixed by the warden image and an
// engine-version pin is silently discarded (and now refused by the backend),
// so the version selector must be disabled + explained.
interface EngineInfo {
  driver: string;
  supports_version_select: boolean;
  vllm_version: string | null;
}

export function TryStackPanel({
  modelId,
  hfRepo,
  maxModelLen,
  tensorParallelSize,
}: {
  modelId: string;
  hfRepo: string;
  maxModelLen: number | null;
  tensorParallelSize: number | null;
}) {
  const key = `/api/models/${modelId}/try-stack`;
  const { data, mutate } = useSWR<TryStackResponse>(key, authFetchJSON);
  const attempts = data?.attempts ?? [];
  const latest = attempts.length > 0 ? attempts[attempts.length - 1] : null;

  const [channel, setChannel] = useState<string>(CHANNELS[0]);
  const [version, setVersion] = useState("");

  // #177: published vLLM versions that resolve to an image for the selected
  // channel. SWR-keyed on `channel` so switching channels refetches; the
  // backend serves this from a 6h family-keyed cache, so it's cheap to call
  // and keeps the field a typeable combobox (escape hatch for an unpublished
  // version or a pinned digest the dropdown won't list). While loading or on
  // an empty/error result, the field still works as free text.
  const { data: engineVersions } = useSWR<EngineVersionsResponse>(
    `/api/templates/engine-versions?channel=${encodeURIComponent(channel)}`,
    authFetchJSON,
  );
  const versionSuggestions: ComboboxSuggestion[] = (engineVersions?.versions ?? []).map(
    (v) => ({ value: v, label: v }),
  );

  // #177: whether this deployment's engine driver can honor a version pin.
  // While loading (engineInfo === undefined) we do NOT disable — the common
  // case is the capable docker driver, and disabling-then-enabling would
  // flash the controls. We only lock the selector once we KNOW it's false.
  const { data: engineInfo } = useSWR<EngineInfo>("/api/system/engine", authFetchJSON);
  const versionSelectDisabled = engineInfo?.supports_version_select === false;

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [failureDetail, setFailureDetail] = useState("");

  async function tryCombo() {
    if (busy || !version.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const r = await authFetch(key, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ channel, vllm_version: version.trim() }),
      });
      if (!r.ok) {
        const d = await r.json().catch(() => null);
        setError((d && typeof d === "object" && "detail" in d ? String(d.detail) : null) ?? `Failed (HTTP ${r.status})`);
        return;
      }
      await mutate();
    } finally {
      setBusy(false);
    }
  }

  // Report whether the pinned combo actually brought the model up. Drives the
  // backend's record_try_stack_result, which on "failed" runs the classifier
  // to annotate the attempt with a category + suggested next combo.
  async function reportResult(result: "ok" | "failed") {
    if (busy || !latest || latest.result !== "pending") return;
    setBusy(true);
    setError(null);
    try {
      const body: Record<string, unknown> = { result };
      if (result === "failed" && failureDetail.trim()) {
        body.error = failureDetail.trim();
      }
      const r = await authFetch(`${key}/${latest.id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const d = await r.json().catch(() => null);
        setError((d && typeof d === "object" && "detail" in d ? String(d.detail) : null) ?? `Failed to report result (HTTP ${r.status})`);
        return;
      }
      setFailureDetail("");
      await mutate();
    } finally {
      setBusy(false);
    }
  }

  async function saveAsTemplate() {
    if (!latest || latest.result !== "ok") return;
    setBusy(true);
    setError(null);
    try {
      const id = `${modelId}-${latest.channel}-${latest.vllm_version}`;
      // Capture the model's actual tuning when known so the saved template
      // reproduces the combo that worked; omit when unset (backend defaults).
      const body: Record<string, unknown> = {
        id,
        label: `${hfRepo || modelId} on ${latest.channel} vLLM ${latest.vllm_version}`,
        hf_repo: hfRepo,
        // #170: hand the backend the live model id so it sources the model's
        // actual extra_args + gpu_memory_utilization from the row (the panel
        // doesn't carry those props), instead of saving schema defaults.
        model_id: modelId,
        engine: {
          channel: latest.channel,
          vllm_version: latest.vllm_version,
          image: latest.image,
        },
      };
      if (maxModelLen != null) body.max_model_len = maxModelLen;
      if (tensorParallelSize != null) body.tensor_parallel_size = tensorParallelSize;
      const r = await authFetch("/api/models/templates", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const d = await r.json().catch(() => null);
        setError((d && typeof d === "object" && "detail" in d ? String(d.detail) : null) ?? `Failed to save template (HTTP ${r.status})`);
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-3">
      {versionSelectDisabled && (
        <p className="text-xs text-amber-300" data-testid="try-stack-driver-note">
          This deployment runs the in-container engine
          {engineInfo?.vllm_version ? ` (vLLM ${engineInfo.vllm_version})` : ""}; the
          engine version is fixed by the image. Version selection requires the docker
          engine driver.
        </p>
      )}
      <div className="flex flex-wrap items-end gap-2">
        <label className="space-y-1">
          <span className="block text-xs text-slate-400">Channel</span>
          <select
            data-testid="try-stack-channel"
            className="rounded-md border border-slate-700 bg-slate-900 p-2 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500 disabled:cursor-not-allowed disabled:opacity-50"
            value={channel}
            onChange={(e) => setChannel(e.target.value)}
            disabled={versionSelectDisabled}
          >
            {CHANNELS.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </label>
        <label className="space-y-1">
          <span className="block text-xs text-slate-400">vLLM version</span>
          {/* #177: typeable dropdown of published image-resolving versions.
              Stays free-text so an operator can still enter an unpublished
              version or a pinned digest the catalog won't list. */}
          <Combobox
            suggestions={versionSuggestions}
            value={version}
            onChange={setVersion}
            placeholder="0.20.0"
            className="w-32"
            ariaLabel="vLLM version"
            data-testid="try-stack-version"
            disabled={versionSelectDisabled}
          />
        </label>
        <Button
          size="sm"
          onClick={() => void tryCombo()}
          disabled={busy || !version.trim() || versionSelectDisabled}
          data-testid="try-stack-submit"
        >
          {busy ? "Working…" : "Try combo"}
        </Button>
        {latest?.result === "ok" && (
          <Button
            size="sm"
            variant="secondary"
            onClick={() => void saveAsTemplate()}
            disabled={busy}
            data-testid="try-stack-save-template"
          >
            Save working combo as template
          </Button>
        )}
      </div>

      {latest?.result === "pending" && (
        <div
          className="flex flex-wrap items-end gap-2 rounded-md border border-slate-700 bg-slate-900/40 p-3"
          data-testid="try-stack-report"
        >
          <label className="space-y-1">
            <span className="block text-xs text-slate-400">
              Did the model come up? (combo {latest.channel} · vLLM {latest.vllm_version})
            </span>
            <Input
              data-testid="try-stack-report-error"
              value={failureDetail}
              onChange={(e) => setFailureDetail(e.target.value)}
              placeholder="Failure detail (optional, for failed)"
              className="w-72"
            />
          </label>
          <Button
            size="sm"
            onClick={() => void reportResult("ok")}
            disabled={busy}
            data-testid="try-stack-report-ok"
          >
            Mark working
          </Button>
          <Button
            size="sm"
            variant="secondary"
            onClick={() => void reportResult("failed")}
            disabled={busy}
            data-testid="try-stack-report-failed"
          >
            Mark failed
          </Button>
        </div>
      )}

      {error && <p className="text-sm text-red-400">{error}</p>}

      <TryStackHistory attempts={attempts} />
    </div>
  );
}
