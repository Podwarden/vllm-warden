"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { use, useRef, useState } from "react";
import useSWR, { useSWRConfig } from "swr";
import { authFetch, authFetchJSON } from "@/lib/auth-fetch";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { LogStream } from "@/components/models/log-stream";
import { PullProgress } from "@/components/models/pull-progress";
import { DeleteModelModal } from "@/components/models/delete-model-modal";
import { TryStackPanel } from "@/components/models/try-stack-panel";

interface ModelDetail {
  id: string;
  served_model_name: string;
  hf_repo: string;
  hf_revision: string;
  gpu_indices: number[];
  tensor_parallel_size: number | null;
  dtype: string | null;
  max_model_len: number | null;
  gpu_memory_utilization: number | null;
  trust_remote_code: boolean;
  extra_args: string[];
  extra_env: Record<string, string>;
  status:
    | "registered"
    | "pulling"
    | "pulled"
    | "loading"
    | "loaded"
    | "unloading"
    | "failed";
  pulled_bytes: number | null;
  pulled_total: number | null;
  last_error: string | null;
}

type ModelStatus = ModelDetail["status"];

function badgeVariantForStatus(
  status: ModelStatus,
): "default" | "success" | "warning" | "error" | "info" {
  switch (status) {
    case "loaded":
    case "pulled":
      return "success";
    case "loading":
    case "pulling":
    case "unloading":
      return "info";
    case "failed":
      return "error";
    case "registered":
    default:
      return "default";
  }
}

// Backend gates on these in app/models/routes_api.py:
//  - load:   row.status must be in ("pulled", "failed")  → 409 otherwise
//  - unload: row.status must be in ("loaded", "failed")  → 409 otherwise
//  - delete: row.status must NOT be in ("loaded", "loading", "unloading", "pulling") → 409
//
// Mirroring those rules here keeps disabled buttons in sync with the server
// contract; the actual response still wins (we surface errors below) but a
// disabled button is the right UX hint before the round-trip.
function canLoad(s: ModelStatus): boolean {
  return s === "pulled" || s === "failed";
}
function canUnload(s: ModelStatus): boolean {
  return s === "loaded" || s === "failed";
}
function canDelete(s: ModelStatus): boolean {
  return !["loaded", "loading", "unloading", "pulling"].includes(s);
}

export default function ModelDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  // Next.js 15 dynamic params are exposed as a Promise so the framework can
  // suspend the page until they're known. `use()` unwraps inside a client
  // component without forcing the whole subtree to be a server component.
  const { id } = use(params);
  const router = useRouter();
  const { mutate } = useSWRConfig();
  const key = `/api/models/${id}`;
  const { data, error, isLoading } = useSWR<ModelDetail>(key, authFetchJSON, {
    // 2s aligns with the cadence the plan calls for. Pause polling while
    // the tab is hidden (mirror the list page's visibility guard) so the
    // detail tab doesn't keep hammering the API while the operator is in
    // another tab. Coming back triggers SWR's focus-revalidate so the
    // first frame after returning is fresh.
    refreshInterval: () =>
      typeof document !== "undefined" && document.hidden ? 0 : 2000,
  });
  const [actionError, setActionError] = useState<string | null>(null);
  const [busy, setBusy] = useState<null | "load" | "unload">(null);
  const [deleteOpen, setDeleteOpen] = useState(false);
  // Synchronous guard against double-clicks. setBusy is React state — it
  // doesn't take effect until the next render, which means a fast second
  // click can fire a concurrent POST before the disabled prop lands in
  // the DOM. The backend returns 409 on the duplicate call and we'd
  // flash a misleading error on what was actually a successful action.
  // A useRef flag is mutated in the same task as the first click, so the
  // second click sees the in-flight marker and bails out.
  //
  // S6 (epic/overhaul, #105): delete now goes through DeleteModelModal
  // which handles its own in-flight state, error surface, and the
  // optional "free cache" chained DELETE. The runAction path here only
  // handles load/unload.
  const inflight = useRef<"load" | "unload" | null>(null);

  async function runAction(action: "load" | "unload") {
    if (inflight.current) return;
    inflight.current = action;
    setActionError(null);
    setBusy(action);
    try {
      const r = await authFetch(`/api/models/${id}/${action}`, {
        method: "POST",
      });
      if (!r.ok) {
        let detail = `HTTP ${r.status}`;
        try {
          const body = await r.json();
          if (body && typeof body.detail === "string") {
            detail = body.detail;
          } else if (
            body &&
            typeof body.detail === "object" &&
            body.detail !== null &&
            typeof body.detail.message === "string"
          ) {
            detail = body.detail.message;
          }
        } catch {
          /* non-JSON error body — fall back to status code */
        }
        setActionError(detail);
        return;
      }
      // For load/unload, just refresh the row so the new status reflects.
      await mutate(key);
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e));
    } finally {
      inflight.current = null;
      setBusy(null);
    }
  }

  async function handleDeleted() {
    // Modal's own busy guard prevented a double-click; the row is
    // gone, and we may have also cleared the cache row. Refresh the
    // list cache and navigate back. The modal stays mounted until the
    // navigation lands so its cache-failed notice can render.
    setDeleteOpen(false);
    await mutate("/api/models");
    router.push("/models");
  }

  // 404 — model was deleted out from under us, or the URL is bogus.
  // SWR's authFetchJSON throws an Error with .status; surface a clean
  // dead-end page rather than a generic crash.
  const status = (error as (Error & { status?: number }) | undefined)?.status;
  if (status === 404) {
    return (
      <div className="space-y-4">
        <p className="text-sm text-slate-400">
          <Link href="/models" className="hover:underline">
            ← Back to models
          </Link>
        </p>
        <Card>
          <CardHeader>
            <CardTitle>Model not found</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-slate-400">
            The model <span className="font-mono">{id}</span> does not exist.
            It may have been deleted.
          </CardContent>
        </Card>
      </div>
    );
  }

  if (isLoading || (!data && !error)) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-48 w-full" />
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="space-y-4">
        <p className="text-sm text-slate-400">
          <Link href="/models" className="hover:underline">
            ← Back to models
          </Link>
        </p>
        <div className="rounded-md border border-red-700 bg-red-900/30 p-4 text-sm text-red-200">
          Failed to load model
          {error instanceof Error ? `: ${error.message}` : "."}
        </div>
      </div>
    );
  }

  // After the loading/404/error guards above, `data` is defined. The TS
  // narrowing flow doesn't track this through `!data && !error` so help
  // the checker with a runtime guard that doubles as defensive depth.
  if (!data) return null;

  return (
    <div className="space-y-6">
      <p className="text-sm text-slate-400">
        <Link href="/models" className="hover:underline">
          ← Back to models
        </Link>
      </p>

      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h1 className="truncate text-2xl font-semibold">
            {data.served_model_name}
          </h1>
          <p className="mt-1 truncate text-xs text-slate-400">
            <span className="font-mono">{data.hf_repo}</span>
            <span className="mx-1 text-slate-500">@</span>
            <span className="font-mono">{data.hf_revision}</span>
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button
            size="sm"
            onClick={() => runAction("load")}
            disabled={!canLoad(data.status) || busy !== null}
          >
            {busy === "load" ? "Loading…" : "Load"}
          </Button>
          <Button
            size="sm"
            variant="secondary"
            onClick={() => runAction("unload")}
            disabled={!canUnload(data.status) || busy !== null}
          >
            {busy === "unload" ? "Unloading…" : "Unload"}
          </Button>
          {/* Conditionally render the Link wrapper. A wrapped <Link> with a
              disabled <Button> still navigates: Button applies
              `disabled:pointer-events-none` on the inner <button> so the
              click falls through to the enclosing <a>. Splitting the two
              cases keeps the disabled state honest. */}
          {busy !== null ? (
            <Button size="sm" variant="outline" disabled>
              Settings
            </Button>
          ) : (
            <Link href={`/models/${id}/settings`}>
              <Button size="sm" variant="outline">
                Settings
              </Button>
            </Link>
          )}
          <Button
            size="sm"
            variant="destructive"
            onClick={() => setDeleteOpen(true)}
            disabled={!canDelete(data.status) || busy !== null}
          >
            Delete
          </Button>
        </div>
      </div>

      <DeleteModelModal
        open={deleteOpen}
        onClose={() => setDeleteOpen(false)}
        modelId={data.id}
        servedModelName={data.served_model_name}
        hfRepo={data.hf_repo}
        onDeleted={handleDeleted}
      />

      {actionError && (
        <div
          role="alert"
          className="rounded-md border border-red-700 bg-red-900/30 p-3 text-sm text-red-200"
        >
          {actionError}
        </div>
      )}

      {/* S1 (epic/overhaul) collapsed the Benchmark tab; the detail page
          now renders a single Overview view inline. If a future slice
          re-introduces multiple tabs, restore the <Tabs> wrapper from
          the pre-overhaul history (commit before this one). */}
      <div className="space-y-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0">
            <CardTitle>Status</CardTitle>
            <Badge variant={badgeVariantForStatus(data.status)}>
              {data.status}
            </Badge>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-slate-300">
            <p>
              GPUs:{" "}
              {data.gpu_indices.length > 0 ? (
                <span className="font-mono">
                  {data.gpu_indices.join(", ")}
                </span>
              ) : (
                <span className="text-slate-500">none</span>
              )}
            </p>
            {data.tensor_parallel_size !== null && (
              <p>
                Tensor parallel size:{" "}
                <span className="font-mono">
                  {data.tensor_parallel_size}
                </span>
              </p>
            )}
            {data.dtype && (
              <p>
                dtype: <span className="font-mono">{data.dtype}</span>
              </p>
            )}
            {data.max_model_len !== null && (
              <p>
                max_model_len:{" "}
                <span className="font-mono">{data.max_model_len}</span>
              </p>
            )}
            {data.gpu_memory_utilization !== null && (
              <p>
                gpu_memory_utilization:{" "}
                <span className="font-mono">
                  {data.gpu_memory_utilization}
                </span>
              </p>
            )}
          </CardContent>
        </Card>

        {/* Try-stack (#162): trial-and-error engine-combo loop. Pins a
            (channel, vLLM version) onto the model, records the attempt, lets
            the operator report ok/failed (classifier suggests the next combo
            on failure), and saves a working combo as a reusable template. */}
        <Card>
          <CardHeader>
            <CardTitle>Try stack</CardTitle>
          </CardHeader>
          <CardContent>
            <TryStackPanel
              modelId={id}
              hfRepo={data.hf_repo}
              maxModelLen={data.max_model_len}
              tensorParallelSize={data.tensor_parallel_size}
              modelStatus={data.status}
            />
          </CardContent>
        </Card>

        {/* PullProgress decides internally whether to render — keeps the
            detail page's layout straightforward (no conditional Card). */}
        {(data.status === "pulling" || data.status === "registered") && (
          <Card>
            <CardHeader>
              <CardTitle>Pull progress</CardTitle>
            </CardHeader>
            <CardContent>
              <PullProgress
                status={data.status}
                pulledBytes={data.pulled_bytes}
                pulledTotal={data.pulled_total}
              />
            </CardContent>
          </Card>
        )}

        {data.last_error && (
          <div
            role="alert"
            className="rounded-md border border-red-700 bg-red-900/30 p-3 text-sm text-red-200"
          >
            <span className="font-semibold">Last error:</span>{" "}
            {data.last_error}
          </div>
        )}

        <Card>
          <CardHeader>
            <CardTitle>Live logs</CardTitle>
          </CardHeader>
          <CardContent>
            {/* Key on modelId only — issue #52/#53. The earlier
                `${id}:${data.status}` key (commit 7df62a3,
                v17.2) forced LogStream to fully remount on
                every status transition so each flip got a fresh
                SSE connect + 200-line backfill. Cost: every
                remount tore down EventSource, minted a new
                ticket, and re-opened — and the Next.js rewrite
                proxy surfaces the upstream socket teardown as
                503 during the `pulled → loading` window, which
                the SSE hook then drives into reconnect/backoff
                while the panel sits blank.

                devops empirically verified the in-pod tail path
                is fine (file grows, fd tracks, curling SSE from
                inside the API container during an active load
                delivers live vLLM startup lines). The browser-
                facing symptom is the proxy churn caused by the
                status-keyed remount, NOT a backend tail bug.
                De-keying lets LogStream keep a single
                EventSource open across the lifecycle so the
                proxy never sees the teardown.

                The genuine remount case (operator switches to a
                different model — different id, different key)
                still triggers a fresh connect; the backend's
                per-connect 200-line backfill replays anything
                produced before the new mount.

                We hand the current status down to LogStream so
                it can skip opening the EventSource for non-log-
                producing states (`registered`) up front. */}
            <LogStream key={id} modelId={id} status={data.status} />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
