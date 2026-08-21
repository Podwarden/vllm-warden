"use client";

// /godmode — live prompt/output viewer ("god mode").
//
// Admin-only surface (the SSE endpoint gates on require_jwt, same as the
// active-requests + model-log streams). Shows, in real time, the prompts and
// model output flowing through the vLLM warden proxy. Config-gated on the
// backend by VW_GODMODE_ENABLED — when off, the viewer renders a disabled
// placeholder rather than a dead stream (see GodModeViewer).
//
// This page is part of the authenticated UI surface; there is no separate
// content for unauthenticated users (defense-in-depth — the real gate is the
// backend's require_jwt on the stream + ticket mint).

import Link from "next/link";
import { GodModeViewer } from "@/components/godmode/godmode-viewer";

export default function GodModePage() {
  return (
    <div className="space-y-4" data-testid="godmode-page">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold">God Mode</h1>
          <p className="mt-1 text-xs text-slate-400">
            Live prompts &amp; model output flowing through the proxy. In-memory only —
            nothing here is persisted.
          </p>
        </div>
        <Link
          href="/stats"
          className="text-xs text-slate-400 underline-offset-2 hover:text-slate-200 hover:underline"
        >
          ← Back to stats
        </Link>
      </div>

      <GodModeViewer />
    </div>
  );
}
