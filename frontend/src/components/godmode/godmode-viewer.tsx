"use client";

// God-mode live viewer — real-time prompts + model output flowing through the
// LLM Warden proxy. Admin-only (the SSE endpoint gates on require_jwt).
//
// This mirrors models/log-stream.tsx deliberately: <Virtuoso> + the shared
// `useStickyBottom` hook give us live/explore scroll, "Jump to latest", the
// elided-count banner, AND the fast-load fix (followOutput is always "auto",
// decoupled from the stick/free latch) for free. We consume the god-mode SSE
// exactly like LogStream — same single-use ticket flow via `useEventSource`.
//
// Rendering: events are grouped by `req_id` into request blocks. Each block is
// one Virtuoso row — a header (token label · model · client · finish_reason)
// with the completion streaming in beneath it. `channel:"reasoning"` deltas
// render dimmed/italic, distinct from `channel:"content"`. Every request gets
// a stable per-session color (see req-color.ts) on its header + a left-edge
// bar so concurrent, interleaved sessions are separable by eye.

import { forwardRef, useCallback, useMemo, useRef, useState } from "react";
import { Virtuoso, type VirtuosoHandle } from "react-virtuoso";

import { Button } from "@/components/ui/button";
import { useStickyBottom } from "@/components/shared/use-sticky-bottom";
import { useEventSource, MAX_RECONNECT, type SseState } from "@/lib/sse";
import { useHeaderMetrics } from "@/lib/header-metrics-stream";
import { activeModelsOf } from "@/lib/header-models";
import { useModelSelection } from "@/lib/model-selection";
import { ModelSelector } from "@/components/stats/model-selector";
import { cn } from "@/lib/utils";
import { reqColor } from "./req-color";
import { MediaStrip } from "./godmode-media";

// SSE endpoint — same admin-gated ticket path convention as the model log
// stream. useEventSource mints a single-use ticket via POST /api/auth/sse-ticket
// scoped to this path before opening the EventSource.
const GODMODE_STREAM_PATH = "/api/admin/godmode/stream";

// Bounded in-memory FIFO ring, mirroring log-stream's MAX_LINES. A long-open
// god-mode tab can't bloat the heap: past this many events the oldest are
// evicted and surfaced via the "… N older events elided" banner. (The backend
// hub has its own ring; this is the client-side cap independent of it.)
export const MAX_EVENTS = 4000;

// ---------------------------------------------------------------------------
// Event shape (matches app/proxy/godmode.py — see spec §3)
// ---------------------------------------------------------------------------

export interface RequestStartEvent {
  seq: number;
  type: "request_start";
  req_id: string;
  ts: number;
  token_label: string;
  token_id: string;
  model: string;
  served_name: string;
  client_ip: string;
  stream: boolean;
  prompt: string;
  /** Backend marked the captured prompt's middle as elided (head+tail window).
   *  Optional — older backends omit it; the viewer also detects the elision
   *  marker in the text itself, so it must work whether or not this is set. */
  prompt_elided?: boolean;
  /** Captured image_url parts (spec 2026-08-03) — absent on text-only
   *  requests and on events from older backends. */
  media?: MediaEntry[];
}

export interface MediaEntry {
  kind: "image";
  /** Present for store-backed items — fetch /api/admin/godmode/media/{id}. */
  media_id?: string;
  mime?: string;
  /** base64 length of the stored payload (decoded bytes ≈ chars * 3/4). */
  chars?: number;
  /** Present for remote images — hotlinked directly. */
  url?: string;
  dropped?: "too_large" | "count";
  count?: number;
}

export interface DeltaEvent {
  seq: number;
  type: "delta";
  req_id: string;
  ts: number;
  channel: "content" | "reasoning";
  text: string;
}

export interface RequestEndEvent {
  seq: number;
  type: "request_end";
  req_id: string;
  ts: number;
  finish_reason: string | null;
  prompt_tokens: number;
  completion_tokens: number;
}

export type GodModeEvent = RequestStartEvent | DeltaEvent | RequestEndEvent;

// ---------------------------------------------------------------------------
// Ring state
// ---------------------------------------------------------------------------

interface GmState {
  events: GodModeEvent[];
  /** Total events dropped by the FIFO so far — surfaced as a banner row. */
  elided: number;
  /** Stable per-session index for each req_id, assigned on first sight and
   *  never reclaimed, so a request's golden-angle color survives ring
   *  eviction. */
  reqIndex: Record<string, number>;
  /** Next session index to hand out. */
  nextIndex: number;
}

const INITIAL_STATE: GmState = { events: [], elided: 0, reqIndex: {}, nextIndex: 0 };

export function appendEvent(prev: GmState, ev: GodModeEvent): GmState {
  let reqIndex = prev.reqIndex;
  let nextIndex = prev.nextIndex;
  if (reqIndex[ev.req_id] === undefined) {
    reqIndex = { ...reqIndex, [ev.req_id]: nextIndex };
    nextIndex += 1;
  }

  if (prev.events.length >= MAX_EVENTS) {
    return {
      events: [...prev.events.slice(prev.events.length - MAX_EVENTS + 1), ev],
      elided: prev.elided + 1,
      reqIndex,
      nextIndex,
    };
  }
  return { events: [...prev.events, ev], elided: prev.elided, reqIndex, nextIndex };
}

// ---------------------------------------------------------------------------
// Grouping
// ---------------------------------------------------------------------------

export interface RequestBlockData {
  reqId: string;
  start?: RequestStartEvent;
  deltas: DeltaEvent[];
  end?: RequestEndEvent;
}

/** Fold the flat event ring into request blocks, preserving first-seen order
 *  (which is start-order in practice, since request_start is a request's
 *  first event). O(n) over the window. */
export function groupBlocks(events: GodModeEvent[]): RequestBlockData[] {
  const byId = new Map<string, RequestBlockData>();
  const order: string[] = [];
  for (const ev of events) {
    let block = byId.get(ev.req_id);
    if (!block) {
      block = { reqId: ev.req_id, deltas: [] };
      byId.set(ev.req_id, block);
      order.push(ev.req_id);
    }
    if (ev.type === "request_start") block.start = ev;
    else if (ev.type === "delta") block.deltas.push(ev);
    else if (ev.type === "request_end") block.end = ev;
  }
  return order.map((id) => byId.get(id)!);
}

/**
 * Narrow request blocks to a model selection.
 *
 * `modelIds` of `null` means no filter. Filtering happens HERE, in the client,
 * and not in the SSE: the hub is a shared broadcast ring with a replay
 * snapshot, and a per-subscriber server-side filter would drop a
 * `request_start` while still delivering that request's `delta` and
 * `request_end` frames — leaving orphan blocks with no prompt and no model.
 * Correlation is by `req_id`, so the whole conversation has to arrive together
 * and be grouped before anything can be attributed to a model at all.
 *
 * The god-mode env gate is untouched by this: the stream is as gated as it
 * ever was, and this only decides what a viewer who already has it renders.
 *
 * A block with NO `request_start` is KEPT. Its start was evicted from the ring,
 * so its model is unknowable — and silently dropping traffic we cannot
 * attribute would make a narrowed god-mode view quietly incomplete, which is
 * the one thing a forensic tool must never be.
 */
export function filterBlocksByModel(
  blocks: RequestBlockData[],
  modelIds: readonly string[] | null,
): RequestBlockData[] {
  if (modelIds === null) return blocks;
  const wanted = new Set(modelIds);
  return blocks.filter((b) => !b.start || wanted.has(b.start.model));
}

// ---------------------------------------------------------------------------
// Repeated-system-prompt collapse
// ---------------------------------------------------------------------------
//
// A god-mode viewer watching an agent loop sees the SAME lengthy system prompt
// re-sent on every request. To keep the genuinely-new turn readable, we diff a
// request's prompt against the PREVIOUS request from the same token identity
// and collapse the shared leading portion behind a toggle, leaving only the
// divergent remainder expanded. The diff is line-aware (split on lines, take
// the longest matching leading run) so we never cut mid-line.

/** Backend's elision marker, e.g. `…[1234 chars elided]…`. When a capture was
 *  windowed (head+tail), we must NOT diff across the gap — restrict the
 *  comparable region to the head that precedes this marker. */
const ELISION_MARKER_RE = /…\[\d+ chars elided\]…/;

function headBeforeElision(s: string): string {
  const m = s.match(ELISION_MARKER_RE);
  return m && m.index !== undefined ? s.slice(0, m.index) : s;
}

/** Longest common leading run of whole lines between `a` and `b`, returned as
 *  the exact leading substring of `a` (so `a.slice(prefix.length)` is the
 *  divergent remainder). "" when the first line already differs. */
export function commonLinePrefix(a: string, b: string): string {
  if (!a || !b) return "";
  const al = a.split("\n");
  const bl = b.split("\n");
  const n = Math.min(al.length, bl.length);
  let matched = 0;
  while (matched < n && al[matched] === bl[matched]) matched += 1;
  if (matched === 0) return "";
  let prefix = al.slice(0, matched).join("\n");
  // Re-attach the newline that separates the matched run from `a`'s remainder,
  // but only when there IS more content after it — so prefix+remainder === a.
  if (matched < al.length) prefix += "\n";
  return prefix;
}

export interface PromptCollapse {
  /** Shared leading portion, hidden by default behind the toggle. */
  collapsed: string;
  /** Divergent NEW content, always shown. */
  remainder: string;
}

/** Decide whether a prompt's shared head with the previous same-token prompt is
 *  substantial enough to collapse. Guarded so a short or mostly-new prompt is
 *  left fully visible. Returns null when no collapse should happen. */
export function computePromptCollapse(
  current: string,
  previous: string | undefined,
  opts?: { minChars?: number; minRatio?: number },
): PromptCollapse | null {
  if (!current || !previous) return null;
  const minChars = opts?.minChars ?? 400;
  const minRatio = opts?.minRatio ?? 0.4;
  // Never diff past an elision gap — compare only the heads. A prefix of the
  // head is still a prefix of the full current prompt, so the marker + tail
  // fall into `remainder` and stay visible (never claimed as "unchanged").
  const prefix = commonLinePrefix(headBeforeElision(current), headBeforeElision(previous));
  if (prefix.length < minChars) return null;
  const shorter = Math.min(current.length, previous.length);
  if (shorter === 0 || prefix.length < shorter * minRatio) return null;
  return { collapsed: prefix, remainder: current.slice(prefix.length) };
}

// ---------------------------------------------------------------------------
// Status message (mirrors log-stream.renderStatusMessage)
// ---------------------------------------------------------------------------

function renderStatusMessage(s: SseState): { text: string; tone: "info" | "warn" | "error" } | null {
  switch (s.status) {
    case "connected":
      return null;
    case "connecting":
      return { text: "Connecting to god-mode stream…", tone: "info" };
    case "reconnecting":
      return {
        text: `Connection lost — retrying (${s.attempts}/${MAX_RECONNECT})…`,
        tone: "warn",
      };
    case "terminal-error": {
      const code = s.errorCode;
      let text: string;
      if (code === 401 || code === 403) {
        text = "Stream unavailable — your session expired. Please re-login.";
      } else if (code !== null) {
        text = `Stream unavailable (HTTP ${code}). Please refresh.`;
      } else {
        text = `Stream unavailable after ${MAX_RECONNECT} retries. Please refresh.`;
      }
      return { text, tone: "error" };
    }
  }
}

// A terminal-error whose ticket mint returned 404/409 means the backend has
// god mode switched off (spec §5). Show a purpose-built placeholder rather
// than a generic "stream unavailable" so the operator knows it's a config
// flag, not a fault.
function isDisabledResponse(s: SseState): boolean {
  return s.status === "terminal-error" && (s.errorCode === 404 || s.errorCode === 409);
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface GodModeViewerProps {
  className?: string;
  /** Fixed list height in px. */
  heightPx?: number;
}

export function GodModeViewer({ className, heightPx = 560 }: GodModeViewerProps) {
  const [state, setState] = useState<GmState>(INITIAL_STATE);

  const onMessage = useCallback((ev: GodModeEvent) => {
    // Defensive: ignore anything without the fields we key on. A malformed
    // frame shouldn't tear down the view.
    if (!ev || typeof ev.seq !== "number" || typeof ev.req_id !== "string") return;
    setState((prev) => appendEvent(prev, ev));
  }, []);

  const sse = useEventSource<GodModeEvent>(GODMODE_STREAM_PATH, { onMessage });

  const sticky = useStickyBottom("stick");
  const virtuosoRef = useRef<VirtuosoHandle>(null);

  // The loaded-model list comes from the header-metrics stream, which is
  // already open in this tab (NavBar mounts it everywhere but /login and
  // /setup) and is ref-counted, so subscribing here costs no new connection
  // and no new endpoint. It is also exactly the right list: the models the
  // box is currently serving.
  const header = useHeaderMetrics();
  const loadedModels = useMemo(
    () =>
      activeModelsOf(header.frame).map((m) => ({
        id: m.id,
        served_model_name: m.served_model_name,
      })),
    [header.frame],
  );
  const modelIds = useMemo(() => loadedModels.map((m) => m.id), [loadedModels]);
  const selection = useModelSelection(modelIds);

  const allBlocks = useMemo(() => groupBlocks(state.events), [state.events]);
  const blocks = useMemo(
    () =>
      filterBlocksByModel(
        allBlocks,
        // No selection yet (still resolving, or nothing loaded) means no
        // filter — never an empty allow-list, which would blank the view.
        selection.selected.length > 0 ? selection.selected : null,
      ),
    [allBlocks, selection.selected],
  );

  // Per-request collapse decision: diff each request's prompt against the
  // PREVIOUS request from the same token identity (label preferred, id fallback)
  // and collapse the shared head. Built in first-seen (start) order so each
  // block sees only requests that preceded it. Keyed by reqId for the row.
  const collapseByReq = useMemo(() => {
    const out: Record<string, PromptCollapse | null> = {};
    const lastPromptByToken: Record<string, string> = {};
    for (const block of blocks) {
      const start = block.start;
      if (!start || !start.prompt) {
        if (start) out[block.reqId] = null;
        continue;
      }
      const identity = start.token_label || start.token_id || "";
      const prev = identity ? lastPromptByToken[identity] : undefined;
      out[block.reqId] = computePromptCollapse(start.prompt, prev);
      if (identity) lastPromptByToken[identity] = start.prompt;
    }
    return out;
  }, [blocks]);

  // Disabled placeholder — god mode is off on the backend.
  if (isDisabledResponse(sse)) {
    return (
      <div
        role="status"
        className={cn(
          "rounded border p-4 text-sm",
          "border-slate-700 bg-slate-900/50 text-slate-400",
          className,
        )}
      >
        God mode is disabled (<code className="font-mono text-slate-300">VW_GODMODE_ENABLED</code>).
      </div>
    );
  }

  const statusBar = renderStatusMessage(sse);
  const showStatusBar =
    statusBar !== null && (state.events.length === 0 || sse.status !== "connected");

  // Connected-but-empty — the stream is live but nothing is flowing yet.
  const showEmptyPlaceholder = sse.status === "connected" && state.events.length === 0;
  if (showEmptyPlaceholder) {
    return (
      <div
        role="status"
        className={cn(
          "rounded border p-4 text-sm",
          "border-slate-700 bg-slate-900/50 text-slate-400",
          className,
        )}
      >
        Waiting for requests… (no traffic through the proxy yet)
      </div>
    );
  }

  if (state.events.length === 0 && statusBar !== null) {
    const role = statusBar.tone === "error" ? "alert" : "status";
    const toneClass =
      statusBar.tone === "error"
        ? "border-red-700/60 bg-red-950/40 text-red-200"
        : statusBar.tone === "warn"
          ? "border-amber-700/60 bg-amber-950/30 text-amber-200"
          : "border-slate-700 bg-slate-900/50 text-slate-400";
    return (
      <div role={role} className={cn("rounded border p-4 text-sm", toneClass, className)}>
        {statusBar.text}
      </div>
    );
  }

  return (
    <div className={cn("space-y-2", className)}>
      {/* The same selection as /stats. God mode's own gate is
          untouched — this decides only what a viewer who already has the
          stream renders. */}
      <ModelSelector models={loadedModels} selection={selection} />

      {showStatusBar && statusBar !== null && (
        <div
          role={statusBar.tone === "error" ? "alert" : "status"}
          className={cn(
            "rounded border px-3 py-1.5 text-xs",
            statusBar.tone === "error"
              ? "border-red-700/60 bg-red-950/40 text-red-200"
              : statusBar.tone === "warn"
                ? "border-amber-700/60 bg-amber-950/30 text-amber-200"
                : "border-slate-700 bg-slate-900/50 text-slate-400",
          )}
        >
          {statusBar.text}
        </div>
      )}

      {state.elided > 0 && (
        <div
          role="status"
          className="rounded-t border border-b-0 border-slate-700 bg-slate-900/70 px-3 py-1 font-mono text-[11px] text-slate-400"
        >
          … {state.elided} older event{state.elided === 1 ? "" : "s"} elided
        </div>
      )}

      <div
        className={cn(
          "relative rounded border border-slate-700 bg-slate-950",
          state.elided > 0 ? "rounded-t-none border-t-0" : undefined,
        )}
      >
        <Virtuoso
          ref={virtuosoRef}
          components={{ List: GodModeList }}
          style={{ height: heightPx }}
          data={blocks}
          followOutput={sticky.followOutput}
          // Match log-stream's generous tolerance so a mid-burst frame where a
          // freshly-appended block sits below the fold doesn't flap the
          // "Jump to latest" button (see use-sticky-bottom rationale).
          atBottomThreshold={64}
          atBottomStateChange={sticky.onAtBottomStateChange}
          computeItemKey={(_idx, block) => block.reqId}
          itemContent={(_idx, block) => (
            <RequestBlock
              block={block}
              colorIndex={state.reqIndex[block.reqId]}
              collapse={collapseByReq[block.reqId]}
            />
          )}
        />

        {sticky.mode === "free" && blocks.length > 0 && (
          <Button
            type="button"
            size="sm"
            variant="secondary"
            className="absolute bottom-2 right-2 shadow-lg"
            onClick={() => {
              virtuosoRef.current?.scrollToIndex({
                index: blocks.length - 1,
                behavior: "smooth",
              });
              sticky.jumpToLatest();
            }}
          >
            Jump to latest
          </Button>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Request block row
// ---------------------------------------------------------------------------

interface RequestBlockProps {
  block: RequestBlockData;
  /** Per-session index for golden-angle color; undefined only if the block's
   *  start was elided before we assigned one (falls back to a hash of the id). */
  colorIndex?: number;
  /** Repeated-system-prompt collapse for this request's prompt, or null/undefined
   *  to render the prompt in full (first request from a token, or no match). */
  collapse?: PromptCollapse | null;
}

function RequestBlock({ block, colorIndex, collapse }: RequestBlockProps) {
  const color = reqColor(block.reqId, colorIndex);
  const start = block.start;
  const label = start?.token_label || start?.token_id || block.reqId.slice(0, 8);

  return (
    <div
      data-testid="godmode-block"
      data-req-id={block.reqId}
      className="mb-2 rounded border-l-4 border-slate-800 bg-slate-900/30"
      style={{ borderLeftColor: color.accent }}
    >
      <div
        className="flex flex-wrap items-center gap-x-2 gap-y-0.5 rounded-tr px-3 py-1.5 text-xs"
        style={{ backgroundColor: color.headerBg }}
      >
        <span className="font-semibold" style={{ color: color.text }}>
          {label}
        </span>
        {start?.model && (
          <>
            <span aria-hidden className="text-slate-600">·</span>
            <span className="font-mono text-slate-300">{start.model}</span>
          </>
        )}
        {start?.client_ip && (
          <>
            <span aria-hidden className="text-slate-600">·</span>
            <span className="font-mono text-slate-400">{start.client_ip}</span>
          </>
        )}
        {block.end && (
          <>
            <span aria-hidden className="text-slate-600">·</span>
            <span
              data-testid="godmode-finish"
              className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[10px] uppercase text-slate-300"
            >
              {block.end.finish_reason ?? "ended"}
            </span>
          </>
        )}
      </div>

      {start?.media && start.media.length > 0 && <MediaStrip media={start.media} />}

      {start?.prompt && <PromptView prompt={start.prompt} collapse={collapse} />}

      <div className="whitespace-pre-wrap px-3 py-1.5 font-mono text-xs leading-5 text-slate-100">
        {block.deltas.map((d) => (
          <span
            key={d.seq}
            data-channel={d.channel}
            className={cn(d.channel === "reasoning" && "italic text-slate-500")}
          >
            {d.text}
          </span>
        ))}
      </div>
    </div>
  );
}

// Prompt row — collapses a repeated system-prompt head behind a toggle when
// `collapse` is set, otherwise renders the prompt in full (matching the prior
// look). The collapsed head starts hidden; the divergent remainder is always
// visible so the newest turn reads at a glance.
function PromptView({ prompt, collapse }: { prompt: string; collapse?: PromptCollapse | null }) {
  const [expanded, setExpanded] = useState(false);

  if (!collapse) {
    return (
      <div className="whitespace-pre-wrap border-b border-slate-800/60 px-3 py-1.5 font-mono text-xs text-slate-400">
        <span className="mr-1 select-none text-slate-600">▸</span>
        {prompt}
      </div>
    );
  }

  const chars = collapse.collapsed.length;
  return (
    <div className="whitespace-pre-wrap border-b border-slate-800/60 px-3 py-1.5 font-mono text-xs text-slate-400">
      <div className="mb-1 flex flex-wrap items-center gap-x-2 gap-y-0.5">
        <button
          type="button"
          data-testid="godmode-prompt-toggle"
          aria-expanded={expanded}
          onClick={() => setExpanded((v) => !v)}
          className="inline-flex select-none items-center rounded bg-slate-800/80 px-1.5 py-0.5 text-[10px] font-medium text-slate-300 hover:bg-slate-700/80"
        >
          (system prompt [{expanded ? "−" : "+"}])
        </button>
        <span className="select-none text-[10px] text-slate-500">
          system prompt · {chars.toLocaleString()} chars · unchanged from previous
        </span>
      </div>
      {expanded && (
        <div data-testid="godmode-prompt-collapsed" className="text-slate-500">
          {collapse.collapsed}
        </div>
      )}
      <div data-testid="godmode-prompt-remainder">
        <span className="mr-1 select-none text-slate-600">▸</span>
        {collapse.remainder}
      </div>
    </div>
  );
}

// Virtuoso List override — stamps role="log" so the virtualized scroll
// container is a polite live region for screen readers (matches LogStream).
const GodModeList = forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement> & { context?: unknown }
>(function GodModeListImpl(props, ref) {
  const { context: _context, ...rest } = props;
  return <div ref={ref} role="log" aria-label="God mode live stream" {...rest} />;
});
