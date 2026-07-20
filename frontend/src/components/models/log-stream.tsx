"use client";

// Live vLLM stdout/stderr tail with dual-mode "stick / free" scrolling.
//
// What changed in v17.11 #75:
//   - Switched from a plain scroll-container <pre> to <Virtuoso>, so the
//     5000-line ring buffer renders only the visible window. Pre-#75 the
//     full join('\n') was passed to dangerouslySetInnerHTML on every new
//     line, which spent ~80ms in HTML parsing on a 5k-line buffer under
//     a 200-line/s vLLM startup burst.
//   - Added an `elided_count` so the operator knows when the FIFO has
//     evicted lines. A 1-row "… N older lines elided" banner replaces
//     the silent drop.
//   - Sticky-bottom is delegated to `shared/use-sticky-bottom.ts` so the
//     "Jump to latest" semantics stay consistent across future streaming
//     views. (The original sibling consumer, bench/events-tab.tsx, was
//     removed in epic/overhaul S1.)
//
// Each log line is its own row in Virtuoso. We render ANSI escapes per
// line (not per buffer) — small per-line work, but it means each row is
// independently memoizable and Virtuoso can hand the operator a steady
// 60 fps even during a vLLM module-import burst.

import { forwardRef, useCallback, useRef, useState } from "react";
import { Virtuoso, type VirtuosoHandle } from "react-virtuoso";

import { AnsiLog } from "@/components/ansi-log";
import { Button } from "@/components/ui/button";
import { useStickyBottom } from "@/components/shared/use-sticky-bottom";
import { useEventSource, MAX_RECONNECT, type SseState } from "@/lib/sse";
import { cn } from "@/lib/utils";

/**
 * Statuses where no new log content is being produced and opening an
 * EventSource would only churn single-use SSE tickets through the proxy:
 *   - "registered": log file may not exist yet
 *   - "unloading":  subprocess has been signaled, supervisor is tearing down
 * "failed" is deliberately NOT here — its log file contains the failure
 * traceback, and the 200-line backfill is operator-useful.
 *
 * Pull progress is rendered by the dedicated PullProgress card during
 * `pulling`, and load progress is implicit during `loading` — both
 * states already have richer UX than what live logs would add, but
 * vLLM does produce stdout during loading so we keep the stream open
 * for those. Operators have explicitly asked to see vLLM's stdout
 * during the 20GB-weight-load window so we MUST stream `loading`.
 *
 * Issue #53 — confining the enabled-states list keeps the EventSource
 * lifecycle minimal during the pre-load workflow.
 */
const NON_LOG_PRODUCING_STATUSES = new Set<string>([
  "registered",
  "unloading",
]);

interface LogStreamProps {
  modelId: string;
  /** Current model lifecycle status. When in a non-log-producing
   *  state (see NON_LOG_PRODUCING_STATUSES) the EventSource is NOT
   *  opened — we render an explanatory placeholder instead. Optional
   *  for backwards compatibility with callers that haven't been
   *  updated yet (treated as "stream-eligible"). */
  status?: string;
  className?: string;
  /** Fixed list height in px. Default matches the pre-#75 max-h-[480px]
   *  so the layout doesn't reflow on existing pages. */
  heightPx?: number;
}

// Cap the in-memory line buffer so a long-running tab can't bloat heap.
// Matches the plan's prev.slice(-4999) figure: 5000 lines is roughly
// the window an operator can reasonably scroll through, and at ~200B
// per line it tops out at ~1MB of strings — well below tab budgets.
export const MAX_LINES = 5000;

interface LogEvent {
  line: string;
}

/**
 * Map the SSE hook's state to a status-bar message + tone. Pulled out
 * of the component so the test can target the rendered string for each
 * status without driving the full SSE plumbing.
 */
function renderStatusMessage(s: SseState): { text: string; tone: "info" | "warn" | "error" } | null {
  switch (s.status) {
    case "connected":
      // Once the stream is live the bar disappears so the log lines can
      // own the vertical space. The caller distinguishes
      // connected-but-empty from connected-with-data and renders a
      // dedicated "(no log lines yet)" placeholder for the former (see
      // showEmptyPlaceholder below) so an operator never stares at a
      // blank role="log" div wondering whether the stream is wedged.
      return null;
    case "connecting":
      return { text: "Connecting to log stream…", tone: "info" };
    case "reconnecting":
      return {
        text: `Connection lost — retrying (${s.attempts}/${MAX_RECONNECT})…`,
        tone: "warn",
      };
    case "terminal-error": {
      // Distinguish auth failures from "stream gone" so the operator
      // has a real next step. errorCode === null means EventSource
      // gave up after MAX_RECONNECT without us getting an HTTP status.
      const code = s.errorCode;
      let text: string;
      if (code === 401 || code === 403) {
        text = "Stream unavailable — your session expired. Please re-login.";
      } else if (code === 404) {
        text = "Log stream not found for this model.";
      } else if (code !== null) {
        text = `Stream unavailable (HTTP ${code}). Please refresh.`;
      } else {
        text = `Stream unavailable after ${MAX_RECONNECT} retries. Please refresh.`;
      }
      return { text, tone: "error" };
    }
  }
}

// ---------------------------------------------------------------------------
// Row state
// ---------------------------------------------------------------------------

/** Internal log line — every row carries a stable id so Virtuoso doesn't
 *  re-mount existing rows when the FIFO evicts the head. The id is a
 *  monotonically-increasing counter; the row's index in the array shifts
 *  on eviction, but its `id` does not. */
interface LogLine {
  id: number;
  text: string;
}

interface LogState {
  lines: LogLine[];
  /** Total lines elided by the FIFO so far. Surfaced as a "… N older
   *  lines elided" banner row at the head of the list. */
  elided: number;
  /** Monotonic counter for the next line's id. Survives eviction. */
  nextId: number;
}

const INITIAL_STATE: LogState = { lines: [], elided: 0, nextId: 0 };

function appendLine(prev: LogState, text: string): LogState {
  const next: LogLine = { id: prev.nextId, text };
  if (prev.lines.length >= MAX_LINES) {
    // FIFO eviction. Slice off the head and bump elided so the banner
    // tells the operator "+1 older line dropped".
    return {
      lines: [...prev.lines.slice(prev.lines.length - MAX_LINES + 1), next],
      elided: prev.elided + 1,
      nextId: prev.nextId + 1,
    };
  }
  return {
    lines: [...prev.lines, next],
    elided: prev.elided,
    nextId: prev.nextId + 1,
  };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function LogStream({ modelId, status, className, heightPx = 480 }: LogStreamProps) {
  const [state, setState] = useState<LogState>(INITIAL_STATE);

  // Memoize the message handler so useEventSource's effect-deps array
  // (which deliberately omits onMessage — see sse.ts) doesn't reopen
  // the connection on every parent rerender.
  const onMessage = useCallback((m: LogEvent) => {
    // Defensive: backend pin guarantees {line: string}, but a malformed
    // payload (e.g., bug in a future log filter) shouldn't crash the
    // render. Treat anything non-string as an empty line.
    const next = typeof m?.line === "string" ? m.line : "";
    setState((prev) => appendLine(prev, next));
  }, []);

  // Issue #53 — gate the SSE handshake on the model's lifecycle status.
  const streamEligible = !NON_LOG_PRODUCING_STATUSES.has(status ?? "");

  const sse = useEventSource<LogEvent>(`/api/models/${modelId}/logs/stream`, {
    onMessage,
    enabled: streamEligible,
  });

  // Sticky-bottom hook + virtuoso ref for programmatic re-stick.
  const sticky = useStickyBottom("stick");
  const virtuosoRef = useRef<VirtuosoHandle>(null);

  if (!streamEligible) {
    // Non-log-producing placeholder (registered / unloading).
    return (
      <div
        role="status"
        className={cn(
          "rounded border p-3 text-xs",
          "border-slate-700 bg-slate-900/50 text-slate-400",
          className,
        )}
      >
        Log stream paused (no active subprocess).
      </div>
    );
  }

  const statusBar = renderStatusMessage(sse);

  // Show the status bar when there's nothing in the buffer yet OR when
  // the stream entered a non-connected state after the fact.
  const showStatusBar =
    statusBar !== null && (state.lines.length === 0 || sse.status !== "connected");

  // Connected-but-empty placeholder (subprocess hasn't produced stdout
  // yet — e.g. vLLM loading 20GB of weights).
  const showEmptyPlaceholder = sse.status === "connected" && state.lines.length === 0;

  if (showEmptyPlaceholder) {
    return (
      <div
        role="status"
        className={cn(
          "rounded border p-3 text-xs",
          "border-slate-700 bg-slate-900/50 text-slate-400",
          className,
        )}
      >
        (no log lines yet)
      </div>
    );
  }

  if (state.lines.length === 0 && statusBar !== null) {
    // Pre-first-line placeholder.
    const role = statusBar.tone === "error" ? "alert" : "status";
    const toneClass =
      statusBar.tone === "error"
        ? "border-red-700/60 bg-red-950/40 text-red-200"
        : statusBar.tone === "warn"
          ? "border-amber-700/60 bg-amber-950/30 text-amber-200"
          : "border-slate-700 bg-slate-900/50 text-slate-400";
    return (
      <div
        role={role}
        className={cn("rounded border p-3 text-xs", toneClass, className)}
      >
        {statusBar.text}
      </div>
    );
  }

  return (
    <div className={cn("space-y-2", className)}>
      {showStatusBar && statusBar !== null && (
        // Inline status bar above the log when we already have lines.
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
        // Eviction marker. Lives outside the virtuoso list (as a fixed
        // header row above the scroll region) so it's always visible
        // regardless of scroll position — the operator should never be
        // surprised that older lines were dropped silently.
        <div
          role="status"
          className="rounded-t border border-b-0 border-slate-700 bg-slate-900/70 px-3 py-1 font-mono text-[11px] text-slate-400"
        >
          … {state.elided} older line{state.elided === 1 ? "" : "s"} elided
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
          // role="log" applied via Components.List so screen readers see
          // the virtualized inner scroll container as the live region
          // (implicit aria-live=polite, aria-atomic=false). Without this
          // override Virtuoso wraps everything in a plain <div>.
          components={{
            List: LogList,
          }}
          style={{ height: heightPx }}
          data={state.lines}
          followOutput={sticky.followOutput}
          // Generous bottom tolerance (default is 4px; a log row is ~20px).
          // During a fast append burst a freshly-rendered row briefly sits
          // below the viewport before the follow-scroll lands; without slack
          // Virtuoso would report not-at-bottom for that frame and pop the
          // "Jump to latest" button on/off. ~3 rows of tolerance swallows it.
          atBottomThreshold={64}
          atBottomStateChange={sticky.onAtBottomStateChange}
          computeItemKey={(_idx, line) => line.id}
          itemContent={(_idx, line) => (
            <div className="px-3 py-0 font-mono text-xs leading-5">
              <AnsiLog text={line.text} />
            </div>
          )}
        />

        {sticky.mode === "free" && state.lines.length > 0 && (
          <Button
            type="button"
            size="sm"
            variant="secondary"
            className="absolute bottom-2 right-2 shadow-lg"
            onClick={() => {
              virtuosoRef.current?.scrollToIndex({
                index: state.lines.length - 1,
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

// Virtuoso's `List` component override. We stamp role="log" + aria-label
// onto the scroll container so screen readers identify it as a polite
// live region, matching the pre-#75 component contract.
const LogList = forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement> & { context?: unknown }
>(function LogListImpl(props, ref) {
  const { context: _context, ...rest } = props;
  return (
    <div
      ref={ref}
      role="log"
      aria-label="Model log stream"
      {...rest}
    />
  );
});
