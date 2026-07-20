'use client';
// Message stream view for /chat (S8 of the vllm-warden overhaul).
//
// Layout decisions follow the frontend-design skill output captured in
// the slice plan:
//
//  - Operator-cockpit aesthetic: utilitarian, monospace body, emerald
//    accents on slate-900. NOT a luxury chat experience — this is a
//    diagnostic playground for model engineers.
//  - Density: 14px/1.5 mono body; assistant + user share the same
//    layout and only differ in marker (left rail color + role label).
//    A bubble layout would suggest "messaging app", which leads users
//    to expect persistence — we deliberately do NOT persist.
//  - Streaming cursor: blinking emerald block character (▎) appended
//    to the assistant's last message while phase === 'streaming'.
//    Picked over an ellipsis because vLLM emits tokens at human-
//    readable rates and the cursor moves naturally character-by-
//    character, making the live wire feel tangible.
//  - Secondary controls (Copy, Regenerate-last) are hover-only on
//    assistant messages. Always-visible buttons crowd the view at
//    high message density and the playground's user is keyboard-
//    fluent enough to discover hover affordances.

import { Copy, RotateCcw } from 'lucide-react';
import { useCallback, useState } from 'react';
import type { ChatMessage } from '@/lib/use-chat-stream';
import { copyToClipboard } from '@/lib/utils';

interface MessageListProps {
  messages: ChatMessage[];
  /** Streaming partial text — rendered as an extra assistant message. */
  streamingText: string;
  /** True iff a stream is currently in-flight (drives cursor + dim state). */
  isStreaming: boolean;
  /**
   * Regenerate the last assistant turn. The page handler drops the last
   * assistant message and resends the user message above it. We surface
   * it as a button on each assistant message but only the most recent
   * one is meaningful — older ones would require dropping intervening
   * turns which is a non-obvious destructive operation we'd rather not
   * tempt the operator into.
   */
  onRegenerate: () => void;
  /** Index of the last assistant message (or -1 if none). */
  lastAssistantIndex: number;
}

function useClipboardCopy(): {
  copy: (text: string) => Promise<void>;
  /** index of the last-copied message, briefly. */
  copiedIndex: number | null;
  setCopiedIndex: (i: number | null) => void;
} {
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const copy = useCallback(async (text: string) => {
    // copyToClipboard (lib/utils.ts) handles the navigator.clipboard ->
    // execCommand fallback for non-secure contexts (#149). It throws on
    // total failure; we swallow that here because the chat copy button
    // is a tertiary, hover-only affordance and there is no inline
    // toast surface to render an error into without crowding the
    // message column.
    try {
      await copyToClipboard(text);
    } catch {
      return;
    }
  }, []);
  return { copy, copiedIndex, setCopiedIndex };
}

export function MessageList({
  messages,
  streamingText,
  isStreaming,
  onRegenerate,
  lastAssistantIndex,
}: MessageListProps) {
  const { copy, copiedIndex, setCopiedIndex } = useClipboardCopy();

  if (messages.length === 0 && !isStreaming) {
    return <EmptyState />;
  }

  return (
    <div className="space-y-4 px-4 py-6">
      {messages.map((msg, idx) => {
        const isAssistant = msg.role === 'assistant';
        const isLastAssistant =
          isAssistant && idx === lastAssistantIndex && !isStreaming;
        const justCopied = copiedIndex === idx;
        return (
          <article
            key={idx}
            data-role={msg.role}
            data-testid="chat-message"
            className="group relative grid grid-cols-[6px_1fr] gap-3"
          >
            <RoleRail role={msg.role} />
            <div className="min-w-0">
              <RoleLabel role={msg.role} />
              <div className="font-mono text-sm leading-relaxed text-slate-100 whitespace-pre-wrap break-words">
                {msg.content}
              </div>
              {isAssistant && (
                <div className="mt-2 flex gap-2 opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100">
                  <button
                    type="button"
                    onClick={async () => {
                      await copy(msg.content);
                      setCopiedIndex(idx);
                      window.setTimeout(() => setCopiedIndex(null), 1200);
                    }}
                    className="inline-flex items-center gap-1 rounded border border-slate-700 bg-slate-800/60 px-2 py-1 text-xs text-slate-300 hover:border-emerald-500 hover:text-emerald-300 transition-colors"
                    aria-label="Copy message"
                  >
                    <Copy className="h-3 w-3" aria-hidden="true" />
                    <span aria-live="polite">{justCopied ? 'Copied' : 'Copy'}</span>
                  </button>
                  {isLastAssistant && (
                    <button
                      type="button"
                      onClick={onRegenerate}
                      className="inline-flex items-center gap-1 rounded border border-slate-700 bg-slate-800/60 px-2 py-1 text-xs text-slate-300 hover:border-emerald-500 hover:text-emerald-300 transition-colors"
                      aria-label="Regenerate last response"
                    >
                      <RotateCcw className="h-3 w-3" aria-hidden="true" />
                      <span>Regenerate</span>
                    </button>
                  )}
                </div>
              )}
            </div>
          </article>
        );
      })}

      {isStreaming && (
        <article
          data-role="assistant"
          data-testid="chat-streaming"
          className="grid grid-cols-[6px_1fr] gap-3"
        >
          <RoleRail role="assistant" />
          <div className="min-w-0">
            <RoleLabel role="assistant" />
            <div className="font-mono text-sm leading-relaxed text-slate-100 whitespace-pre-wrap break-words">
              {streamingText}
              <span
                className="ml-0.5 inline-block w-[0.55em] -mb-[2px] h-[1.05em] align-text-bottom bg-emerald-400 animate-pulse"
                aria-hidden="true"
              />
            </div>
          </div>
        </article>
      )}
    </div>
  );
}

function RoleRail({ role }: { role: ChatMessage['role'] }) {
  const color =
    role === 'user'
      ? 'bg-slate-600'
      : role === 'assistant'
        ? 'bg-emerald-500'
        : 'bg-amber-500';
  return (
    <div
      aria-hidden="true"
      className={`h-full w-[3px] rounded-full ${color}`}
    />
  );
}

function RoleLabel({ role }: { role: ChatMessage['role'] }) {
  const label = role === 'user' ? 'YOU' : role === 'assistant' ? 'MODEL' : 'SYS';
  return (
    <div className="mb-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-500">
      {label}
    </div>
  );
}

function EmptyState() {
  // The empty state intentionally avoids "Ask anything..." or marketing
  // copy. The operator is here to probe a model — give them the prompt
  // structure they actually need: pick a model, set the knobs, send.
  return (
    <div
      data-testid="chat-empty"
      className="flex h-full min-h-[40vh] flex-col items-center justify-center px-6 py-12 text-center"
    >
      <div className="font-mono text-xs uppercase tracking-[0.22em] text-slate-500">
        Playground · Session-only · No persistence
      </div>
      <div className="mt-4 max-w-md text-sm text-slate-400">
        Select a loaded model, tune temperature and max tokens, then send a
        prompt. Conversations are kept in this tab only and discarded on
        refresh.
      </div>
    </div>
  );
}
