'use client';
// SSE chat-stream hook for the /chat playground (S8 of the
// vllm-warden overhaul, see plan §S8).
//
// The OpenAI-compatible upstream emits chat-completions over SSE in this
// shape:
//
//   data: {"choices":[{"delta":{"content":"Hi"}, ...}], ...}\n\n
//   data: {"choices":[{"delta":{"content":" there"}, ...}], ...}\n\n
//   data: [DONE]\n\n
//
// We POST to the server-side proxy at `/api/chat/completions` (JWT- and
// CSRF-gated by authFetch) and stream the body. EventSource is NOT used
// because:
//
//  - EventSource is GET-only; chat needs POST with a JSON body.
//  - EventSource silently auto-reconnects on disconnect, which would
//    corrupt the conversation state (re-deliver tokens). We need
//    deterministic single-shot streams that we OWN the lifecycle of.
//  - EventSource has no AbortController integration — the "Stop"
//    affordance in the composer hinges on abort.
//
// Cancellation contract: when `abort()` is called the AbortController
// fires, fetch's underlying socket closes, the server-side generator's
// finally block runs (releasing the priority scheduler slot and
// decrementing app.state.chat_active_requests), and the partial assistant
// text already received is preserved in the messages array — this is
// session-only history, so half a reply still has value to the user.

import { useCallback, useRef, useState } from 'react';
import { authFetch } from '@/lib/auth-fetch';

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

interface ChatStreamParams {
  model: string;
  messages: ChatMessage[];
  temperature: number;
  max_tokens: number;
}

type ChatPhase = 'idle' | 'streaming' | 'aborted' | 'error';

interface UseChatStreamReturn {
  /** Current phase — drives composer/abort button enablement. */
  phase: ChatPhase;
  /** Latest assistant chunk text being streamed (live token-by-token). */
  streamingText: string;
  /** Human-readable error message when phase === 'error', else null. */
  errorMessage: string | null;
  /**
   * Kick off a streaming completion.
   *
   * Resolves to the FULL accumulated assistant text once the upstream
   * sends `[DONE]` or the stream ends. On abort, resolves to whatever
   * partial text we had — the caller decides whether to keep it or
   * drop it (the /chat page keeps it, per S8 plan).
   *
   * On unrecoverable error (HTTP 4xx/5xx with body, network drop
   * before any tokens) it sets phase='error' and resolves to ''.
   */
  send: (params: ChatStreamParams) => Promise<string>;
  /** Cancel the in-flight stream. No-op if not streaming. */
  abort: () => void;
}

/**
 * Parse an OpenAI-style chat-completion SSE delta blob.
 *
 * The proxy preserves the upstream JSON verbatim, so we only need to
 * pluck `choices[0].delta.content`. Other fields (finish_reason, usage)
 * are ignored — the FE rolls token counters off the assistant text
 * length only when it cares (which today it doesn't, see plan §S8 OUT
 * OF SCOPE).
 *
 * Tolerant by design: a malformed chunk produces an empty delta rather
 * than aborting the stream. Upstream has been known to emit keep-alive
 * blanks (especially during long prefill) and we don't want them
 * surfaced as errors.
 */
function extractDelta(payload: string): string {
  try {
    const obj = JSON.parse(payload) as {
      choices?: Array<{ delta?: { content?: string | null } }>;
    };
    return obj.choices?.[0]?.delta?.content ?? '';
  } catch {
    return '';
  }
}

/**
 * Split a multi-event SSE buffer into (events, leftover) where leftover
 * is the trailing partial-event bytes (no terminating "\n\n" yet).
 *
 * SSE separator is "\n\n" per the spec, but the upstream uses "\r\n\r\n"
 * sometimes — we accept either by normalising "\r" out of the buffer
 * before splitting. The trade-off: we destroy "\r" inside event data
 * (which OpenAI never emits inside JSON anyway) in exchange for
 * single-codepath parsing.
 */
function splitSseEvents(buffer: string): { events: string[]; leftover: string } {
  const normalised = buffer.replace(/\r/g, '');
  const parts = normalised.split('\n\n');
  const leftover = parts.pop() ?? '';
  return { events: parts, leftover };
}

/**
 * Pull a `data:` line's payload out of a single SSE event. SSE events
 * can span multiple lines and may carry comments (`:` prefix), event
 * names (`event:`), and IDs (`id:`). OpenAI's chat-completions stream
 * only uses `data:` so we collect those and concatenate.
 */
function extractDataPayload(event: string): string | null {
  const lines = event.split('\n');
  const dataLines: string[] = [];
  for (const line of lines) {
    if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).trimStart());
    }
  }
  if (dataLines.length === 0) return null;
  return dataLines.join('\n');
}

export function useChatStream(): UseChatStreamReturn {
  const [phase, setPhase] = useState<ChatPhase>('idle');
  const [streamingText, setStreamingText] = useState('');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const controllerRef = useRef<AbortController | null>(null);
  // Mutable accumulator — useState would trigger a render per token
  // which is fine for short replies but tanks at 50+ tok/s. We mirror
  // the final value into streamingText on each chunk for the live
  // view, but keep this ref as the authoritative accumulator for the
  // resolved Promise (so a stale closure can't return partial text).
  const accumulatorRef = useRef('');

  const abort = useCallback(() => {
    controllerRef.current?.abort();
  }, []);

  const send = useCallback(
    async (params: ChatStreamParams): Promise<string> => {
      // Re-entrancy guard. If a previous stream is still live (caller
      // bug — should call abort() first), abort it. We don't reject
      // the new send — that surprised users in pilot testing.
      controllerRef.current?.abort();

      const controller = new AbortController();
      controllerRef.current = controller;
      accumulatorRef.current = '';
      setStreamingText('');
      setErrorMessage(null);
      setPhase('streaming');

      let response: Response;
      try {
        response = await authFetch('/api/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ...params,
            stream: true,
          }),
          signal: controller.signal,
        });
      } catch (err) {
        // AbortError lands here when abort() fires before headers arrive
        // — treat as a clean cancel, not an error.
        if ((err as { name?: string }).name === 'AbortError') {
          setPhase('aborted');
          return accumulatorRef.current;
        }
        setErrorMessage(
          err instanceof Error ? err.message : 'network error',
        );
        setPhase('error');
        return '';
      }

      if (!response.ok) {
        // Surface the proxy's HTTPException detail when present.
        // 409 = playground token not initialised (UI should retry
        // ensure(); we surface a friendly message). 401/403 are
        // handled by authFetch's replay path already.
        let detail = `request failed (HTTP ${response.status})`;
        try {
          const body = (await response.json()) as { detail?: unknown };
          if (typeof body.detail === 'string') {
            detail = body.detail;
          } else if (body.detail) {
            detail = JSON.stringify(body.detail);
          }
        } catch {
          /* body wasn't JSON — keep the generic message */
        }
        setErrorMessage(detail);
        setPhase('error');
        return '';
      }

      const body = response.body;
      if (!body) {
        // Headers say 200 but no body — should never happen with our
        // proxy but guard so we don't hang the UI.
        setErrorMessage('no response body');
        setPhase('error');
        return '';
      }

      const reader = body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      let done = false;
      let sawDone = false;

      try {
        while (!done) {
          const { value, done: readerDone } = await reader.read();
          done = readerDone;
          if (value) {
            buffer += decoder.decode(value, { stream: !done });
            const { events, leftover } = splitSseEvents(buffer);
            buffer = leftover;
            for (const ev of events) {
              const payload = extractDataPayload(ev);
              if (payload === null) continue;
              if (payload === '[DONE]') {
                sawDone = true;
                continue;
              }
              const delta = extractDelta(payload);
              if (delta) {
                accumulatorRef.current += delta;
                setStreamingText(accumulatorRef.current);
              }
            }
          }
        }
      } catch (err) {
        // AbortError on read() — clean cancel. The reader is auto-
        // released by fetch's internals when the controller aborts;
        // we still attempt to cancel it explicitly so future
        // throwing of the reader doesn't break.
        if ((err as { name?: string }).name === 'AbortError') {
          try {
            await reader.cancel();
          } catch {
            /* already done */
          }
          setPhase('aborted');
          return accumulatorRef.current;
        }
        setErrorMessage(
          err instanceof Error ? err.message : 'stream interrupted',
        );
        setPhase('error');
        return accumulatorRef.current;
      } finally {
        controllerRef.current = null;
      }

      // Reaching here means the upstream closed the stream cleanly.
      // `[DONE]` arriving exactly once is the happy path; missing it
      // (some proxies omit it on shorter completions) is acceptable
      // as long as we got at least one token.
      void sawDone;
      setPhase('idle');
      return accumulatorRef.current;
    },
    [],
  );

  return { phase, streamingText, errorMessage, send, abort };
}
