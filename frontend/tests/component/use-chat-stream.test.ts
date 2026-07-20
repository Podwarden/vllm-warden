// Tests for the use-chat-stream hook (S8 of the vllm-warden overhaul).
//
// The hook is the SSE parsing + cancellation kernel for /chat. Its three
// critical contracts:
//
//   1. SSE delta parsing — produces the accumulated assistant text in
//      `streamingText`, ignoring "[DONE]" sentinels and tolerating
//      keep-alive blanks.
//   2. Abort — calling abort() during a stream resolves the promise to
//      whatever partial text we had, leaves phase = 'aborted', and does
//      NOT throw.
//   3. Error envelopes — a non-2xx response is surfaced as
//      phase = 'error' with the detail string.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';
import { useChatStream } from '@/lib/use-chat-stream';

// Build a Response whose body is a stream we can push chunks into. The
// returned `push` function appends a UTF-8 encoded chunk; `close` ends
// the stream; `error` rejects the next pending read() with the supplied
// reason (used to simulate AbortError propagation since real fetch's
// signal-to-body wiring is not modelled by our stub). We use this in
// lieu of msw / nock because the hook only touches the global fetch and
// the test is about parsing behaviour.
function makeStreamingResponse(): {
  response: Response;
  push: (chunk: string) => void;
  close: () => void;
  error: (reason: unknown) => void;
} {
  let controller!: ReadableStreamDefaultController<Uint8Array>;
  const enc = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(c) {
      controller = c;
    },
  });
  return {
    response: new Response(body, {
      status: 200,
      headers: { 'Content-Type': 'text/event-stream' },
    }),
    push: (chunk: string) => controller.enqueue(enc.encode(chunk)),
    close: () => controller.close(),
    error: (reason: unknown) => controller.error(reason),
  };
}

beforeEach(() => {
  // Skip the auth-fetch refresh dance by seeding both creds.
  setAccessToken('test-jwt', 3600);
  setCsrfToken('test-csrf');
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('useChatStream: SSE parsing', () => {
  it('accumulates content deltas and resolves to the full text', async () => {
    const { response, push, close } = makeStreamingResponse();
    vi.stubGlobal('fetch', vi.fn(async () => response));

    const { result } = renderHook(() => useChatStream());

    let sendPromise!: Promise<string>;
    act(() => {
      sendPromise = result.current.send({
        model: 'm-1',
        messages: [{ role: 'user', content: 'hi' }],
        temperature: 0.7,
        max_tokens: 32,
      });
    });

    push('data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n');
    push('data: {"choices":[{"delta":{"content":" world"}}]}\n\n');
    push('data: [DONE]\n\n');
    close();

    const full = await sendPromise;
    expect(full).toBe('Hello world');
    // setPhase('idle') runs INSIDE the send() promise resolution, but
    // React batches state updates so the hook's externally observable
    // phase may still be 'streaming' for one microtask. waitFor() yields
    // until the test renderer flushes the queued update.
    await waitFor(() => {
      expect(result.current.phase).toBe('idle');
    });
  });

  it('tolerates split-chunk events (multi-byte chunk boundary)', async () => {
    const { response, push, close } = makeStreamingResponse();
    vi.stubGlobal('fetch', vi.fn(async () => response));

    const { result } = renderHook(() => useChatStream());

    let sendPromise!: Promise<string>;
    act(() => {
      sendPromise = result.current.send({
        model: 'm-1',
        messages: [{ role: 'user', content: 'hi' }],
        temperature: 0.7,
        max_tokens: 32,
      });
    });

    // Split the first event across two pushes — buffering must reassemble.
    push('data: {"choices":[{"delta":{"con');
    push('tent":"A"}}]}\n\ndata: {"choices":[{"delta":{"content":"B"}}]}\n\n');
    push('data: [DONE]\n\n');
    close();

    const full = await sendPromise;
    expect(full).toBe('AB');
  });

  it('ignores non-data SSE fields and malformed JSON', async () => {
    const { response, push, close } = makeStreamingResponse();
    vi.stubGlobal('fetch', vi.fn(async () => response));

    const { result } = renderHook(() => useChatStream());

    let sendPromise!: Promise<string>;
    act(() => {
      sendPromise = result.current.send({
        model: 'm-1',
        messages: [{ role: 'user', content: 'hi' }],
        temperature: 0.7,
        max_tokens: 32,
      });
    });

    push(': keep-alive comment\n\n');
    push('event: ping\ndata: {"not":"a delta"}\n\n');
    push('data: not-json-at-all\n\n');
    push('data: {"choices":[{"delta":{"content":"ok"}}]}\n\n');
    push('data: [DONE]\n\n');
    close();

    const full = await sendPromise;
    expect(full).toBe('ok');
  });
});

describe('useChatStream: abort', () => {
  it('preserves partial text on abort and sets phase=aborted', async () => {
    const { response, push, error } = makeStreamingResponse();
    // Hook the AbortSignal up to the ReadableStream controller: when the
    // hook calls abort() on its internal controller, the signal fires
    // here, and we error() the source stream — which is what real fetch
    // would do internally. We cannot call response.body.cancel() because
    // the hook's reader has already locked it via getReader(); attempting
    // to cancel a locked stream raises "Invalid state: ReadableStream is
    // locked", which surfaces as an unhandled rejection and trips Vitest.
    vi.stubGlobal('fetch', vi.fn(async (_url, init) => {
      const signal = (init as RequestInit | undefined)?.signal;
      if (signal) {
        signal.addEventListener('abort', () => {
          const reason = new DOMException('aborted', 'AbortError');
          error(reason);
        });
      }
      return response;
    }));

    const { result } = renderHook(() => useChatStream());

    let sendPromise!: Promise<string>;
    act(() => {
      sendPromise = result.current.send({
        model: 'm-1',
        messages: [{ role: 'user', content: 'hi' }],
        temperature: 0.7,
        max_tokens: 32,
      });
    });

    push('data: {"choices":[{"delta":{"content":"partial"}}]}\n\n');

    await waitFor(() => {
      expect(result.current.streamingText).toBe('partial');
    });

    act(() => {
      result.current.abort();
    });

    const final = await sendPromise;
    expect(final).toBe('partial');
    await waitFor(() => {
      expect(result.current.phase).toBe('aborted');
    });
  });
});

describe('useChatStream: error envelopes', () => {
  it('surfaces 409 detail when ensure() has not run', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) => {
        if (url === '/api/csrf') {
          return new Response(JSON.stringify({ csrf: 'test-csrf' }), {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          });
        }
        return new Response(
          JSON.stringify({ detail: 'playground token not initialised' }),
          {
            status: 409,
            headers: { 'Content-Type': 'application/json' },
          },
        );
      }),
    );

    const { result } = renderHook(() => useChatStream());

    let sendPromise!: Promise<string>;
    act(() => {
      sendPromise = result.current.send({
        model: 'm-1',
        messages: [{ role: 'user', content: 'hi' }],
        temperature: 0.7,
        max_tokens: 32,
      });
    });

    const out = await sendPromise;
    expect(out).toBe('');
    await waitFor(() => {
      expect(result.current.phase).toBe('error');
      expect(result.current.errorMessage).toBe(
        'playground token not initialised',
      );
    });
  });
});
