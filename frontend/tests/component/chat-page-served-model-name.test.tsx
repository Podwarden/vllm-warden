// Regression test for the /chat playground served_model_name wire fix.
//
// Bug history: handleSubmit and handleRegenerate posted `model: modelId`
// where modelId is the picker's internal selection key (the row's `id`,
// an opaque hash like "dfd0982d2d9e5478"). The backend proxy at
// app/proxy/routes.py looks up engines by served_model_name, so the
// request 404'd with `model 'dfd0982d2d9e5478' is not loaded` for any
// model whose served_model_name differs from its id.
//
// The bug was latent for the entire history of the chat playground
// because all prior models on the test fleet had served_model_name == id.
// Qwen3.6-27B was deployed 2026-05-22 with served_model_name="qwen3.6-27b"
// distinct from its id, and every chat send started 404'ing on
// https://vllm.protrener.com/ui/chat.
//
// Fix: resolve modelId -> served_model_name via the already-loaded
// `loadedModels` list before placing it on the wire. The picker key
// (state) remains the id; only the wire field changes.
//
// This test renders the page with a stubbed models list whose id and
// served_model_name are deliberately distinct, triggers a send, and
// asserts the outgoing POST body uses served_model_name, NOT the id.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  render,
  screen,
  cleanup,
  waitFor,
  fireEvent,
} from '@testing-library/react';
import { SWRConfig } from 'swr';
import ChatPage from '@/app/chat/page';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';

function renderPage() {
  return render(
    <SWRConfig
      value={{
        provider: () => new Map(),
        dedupingInterval: 0,
        revalidateOnFocus: false,
        revalidateOnReconnect: false,
      }}
    >
      <ChatPage />
    </SWRConfig>,
  );
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

// Build a streaming SSE Response whose body emits a single
// chat-completion chunk followed by [DONE]. The hook reads the body
// via getReader().read() in a loop; an empty stream would leave it
// hanging, so we provide one synthetic delta and immediately close.
function sseResponse(): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(
        encoder.encode(
          'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n',
        ),
      );
      controller.enqueue(encoder.encode('data: [DONE]\n\n'));
      controller.close();
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { 'Content-Type': 'text/event-stream' },
  });
}

beforeEach(() => {
  setAccessToken('test-jwt', 3600);
  setCsrfToken('test-csrf');
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  setAccessToken(null);
  setCsrfToken(null);
});

describe('ChatPage — sends served_model_name on the wire, not the internal id', () => {
  it('handleSubmit posts model = served_model_name', async () => {
    // Capture every call to /api/chat/completions for assertion.
    const completionCalls: Array<{ url: string; body: unknown }> = [];

    const fetchMock = vi.fn(
      async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === 'string' ? input : input.toString();
        const method = (init?.method ?? 'GET').toUpperCase();

        if (url === '/api/auth/refresh') {
          return json({ access_token: 'test-jwt-refreshed' });
        }
        if (url === '/api/csrf') {
          return json({ csrf: 'test-csrf' });
        }
        if (url === '/api/chat/playground/ensure' && method === 'POST') {
          return json({ token_id: 'tok-test', created: false });
        }
        if (url === '/api/models') {
          // Deliberately distinct id and served_model_name — this is
          // the exact shape that broke production on 2026-05-22 when
          // Qwen3.6-27B was deployed with served_model_name="qwen3.6-27b"
          // and id="dfd0982d2d9e5478".
          return json({
            models: [
              {
                id: 'abc123',
                served_model_name: 'friendly-name',
                status: 'loaded',
              },
            ],
          });
        }
        if (url === '/api/chat/completions' && method === 'POST') {
          const raw = init?.body;
          const parsed =
            typeof raw === 'string' ? JSON.parse(raw) : { _nonString: true };
          completionCalls.push({ url, body: parsed });
          return sseResponse();
        }
        return json({}, 404);
      },
    );
    vi.stubGlobal('fetch', fetchMock);

    renderPage();

    // Wait for the picker to populate (proves ensure() + /api/models
    // both resolved and the model auto-selected).
    const picker = (await screen.findByTestId(
      'chat-model-picker',
    )) as HTMLSelectElement;
    await waitFor(() => {
      expect(picker.value).toBe('abc123');
    });

    // Type a prompt and click send. We type via fireEvent.change rather
    // than typing key-by-key — this is a wire-shape test, not a UX test,
    // so the simpler path is sufficient.
    const input = (await screen.findByTestId(
      'chat-input',
    )) as HTMLTextAreaElement;
    fireEvent.change(input, { target: { value: 'hello' } });

    const sendButton = await screen.findByTestId('chat-send');
    fireEvent.click(sendButton);

    // Wait for the proxy call to happen. The hook posts as soon as the
    // user message is appended, so this resolves on the next tick.
    await waitFor(() => {
      expect(completionCalls.length).toBeGreaterThanOrEqual(1);
    });

    const body = completionCalls[0].body as { model?: unknown };
    expect(body.model).toBe('friendly-name');
    expect(body.model).not.toBe('abc123');
  });
});
