// Component tests for the /chat playground page wiring (#147 regression).
//
// The chat page calls `GET /api/models` and feeds the result into the
// Composer's model picker. The original S8 wiring expected a bare
// array but the backend has always returned `{ models: [...] }` —
// every model row was silently filtered out and the picker showed
// "no loaded models" no matter how many engines were live (#147).
//
// These tests pin the envelope contract end-to-end at the page level:
//  - happy path: one loaded model surfaces in the picker
//  - filtering:  non-loaded statuses are excluded
//  - empty:      truly-empty backend keeps the "no loaded models" copy
//  - defensive:  malformed wire payload (bare array) doesn't crash
//
// Each test renders <ChatPage /> inside a fresh SWRConfig (per
// stats-page.test.tsx) so the SWR cache doesn't bleed across tests,
// and stubs `fetch` to answer `/api/auth/refresh`, `/api/csrf`,
// `/api/chat/playground/ensure`, and `/api/models`.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  render,
  screen,
  cleanup,
  waitFor,
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

interface ChatFixtures {
  // What `/api/models` returns. Test passes either the envelope shape,
  // a bare array (defensive case), or an explicit payload override.
  modelsPayload?: unknown;
  // What `/api/chat/playground/ensure` returns. Defaults to a stub
  // success envelope so the page enters its "ensured" state quickly.
  ensurePayload?: unknown;
  ensureStatus?: number;
}

function installFetchStub(fixtures: ChatFixtures = {}) {
  const ensurePayload =
    fixtures.ensurePayload ?? { token_id: 'tok-test', created: false };
  const modelsPayload = fixtures.modelsPayload ?? { models: [] };
  const mock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : input.toString();
    const method = (init?.method ?? 'GET').toUpperCase();
    if (url === '/api/auth/refresh') {
      return json({ access_token: 'test-jwt-refreshed' });
    }
    if (url === '/api/csrf') {
      return json({ csrf: 'test-csrf' });
    }
    if (url === '/api/chat/playground/ensure' && method === 'POST') {
      return json(ensurePayload, fixtures.ensureStatus ?? 200);
    }
    if (url === '/api/models') {
      return json(modelsPayload);
    }
    return json({}, 404);
  });
  vi.stubGlobal('fetch', mock);
  return mock;
}

beforeEach(() => {
  // Preload an access token + CSRF so authFetch doesn't run a refresh
  // round-trip during the test (the fetch stub still answers /refresh
  // for any caller that does ask, as belt-and-suspenders).
  setAccessToken('test-jwt', 3600);
  setCsrfToken('test-csrf');
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  setAccessToken(null);
  setCsrfToken(null);
});

describe('ChatPage — model selector wiring (#147)', () => {
  it('populates the picker from `{ models: [...] }` when a model is loaded', async () => {
    installFetchStub({
      modelsPayload: {
        models: [
          {
            id: 'mdl-1',
            served_model_name: 'qwen-tiny',
            status: 'loaded',
          },
        ],
      },
    });

    renderPage();

    // Picker is rendered immediately (the page renders Composer
    // unconditionally), so the assertion is on the option content
    // after SWR resolves the /api/models payload.
    const picker = (await screen.findByTestId(
      'chat-model-picker',
    )) as HTMLSelectElement;

    await waitFor(() => {
      // Option list should contain the served_model_name we returned.
      // We assert via the textContent rather than the value because
      // Composer auto-selects the first loaded model and the test is
      // primarily pinning "the picker is not empty".
      expect(picker.textContent).toContain('qwen-tiny');
    });

    // The Composer renders "no loaded models" as the only option when
    // models.length === 0 — the regression. Pin that copy is GONE
    // once the loaded model is in the picker.
    expect(picker.textContent).not.toContain('no loaded models');

    // Auto-selection of the first loaded model means the picker's
    // value should be the model id, not the empty placeholder.
    await waitFor(() => {
      expect(picker.value).toBe('mdl-1');
    });
  });

  it('excludes models whose status is not "loaded"', async () => {
    installFetchStub({
      modelsPayload: {
        models: [
          { id: 'm-loaded', served_model_name: 'qwen', status: 'loaded' },
          { id: 'm-pulled', served_model_name: 'mistral', status: 'pulled' },
          { id: 'm-loading', served_model_name: 'llama', status: 'loading' },
          {
            id: 'm-failed',
            served_model_name: 'broken',
            status: 'failed',
          },
        ],
      },
    });

    renderPage();

    const picker = (await screen.findByTestId(
      'chat-model-picker',
    )) as HTMLSelectElement;

    await waitFor(() => {
      expect(picker.textContent).toContain('qwen');
    });
    expect(picker.textContent).not.toContain('mistral');
    expect(picker.textContent).not.toContain('llama');
    expect(picker.textContent).not.toContain('broken');
  });

  it('renders the empty-state copy when no models are loaded', async () => {
    installFetchStub({
      modelsPayload: {
        models: [
          { id: 'm-pulled', served_model_name: 'mistral', status: 'pulled' },
        ],
      },
    });

    renderPage();

    const picker = (await screen.findByTestId(
      'chat-model-picker',
    )) as HTMLSelectElement;

    // Composer renders a single <option value="">no loaded models</option>
    // when there are zero candidates. Pin that contract is preserved
    // for the genuine empty case (the regression flipped this from
    // "false negative" to "always negative").
    await waitFor(() => {
      expect(picker.textContent).toContain('no loaded models');
    });
    expect(picker.value).toBe('');
  });

  it('does not crash when the backend returns a bare array instead of the envelope', async () => {
    // Defensive: if a future contract change accidentally ships a bare
    // array, the page should degrade to "no models" gracefully rather
    // than throw inside a render path.
    installFetchStub({
      modelsPayload: [
        { id: 'mdl-1', served_model_name: 'qwen', status: 'loaded' },
      ],
    });

    renderPage();

    const picker = (await screen.findByTestId(
      'chat-model-picker',
    )) as HTMLSelectElement;
    await waitFor(() => {
      expect(picker.textContent).toContain('no loaded models');
    });
  });
});
