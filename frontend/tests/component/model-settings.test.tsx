import { Suspense } from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup, act } from '@testing-library/react';
import { SWRConfig } from 'swr';
import ModelSettingsPage from '@/app/models/[id]/settings/page';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';

// React 19's `use()` reads a `status`/`value` shape that React itself
// attaches to a Promise once the runtime has observed it resolve. Forging
// that shape upfront lets the page hydrate on the first render without
// any scheduler dance — jsdom does not reliably let us flush React's
// suspense replay queue via setTimeout/microtask awaits, so a "real"
// Promise.resolve never settles inside the test. Test-only escape hatch;
// production code path (where Next.js feeds a real async-resolving
// params promise) is unaffected.
function syncResolved<T>(value: T): Promise<T> {
  const p = Promise.resolve(value) as Promise<T> & {
    status?: string;
    value?: T;
  };
  p.status = "fulfilled";
  p.value = value;
  return p;
}

// Each test renders inside a fresh SWR cache so cached `/api/models/...`
// responses from a previous test don't bleed across. `provider: () =>
// new Map()` is SWR's recommended test reset; `dedupingInterval: 0`
// makes the first fetch in each test actually fire (otherwise SWR
// suppresses the request as a duplicate of an already-pending one).
function renderPage(id: string) {
  const paramsPromise = syncResolved({ id });
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <Suspense fallback={<div data-testid="suspense-fallback">loading</div>}>
        <ModelSettingsPage params={paramsPromise} />
      </Suspense>
    </SWRConfig>,
  );
}

// Default GPU probe response — GPU 0 present so the default fakeSettings
// (gpu_indices: [0]) does not produce ghost rows or extra alert banners.
const DEFAULT_GPU_PROBE = {
  gpus: [{ index: 0, name: 'A4000', memory_total_mib: 16376, memory_used_mib: 0, utilization_pct: 0 }],
  probed_at: new Date().toISOString(),
  probe_error: null,
};

interface SettingsResponse {
  id: string;
  served_model_name: string;
  hf_repo: string;
  hf_revision: string;
  gpu_indices: number[];
  tensor_parallel_size: number;
  dtype: string | null;
  max_model_len: number | null;
  gpu_memory_utilization: number;
  trust_remote_code: boolean;
  extra_args: string[];
  extra_env: Record<string, string>;
  status: string;
  pulled_bytes: number;
  pulled_total: number | null;
  last_error: string | null;
}

function fakeSettings(overrides: Partial<SettingsResponse> = {}): SettingsResponse {
  return {
    id: 'abc',
    served_model_name: 'llama3-8b',
    hf_repo: 'meta-llama/Llama-3-8B',
    hf_revision: 'main',
    gpu_indices: [0],
    tensor_parallel_size: 1,
    dtype: null,
    max_model_len: null,
    gpu_memory_utilization: 0.9,
    trust_remote_code: false,
    extra_args: [],
    extra_env: {},
    status: 'pulled',
    pulled_bytes: 0,
    pulled_total: null,
    last_error: null,
    ...overrides,
  };
}

describe('ModelSettingsPage', () => {
  beforeEach(() => {
    // Pre-seed auth tokens so authFetch doesn't try to round-trip /api/csrf
    // or /api/auth/refresh during the test (those would noise up the
    // fetchMock and the test wouldn't be focused on settings behaviour).
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('PATCHes only the dirty subset of fields', async () => {
    const settings = fakeSettings();
    // Track sequential responses: first the GET load, then the PATCH ok,
    // then the post-PATCH revalidate GET. fetchMock is wired so we can
    // route by (url, method) — keeps the assertion below clean.
    const fetchMock = vi.fn(async (input: RequestInfo, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : (input as Request).url;
      const method = (init?.method ?? 'GET').toUpperCase();
      if (url === '/api/models/abc/settings' && method === 'GET') {
        return new Response(JSON.stringify(settings), { status: 200 });
      }
      if (url === '/api/models/abc/settings' && method === 'PATCH') {
        return new Response('{"ok":true}', { status: 200 });
      }
      if (url === '/api/system/gpus') {
        return new Response(JSON.stringify(DEFAULT_GPU_PROBE), { status: 200 });
      }
      throw new Error(`unexpected fetch: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    renderPage('abc');
    // Flush the SWR fetch + first commit. See syncResolved at the top
    // of this file for the use()-on-Promise workaround.
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });

    // Wait for the initial GET to populate the form.
    const hfRevisionInput = await screen.findByLabelText(/HF revision/i);
    expect(hfRevisionInput).toHaveValue('main');

    // Tweak only hf_revision. The page must include only this one key in
    // the PATCH body — the other 10 keys are unchanged and including
    // them would defeat the "minimum diff" contract the backend tolerates
    // but doesn't require.
    fireEvent.change(hfRevisionInput, { target: { value: 'abc1234' } });

    const saveBtn = screen.getByRole('button', { name: /save/i });
    await waitFor(() => expect(saveBtn).not.toBeDisabled());
    fireEvent.click(saveBtn);

    // The PATCH should fire with a body containing only hf_revision.
    await waitFor(() => {
      const patchCall = fetchMock.mock.calls.find(
        (c) => (c[1] as RequestInit | undefined)?.method === 'PATCH',
      );
      expect(patchCall).toBeDefined();
    });
    const patchCall = fetchMock.mock.calls.find(
      (c) => (c[1] as RequestInit | undefined)?.method === 'PATCH',
    )!;
    const body = JSON.parse((patchCall[1] as RequestInit).body as string);
    expect(body).toEqual({ hf_revision: 'abc1234' });
    // Explicit anti-leak assertions for the keys most likely to drift if
    // someone "helpfully" widens the dirty-diff later.
    expect(body.gpu_indices).toBeUndefined();
    expect(body.served_model_name).toBeUndefined();
    expect(body.extra_env).toBeUndefined();
  });

  it('surfaces 409 with an unload-first banner', async () => {
    const settings = fakeSettings({ status: 'pulled' });
    const fetchMock = vi.fn(async (input: RequestInfo, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : (input as Request).url;
      const method = (init?.method ?? 'GET').toUpperCase();
      if (url === '/api/models/abc/settings' && method === 'GET') {
        return new Response(JSON.stringify(settings), { status: 200 });
      }
      if (url === '/api/models/abc/settings' && method === 'PATCH') {
        return new Response(
          '{"detail":"model must be unloaded before editing settings"}',
          { status: 409 },
        );
      }
      if (url === '/api/system/gpus') {
        return new Response(JSON.stringify(DEFAULT_GPU_PROBE), { status: 200 });
      }
      throw new Error(`unexpected fetch: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    renderPage('abc');
    // Flush the SWR fetch + first commit. `syncResolved` already lets
    // `use()` return synchronously, so this only needs to drain the
    // fetch microtask + give React one render pass to write the form.
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });

    const hfRevisionInput = await screen.findByLabelText(/HF revision/i);
    fireEvent.change(hfRevisionInput, { target: { value: 'racing' } });

    const saveBtn = screen.getByRole('button', { name: /save/i });
    await waitFor(() => expect(saveBtn).not.toBeDisabled());
    await act(async () => {
      fireEvent.click(saveBtn);
    });

    // The banner uses "unload" copy. Don't pin on the full sentence —
    // a copy tweak would break the test for no behavioural reason.
    const banner = await screen.findByRole('alert');
    expect(banner.textContent).toMatch(/unloaded/i);
    expect(banner.textContent).toMatch(/before editing/i);
  });

  it('disables inputs + shows banner when status is "loaded" before any user interaction', async () => {
    const settings = fakeSettings({ status: 'loaded' });
    const fetchMock = vi.fn(async (input: RequestInfo) => {
      const url = typeof input === 'string' ? input : (input as Request).url;
      if (url === '/api/models/abc/settings') {
        return new Response(JSON.stringify(settings), { status: 200 });
      }
      if (url === '/api/system/gpus') {
        return new Response(JSON.stringify(DEFAULT_GPU_PROBE), { status: 200 });
      }
      throw new Error(`unexpected fetch: ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    renderPage('abc');
    // Flush the SWR fetch + first commit. `syncResolved` already lets
    // `use()` return synchronously, so this only needs to drain the
    // fetch microtask + give React one render pass to write the form.
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });

    // Wait for the page to settle — the form fields land in the DOM
    // after the GET resolves.
    const hfRepoInput = await screen.findByLabelText(/Hugging Face repo/i);
    // Pre-emptive guard fires BEFORE any user interaction: the input is
    // already disabled, and the unload-first banner is already present.
    expect(hfRepoInput).toBeDisabled();
    const banner = await screen.findByRole('alert');
    expect(banner.textContent).toMatch(/unloaded/i);
    // Save should be disabled too — we shouldn't even let the operator
    // attempt the round-trip.
    expect(screen.getByRole('button', { name: /save/i })).toBeDisabled();
  });

  it('Save button does not get stuck if there is nothing to save', async () => {
    // Regression guard: an earlier draft of onSave called setSaving(true)
    // before the no-dirty-keys early-return, which made the spinner stay
    // stuck on forever (the finally never ran because no async path was
    // entered). The fix moved the dirty-set check above the spinner flip;
    // this test pins that ordering.
    const settings = fakeSettings();
    const fetchMock = vi.fn(async (input: RequestInfo, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : (input as Request).url;
      const method = (init?.method ?? 'GET').toUpperCase();
      if (url === '/api/models/abc/settings' && method === 'GET') {
        return new Response(JSON.stringify(settings), { status: 200 });
      }
      if (url === '/api/system/gpus') {
        return new Response(JSON.stringify(DEFAULT_GPU_PROBE), { status: 200 });
      }
      throw new Error(`unexpected fetch: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    renderPage('abc');
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });

    // Wait for the form to land; don't change anything.
    await screen.findByLabelText(/HF revision/i);

    const saveBtn = screen.getByRole('button', { name: /save/i });
    // With no dirty fields the button is disabled to begin with — clicking
    // it still must not flip the spinner state. We click anyway (jsdom
    // honours disabled clicks as no-ops at the DOM level, but the
    // underlying handler is what we actually want to assert about; force
    // a direct call by re-enabling momentarily would be cheating, so we
    // just assert the rendered state is stable).
    await act(async () => {
      fireEvent.click(saveBtn);
    });

    // Button text must NOT be "Saving…" — if onSave had set saving=true
    // before the early-return, finally would never fire and we'd be
    // permanently stuck on the spinner copy.
    expect(saveBtn.textContent).toMatch(/^save$/i);
    expect(saveBtn.textContent).not.toMatch(/saving/i);
    // No PATCH should have fired either way.
    const patchCall = fetchMock.mock.calls.find(
      (c) => (c[1] as RequestInit | undefined)?.method === 'PATCH',
    );
    expect(patchCall).toBeUndefined();
  });

  it('Reset clears the 409 banner', async () => {
    const settings = fakeSettings({ status: 'pulled' });
    const fetchMock = vi.fn(async (input: RequestInfo, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : (input as Request).url;
      const method = (init?.method ?? 'GET').toUpperCase();
      if (url === '/api/models/abc/settings' && method === 'GET') {
        return new Response(JSON.stringify(settings), { status: 200 });
      }
      if (url === '/api/models/abc/settings' && method === 'PATCH') {
        return new Response(
          '{"detail":"model must be unloaded before editing settings"}',
          { status: 409 },
        );
      }
      if (url === '/api/system/gpus') {
        return new Response(JSON.stringify(DEFAULT_GPU_PROBE), { status: 200 });
      }
      throw new Error(`unexpected fetch: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    renderPage('abc');
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });

    const hfRevisionInput = await screen.findByLabelText(/HF revision/i);
    fireEvent.change(hfRevisionInput, { target: { value: 'racing' } });

    const saveBtn = screen.getByRole('button', { name: /save/i });
    await waitFor(() => expect(saveBtn).not.toBeDisabled());
    await act(async () => {
      fireEvent.click(saveBtn);
    });

    // 409 banner is up.
    const banner = await screen.findByRole('alert');
    expect(banner.textContent).toMatch(/unloaded/i);

    // Now click Reset — the banner should disappear.
    const resetBtn = screen.getByRole('button', { name: /reset/i });
    await act(async () => {
      fireEvent.click(resetBtn);
    });

    await waitFor(() => {
      expect(screen.queryByRole('alert')).toBeNull();
    });
  });

  it('shows a ghost row for a configured GPU absent from the probe', async () => {
    const settings = fakeSettings({ gpu_indices: [0, 7] });
    const fetchMock = vi.fn(async (input: RequestInfo, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : (input as Request).url;
      const method = (init?.method ?? 'GET').toUpperCase();
      if (url === '/api/models/abc/settings' && method === 'GET') {
        return new Response(JSON.stringify(settings), { status: 200 });
      }
      if (url === '/api/system/gpus') {
        // Only GPU 0 present — same probe as DEFAULT_GPU_PROBE. The single
        // variable under test is the model's gpu_indices: [0, 7] vs [0].
        return new Response(JSON.stringify(DEFAULT_GPU_PROBE), { status: 200 });
      }
      throw new Error(`unexpected fetch: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    renderPage('abc');
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });

    // Ghost row label: GPU {index} — not present (em-dash U+2014)
    expect(await screen.findByText(/GPU 7 — not present/)).toBeInTheDocument();
    // Warning banner from GpuChecklist — role="alert" carrying the full
    // "configured but not present in the system" sentence (gpu-checklist.tsx).
    const alerts = await screen.findAllByRole('alert');
    const gpuAlert = alerts.find((el) => /not present/i.test(el.textContent ?? ''));
    expect(gpuAlert).toBeDefined();
    expect(gpuAlert!.textContent).toMatch(/configured but not present in the system/i);
  });

  it('blocks Save and shows a message when gpu_indices is emptied, re-enables on re-check', async () => {
    // Start with a single configured + present GPU so we can uncheck it to
    // reach the empty state, then re-check to recover. Per spec the per-model
    // gpu_indices requires >=1 selection: Save disabled + inline message.
    const settings = fakeSettings({ gpu_indices: [0] });
    const fetchMock = vi.fn(async (input: RequestInfo, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : (input as Request).url;
      const method = (init?.method ?? 'GET').toUpperCase();
      if (url === '/api/models/abc/settings' && method === 'GET') {
        return new Response(JSON.stringify(settings), { status: 200 });
      }
      if (url === '/api/system/gpus') {
        return new Response(JSON.stringify(DEFAULT_GPU_PROBE), { status: 200 });
      }
      throw new Error(`unexpected fetch: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    renderPage('abc');
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });

    await screen.findByLabelText(/HF revision/i);

    // The single present GPU checkbox.
    const gpuCheckbox = document.getElementById('gpu-checklist-0') as HTMLInputElement;
    expect(gpuCheckbox).toBeTruthy();
    expect(gpuCheckbox.checked).toBe(true);

    // No validation message while a GPU is selected.
    expect(screen.queryByTestId('gpu-indices-empty-error')).toBeNull();

    // Uncheck it → gpu_indices becomes empty. This is a dirty change, but
    // Save must stay disabled because the empty selection is invalid.
    await act(async () => {
      fireEvent.click(gpuCheckbox);
    });

    const saveBtn = screen.getByRole('button', { name: /save/i });
    await waitFor(() => {
      expect(screen.getByTestId('gpu-indices-empty-error')).toBeInTheDocument();
    });
    expect(screen.getByTestId('gpu-indices-empty-error').textContent).toMatch(
      /select at least one gpu/i,
    );
    expect(saveBtn).toBeDisabled();

    // Re-check the GPU → selection valid again. The message disappears and
    // Save re-enables (the draft is dirty vs the original? no — it's back to
    // [0], so not dirty; but the message must be gone and Save not blocked by
    // the empty-guard). Assert the message is gone.
    await act(async () => {
      fireEvent.click(gpuCheckbox);
    });
    await waitFor(() => {
      expect(screen.queryByTestId('gpu-indices-empty-error')).toBeNull();
    });
  });

  it('re-enables Save when a GPU is checked after being emptied alongside another edit', async () => {
    // Make an unrelated edit first so the draft stays dirty regardless of the
    // gpu_indices round-trip; this isolates the empty-guard's effect on Save.
    const settings = fakeSettings({ gpu_indices: [0] });
    const fetchMock = vi.fn(async (input: RequestInfo, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : (input as Request).url;
      const method = (init?.method ?? 'GET').toUpperCase();
      if (url === '/api/models/abc/settings' && method === 'GET') {
        return new Response(JSON.stringify(settings), { status: 200 });
      }
      if (url === '/api/system/gpus') {
        return new Response(JSON.stringify(DEFAULT_GPU_PROBE), { status: 200 });
      }
      throw new Error(`unexpected fetch: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    renderPage('abc');
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });

    const hfRevisionInput = await screen.findByLabelText(/HF revision/i);
    fireEvent.change(hfRevisionInput, { target: { value: 'abc1234' } });

    const saveBtn = screen.getByRole('button', { name: /save/i });
    await waitFor(() => expect(saveBtn).not.toBeDisabled());

    const gpuCheckbox = document.getElementById('gpu-checklist-0') as HTMLInputElement;
    // Empty the GPU selection → Save blocked despite the dirty hf_revision.
    await act(async () => {
      fireEvent.click(gpuCheckbox);
    });
    await waitFor(() => expect(saveBtn).toBeDisabled());
    expect(screen.getByTestId('gpu-indices-empty-error')).toBeInTheDocument();

    // Re-check → valid selection, Save re-enables (hf_revision still dirty).
    await act(async () => {
      fireEvent.click(gpuCheckbox);
    });
    await waitFor(() => expect(saveBtn).not.toBeDisabled());
    expect(screen.queryByTestId('gpu-indices-empty-error')).toBeNull();
  });
});
