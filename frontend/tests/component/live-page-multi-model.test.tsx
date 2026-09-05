// /stats/live with several models, and with a selection.
//
// The page read one frame and rendered one set of panels. `_loaded_model` had
// LIMIT 1 behind it, so on the operator's two-engine box it showed one of them
// and which one was whatever SQLite returned first.
//
// Two things are pinned here:
//   * one section per SELECTED model, with the engine-level numbers staying
//     per-model — a p99 across two engines is not a p99 of anything;
//   * the one place figures ARE combined does not treat an unreported metric
//     as zero. llama.cpp publishes no KV gauge and no throughput counters;
//     summing its silence as 0 produces a fleet total that is short by an
//     unknown amount and looks authoritative.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react';
import { SWRConfig } from 'swr';
import type { LiveEngineFrame } from '@/lib/live-stats';
import type { LiveStatsState } from '@/lib/live-stats-stream';

let mockState: LiveStatsState = {
  status: 'connecting',
  frame: null,
  errorCode: null,
};

vi.mock('@/lib/live-stats-stream', async (orig) => {
  const actual = await orig<typeof import('@/lib/live-stats-stream')>();
  return { ...actual, useLiveStats: () => mockState };
});

// eslint-disable-next-line @typescript-eslint/no-require-imports
import LiveStatsPage from '@/app/stats/live/page';

class FakeResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

function block(
  name: string,
  overrides: Partial<LiveEngineFrame> = {},
): LiveEngineFrame {
  return {
    ts: '2026-09-02T19:00:00Z',
    model: name,
    model_id: `id-${name}`,
    backend: 'vllm',
    max_model_len: 8192,
    engine: {
      num_requests_running: 2,
      num_requests_waiting: 1,
      waiting_by_reason: {},
      kv_cache_usage_perc: 0.4,
      kv_tokens_used: 100,
      kv_tokens_total: 250,
      engine_sleep_state: 0,
      preemptions_total: 0,
      preemptions_per_s: 0,
    },
    throughput: {
      prompt_tokens_per_s: 10,
      generation_tokens_per_s: 20,
      prompt_tokens_total: 1000,
      generation_tokens_total: 2000,
    },
    cache: {
      prefix_hit_rate: null,
      prefix_hit_rate_cumulative: null,
      mm_hit_rate_cumulative: null,
      external_prefix_hit_rate_cumulative: null,
    },
    latency: {
      ttft_p50: null, ttft_p90: null, ttft_p99: null, ttft_mean: null,
      itl_p50: null, itl_p99: null, tpot_p50: null,
      e2e_p50: null, e2e_p90: null, e2e_p99: null,
    },
    mfu: null,
    finished: {},
    scrape_error: null,
    ...overrides,
  };
}

const SILENT_LLAMACPP = block('qwen3.8-27b', {
  backend: 'llamacpp',
  engine: {
    num_requests_running: 1,
    num_requests_waiting: 0,
    waiting_by_reason: {},
    // llama.cpp publishes no KV gauge and no preemption counter.
    kv_cache_usage_perc: null,
    kv_tokens_used: null,
    kv_tokens_total: null,
    engine_sleep_state: null,
    preemptions_total: null,
    preemptions_per_s: null,
  },
  throughput: {
    prompt_tokens_per_s: null,
    generation_tokens_per_s: null,
    prompt_tokens_total: null,
    generation_tokens_total: null,
  },
});

function frameWith(blocks: LiveEngineFrame[]): LiveEngineFrame {
  return { ...blocks[0], models: blocks };
}

function json(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });
}

const REQUESTS = {
  ts: '2026-09-02T19:00:00Z',
  count: 2,
  requests: [
    {
      id: 'r1', token_name: 'key-a', client_ip: '10.0.0.1',
      model: 'llama-3.1-8b', path: '/v1/chat/completions',
      prompt_tokens: 100, completion_tokens: 10, context_tokens: 110,
      max_model_len: 8192, context_pct: 0.013, elapsed_s: 1, phase: 'decode',
      orphan: false,
    },
    {
      id: 'r2', token_name: 'key-b', client_ip: '10.0.0.2',
      model: 'qwen3.8-27b', path: '/v1/chat/completions',
      prompt_tokens: 200, completion_tokens: 20, context_tokens: 220,
      max_model_len: 8192, context_pct: 0.027, elapsed_s: 2, phase: 'decode',
      orphan: false,
    },
  ],
  by_token: [],
  by_ip: [],
};

function installFetchStub() {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.startsWith('/api/stats/requests')) return json(REQUESTS);
      if (url === '/api/auth/refresh') return json({ access_token: 't' });
      if (url === '/api/csrf') return json({ csrf: 'c' });
      return json({});
    }),
  );
}

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
      <LiveStatsPage />
    </SWRConfig>,
  );
}

function boxFor(id: string): HTMLInputElement {
  const opt = screen
    .getAllByTestId('model-selector-option')
    .find((o) => o.getAttribute('data-model-id') === id);
  if (!opt) throw new Error(`no option for ${id}`);
  return opt.querySelector('input') as HTMLInputElement;
}

describe('LiveStatsPage — several models', () => {
  beforeEach(() => {
    window.localStorage.clear();
    vi.stubGlobal('ResizeObserver', FakeResizeObserver);
    installFetchStub();
    mockState = {
      status: 'connected',
      frame: frameWith([block('llama-3.1-8b'), SILENT_LLAMACPP]),
      errorCode: null,
    };
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('renders one section per loaded model, not just the first', async () => {
    renderPage();
    await waitFor(() =>
      expect(screen.getAllByTestId('live-model-section')).toHaveLength(2),
    );
    const names = screen
      .getAllByTestId('live-model-name')
      .map((n) => n.textContent);
    expect(names).toEqual(['llama-3.1-8b', 'qwen3.8-27b']);
  });

  it('offers the same selector as /stats', async () => {
    renderPage();
    await waitFor(() =>
      expect(screen.getAllByTestId('model-selector-option')).toHaveLength(2),
    );
  });

  it('drops a section when its model is deselected', async () => {
    renderPage();
    await waitFor(() =>
      expect(screen.getAllByTestId('model-selector-option')).toHaveLength(2),
    );
    fireEvent.click(boxFor('id-qwen3.8-27b'));
    await waitFor(() =>
      expect(screen.getAllByTestId('live-model-section')).toHaveLength(1),
    );
    expect(screen.getByTestId('live-model-name').textContent).toBe('llama-3.1-8b');
  });

  it('cannot be emptied — the last section always stands', async () => {
    renderPage();
    await waitFor(() =>
      expect(screen.getAllByTestId('model-selector-option')).toHaveLength(2),
    );
    fireEvent.click(boxFor('id-qwen3.8-27b'));
    await waitFor(() => expect(boxFor('id-llama-3.1-8b').disabled).toBe(true));
    fireEvent.click(boxFor('id-llama-3.1-8b'));
    expect(screen.getAllByTestId('live-model-section')).toHaveLength(1);
  });

  it('narrows the live request list to the selected models', async () => {
    // The live view must reflect the selection, not only the numbers above it.
    // The IP appears in the request row AND in the by-IP rollup, so both are
    // asserted by count rather than by a single-element query.
    renderPage();
    await waitFor(() =>
      expect(screen.getAllByText('10.0.0.2').length).toBeGreaterThan(0),
    );
    fireEvent.click(boxFor('id-qwen3.8-27b'));
    await waitFor(() => expect(screen.queryAllByText('10.0.0.2')).toHaveLength(0));
    expect(screen.getAllByText('10.0.0.1').length).toBeGreaterThan(0);
  });

  it('recomputes the by-token rollup from the narrowed rows', async () => {
    // The server aggregates over EVERY in-flight request. Leaving that alone
    // would leave two views of the same instant disagreeing about which
    // requests exist.
    renderPage();
    await waitFor(() =>
      expect(screen.getAllByText('key-b').length).toBeGreaterThan(0),
    );
    fireEvent.click(boxFor('id-qwen3.8-27b'));
    await waitFor(() => expect(screen.queryAllByText('key-b')).toHaveLength(0));
    expect(screen.getAllByText('key-a').length).toBeGreaterThan(0);
  });
});

describe('LiveStatsPage — combining a mixed selection', () => {
  beforeEach(() => {
    window.localStorage.clear();
    vi.stubGlobal('ResizeObserver', FakeResizeObserver);
    installFetchStub();
    mockState = {
      status: 'connected',
      frame: frameWith([block('llama-3.1-8b'), SILENT_LLAMACPP]),
      errorCode: null,
    };
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('shows fleet totals only when several models are selected', async () => {
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId('live-combined')).toBeInTheDocument(),
    );
    fireEvent.click(boxFor('id-qwen3.8-27b'));
    // With one model the row would only restate the panel below it.
    await waitFor(() => expect(screen.queryByTestId('live-combined')).toBeNull());
  });

  it('does not count the silent engine as zero', async () => {
    // vLLM reports 20 tok/s; llama.cpp reports nothing. The total is 20 FROM
    // ONE MODEL — not 20 out of two, and certainly not an average of 10.
    renderPage();
    const tile = await screen.findByTestId('combined-generation');
    expect(tile.textContent).toContain('20');
    expect(
      screen.getByTestId('combined-generation-partial').textContent,
    ).toMatch(/1 of 2 models report this/i);
  });

  it('adds the counters both engines do report', async () => {
    // running: 2 (vLLM) + 1 (llama.cpp) = 3. Both publish it, so no caveat.
    renderPage();
    const tile = await screen.findByTestId('combined-running');
    expect(tile.textContent).toContain('3');
    expect(screen.queryByTestId('combined-running-partial')).toBeNull();
  });

  it('renders an em dash, never 0, when no selected engine reports it', async () => {
    // 0 would read as "measured, and idle". The truth is "nobody here
    // measures this" — the exact confusion sub-project C fixed for three
    // panels, which a multi-model selection can reintroduce once per tile.
    mockState = {
      status: 'connected',
      frame: frameWith([
        SILENT_LLAMACPP,
        { ...SILENT_LLAMACPP, model: 'other', model_id: 'id-other' },
      ]),
      errorCode: null,
    };
    renderPage();
    const tile = await screen.findByTestId('combined-generation');
    expect(tile.textContent).toContain('—');
    expect(tile.textContent).not.toMatch(/\b0\b/);
    // And it says so, rather than leaving a bare dash to be read as a bug.
    expect(tile.textContent).toMatch(/not reported/i);
  });
});
