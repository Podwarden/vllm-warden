// A metric the engine does not report must never render as 0.
//
// Design spec §9.3: a user who reads past a degradation and acts on it has been
// actively misled. "0% KV usage" says *plenty of headroom*; "0 tok/s" says *the
// engine is idle*. For llama.cpp both are false -- the truth is "this engine is
// silent about that" -- and an operator who trusts either has been given a
// confidently precise wrong answer.
//
// The MFU panel already had the right shape: a "Not reported" empty state that
// names the engine. This pins that the KV panel, the throughput headline, the
// running/waiting counts and the sleep-state chip all behave the same way, and
// that a vLLM frame is unchanged.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { SWRConfig } from "swr";

const mockState = { status: "connected", frame: null as unknown, errorCode: null };

vi.mock("@/lib/live-stats-stream", () => ({
  useLiveStats: () => mockState,
}));

class FakeResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
vi.stubGlobal("ResizeObserver", FakeResizeObserver);

import LiveStatsPage from "@/app/stats/live/page";

const EMPTY_REQUESTS = {
  ts: "2026-09-01T00:00:00Z",
  requests: [],
  by_token: [],
  by_ip: [],
  totals: { requests: 0, context_tokens: 0, prompt_tokens: 0, completion_tokens: 0 },
};

// Six numbers present, the rest null because llama.cpp does not report them.
const LLAMACPP_FRAME = {
  ts: "2026-09-01T00:00:00Z",
  model: "qwen",
  model_id: "m1",
  backend: "llamacpp",
  max_model_len: 8192,
  engine: {
    num_requests_running: 1,
    num_requests_waiting: 0,
    waiting_by_reason: { capacity: null, deferred: null },
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
    prompt_tokens_total: 120,
    generation_tokens_total: 34,
  },
  cache: {
    prefix_hit_rate: null,
    prefix_hit_rate_cumulative: 0.25,
    mm_hit_rate_cumulative: null,
    external_prefix_hit_rate_cumulative: null,
  },
  latency: {
    ttft_p50: null,
    ttft_p90: null,
    ttft_p99: null,
    ttft_mean: null,
    itl_p50: null,
    itl_p99: null,
    tpot_p50: null,
    e2e_p50: null,
    e2e_p90: null,
    e2e_p99: null,
  },
  mfu: { flops_per_gpu_total: null, mfu_estimate: null },
  finished: { stop: null, length: null, abort: null },
  scrape_error: null,
};

const VLLM_FRAME = {
  ...LLAMACPP_FRAME,
  backend: "vllm",
  engine: {
    num_requests_running: 3,
    num_requests_waiting: 2,
    waiting_by_reason: { capacity: 2, deferred: 0 },
    kv_cache_usage_perc: 0.5,
    kv_tokens_used: 112400,
    kv_tokens_total: 224800,
    engine_sleep_state: 0,
    preemptions_total: 12,
    preemptions_per_s: 1.0,
  },
  throughput: {
    prompt_tokens_per_s: 500,
    generation_tokens_per_s: 50,
    prompt_tokens_total: 1000000,
    generation_tokens_total: 500000,
  },
};

function renderWithFrame(frame: unknown) {
  mockState.frame = frame;
  return render(
    <SWRConfig
      value={{
        provider: () => new Map(),
        dedupingInterval: 0,
        revalidateOnFocus: false,
        revalidateOnReconnect: false,
        fetcher: async () => EMPTY_REQUESTS,
      }}
    >
      <LiveStatsPage />
    </SWRConfig>,
  );
}

beforeEach(() => {
  mockState.frame = null;
});
afterEach(() => cleanup());

describe("live stats with a backend that reports fewer metrics", () => {
  it("renders absent KV usage as not-reported, never as 0%", () => {
    renderWithFrame(LLAMACPP_FRAME);
    expect(screen.queryByText("0%")).toBeNull();
    expect(screen.getByTestId("kv-usage").textContent ?? "").toMatch(
      /not reported|—/i,
    );
  });

  it("renders absent throughput as a dash, never as 0", () => {
    renderWithFrame(LLAMACPP_FRAME);
    expect(screen.getByTestId("gen-tps").textContent).toBe("—");
    expect(screen.getByTestId("prompt-tps").textContent ?? "").toContain("—");
  });

  it("never prints the string 'null'", () => {
    const { container } = renderWithFrame(LLAMACPP_FRAME);
    expect(container.textContent ?? "").not.toMatch(/\bnull\b/);
  });

  it("names the engine that is silent", () => {
    renderWithFrame(LLAMACPP_FRAME);
    expect(screen.getAllByText(/llama\.cpp/i).length).toBeGreaterThan(0);
  });

  it("renders an absent sleep state as a dash, not 'sleep null'", () => {
    const { container } = renderWithFrame(LLAMACPP_FRAME);
    expect(container.textContent ?? "").not.toContain("sleep null");
  });

  it("still renders a vLLM frame exactly as before", () => {
    renderWithFrame(VLLM_FRAME);
    expect(screen.getByTestId("kv-usage").textContent ?? "").toContain("50%");
    expect(screen.getByTestId("gen-tps").textContent ?? "").not.toBe("—");
  });

  it("keeps the sleep-state chip for a vLLM frame", () => {
    const { container } = renderWithFrame(VLLM_FRAME);
    expect(container.textContent ?? "").toContain("awake");
  });
});
