// Component tests for the rebuilt /stats page (S7, #124).
//
// Drives the page through the SWR-backed fetch boundary the same way the
// rest of the suite does (see cache-table.test.tsx). We stub `fetch`,
// answer the two /api/stats/v2/* endpoints with documented fixtures, and
// assert the current-row tiles + active-model strip + per-key table
// populate correctly. Recharts itself is rendered into jsdom under a
// ResizeObserver stub — we deliberately do NOT assert on chart contents
// (those are exercised by the underlying recharts test suite); we only
// pin that the page mounts and the chart panels are present.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  render,
  screen,
  cleanup,
  waitFor,
  fireEvent,
  act,
} from "@testing-library/react";
import { SWRConfig } from "swr";
import StatsPage from "@/app/stats/page";
import { setAccessToken, setCsrfToken } from "@/lib/auth-fetch";

// SWR maintains a module-level cache that persists across renders within
// the same vitest worker — so a payload supplied by test A bleeds into
// test B. Wrap each render in a fresh SWRConfig provider so the cache is
// per-test. We also disable revalidate-on-mount so the wait-for-fetch
// path is deterministic.
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
      <StatsPage />
    </SWRConfig>,
  );
}

// Recharts uses ResizeObserver inside ResponsiveContainer; jsdom has no
// implementation, so chart components silently render nothing. Stub a
// no-op so the page mounts cleanly under the test runner.
class FakeResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
vi.stubGlobal("ResizeObserver", FakeResizeObserver);

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

interface FixtureSet {
  overview?: unknown;
  tokensPerKey?: unknown;
}

function installFetchStub(fixtures: FixtureSet) {
  const mock = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url === "/api/auth/refresh") {
      return json({ access_token: "test-jwt-refreshed" });
    }
    if (url === "/api/csrf") {
      return json({ csrf: "test-csrf" });
    }
    if (url.startsWith("/api/stats/v2/overview")) {
      return json(fixtures.overview ?? {});
    }
    if (url.startsWith("/api/stats/v2/tokens-per-key")) {
      return json(fixtures.tokensPerKey ?? { range: "1h", since_minute: 0, rows: [] });
    }
    return json({}, 404);
  });
  vi.stubGlobal("fetch", mock);
  return mock;
}

const FIXTURE_OVERVIEW = {
  range: "1h",
  now_minute: 31538880,
  since_minute: 31538820,
  current: {
    vram_used_mib: 12000,
    vram_total_mib: 32000,
    vram_pct: 38,
    gpu_util_pct: 80,
    power_w: 250.0,
    tps: 14.0,
  },
  active_models: [
    { id: "m-active", served_model_name: "served-name-1" },
  ],
  series: {
    vram: [{ minute: 31538879, used_mib: 3000, total_mib: 32000 }],
    util: [{ minute: 31538879, max_pct: 50 }],
    power: [{ minute: 31538879, watts: 230.0 }],
    tokens: [{ minute: 31538879, prompt: 1000, completion: 500 }],
  },
};

const FIXTURE_TPK = {
  range: "1h",
  since_minute: 31538820,
  rows: [
    {
      token_id: "t-heavy",
      name: "Heavy Key",
      prefix: "pwm_aaa",
      requests: 15,
      prompt_tokens: 1500,
      completion_tokens: 750,
      total_tokens: 2250,
    },
    {
      token_id: "t-orphan",
      name: "(unknown)",
      prefix: null,
      requests: 1,
      prompt_tokens: 10,
      completion_tokens: 5,
      total_tokens: 15,
    },
  ],
};

describe("StatsPage", () => {
  beforeEach(() => {
    window.localStorage.clear();
    setAccessToken("test-jwt", 900);
    setCsrfToken("test-csrf");
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("renders the four current-row tiles populated from the overview payload", async () => {
    installFetchStub({
      overview: FIXTURE_OVERVIEW,
      tokensPerKey: FIXTURE_TPK,
    });
    renderPage();

    await waitFor(() => {
      expect(screen.getByTestId("tile-vram-value")).toBeInTheDocument();
    });

    // VRAM tile: 12000/32000 MiB → 11.7 / 31.3 GiB
    expect(screen.getByTestId("tile-vram-value").textContent).toContain("11.7");
    expect(screen.getByTestId("tile-vram-value").textContent).toContain("31.3");

    // GPU util tile
    expect(screen.getByTestId("tile-util-value").textContent).toBe("80");

    // Power tile — 250W → "250" (≥100 → integer)
    expect(screen.getByTestId("tile-power-value").textContent).toBe("250");

    // TPS tile — 14 → "14" (≥10 → integer)
    expect(screen.getByTestId("tile-tps-value").textContent).toBe("14");
  });

  it("renders the active-model strip", async () => {
    installFetchStub({ overview: FIXTURE_OVERVIEW });
    renderPage();
    await waitFor(() => {
      expect(screen.getByTestId("active-models")).toBeInTheDocument();
    });
    expect(screen.getByTestId("active-models").textContent).toContain(
      "served-name-1",
    );
  });

  it("renders the per-key tokens table sorted total_tokens DESC by default", async () => {
    installFetchStub({
      overview: FIXTURE_OVERVIEW,
      tokensPerKey: FIXTURE_TPK,
    });
    renderPage();

    await waitFor(() => {
      expect(screen.getByTestId("tokens-per-key-table")).toBeInTheDocument();
    });

    const rows = screen.getAllByTestId("tokens-per-key-row");
    expect(rows).toHaveLength(2);
    // Heavy Key first (2250 total_tokens) — backend already sorts but the
    // page also re-sorts client-side; assert the visible order.
    expect(rows[0].textContent).toContain("Heavy Key");
    expect(rows[0].textContent).toContain("2,250");
    expect(rows[1].textContent).toContain("(unknown)");
    expect(rows[1].textContent).toContain("orphan");
  });

  it("renders the four chart panels", async () => {
    installFetchStub({ overview: FIXTURE_OVERVIEW });
    renderPage();
    await waitFor(() => {
      expect(screen.getByTestId("chart-vram")).toBeInTheDocument();
    });
    expect(screen.getByTestId("chart-util")).toBeInTheDocument();
    expect(screen.getByTestId("chart-power")).toBeInTheDocument();
    expect(screen.getByTestId("chart-tokens")).toBeInTheDocument();
  });

  it("persists the range selection to localStorage under 'vw.stats.range'", async () => {
    installFetchStub({ overview: FIXTURE_OVERVIEW });
    renderPage();
    await waitFor(() => {
      expect(screen.getByTestId("range-selector")).toBeInTheDocument();
    });
    // Click the '7d' button.
    const sevenDay = screen.getByRole("button", { name: "7d" });
    act(() => {
      fireEvent.click(sevenDay);
    });
    expect(window.localStorage.getItem("vw.stats.range")).toBe("7d");
  });

  it("renders the power tile as '—' when telemetry is unavailable", async () => {
    installFetchStub({
      overview: {
        ...FIXTURE_OVERVIEW,
        current: { ...FIXTURE_OVERVIEW.current, power_w: null },
        series: { ...FIXTURE_OVERVIEW.series, power: [] },
      },
    });
    renderPage();
    await waitFor(() => {
      expect(screen.getByTestId("tile-power-value")).toBeInTheDocument();
    });
    expect(screen.getByTestId("tile-power-value").textContent).toBe("—");
  });

  it("hides the active-model strip when no model is loaded", async () => {
    installFetchStub({
      overview: { ...FIXTURE_OVERVIEW, active_models: [] },
    });
    renderPage();
    await waitFor(() => {
      expect(screen.getByTestId("tile-vram-value")).toBeInTheDocument();
    });
    expect(screen.queryByTestId("active-models")).toBeNull();
  });

  it("shows an empty-state when the per-key table has no rows", async () => {
    installFetchStub({
      overview: FIXTURE_OVERVIEW,
      tokensPerKey: { range: "1h", since_minute: 0, rows: [] },
    });
    renderPage();
    await waitFor(() => {
      expect(screen.getByTestId("tile-vram-value")).toBeInTheDocument();
    });
    expect(screen.queryByTestId("tokens-per-key-table")).toBeNull();
    expect(screen.getByText(/no token usage in this window/i)).toBeInTheDocument();
  });
});
