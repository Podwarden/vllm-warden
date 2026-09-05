// Component tests for the stress-test modal.
//
// What warrants pinning down here is not the markup, it is the promises the
// design makes about this UI:
//
//   * the acknowledgement is an explicit choice, and the POST body carries it
//   * no mode is presented as crash-free
//   * progress is POLLED, and an indeterminate bar omits aria-valuenow
//   * the result LEADS with a setting, marks recommended_config as not-live,
//     and never shows a limit as a single scalar
//   * a failure says WHICH failure, because the remedies differ
//   * closing resets, so the next open is a fresh confirm screen
//
// Fetch is stubbed globally so the auth-fetch wrapper runs end to end,
// mirroring delete-model-modal.test.tsx.
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  render,
  screen,
  fireEvent,
  waitFor,
  cleanup,
  within,
} from "@testing-library/react";
import { StressTestModal } from "@/components/stress/stress-test-modal";
import { setAccessToken, setCsrfToken } from "@/lib/auth-fetch";

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

interface Call {
  url: string;
  method: string;
  body: string | null;
}

function installFetchStub(
  handler: (url: string, init?: RequestInit) => Response | Promise<Response>,
): Call[] {
  const calls: Call[] = [];
  const mock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    calls.push({
      url,
      method: init?.method ?? "GET",
      body: typeof init?.body === "string" ? init.body : null,
    });
    if (url === "/api/auth/refresh") return json({ access_token: "jwt2" });
    if (url === "/api/csrf") return json({ csrf: "test-csrf" });
    return handler(url, init);
  });
  vi.stubGlobal("fetch", mock);
  return calls;
}

function renderModal(props: Record<string, unknown> = {}) {
  return render(
    <StressTestModal
      open
      onClose={() => {}}
      modelId="m-1"
      servedModelName="qwen3-8b"
      maxModelLen={32768}
      // Fast enough that waitFor sees several polls without fake timers.
      pollIntervalMs={5}
      {...props}
    />,
  );
}

/** No runs at all — the usual state when the modal first opens. */
const NO_RUNS = { limits: {}, recommended_config: null, runs: [] };

describe("StressTestModal — confirm phase", () => {
  beforeEach(() => {
    setAccessToken("test-jwt", 900);
    setCsrfToken("test-csrf");
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("keeps Start disabled until disruption is explicitly acknowledged", async () => {
    installFetchStub((url) => {
      if (url.endsWith("/capabilities")) return json(NO_RUNS);
      throw new Error(`Unmocked: ${url}`);
    });
    renderModal();

    const ack = screen.getByTestId("stress-acknowledge") as HTMLInputElement;
    // Explicitly NOT pre-ticked: a default-on acknowledgement is not one.
    expect(ack.checked).toBe(false);
    expect(screen.getByTestId("stress-start")).toBeDisabled();

    fireEvent.click(ack);
    expect(screen.getByTestId("stress-start")).not.toBeDisabled();
  });

  it("never presents a mode as crash-free, and shows honest estimates", () => {
    installFetchStub((url) => {
      if (url.endsWith("/capabilities")) return json(NO_RUNS);
      throw new Error(`Unmocked: ${url}`);
    });
    renderModal();

    // The warning is unconditional — not gated on which mode is selected.
    expect(screen.getByTestId("stress-disruption-warning")).toHaveTextContent(
      /may crash and be restarted/i,
    );
    expect(screen.getByTestId("stress-disruption-warning")).toHaveTextContent(
      /No mode can promise otherwise/i,
    );
    // v1 called the default mode "safe" and promised no crash. Nothing in the
    // confirm screen may say that, in any mode.
    const body = screen.getByTestId("stress-disruption-warning").textContent ?? "";
    expect(body).not.toMatch(/never crash|no crash|crash-free|safe mode/i);

    expect(screen.getByText("~15 min")).toBeInTheDocument();
    expect(screen.getByText("~30–45 min")).toBeInTheDocument();
    expect(screen.getByText("2.5–4 h")).toBeInTheDocument();
  });

  it("POSTs the selected mode with acknowledge_disruption and force:false", async () => {
    const calls = installFetchStub((url, init) => {
      if (url.endsWith("/capabilities")) return json(NO_RUNS);
      if (url === "/api/models/m-1/stress" && init?.method === "POST") {
        return json({ run_id: "r-1" }, 202);
      }
      throw new Error(`Unmocked: ${init?.method} ${url}`);
    });
    renderModal();

    fireEvent.click(screen.getByTestId("stress-mode-thorough"));
    fireEvent.click(screen.getByTestId("stress-acknowledge"));
    fireEvent.click(screen.getByTestId("stress-start"));

    await waitFor(() =>
      expect(screen.getByTestId("stress-running")).toBeInTheDocument(),
    );
    const post = calls.find((c) => c.method === "POST" && c.url.endsWith("/stress"));
    expect(post).toBeDefined();
    expect(JSON.parse(post!.body!)).toEqual({
      mode: "thorough",
      acknowledge_disruption: true,
      force: false,
      // An ordinary start never wipes. `reset` is set only by "Wipe and run
      // again" on the results screen: the published record deliberately falls
      // back past a run that measured nothing, so a FAILED run must not
      // withdraw a good older measurement, and only the operator can tell that
      // case from a deliberate replacement.
      reset: false,
    });
  });

  it("reveals the force affordance only after the server refuses with 409", async () => {
    let refused = false;
    const calls = installFetchStub((url, init) => {
      if (url.endsWith("/capabilities")) return json(NO_RUNS);
      if (url === "/api/models/m-1/stress" && init?.method === "POST") {
        if (!refused) {
          refused = true;
          return json({ detail: "model is serving traffic" }, 409);
        }
        return json({ run_id: "r-9" }, 202);
      }
      throw new Error(`Unmocked: ${init?.method} ${url}`);
    });
    renderModal();

    // `force` is not an ambient checkbox — it does not exist until refused.
    expect(screen.queryByTestId("stress-force")).not.toBeInTheDocument();

    fireEvent.click(screen.getByTestId("stress-acknowledge"));
    fireEvent.click(screen.getByTestId("stress-start"));

    await waitFor(() =>
      expect(screen.getByTestId("stress-conflict")).toHaveTextContent(
        /serving traffic/i,
      ),
    );
    // Refused and not yet forced: Start must not fire the same request again.
    expect(screen.getByTestId("stress-start")).toBeDisabled();

    fireEvent.click(screen.getByTestId("stress-force"));
    fireEvent.click(screen.getByTestId("stress-start"));

    await waitFor(() =>
      expect(screen.getByTestId("stress-running")).toBeInTheDocument(),
    );
    const posts = calls.filter(
      (c) => c.method === "POST" && c.url.endsWith("/stress"),
    );
    expect(posts).toHaveLength(2);
    expect(JSON.parse(posts[1].body!).force).toBe(true);
  });
});

describe("StressTestModal — running phase", () => {
  beforeEach(() => {
    setAccessToken("test-jwt", 900);
    setCsrfToken("test-csrf");
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  async function startRun(
    capabilitiesFor: (n: number) => unknown,
  ): Promise<Call[]> {
    let polls = 0;
    const calls = installFetchStub((url, init) => {
      if (url.endsWith("/capabilities")) {
        const body = capabilitiesFor(polls);
        polls += 1;
        return json(body);
      }
      if (url === "/api/models/m-1/stress" && init?.method === "POST") {
        return json({ run_id: "r-1" }, 202);
      }
      throw new Error(`Unmocked: ${init?.method} ${url}`);
    });
    renderModal();
    fireEvent.click(screen.getByTestId("stress-acknowledge"));
    fireEvent.click(screen.getByTestId("stress-start"));
    await waitFor(() =>
      expect(screen.getByTestId("stress-running")).toBeInTheDocument(),
    );
    return calls;
  }

  it("polls the run row (no SSE) and renders probe outcomes as badges", async () => {
    const calls = await startRun((n) => ({
      runs: [
        {
          id: "r-1",
          status: "running",
          progress: { completed: n, total: 12, phase: "bracket" },
          observations: [
            { axis: "length", value: 1024, cls: "pass" },
            { axis: "length", value: 2048, cls: "degraded", gate_tripped: "abrupt" },
          ],
        },
      ],
    }));

    await waitFor(() =>
      expect(screen.getAllByTestId("stress-probe-row")).toHaveLength(2),
    );
    expect(screen.getByText("pass")).toBeInTheDocument();
    expect(screen.getByText("degraded")).toBeInTheDocument();
    expect(screen.getByTestId("stress-phase")).toHaveTextContent("bracket");

    // More than one poll, and every one of them a plain GET — the design
    // rejects SSE replay for progress outright.
    await waitFor(() => {
      const polls = calls.filter((c) => c.url.endsWith("/capabilities"));
      expect(polls.length).toBeGreaterThan(1);
      expect(polls.every((c) => c.method === "GET")).toBe(true);
    });
    expect(vi.isMockFunction(globalThis.EventSource)).toBe(false);
  });

  it("omits aria-valuenow while no probe total is known", async () => {
    await startRun(() => ({
      runs: [{ id: "r-1", status: "running", observations: [] }],
    }));
    const bar = await screen.findByRole("progressbar");
    // Indeterminate ARIA: announcing 0% would be a lie, not a default.
    expect(bar).not.toHaveAttribute("aria-valuenow");

    cleanup();

    await startRun(() => ({
      runs: [
        {
          id: "r-1",
          status: "running",
          progress: { completed: 3, total: 12 },
          observations: [],
        },
      ],
    }));
    await waitFor(() =>
      expect(screen.getByRole("progressbar")).toHaveAttribute(
        "aria-valuenow",
        "25",
      ),
    );
  });

  it("keeps polling through a transient poll failure rather than declaring failure", async () => {
    let polls = 0;
    installFetchStub((url, init) => {
      if (url.endsWith("/capabilities")) {
        polls += 1;
        // The warden restarts the engine after a crash; the API can blip.
        if (polls === 2) return json({ detail: "backend restarting" }, 503);
        return json({ runs: [{ id: "r-1", status: "running", observations: [] }] });
      }
      if (url === "/api/models/m-1/stress" && init?.method === "POST") {
        return json({ run_id: "r-1" }, 202);
      }
      throw new Error(`Unmocked: ${init?.method} ${url}`);
    });
    renderModal();
    fireEvent.click(screen.getByTestId("stress-acknowledge"));
    fireEvent.click(screen.getByTestId("stress-start"));

    await waitFor(() =>
      expect(screen.getByTestId("stress-poll-error")).toBeInTheDocument(),
    );
    // Still running, and the notice clears once the next poll succeeds.
    expect(screen.getByTestId("stress-running")).toBeInTheDocument();
    await waitFor(() =>
      expect(screen.queryByTestId("stress-poll-error")).not.toBeInTheDocument(),
    );
  });

  it("adopts a run that is already in flight instead of offering to start another", async () => {
    installFetchStub((url) => {
      if (url.endsWith("/capabilities")) {
        return json({
          runs: [{ id: "r-existing", status: "running", observations: [] }],
        });
      }
      throw new Error(`Unmocked: ${url}`);
    });
    renderModal();
    await waitFor(() =>
      expect(screen.getByTestId("stress-running")).toBeInTheDocument(),
    );
    expect(screen.queryByTestId("stress-start")).not.toBeInTheDocument();
  });
});

describe("StressTestModal — done phase", () => {
  beforeEach(() => {
    setAccessToken("test-jwt", 900);
    setCsrfToken("test-csrf");
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  function completedRun(overrides: Record<string, unknown> = {}) {
    return {
      id: "r-1",
      status: "completed",
      limits: {
        quality_context: {
          value: 122880,
          unit: "tokens",
          provenance: "measured",
          limited_by: "quality",
          gate_tripped: "abrupt",
          raw_confirmed: 136533,
          first_observed_failure: 147456,
          oracle: { n_consecutive: 3, contour_p: 0.21, confirm_m: 5, safety_factor: 0.9 },
        },
      },
      recommended_config: {
        max_model_len: 65536,
        limited_by: "load",
        next_failed_at: 98304,
      },
      observations: [],
      ...overrides,
    };
  }

  async function renderDone(run: Record<string, unknown>) {
    installFetchStub((url, init) => {
      if (url.endsWith("/capabilities")) return json({ runs: [run] });
      if (url === "/api/models/m-1/stress" && init?.method === "POST") {
        return json({ run_id: "r-1" }, 202);
      }
      throw new Error(`Unmocked: ${init?.method} ${url}`);
    });
    renderModal();
    // Through the real flow: start, then let the first poll observe a run row
    // that is no longer `running`.
    fireEvent.click(screen.getByTestId("stress-acknowledge"));
    fireEvent.click(screen.getByTestId("stress-start"));
    await waitFor(() =>
      expect(screen.getByTestId("stress-done")).toBeInTheDocument(),
    );
  }

  it("leads with the setting to apply, and says the config is not the live one", async () => {
    await renderDone(completedRun());
    // The headline is an action, not a breaking point.
    expect(screen.getByTestId("stress-recommendation")).toHaveTextContent(
      "Set max_model_len to 65,536",
    );
    expect(screen.getByTestId("stress-not-live-caveat")).toHaveTextContent(
      /not currently running/i,
    );
    // limited_by drives the remedy: a memory wall is not a quality wall.
    expect(screen.getByTestId("stress-limited-by")).toHaveTextContent(
      "memory at load",
    );
    expect(screen.getByText(/could not allocate its KV pool/i)).toBeInTheDocument();
  });

  it("shows all three numbers for a limit, never a single scalar", async () => {
    await renderDone(completedRun());
    const row = screen.getByTestId("stress-limit-quality_context");
    expect(row).toHaveTextContent("122,880");
    expect(row).toHaveTextContent("136,533"); // raw_confirmed
    expect(row).toHaveTextContent("147,456"); // first_observed_failure
    expect(row).toHaveTextContent("3/3 consecutive");
  });

  it("falls back to the measured context when no reload sweep ran", async () => {
    // conservative/quick modes never sweep, so there is no better config to
    // recommend — and inventing one would be worse than saying so.
    await renderDone(completedRun({ recommended_config: null }));
    expect(screen.getByTestId("stress-recommendation")).toHaveTextContent(
      "Keep prompts under 122,880 tokens",
    );
    expect(screen.queryByTestId("stress-not-live-caveat")).not.toBeInTheDocument();
    expect(screen.getByText(/thorough/i)).toBeInTheDocument();
  });

  it("flags a completed-but-truncated run as a lower bound", async () => {
    await renderDone(completedRun({ truncated_by: "crash_budget" }));
    expect(screen.getByTestId("stress-truncated")).toHaveTextContent(
      /lower bound/i,
    );
  });
});

describe("StressTestModal — failure phase", () => {
  beforeEach(() => {
    setAccessToken("test-jwt", 900);
    setCsrfToken("test-csrf");
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  async function renderFailed(run: Record<string, unknown>) {
    installFetchStub((url, init) => {
      if (url.endsWith("/capabilities")) return json({ runs: [run] });
      if (url === "/api/models/m-1/stress" && init?.method === "POST") {
        return json({ run_id: "r-1" }, 202);
      }
      throw new Error(`Unmocked: ${init?.method} ${url}`);
    });
    renderModal();
    fireEvent.click(screen.getByTestId("stress-acknowledge"));
    fireEvent.click(screen.getByTestId("stress-start"));
    await waitFor(() =>
      expect(screen.getByTestId("stress-failed")).toBeInTheDocument(),
    );
  }

  it("names a non-monotone verdict as unsound, not as a limit", async () => {
    await renderFailed({
      id: "r-1",
      status: "aborted",
      non_monotone: true,
      observations: [],
    });
    expect(screen.getByTestId("stress-failure-title")).toHaveTextContent(
      /unsound/i,
    );
    expect(screen.getByTestId("stress-failed")).toHaveTextContent(
      /not upward-closed/i,
    );
  });

  it("distinguishes a neighbour impact from a wedged engine", async () => {
    await renderFailed({
      id: "r-1",
      status: "aborted",
      truncated_by: "neighbour_impact",
      observations: [],
    });
    expect(screen.getByTestId("stress-failure-title")).toHaveTextContent(
      /co-resident/i,
    );
    cleanup();

    await renderFailed({
      id: "r-1",
      status: "aborted",
      truncated_by: "wedged",
      observations: [],
    });
    expect(screen.getByTestId("stress-failure-title")).toHaveTextContent(
      /wedged/i,
    );
  });

  it("says an interrupted run published nothing", async () => {
    await renderFailed({
      id: "r-1",
      status: "interrupted",
      observations: [],
      last_error: null,
    });
    expect(screen.getByTestId("stress-failed")).toHaveTextContent(
      /never published/i,
    );
  });
});

describe("StressTestModal — close", () => {
  beforeEach(() => {
    setAccessToken("test-jwt", 900);
    setCsrfToken("test-csrf");
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("resets phase state so the next open is a fresh confirm screen", async () => {
    installFetchStub((url, init) => {
      if (url.endsWith("/capabilities")) {
        return json({
          runs: [
            { id: "r-1", status: "completed", limits: {}, observations: [] },
          ],
        });
      }
      if (url === "/api/models/m-1/stress" && init?.method === "POST") {
        return json({ run_id: "r-1" }, 202);
      }
      throw new Error(`Unmocked: ${init?.method} ${url}`);
    });
    function modal(open: boolean) {
      return (
        <StressTestModal
          open={open}
          onClose={() => {}}
          modelId="m-1"
          servedModelName="qwen3-8b"
          maxModelLen={32768}
          pollIntervalMs={5}
        />
      );
    }
    const { rerender } = render(modal(true));
    fireEvent.click(screen.getByTestId("stress-acknowledge"));
    fireEvent.click(screen.getByTestId("stress-start"));
    await waitFor(() =>
      expect(screen.getByTestId("stress-done")).toBeInTheDocument(),
    );

    // Scoped to the result body: the Modal chrome has its own "Close".
    fireEvent.click(
      within(screen.getByTestId("stress-done")).getByRole("button", {
        name: "Close",
      }),
    );
    rerender(modal(false));
    rerender(modal(true));
    // A finished run is not adopted on reopen — only a running one is — so the
    // operator lands back on the confirm screen with the box unticked.
    await waitFor(() =>
      expect(screen.getByTestId("stress-acknowledge")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("stress-acknowledge")).not.toBeChecked();
  });
});
