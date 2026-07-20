// Component tests for the System Configuration panel (#148).
//
// Drives the section through the SWR + authFetchJSON boundary with a
// stubbed `fetch`, the same pattern stats-page.test.tsx uses. We assert
// the cards populate correctly across three representative payloads:
//   * full host (CPU + RAM + 2 GPUs + Docker)
//   * dev workstation with no GPU and no docker socket
//   * partial host where /proc readers returned null
//
// We deliberately exercise the section in isolation (not via StatsPage)
// so a regression here doesn't have to wait for the full v2 overview
// fixture to also render — focused failure messages > integration soup.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  render,
  screen,
  cleanup,
  waitFor,
} from "@testing-library/react";
import { SWRConfig } from "swr";
import { SystemConfigSection } from "@/components/stats/system-config-section";
import { setAccessToken, setCsrfToken } from "@/lib/auth-fetch";

function renderSection() {
  return render(
    <SWRConfig
      value={{
        provider: () => new Map(),
        dedupingInterval: 0,
        revalidateOnFocus: false,
        revalidateOnReconnect: false,
      }}
    >
      <SystemConfigSection />
    </SWRConfig>,
  );
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function installFetchStub(payload: unknown, status = 200) {
  const mock = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url === "/api/auth/refresh") {
      return json({ access_token: "test-jwt-refreshed" });
    }
    if (url === "/api/csrf") {
      return json({ csrf: "test-csrf" });
    }
    if (url === "/api/system/info") {
      return json(payload, status);
    }
    return json({}, 404);
  });
  vi.stubGlobal("fetch", mock);
  return mock;
}

const FIXTURE_FULL = {
  cpu: {
    model: "Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz",
    physical_cores: 14,
    threads: 28,
  },
  ram: { total_mb: 64277 },
  gpus: [
    {
      index: 0,
      name: "NVIDIA RTX A4000",
      vram_total_mb: 16376,
      driver_version: "550.54.15",
      cuda_version: "12.4",
    },
    {
      index: 1,
      name: "NVIDIA RTX A4000",
      vram_total_mb: 16376,
      driver_version: "550.54.15",
      cuda_version: "12.4",
    },
  ],
  os: { name: "Ubuntu", version: "24.04", kernel: "6.8.0-1008-nvidia" },
  docker: { version: "28.1.1", runtime: "nvidia", available: true },
};

const FIXTURE_NO_GPU_NO_DOCKER = {
  cpu: { model: "AMD Ryzen 9 7950X", physical_cores: 16, threads: 32 },
  ram: { total_mb: 65536 },
  gpus: [],
  os: { name: "Debian", version: "12", kernel: "6.1.0-13-amd64" },
  docker: { version: null, runtime: null, available: false },
};

const FIXTURE_NULL_PROC = {
  cpu: null,
  ram: null,
  gpus: [],
  os: { name: "unknown", version: "unknown", kernel: "unknown" },
  docker: { version: null, runtime: null, available: false },
};

describe("SystemConfigSection", () => {
  beforeEach(() => {
    setAccessToken("test-jwt", 900);
    setCsrfToken("test-csrf");
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("renders the top row (CPU + RAM + OS/Docker) and a card per GPU", async () => {
    installFetchStub(FIXTURE_FULL);
    renderSection();

    await waitFor(() => {
      expect(screen.getByTestId("system-cpu-value")).toBeInTheDocument();
    });

    expect(screen.getByTestId("system-cpu-value").textContent).toContain(
      "Xeon",
    );
    // RAM card: 64277 MB → 62.8 GiB
    expect(screen.getByTestId("system-ram-value").textContent).toBe("62.8");

    // OS card line
    expect(screen.getByTestId("system-os-value").textContent).toBe(
      "Ubuntu 24.04",
    );
    expect(screen.getByTestId("system-os-kernel").textContent).toContain(
      "6.8.0-1008-nvidia",
    );

    // Docker card line
    const dockerLine = screen.getByTestId("system-docker-available").textContent;
    expect(dockerLine).toContain("28.1.1");
    expect(dockerLine).toContain("nvidia");

    // GPU cards — 2 total, indexes 0 and 1.
    const gpuCards = screen.getAllByTestId("system-gpu-card");
    expect(gpuCards).toHaveLength(2);
    expect(gpuCards[0].getAttribute("data-gpu-index")).toBe("0");
    expect(gpuCards[1].getAttribute("data-gpu-index")).toBe("1");

    // Per-GPU details — driver, CUDA, VRAM.
    const drivers = screen.getAllByTestId("system-gpu-driver");
    expect(drivers[0].textContent).toBe("550.54.15");
    const cudas = screen.getAllByTestId("system-gpu-cuda");
    expect(cudas[0].textContent).toBe("12.4");
    const vrams = screen.getAllByTestId("system-gpu-vram");
    // 16376 MB → 16.0 GiB
    expect(vrams[0].textContent).toContain("16.0");
    expect(vrams[0].textContent).toContain("16376");
  });

  it("shows the empty-state and 'not available' Docker line on a no-GPU host", async () => {
    installFetchStub(FIXTURE_NO_GPU_NO_DOCKER);
    renderSection();

    await waitFor(() => {
      expect(screen.getByTestId("system-cpu-value")).toBeInTheDocument();
    });

    expect(screen.getByTestId("system-cpu-value").textContent).toContain("Ryzen");
    expect(screen.getByTestId("system-gpus-empty")).toBeInTheDocument();
    expect(screen.queryByTestId("system-gpu-card")).toBeNull();
    expect(
      screen.getByTestId("system-docker-unavailable").textContent,
    ).toMatch(/not available/i);
  });

  it("renders em-dash placeholders when CPU/RAM are null", async () => {
    installFetchStub(FIXTURE_NULL_PROC);
    renderSection();

    await waitFor(() => {
      expect(screen.getByTestId("system-cpu-value")).toBeInTheDocument();
    });

    expect(screen.getByTestId("system-cpu-value").textContent).toBe("—");
    expect(screen.getByTestId("system-ram-value").textContent).toBe("—");
    // OS reads "unknown" when all three fields are unknown — verifies
    // formatOsName collapses the all-unknown case.
    expect(screen.getByTestId("system-os-value").textContent).toBe("unknown");
  });

  it("renders an error message on fetch failure", async () => {
    installFetchStub({ detail: "boom" }, 500);
    renderSection();

    await waitFor(() => {
      expect(screen.getByTestId("system-config-error")).toBeInTheDocument();
    });
  });
});
