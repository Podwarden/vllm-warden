// The pure helpers behind the merged /stats page.
//
// Three of these are MIRRORS of committed backend code and the tests pin the
// mirrored semantics with the backend's own worked examples:
//
//   busyMedian / peakOf     ← app/stats/history.py (test_history.py)
//   bucketDeltas            ← app/stats/live_engine.py (test_latency_buckets.py)
//   quantileFromBuckets     ← app/stats/prometheus.py (test_live_engine.py)

import { describe, it, expect } from "vitest";
import {
  bucketDeltas,
  bucketIncrements,
  bucketPoints,
  busyMedian,
  combineHistograms,
  combineTimeline,
  hostBusyMinutes,
  peakOf,
  pushSample,
  pushSnapshot,
  quantileFromBuckets,
  windowedDeltas,
  TIMELINE_SPAN_MS,
} from "@/lib/live-history";
import type { HistogramBuckets } from "@/lib/live-stats";

const hist = (
  le: (number | null)[],
  counts: number[],
  count: number | null = counts[counts.length - 1] ?? null,
  sum: number | null = 0,
): HistogramBuckets => ({ le, counts, count, sum });

// ---------------------------------------------------------------------------
// busyMedian / peakOf — history.py mirror
// ---------------------------------------------------------------------------

describe("busyMedian", () => {
  it("takes the median over BUSY samples only", () => {
    // Mostly idle with bursts: a mean over everything would land between the
    // two states and describe neither.
    const values = [0, 0, 300, 320, 0, 340, 0];
    const busy = [false, false, true, true, false, true, false];
    expect(busyMedian(values, busy)).toBe(320);
  });

  it("averages the middle pair for an even count", () => {
    expect(busyMedian([10, 20, 30, 40], [true, true, true, true])).toBe(25);
  });

  it("returns null when nothing was busy — no invented baseline", () => {
    expect(busyMedian([5, 6, 7], [false, false, false])).toBeNull();
  });

  it("narrows, not throws, when the sequences differ in length", () => {
    expect(busyMedian([1, 2, 3], [true])).toBe(1);
  });
});

describe("peakOf", () => {
  it("is over the WHOLE period, busy or not", () => {
    // A spike during an otherwise-quiet minute is still the peak — VRAM does
    // not track business at all, and masking would hide the one event worth
    // seeing.
    expect(peakOf([1, 99, 2])).toBe(99);
  });
  it("is null with no samples", () => {
    expect(peakOf([])).toBeNull();
  });
});

describe("hostBusyMinutes", () => {
  it("is decided once at host level: util OR completions", () => {
    const busy = hostBusyMinutes(
      [
        { minute: 1, max_pct: 80 },
        { minute: 2, max_pct: 0 },
        { minute: 3, max_pct: 3 },
      ],
      [
        { minute: 2, completion: 100 },
        { minute: 3, completion: 0 },
      ],
    );
    expect(busy.has(1)).toBe(true); // util
    expect(busy.has(2)).toBe(true); // completions
    expect(busy.has(3)).toBe(false); // neither
  });
});

// ---------------------------------------------------------------------------
// bucketDeltas — live_engine.py mirror, every refusal included
// ---------------------------------------------------------------------------

describe("bucketDeltas", () => {
  const older = hist([0.1, 1, null], [5, 10, 12], 12, 20);
  const newer = hist([0.1, 1, null], [8, 20, 25], 25, 55);

  it("subtracts cumulative reads into a windowed distribution", () => {
    const d = bucketDeltas(newer, older)!;
    expect(d.counts).toEqual([3, 10, 13]);
    expect(d.count).toBe(13);
    expect(d.sum).toBe(35);
  });

  it("refuses when there is no earlier read", () => {
    // The lifetime total under a "last 5 minutes" heading is precisely the
    // mislabelling the merged page removes.
    expect(bucketDeltas(newer, null)).toBeNull();
  });

  it("refuses counters that went backwards (engine restart)", () => {
    expect(bucketDeltas(older, newer)).toBeNull();
  });

  it("refuses boundaries that moved (engine upgrade)", () => {
    const moved = hist([0.2, 1, null], [8, 20, 25], 25, 55);
    expect(bucketDeltas(moved, older)).toBeNull();
  });

  it("keeps an all-zero window as a real distribution, not null", () => {
    // "No requests in this window" is a different and more useful statement
    // than a blank panel.
    const d = bucketDeltas(older, older)!;
    expect(d.counts).toEqual([0, 0, 0]);
    expect(d.count).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// quantileFromBuckets — prometheus.py mirror, with le:null as +Inf
// ---------------------------------------------------------------------------

describe("quantileFromBuckets", () => {
  // The backend's own worked example (test_hist_quantile_interpolates).
  const e2e = hist([1, 5, 10, 30, 60, null], [0, 20, 60, 90, 99, 100], 100, 0);

  it("interpolates inside the straddling bucket", () => {
    expect(quantileFromBuckets(e2e, 0.5)).toBeCloseTo(8.75);
    expect(quantileFromBuckets(e2e, 0.9)).toBeCloseTo(30.0);
    expect(quantileFromBuckets(e2e, 0.99)).toBeCloseTo(60.0);
  });

  it("returns null for an empty or all-zero histogram", () => {
    expect(quantileFromBuckets(hist([], []), 0.5)).toBeNull();
    expect(quantileFromBuckets(hist([1, null], [0, 0], 0), 0.5)).toBeNull();
    expect(quantileFromBuckets(null, 0.5)).toBeNull();
  });

  it("answers the open-ended top bucket with the last finite boundary", () => {
    const h = hist([1, null], [50, 100], 100, 0);
    expect(quantileFromBuckets(h, 0.99)).toBe(1);
  });
});

describe("bucketIncrements", () => {
  it("turns cumulative counts into per-bar increments", () => {
    expect(bucketIncrements(hist([1, 2, null], [3, 10, 12]))).toEqual([3, 7, 2]);
  });
});

// ---------------------------------------------------------------------------
// combineHistograms — additive only when the edges are identical
// ---------------------------------------------------------------------------

describe("combineHistograms", () => {
  it("sums matching-edged histograms with provenance", () => {
    const a = hist([1, null], [2, 4], 4, 1);
    const b = hist([1, null], [1, 1], 1, 2);
    const out = combineHistograms([a, b]);
    expect(out.combined?.counts).toEqual([3, 5]);
    expect(out.contributing).toBe(2);
    expect(out.total).toBe(2);
  });

  it("excludes a differently-edged histogram rather than mis-binning it", () => {
    const a = hist([1, null], [2, 4], 4, 1);
    const b = hist([2, null], [1, 1], 1, 2);
    const out = combineHistograms([a, b]);
    expect(out.combined?.counts).toEqual([2, 4]);
    expect(out.contributing).toBe(1);
  });

  it("is null with provenance when nothing reports (llama.cpp)", () => {
    const out = combineHistograms([null, null]);
    expect(out.combined).toBeNull();
    expect(out.contributing).toBe(0);
    expect(out.total).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// bucketPoints — the window's bucket-up
// ---------------------------------------------------------------------------

describe("bucketPoints", () => {
  const pts = [
    { minute: 100, v: 1 },
    { minute: 101, v: 5 },
    { minute: 105, v: 3 },
  ];

  it("returns the input untouched for 1-minute buckets", () => {
    expect(bucketPoints(pts, 1, (b) => b[0])).toBe(pts);
  });

  it("folds points into fixed-width buckets keyed on the first minute", () => {
    const out = bucketPoints(pts, 5, (b) => ({
      minute: b[0].minute,
      v: Math.max(...b.map((p) => p.v)),
    }));
    expect(out).toEqual([
      { minute: 100, v: 5 },
      { minute: 105, v: 3 },
    ]);
  });
});

// ---------------------------------------------------------------------------
// the 5-minute rings
// ---------------------------------------------------------------------------

describe("pushSample / combineTimeline", () => {
  it("evicts beyond the span and keeps per-model readings", () => {
    let ring = pushSample([], { at: 0, perModel: { a: { gen: 5, running: 1, waiting: 0 } } });
    ring = pushSample(ring, {
      at: TIMELINE_SPAN_MS + 1000,
      perModel: { a: { gen: 7, running: 2, waiting: 0 } },
    });
    expect(ring).toHaveLength(1);
    expect(ring[0].at).toBe(TIMELINE_SPAN_MS + 1000);
  });

  it("combines selected models null-not-zero", () => {
    const ring = pushSample([], {
      at: 0,
      perModel: {
        a: { gen: 5, running: 1, waiting: 0 },
        b: { gen: null, running: 1, waiting: null },
      },
    });
    const [p] = combineTimeline(ring, ["a", "b"]);
    expect(p.gen).toBe(5); // b's silence contributes nothing, not 0
    expect(p.running).toBe(2);
    expect(p.waiting).toBe(0);
    const [onlyB] = combineTimeline(ring, ["b"]);
    expect(onlyB.gen).toBeNull(); // nobody reports → null, never 0
    expect(onlyB.idle).toBe(false); // running 1
  });

  it("marks idle when every selected model reports zero running", () => {
    const ring = pushSample([], {
      at: 0,
      perModel: { a: { gen: 0, running: 0, waiting: 0 } },
    });
    expect(combineTimeline(ring, ["a"])[0].idle).toBe(true);
  });
});

describe("pushSnapshot / windowedDeltas", () => {
  const at = (s: number) => s * 1000;
  const snap = (s: number, c: number) => ({
    at: at(s),
    buckets: hist([1, null], [c, c], c, 0),
  });

  it("needs two reads before it answers", () => {
    const ring = pushSnapshot([], snap(0, 5));
    expect(windowedDeltas(ring)).toBeNull();
  });

  it("keeps one beyond-window snapshot as the delta baseline", () => {
    let ring = pushSnapshot([], snap(0, 5));
    ring = pushSnapshot(ring, snap(100, 7));
    ring = pushSnapshot(ring, snap(400, 9)); // 0s aged out; 100s is baseline
    expect(ring[0].at).toBe(at(100));
    const d = windowedDeltas(ring)!;
    expect(d.count).toBe(2); // 9 - 7
  });
});
