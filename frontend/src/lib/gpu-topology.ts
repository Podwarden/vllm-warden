// Which card reaches which, and over what — the pure half of the
// interconnect graph on /stats. Input is the `nvidia-smi topo -m` matrix
// as the API relays it (legend codes verbatim); output is a classification
// and a layout the SVG component draws without further judgement.
//
// The legend, and what it means for the picture:
//   X              self
//   NV#            a bonded set of # NVLinks — a DIRECT card-to-card path
//   PIX PXB PHB    no direct link; routed through PCIe at increasing
//   NODE SYS       distance (bridge, host bridge, NUMA node, socket)
//
// Three renderings, and the distinction matters:
//   hub      no NVLink anywhere. Every card runs to the CPU's PCIe host
//            bridge and reaches its neighbours through it. NO card-to-card
//            edge is drawn — PHB means the pair has no direct path, and a
//            line between them would assert hardware that does not exist.
//   bridged  some pairs NVLink-bonded (thick direct edges labelled NV#);
//            every card still has its thin PCIe spoke to the host bridge.
//   fabric   every pair NVLink: the cards are on an NVSwitch. Drawn as the
//            one switch they all join, not a mesh — 8 cards would be 28
//            lines and 16 would be 120, and the switch IS the hardware.

export type TopologyKind = "hub" | "bridged" | "fabric";

export interface NvlinkPair {
  /** Row/column positions into the matrix (NOT GPU indices). */
  a: number;
  b: number;
  /** The legend code, e.g. "NV2". */
  code: string;
}

export interface TopologyClass {
  kind: TopologyKind;
  n: number;
  /** Direct NVLink bonds, each pair once (a < b). Empty for `hub`. */
  nvPairs: NvlinkPair[];
  /** For `fabric`: the code every pair reports (e.g. "NV12"). */
  fabricCode: string | null;
}

const NV_CODE = /^NV\d*$/;

export function isNvlinkCode(code: string): boolean {
  return NV_CODE.test(code.trim());
}

export function classifyTopology(matrix: string[][]): TopologyClass {
  const n = matrix.length;
  const nvPairs: NvlinkPair[] = [];
  let offDiagonal = 0;
  let nvCells = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      offDiagonal++;
      const code = (matrix[i]?.[j] ?? "").trim();
      if (!isNvlinkCode(code)) continue;
      nvCells++;
      if (j > i) nvPairs.push({ a: i, b: j, code });
    }
  }
  // Two bonded cards are a pair, not a switch: a fabric needs three or
  // more cards with every pair linked.
  if (n > 2 && offDiagonal > 0 && nvCells === offDiagonal) {
    return { kind: "fabric", n, nvPairs, fabricCode: nvPairs[0]?.code ?? null };
  }
  if (nvPairs.length > 0) {
    return { kind: "bridged", n, nvPairs, fabricCode: null };
  }
  return { kind: "hub", n, nvPairs: [], fabricCode: null };
}

export interface Point {
  x: number;
  y: number;
}

export interface TopologyLayout {
  width: number;
  height: number;
  hub: Point;
  /** One entry per matrix row, in matrix order. */
  nodes: Point[];
  /** True when the cards sit in a row above and a row below the hub. */
  rows: boolean;
}

export const NODE_W = 92;
export const NODE_H = 38;
export const HUB_W = 132;
export const HUB_H = 42;

// Derived from the boxes, not chosen. A node beside the hub needs half of
// each plus a gap between them, and the worst case is a three-card ring whose
// lower nodes sit at 30 degrees: at RX 120 they cleared by 104px against the
// 124px they need, and overlapped the hub on a real four-card host.
const RING_GAP = 14;
const RING_RX = Math.ceil((NODE_W / 2 + HUB_W / 2 + RING_GAP) / Math.cos(Math.PI / 6));
const RING_RY = 0.66;
const ROW_STEP = 118;

/** Up to four cards sit on an ellipse around the hub. Past that they crowd,
 *  so they split into a row above and a row below — the shape a machine
 *  with eight cards actually has, and it keeps every label legible. There
 *  is no upper bound: 16-GPU hosts exist and MIG can add instances; the
 *  container scrolls horizontally when the rows outgrow it. */
export function layoutTopology(n: number): TopologyLayout {
  const rows = n > 4;
  const perRow = rows ? Math.ceil(n / 2) : n;
  const width = rows
    ? Math.max(560, 60 + perRow * ROW_STEP)
    // Wide enough for the ring plus a whole node either side, so an edge
    // label never lands outside the viewBox.
    : Math.max(460, 2 * (RING_RX + NODE_W / 2) + 48);
  const height = rows ? 320 : 276;
  const cx = width / 2;
  const cy = height / 2;
  const nodes: Point[] = [];
  for (let i = 0; i < n; i++) {
    if (!rows) {
      const angle = ((-90 + (360 / n) * i) * Math.PI) / 180;
      nodes.push({
        x: cx + RING_RX * Math.cos(angle),
        y: cy + RING_RX * Math.sin(angle) * RING_RY,
      });
      continue;
    }
    const top = i < perRow;
    const k = top ? i : i - perRow;
    const count = top ? perRow : n - perRow;
    const span = (count - 1) * ROW_STEP;
    nodes.push({ x: cx - span / 2 + k * ROW_STEP, y: top ? 36 : height - 36 });
  }
  return { width, height, hub: { x: cx, y: cy }, nodes, rows };
}

/** Where a spoke's label sits: the midpoint, nudged off the line. */
export function spokeLabelPoint(from: Point, to: Point, offset = 13): Point {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const len = Math.hypot(dx, dy) || 1;
  return {
    x: (from.x + to.x) / 2 + (-dy / len) * offset,
    y: (from.y + to.y) / 2 + (dx / len) * offset,
  };
}
