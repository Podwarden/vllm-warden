import { describe, it, expect, vi, afterEach } from "vitest";
import React from "react";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import { SettingField } from "@/components/settings/setting-field";
import type { GpuInfo } from "@/components/gpu/gpu-checklist";
import type { FieldHint } from "@/lib/settings-hints";

afterEach(cleanup);

const GPUS: GpuInfo[] = [
  { index: 0, name: "RTX 4090", memory_total_mib: 24564, memory_used_mib: 1200, utilization_pct: 5 },
  { index: 1, name: "RTX 4090", memory_total_mib: 24564, memory_used_mib: 800, utilization_pct: 0 },
];

// SettingField renders its label/hint/restart chrome from a `field: FieldHint`,
// not a bare `label` prop — so the gpu-set tests supply a FieldHint whose
// `label` carries the visible text the spec asserts on.
const FIELD: FieldHint = {
  label: "Default GPU indices",
  hint: "Pre-selected when adding a new model. Comma-separated GPU IDs.",
  restart: "none",
};

describe("SettingField — gpu-set kind", () => {
  it("renders the label and one checkbox per present GPU", () => {
    render(
      <SettingField kind="gpu-set" field={FIELD} value={[0]} gpus={GPUS} onChange={() => {}} />,
    );
    expect(screen.getByText("Default GPU indices")).toBeInTheDocument();
    expect(screen.getAllByRole("checkbox")).toHaveLength(2);
  });

  it("emits a sorted number[] when a GPU is toggled on", () => {
    const onChange = vi.fn();
    render(
      <SettingField kind="gpu-set" field={FIELD} value={[0]} gpus={GPUS} onChange={onChange} />,
    );
    fireEvent.click(screen.getByLabelText(/#1/));
    expect(onChange).toHaveBeenCalledWith([0, 1]);
  });

  it("renders a ghost row + alert for an absent configured index and removes it on uncheck", () => {
    const onChange = vi.fn();
    render(
      <SettingField kind="gpu-set" field={FIELD} value={[0, 5]} gpus={GPUS} onChange={onChange} />,
    );
    const ghost = screen.getByLabelText(/GPU 5 — not present/);
    expect(ghost).toBeInTheDocument();
    expect(ghost).toBeChecked();
    expect(screen.getByRole("alert")).toBeInTheDocument();
    fireEvent.click(ghost);
    expect(onChange).toHaveBeenCalledWith([0]);
  });
});

// ---------------------------------------------------------------------------
// Regression: freeform-typing branches (int-list / string-list / kv-map)
// must preserve in-progress text. The bug was that the displayed value was
// derived from the parsed result on every keystroke, wiping trailing
// commas/newlines and making multi-value entry impossible.
// ---------------------------------------------------------------------------

const FREEFORM_FIELD: FieldHint = {
  label: "L",
  hint: "h",
  restart: "none",
};

// Stateful harness exercises the full controlled round-trip: SettingField's
// parsed onChange feeds back into the `value` prop, so a buggy "display from
// parsed value" branch would snap the text back on the next render.
function Harness({ kind, initial }: { kind: "int-list" | "string-list" | "kv-map"; initial: unknown }) {
  const [v, setV] = React.useState(initial);
  return (
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    <SettingField kind={kind as any} field={FREEFORM_FIELD} value={v as any} onChange={setV as any} />
  );
}

describe("SettingField — freeform typing preserves in-progress text", () => {
  it("int-list: trailing comma and second value survive", () => {
    render(<Harness kind="int-list" initial={[]} />);
    const input = screen.getByRole("textbox") as HTMLInputElement;

    fireEvent.change(input, { target: { value: "0," } });
    expect(input.value).toBe("0,");

    fireEvent.change(input, { target: { value: "0,1" } });
    expect(input.value).toBe("0,1");
  });

  it("string-list: trailing newline and second line survive", () => {
    render(<Harness kind="string-list" initial={[]} />);
    const input = screen.getByRole("textbox") as HTMLTextAreaElement;

    fireEvent.change(input, { target: { value: "--foo\n" } });
    expect(input.value).toBe("--foo\n");

    fireEvent.change(input, { target: { value: "--foo\n--bar" } });
    expect(input.value).toBe("--foo\n--bar");
  });

  it("kv-map: Enter then a second KEY=value survive", () => {
    render(<Harness kind="kv-map" initial={{}} />);
    const input = screen.getByRole("textbox") as HTMLTextAreaElement;

    fireEvent.change(input, { target: { value: "A=1\n" } });
    expect(input.value).toBe("A=1\n");

    fireEvent.change(input, { target: { value: "A=1\nB=2" } });
    expect(input.value).toBe("A=1\nB=2");
  });

  it("int-list: still emits the parsed number[] upward", () => {
    const onChange = vi.fn();
    render(
      <SettingField kind="int-list" field={FREEFORM_FIELD} value={[]} onChange={onChange} />,
    );
    const input = screen.getByRole("textbox") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "0,1" } });
    expect(onChange).toHaveBeenLastCalledWith([0, 1]);
  });

  it("string-list: still emits the parsed string[] upward", () => {
    const onChange = vi.fn();
    render(
      <SettingField kind="string-list" field={FREEFORM_FIELD} value={[]} onChange={onChange} />,
    );
    const input = screen.getByRole("textbox") as HTMLTextAreaElement;
    fireEvent.change(input, { target: { value: "--foo\n--bar" } });
    expect(onChange).toHaveBeenLastCalledWith(["--foo", "--bar"]);
  });

  it("kv-map: still emits the parsed Record upward", () => {
    const onChange = vi.fn();
    render(
      <SettingField kind="kv-map" field={FREEFORM_FIELD} value={{}} onChange={onChange} />,
    );
    const input = screen.getByRole("textbox") as HTMLTextAreaElement;
    fireEvent.change(input, { target: { value: "A=1\nB=2" } });
    expect(onChange).toHaveBeenLastCalledWith({ A: "1", B: "2" });
  });

  it("int-list: adopts an external value reset (e.g. model switch)", () => {
    function ResetHarness() {
      const [v, setV] = React.useState<number[]>([]);
      return (
        <div>
          <button onClick={() => setV([3, 4])}>reset</button>
          <SettingField kind="int-list" field={FREEFORM_FIELD} value={v} onChange={setV} />
        </div>
      );
    }
    render(<ResetHarness />);
    const input = screen.getByRole("textbox") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "0," } });
    expect(input.value).toBe("0,");
    fireEvent.click(screen.getByText("reset"));
    expect(input.value).toBe("3,4");
  });

  it("kv-map: adopts an external value reset to different contents", () => {
    function ResetHarness() {
      const [v, setV] = React.useState<Record<string, string>>({ A: "1" });
      return (
        <div>
          <button onClick={() => setV({ C: "3" })}>reset</button>
          <SettingField kind="kv-map" field={FREEFORM_FIELD} value={v} onChange={setV} />
        </div>
      );
    }
    render(<ResetHarness />);
    const input = screen.getByRole("textbox") as HTMLTextAreaElement;
    expect(input.value).toBe("A=1");
    fireEvent.click(screen.getByText("reset"));
    expect(input.value).toBe("C=3");
  });

  it("kv-map: a reordered-but-equal prop update does NOT clobber typed text", () => {
    // Regression for the order-sensitive guard: after a save→refresh the
    // server may echo the same map in a different key order. With a plain
    // string comparison the effect would fire and snap the textarea to the
    // server's order, resetting the cursor. The canonical (sorted) guard
    // must treat reordered-equal contents as a no-op.
    function ReorderHarness() {
      const [v, setV] = React.useState<Record<string, string>>({});
      return (
        <div>
          <button onClick={() => setV({ B: "2", A: "1" })}>echo</button>
          <SettingField kind="kv-map" field={FREEFORM_FIELD} value={v} onChange={setV} />
        </div>
      );
    }
    render(<ReorderHarness />);
    const input = screen.getByRole("textbox") as HTMLTextAreaElement;
    // User types A then B; onChange emits {A:"1",B:"2"} into the prop.
    fireEvent.change(input, { target: { value: "A=1\nB=2" } });
    expect(input.value).toBe("A=1\nB=2");
    // Server echoes the same contents in reversed key order.
    fireEvent.click(screen.getByText("echo"));
    // Display must stay as the user typed — canonical forms match, no resync.
    expect(input.value).toBe("A=1\nB=2");
  });
});
