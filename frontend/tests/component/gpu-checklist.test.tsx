import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import { GpuChecklist, type GpuInfo } from "@/components/gpu/gpu-checklist";

afterEach(cleanup);

const GPUS: GpuInfo[] = [
  { index: 0, name: "NVIDIA RTX A4000", memory_total_mib: 16376, memory_used_mib: 1024, utilization_pct: 5 },
  { index: 1, name: "NVIDIA RTX A4000", memory_total_mib: 16376, memory_used_mib: 0, utilization_pct: 0 },
];

describe("GpuChecklist", () => {
  it("renders one checkbox per present GPU", () => {
    render(<GpuChecklist gpus={GPUS} selected={[0]} onChange={() => {}} />);
    expect(screen.getByLabelText(/#0/)).toBeChecked();
    expect(screen.getByLabelText(/#1/)).not.toBeChecked();
  });

  it("toggling a GPU emits a sorted number[]", () => {
    const onChange = vi.fn();
    render(<GpuChecklist gpus={GPUS} selected={[1]} onChange={onChange} />);
    fireEvent.click(screen.getByLabelText(/#0/));
    expect(onChange).toHaveBeenCalledWith([0, 1]);
  });

  it("renders a removable ghost row + warning banner for a missing configured index", () => {
    const onChange = vi.fn();
    render(<GpuChecklist gpus={GPUS} selected={[0, 5]} onChange={onChange} />);
    const ghost = screen.getByLabelText(/GPU 5 — not present/);
    expect(ghost).toBeChecked();
    expect(screen.getByRole("alert")).toHaveTextContent(/not present/i);
    fireEvent.click(ghost); // unchecking removes it
    expect(onChange).toHaveBeenCalledWith([0]);
  });

  it("shows an empty-state message when no GPUs are present and none selected", () => {
    render(<GpuChecklist gpus={[]} selected={[]} onChange={() => {}} />);
    expect(screen.getByText(/no gpus detected/i)).toBeInTheDocument();
  });
});
