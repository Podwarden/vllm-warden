import { describe, it, expect } from 'vitest';
import {
  filterBlocksByModel,
  groupBlocks,
  type GodModeEvent,
  type RequestBlockData,
} from '@/components/godmode/godmode-viewer';

// ---------------------------------------------------------------------------
// God mode must reflect the model selection too, not only the stats numbers.
//
// The filtering happens in the CLIENT and that is a design decision, not a
// shortcut. The hub is a shared broadcast ring with a replay snapshot; a
// per-subscriber server-side filter would drop a `request_start` while still
// delivering that request's `delta` and `request_end` frames, leaving orphan
// blocks with no prompt and no model. Correlation is by `req_id`, so a whole
// conversation has to arrive together and be grouped before anything can be
// attributed to a model at all.
//
// The env-var gate on god mode is untouched by any of this — the stream is as
// gated as it ever was; this only decides what a viewer who already has it
// renders.
// ---------------------------------------------------------------------------

function start(reqId: string, model: string): GodModeEvent {
  return {
    type: 'request_start',
    seq: 1,
    req_id: reqId,
    ts: 0,
    token_label: 'k',
    token_id: 't',
    model,
    served_name: model,
    client_ip: '10.0.0.1',
    stream: true,
    prompt: 'hi',
    prompt_elided: false,
  } as GodModeEvent;
}

function delta(reqId: string): GodModeEvent {
  return {
    type: 'delta',
    seq: 2,
    req_id: reqId,
    ts: 0,
    text: 'out',
  } as GodModeEvent;
}

describe('filterBlocksByModel', () => {
  const blocks: RequestBlockData[] = groupBlocks([
    start('r1', 'm-llama'),
    delta('r1'),
    start('r2', 'm-qwen'),
    delta('r2'),
  ]);

  it('passes everything through when there is no selection', () => {
    // Never an empty allow-list while the selection is still resolving — that
    // would blank a forensic view for a beat and read as "no traffic".
    expect(filterBlocksByModel(blocks, null)).toHaveLength(2);
  });

  it('keeps only the selected model', () => {
    const out = filterBlocksByModel(blocks, ['m-llama']);
    expect(out.map((b) => b.reqId)).toEqual(['r1']);
  });

  it('keeps any combination', () => {
    expect(filterBlocksByModel(blocks, ['m-llama', 'm-qwen'])).toHaveLength(2);
  });

  it('keeps a request whose deltas belong to it, not just its start', () => {
    // Correlation is by req_id: filtering must select whole CONVERSATIONS. A
    // rule applied per event would keep the prompt and drop the completion.
    const out = filterBlocksByModel(blocks, ['m-qwen']);
    expect(out[0].deltas).toHaveLength(1);
    expect(out[0].start?.served_name).toBe('m-qwen');
  });

  it('KEEPS a block whose start was evicted from the ring', () => {
    // Its model is unknowable. Dropping it would make a narrowed god-mode view
    // quietly incomplete, which is the one thing a forensic tool must never
    // be — better an extra block than a missing one.
    const orphan = groupBlocks([delta('r3')]);
    const out = filterBlocksByModel([...blocks, ...orphan], ['m-llama']);
    expect(out.map((b) => b.reqId)).toEqual(['r1', 'r3']);
  });

  it('does not mutate the input', () => {
    const before = blocks.length;
    filterBlocksByModel(blocks, ['m-llama']);
    expect(blocks).toHaveLength(before);
  });
});
