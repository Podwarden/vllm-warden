// Component tests for the /chat composer (S8 of the vllm-warden overhaul).
//
// Pin-down behaviours:
//   1. Enter submits, Shift+Enter inserts a newline. (Keyboard fluency
//      is a load-bearing UX commitment in the slice plan; a regression
//      here would surprise every operator returning to /chat.)
//   2. Esc aborts an in-flight stream.
//   3. The submit button flips between Send (emerald) and Stop (rose)
//      depending on `isStreaming`. data-testid switches between
//      `chat-send` and `chat-stop` so Playwright can drive abort by
//      role + testid without timing-sensitive sleeps.
//   4. Submit is disabled when no model is loaded, when input is empty
//      after trim, or when isDisabled is true (ensure() in-flight).
//   5. Picker emits `onModelChange` on selection.

import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import { Composer, type ComposerModel } from '@/components/chat/composer';

const MODELS: ComposerModel[] = [
  { id: 'm-1', served_model_name: 'qwen-tiny' },
  { id: 'm-2', served_model_name: 'mistral-small' },
];

interface RenderProps {
  isStreaming?: boolean;
  isDisabled?: boolean;
  models?: ComposerModel[];
  modelId?: string;
  value?: string;
}

function renderComposer(overrides: RenderProps = {}) {
  const onSubmit = vi.fn();
  const onAbort = vi.fn();
  const onValueChange = vi.fn();
  const onModelChange = vi.fn();
  const onTemperatureChange = vi.fn();
  const onMaxTokensChange = vi.fn();

  render(
    <Composer
      models={overrides.models ?? MODELS}
      modelId={overrides.modelId ?? 'm-1'}
      onModelChange={onModelChange}
      temperature={0.7}
      onTemperatureChange={onTemperatureChange}
      maxTokens={512}
      onMaxTokensChange={onMaxTokensChange}
      value={overrides.value ?? 'hello'}
      onValueChange={onValueChange}
      isStreaming={overrides.isStreaming ?? false}
      isDisabled={overrides.isDisabled ?? false}
      onSubmit={onSubmit}
      onAbort={onAbort}
    />,
  );

  return {
    onSubmit,
    onAbort,
    onValueChange,
    onModelChange,
    onTemperatureChange,
    onMaxTokensChange,
    input: screen.getByTestId('chat-input') as HTMLTextAreaElement,
  };
}

afterEach(() => cleanup());

describe('Composer keyboard', () => {
  it('Enter calls onSubmit when input is non-empty and a model is selected', () => {
    const { onSubmit, input } = renderComposer();
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(onSubmit).toHaveBeenCalledTimes(1);
  });

  it('Shift+Enter does NOT submit (newline goes through to the textarea)', () => {
    const { onSubmit, input } = renderComposer();
    fireEvent.keyDown(input, { key: 'Enter', shiftKey: true });
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it('Enter is a no-op when input is whitespace only', () => {
    const { onSubmit, input } = renderComposer({ value: '   ' });
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it('Enter is a no-op when no model is selected', () => {
    const { onSubmit, input } = renderComposer({ modelId: '' });
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it('Esc calls onAbort while streaming', () => {
    const { onAbort, input } = renderComposer({ isStreaming: true });
    fireEvent.keyDown(input, { key: 'Escape' });
    expect(onAbort).toHaveBeenCalledTimes(1);
  });

  it('Esc is a no-op when not streaming', () => {
    const { onAbort, input } = renderComposer({ isStreaming: false });
    fireEvent.keyDown(input, { key: 'Escape' });
    expect(onAbort).not.toHaveBeenCalled();
  });
});

describe('Composer submit button', () => {
  it('shows the Send button when idle', () => {
    renderComposer();
    expect(screen.getByTestId('chat-send')).toBeInTheDocument();
    expect(screen.queryByTestId('chat-stop')).not.toBeInTheDocument();
  });

  it('flips to the Stop button while streaming', () => {
    renderComposer({ isStreaming: true });
    expect(screen.getByTestId('chat-stop')).toBeInTheDocument();
    expect(screen.queryByTestId('chat-send')).not.toBeInTheDocument();
  });

  it('clicking Stop calls onAbort', () => {
    const { onAbort } = renderComposer({ isStreaming: true });
    fireEvent.click(screen.getByTestId('chat-stop'));
    expect(onAbort).toHaveBeenCalledTimes(1);
  });

  it('clicking Send calls onSubmit', () => {
    const { onSubmit } = renderComposer();
    fireEvent.click(screen.getByTestId('chat-send'));
    expect(onSubmit).toHaveBeenCalledTimes(1);
  });

  it('Send is disabled when no model is loaded', () => {
    renderComposer({ models: [], modelId: '' });
    const btn = screen.getByTestId('chat-send') as HTMLButtonElement;
    expect(btn.disabled).toBe(true);
  });

  it('Send is disabled while ensure() is in-flight (isDisabled)', () => {
    renderComposer({ isDisabled: true });
    const btn = screen.getByTestId('chat-send') as HTMLButtonElement;
    expect(btn.disabled).toBe(true);
  });
});

describe('Composer controls', () => {
  it('model picker emits onModelChange when changed', () => {
    const { onModelChange } = renderComposer();
    const picker = screen.getByTestId('chat-model-picker') as HTMLSelectElement;
    fireEvent.change(picker, { target: { value: 'm-2' } });
    expect(onModelChange).toHaveBeenCalledWith('m-2');
  });

  it('temperature slider emits onTemperatureChange with a number', () => {
    const { onTemperatureChange } = renderComposer();
    const slider = screen.getByTestId('chat-temperature') as HTMLInputElement;
    fireEvent.change(slider, { target: { value: '1.25' } });
    expect(onTemperatureChange).toHaveBeenCalledWith(1.25);
  });

  it('max-tokens slider emits onMaxTokensChange with a number', () => {
    const { onMaxTokensChange } = renderComposer();
    const slider = screen.getByTestId('chat-max-tokens') as HTMLInputElement;
    fireEvent.change(slider, { target: { value: '1024' } });
    expect(onMaxTokensChange).toHaveBeenCalledWith(1024);
  });
});
