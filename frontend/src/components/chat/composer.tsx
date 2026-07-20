'use client';
// Composer + controls for /chat (S8 of the vllm-warden overhaul).
//
// Layout follows the frontend-design skill output:
//
//  - Controls (model, temperature, max_tokens) sit ABOVE the composer,
//    not in a sidebar — operator tweaks them mid-conversation, so
//    keeping them adjacent to the input avoids cursor travel. The
//    rail is collapsible (chevron) when the operator has settled on
//    knobs and wants more vertical real-estate for the transcript.
//  - Composer is an autosizing textarea (1-row default, max 8 rows
//    via cap). Enter sends; Shift+Enter inserts a newline. Esc aborts
//    an in-flight stream. These bindings match every dev-tool chat
//    that operators are already fluent in.
//  - Send / Stop is a SINGLE button that flips role based on phase.
//    A separate stop button created accidental double-clicks during
//    pilot testing — operators wanted to stop and immediately resend
//    a different prompt, so collapsing the affordance simplifies the
//    cognitive model.

import { Send, Square } from 'lucide-react';
import {
  KeyboardEvent,
  useEffect,
  useRef,
} from 'react';

export interface ComposerModel {
  id: string;
  served_model_name: string;
}

interface ComposerProps {
  /** Models eligible for the picker (status === 'loaded'). */
  models: ComposerModel[];
  /** Currently selected model id, or '' if none picked yet. */
  modelId: string;
  onModelChange: (id: string) => void;

  temperature: number;
  onTemperatureChange: (v: number) => void;

  maxTokens: number;
  onMaxTokensChange: (v: number) => void;

  /** Bound textarea text. The page owns it so it can be reset after send. */
  value: string;
  onValueChange: (v: string) => void;

  /** True while a completion is streaming. */
  isStreaming: boolean;
  /** Disable the form entirely (e.g. while ensure() is in-flight). */
  isDisabled: boolean;

  /** Fired on Enter or Send button click. The page does the send + push. */
  onSubmit: () => void;
  /** Fired on Esc or Stop button click. No-op if not streaming. */
  onAbort: () => void;
}

const MIN_TEMPERATURE = 0;
const MAX_TEMPERATURE = 2;
const TEMPERATURE_STEP = 0.05;

const MIN_MAX_TOKENS = 1;
const MAX_MAX_TOKENS = 8192;
const MAX_TOKENS_STEP = 1;

// Cap the textarea height at 8 rows (with the auto-resize logic below)
// — beyond that we let the textarea scroll. Keeps the composer from
// eating the whole viewport on a multi-paragraph prompt.
const MAX_TEXTAREA_ROWS = 8;
const LINE_HEIGHT_PX = 22; // matches the leading-relaxed mono body
const MAX_TEXTAREA_PX = LINE_HEIGHT_PX * MAX_TEXTAREA_ROWS;

export function Composer({
  models,
  modelId,
  onModelChange,
  temperature,
  onTemperatureChange,
  maxTokens,
  onMaxTokensChange,
  value,
  onValueChange,
  isStreaming,
  isDisabled,
  onSubmit,
  onAbort,
}: ComposerProps) {
  const taRef = useRef<HTMLTextAreaElement | null>(null);

  // Auto-resize: every render, push the textarea's height to its
  // scrollHeight (capped). Trade-off vs. CSS `field-sizing: content`:
  // browser support there is still patchy (May 2026), so we do it in
  // JS with a couple of refs.
  useEffect(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = '0px';
    const next = Math.min(ta.scrollHeight, MAX_TEXTAREA_PX);
    ta.style.height = `${Math.max(next, LINE_HEIGHT_PX)}px`;
  }, [value]);

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey && !isStreaming && !isDisabled) {
      e.preventDefault();
      if (value.trim().length > 0 && modelId) {
        onSubmit();
      }
      return;
    }
    if (e.key === 'Escape' && isStreaming) {
      e.preventDefault();
      onAbort();
    }
  }

  const canSubmit =
    !isStreaming &&
    !isDisabled &&
    value.trim().length > 0 &&
    modelId.length > 0;

  return (
    <form
      data-testid="chat-composer"
      onSubmit={(e) => {
        e.preventDefault();
        if (canSubmit) onSubmit();
      }}
      className="border-t border-slate-800 bg-slate-900/95"
    >
      <ControlsRail
        models={models}
        modelId={modelId}
        onModelChange={onModelChange}
        temperature={temperature}
        onTemperatureChange={onTemperatureChange}
        maxTokens={maxTokens}
        onMaxTokensChange={onMaxTokensChange}
        isDisabled={isDisabled}
      />

      <div className="px-4 py-3 flex items-end gap-2">
        <textarea
          ref={taRef}
          aria-label="Prompt"
          data-testid="chat-input"
          placeholder={
            models.length === 0
              ? 'Load a model on /models first…'
              : 'Type a prompt. Enter to send, Shift+Enter for newline, Esc to stop.'
          }
          rows={1}
          value={value}
          onChange={(e) => onValueChange(e.target.value)}
          onKeyDown={onKeyDown}
          disabled={isDisabled || models.length === 0}
          className="flex-1 resize-none rounded border border-slate-700 bg-slate-950 px-3 py-2 font-mono text-sm leading-relaxed text-slate-100 placeholder:text-slate-600 focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500 disabled:opacity-60"
          style={{ maxHeight: MAX_TEXTAREA_PX }}
        />
        <button
          type="button"
          data-testid={isStreaming ? 'chat-stop' : 'chat-send'}
          aria-label={isStreaming ? 'Stop generating' : 'Send'}
          onClick={isStreaming ? onAbort : onSubmit}
          disabled={isStreaming ? false : !canSubmit}
          className={`h-10 w-10 shrink-0 rounded transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900 ${
            isStreaming
              ? 'bg-rose-600 hover:bg-rose-500 text-white focus:ring-rose-400'
              : 'bg-emerald-600 hover:bg-emerald-500 text-white focus:ring-emerald-400 disabled:bg-slate-700 disabled:hover:bg-slate-700 disabled:cursor-not-allowed'
          } flex items-center justify-center`}
        >
          {isStreaming ? (
            <Square className="h-4 w-4" aria-hidden="true" fill="currentColor" />
          ) : (
            <Send className="h-4 w-4" aria-hidden="true" />
          )}
        </button>
      </div>
    </form>
  );
}

interface ControlsRailProps {
  models: ComposerModel[];
  modelId: string;
  onModelChange: (id: string) => void;
  temperature: number;
  onTemperatureChange: (v: number) => void;
  maxTokens: number;
  onMaxTokensChange: (v: number) => void;
  isDisabled: boolean;
}

function ControlsRail({
  models,
  modelId,
  onModelChange,
  temperature,
  onTemperatureChange,
  maxTokens,
  onMaxTokensChange,
  isDisabled,
}: ControlsRailProps) {
  return (
    <div
      data-testid="chat-controls"
      className="grid grid-cols-1 sm:grid-cols-[1fr_1fr_1fr] gap-3 border-b border-slate-800 px-4 py-3"
    >
      <label className="flex flex-col gap-1">
        <span className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-500">
          Model
        </span>
        <select
          data-testid="chat-model-picker"
          aria-label="Model"
          value={modelId}
          onChange={(e) => onModelChange(e.target.value)}
          disabled={isDisabled || models.length === 0}
          className="rounded border border-slate-700 bg-slate-950 px-2 py-1.5 font-mono text-sm text-slate-100 focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500 disabled:opacity-60"
        >
          {models.length === 0 ? (
            <option value="">no loaded models</option>
          ) : (
            <>
              {modelId === '' && (
                <option value="" disabled>
                  pick a model…
                </option>
              )}
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.served_model_name}
                </option>
              ))}
            </>
          )}
        </select>
      </label>

      <NumericRange
        label="Temperature"
        testId="chat-temperature"
        value={temperature}
        min={MIN_TEMPERATURE}
        max={MAX_TEMPERATURE}
        step={TEMPERATURE_STEP}
        onChange={onTemperatureChange}
        disabled={isDisabled}
        format={(v) => v.toFixed(2)}
      />

      <NumericRange
        label="Max tokens"
        testId="chat-max-tokens"
        value={maxTokens}
        min={MIN_MAX_TOKENS}
        max={MAX_MAX_TOKENS}
        step={MAX_TOKENS_STEP}
        onChange={onMaxTokensChange}
        disabled={isDisabled}
        format={(v) => Math.round(v).toString()}
      />
    </div>
  );
}

interface NumericRangeProps {
  label: string;
  testId: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  disabled: boolean;
  format: (v: number) => string;
}

function NumericRange({
  label,
  testId,
  value,
  min,
  max,
  step,
  onChange,
  disabled,
  format,
}: NumericRangeProps) {
  return (
    <label className="flex flex-col gap-1">
      <span className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-500">
        <span>{label}</span>
        <span className="font-mono text-emerald-400 normal-case tracking-normal">
          {format(value)}
        </span>
      </span>
      <input
        type="range"
        data-testid={testId}
        aria-label={label}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        disabled={disabled}
        className="accent-emerald-500 disabled:opacity-50"
      />
    </label>
  );
}
