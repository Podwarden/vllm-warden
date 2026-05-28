'use client';
// /chat — playground page (S8 of the vllm-warden overhaul, see plan §S8).
//
// On mount:
//  1. POST /api/chat/playground/ensure to make sure the `vw-playground`
//     bearer token exists server-side. The plaintext stays server-side;
//     we get back only its id + a created flag (used for telemetry, not UX).
//  2. GET /api/models — used to populate the picker, filtered to
//     status === 'loaded' (you can only talk to a live engine).
//
// Conversation state is in-memory only (this is a React useState array).
// Refresh the page and the transcript is gone — explicit S8 contract
// from the dispatch prompt. We do NOT write to localStorage or to the
// DB; the playground is a probe, not a journal.

import { useCallback, useEffect, useMemo, useState } from 'react';
import useSWR from 'swr';
import { Composer, type ComposerModel } from '@/components/chat/composer';
import { MessageList } from '@/components/chat/message-list';
import { authFetch, authFetchJSON } from '@/lib/auth-fetch';
import { useChatStream, type ChatMessage } from '@/lib/use-chat-stream';

interface ModelListRow {
  id: string;
  served_model_name: string;
  status: string;
}

// `GET /api/models` returns an envelope `{ models: [...] }`, NOT a bare
// array. This matches the contract that `/models` and other consumers
// have always relied on (see `frontend/src/app/models/page.tsx`). The
// chat page originally tried to treat the response as an array, which
// silently produced an empty picker even when a model was loaded
// (#147). Keep the envelope explicit here so the SWR generic and the
// downstream destructure stay in sync.
interface ModelsListResponse {
  models: ModelListRow[];
}

const DEFAULT_TEMPERATURE = 0.7;
const DEFAULT_MAX_TOKENS = 512;

export default function ChatPage() {
  // Persisted (session-only) conversation. The last assistant message
  // is mutable while streaming — see streamingText below.
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [modelId, setModelId] = useState('');
  const [temperature, setTemperature] = useState(DEFAULT_TEMPERATURE);
  const [maxTokens, setMaxTokens] = useState(DEFAULT_MAX_TOKENS);
  // The ensure() flow is fire-and-forget on mount; we disable the
  // composer until it resolves so the FE doesn't race a 409 from
  // /api/chat/completions ("playground token not initialised").
  const [ensured, setEnsured] = useState(false);
  const [ensureError, setEnsureError] = useState<string | null>(null);

  const { phase, streamingText, errorMessage, send, abort } = useChatStream();
  const isStreaming = phase === 'streaming';

  // Step 1 — ensure the playground token on mount.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await authFetch('/api/chat/playground/ensure', {
          method: 'POST',
        });
        if (!r.ok) {
          const detail = await r
            .json()
            .catch(() => ({ detail: `HTTP ${r.status}` }));
          throw new Error(
            typeof detail.detail === 'string'
              ? detail.detail
              : `ensure failed (${r.status})`,
          );
        }
        if (!cancelled) setEnsured(true);
      } catch (err) {
        if (!cancelled) {
          setEnsureError(
            err instanceof Error ? err.message : 'ensure() failed',
          );
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Step 2 — models list. Refresh every 5s so a newly-loaded model
  // shows up without the operator having to reload /chat. SWR's
  // built-in dedupe means we don't hammer the API.
  //
  // The backend returns `{ models: [...] }` (envelope) — NOT a bare
  // array. See ModelsListResponse comment above for the #147
  // regression-fix history.
  const { data: modelsResponse } = useSWR<ModelsListResponse>(
    '/api/models',
    authFetchJSON,
    { refreshInterval: 5000, shouldRetryOnError: false },
  );

  // Filter to loaded models and shape into the picker's contract.
  // The optional-chain + Array.isArray guard tolerates both an
  // in-flight load (modelsResponse is undefined) and a malformed
  // payload (defensive: never trust a wire payload's shape blindly).
  const loadedModels: ComposerModel[] = useMemo(() => {
    const rows = modelsResponse?.models;
    if (!Array.isArray(rows)) return [];
    return rows
      .filter((m) => m.status === 'loaded')
      .map((m) => ({ id: m.id, served_model_name: m.served_model_name }));
  }, [modelsResponse]);

  // Auto-select the first loaded model when the list first arrives, or
  // when the currently-selected model disappears (operator unloaded
  // mid-conversation). We keep the conversation; only the picker resets.
  useEffect(() => {
    if (loadedModels.length === 0) {
      if (modelId !== '') setModelId('');
      return;
    }
    const stillThere = loadedModels.some((m) => m.id === modelId);
    if (!stillThere) {
      setModelId(loadedModels[0].id);
    }
  }, [loadedModels, modelId]);

  // Send a turn. Pushes the user message, then streams the assistant
  // reply into a new message. We append the assistant message ONCE,
  // when the stream resolves — pre-resolution the partial text lives
  // in streamingText only (rendered as the streaming placeholder).
  const handleSubmit = useCallback(async () => {
    const text = input.trim();
    if (!text || !modelId || isStreaming || !ensured) return;
    const userMsg: ChatMessage = { role: 'user', content: text };
    const next = [...messages, userMsg];
    setMessages(next);
    setInput('');
    // Resolve the picker's internal id -> served_model_name on the wire.
    // The backend proxy (app/proxy/routes.py) looks up the engine by
    // served_model_name; posting the internal id 404'd as "model
    // '<hash>' is not loaded". The fallback keeps the previous behaviour
    // for any stray modelId not in loadedModels (defensive only).
    const wireModel =
      loadedModels.find((m) => m.id === modelId)?.served_model_name ?? modelId;
    const reply = await send({
      model: wireModel,
      messages: next,
      temperature,
      max_tokens: maxTokens,
    });
    if (reply) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: reply },
      ]);
    }
  }, [input, modelId, isStreaming, ensured, messages, send, temperature, maxTokens, loadedModels]);

  // Regenerate the last assistant turn: drop the trailing assistant
  // message (if any) and re-stream from the same user prompt above it.
  // No-op if the conversation doesn't end with an assistant message
  // (e.g. the previous send was aborted before any tokens arrived).
  const handleRegenerate = useCallback(async () => {
    if (isStreaming || !modelId || !ensured) return;
    const lastIdx = messages.length - 1;
    if (lastIdx < 0 || messages[lastIdx].role !== 'assistant') return;
    const truncated = messages.slice(0, lastIdx);
    setMessages(truncated);
    // Same picker-id -> served_model_name resolution as handleSubmit.
    // See comment there for the proxy contract.
    const wireModel =
      loadedModels.find((m) => m.id === modelId)?.served_model_name ?? modelId;
    const reply = await send({
      model: wireModel,
      messages: truncated,
      temperature,
      max_tokens: maxTokens,
    });
    if (reply) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: reply },
      ]);
    }
  }, [isStreaming, modelId, ensured, messages, send, temperature, maxTokens, loadedModels]);

  // Index of the last assistant message — passed to MessageList so it
  // can show the Regenerate button on only the freshest one.
  const lastAssistantIndex = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'assistant') return i;
    }
    return -1;
  }, [messages]);

  return (
    <main className="container mx-auto flex h-[calc(100vh-3.5rem)] max-w-4xl flex-col px-4">
      <header className="py-4">
        <h1 className="font-mono text-sm font-semibold uppercase tracking-[0.22em] text-emerald-400">
          Chat playground
        </h1>
        <p className="mt-1 text-xs text-slate-500">
          Probe loaded models with live token streaming. Session-only.
        </p>
      </header>

      {ensureError && (
        <div
          role="alert"
          className="mb-2 rounded border border-rose-700 bg-rose-950/40 px-3 py-2 font-mono text-xs text-rose-300"
        >
          ensure() failed: {ensureError}
        </div>
      )}
      {errorMessage && (
        <div
          role="alert"
          data-testid="chat-error"
          className="mb-2 rounded border border-rose-700 bg-rose-950/40 px-3 py-2 font-mono text-xs text-rose-300"
        >
          {errorMessage}
        </div>
      )}

      <section className="flex-1 overflow-y-auto rounded border border-slate-800 bg-slate-950/40">
        <MessageList
          messages={messages}
          streamingText={streamingText}
          isStreaming={isStreaming}
          onRegenerate={() => {
            void handleRegenerate();
          }}
          lastAssistantIndex={lastAssistantIndex}
        />
      </section>

      <Composer
        models={loadedModels}
        modelId={modelId}
        onModelChange={setModelId}
        temperature={temperature}
        onTemperatureChange={setTemperature}
        maxTokens={maxTokens}
        onMaxTokensChange={setMaxTokens}
        value={input}
        onValueChange={setInput}
        isStreaming={isStreaming}
        isDisabled={!ensured}
        onSubmit={() => {
          void handleSubmit();
        }}
        onAbort={abort}
      />
    </main>
  );
}
