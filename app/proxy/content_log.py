"""Config-gated, per-token-scoped request/response content logger.

Tapped into the proxy forward path (``app/proxy/routes.py::_forward``) to
capture full prompt + completion content for a small allowlist of tokens on a
shared server — the diagnostic tool for chasing runaway / degenerate
generations where we need to see exactly what came out.

Contract:

* **Disabled by default.** When ``settings.content_log_enabled`` is False the
  caller's fast-path gate (:func:`should_log`) short-circuits before any of
  this runs, so the forward path stays byte-identical to a build without this
  module — no extra awaits, no extra work.
* **Hard-scoped.** Even when enabled, a request is logged only if its token id
  is in ``settings.content_log_tokens``. There is deliberately no "log
  everything" mode: this runs on shared production.
* **Never breaks or delays the response.** Every public entry point swallows
  exceptions and at most logs a warning. The caller does the write only after
  the response content is in hand and the scheduler slot is released.

One JSON object per line (JSONL). ``prompt`` and ``completion`` are capped at
``settings.content_log_max_chars`` each so the file can't grow unbounded on the
tens-of-thousands-of-token generations we're chasing.
"""

import json
import logging
import time

log = logging.getLogger("vllm_warden.content_log")


def should_log(settings, token_id) -> bool:
    """Cheap gate: True only when logging is enabled AND this token is allowlisted.

    Called first in the hot path so a disabled logger costs one attribute read
    plus, at most, a set-membership test — no I/O, no parsing.
    """
    return (
        settings.content_log_enabled
        and token_id is not None
        and token_id in settings.content_log_tokens
    )


def _cap(text: str | None, max_chars: int) -> str | None:
    """Truncate ``text`` to ``max_chars``, appending a marker noting how much
    was dropped. ``None`` passes through unchanged."""
    if text is None:
        return None
    if max_chars >= 0 and len(text) > max_chars:
        dropped = len(text) - max_chars
        return text[:max_chars] + f"…[truncated {dropped} chars]"
    return text


def parse_nonstream(content: bytes, is_chat: bool) -> tuple[str | None, str | None]:
    """Extract ``(completion_text, finish_reason)`` from a non-streaming body.

    ``is_chat`` selects ``choices[0].message.content`` (chat) vs
    ``choices[0].text`` (/completions). Best-effort: any parse failure yields
    ``(None, None)`` rather than raising.
    """
    try:
        data = json.loads(content)
        choice = (data.get("choices") or [{}])[0] or {}
        if is_chat:
            completion = (choice.get("message") or {}).get("content")
        else:
            completion = choice.get("text")
        return completion, choice.get("finish_reason")
    except Exception:
        return None, None


def parse_sse_finish(line: bytes) -> str | None:
    """Return ``choices[0].finish_reason`` from one SSE ``data:`` frame, if any.

    Companion to ``routes._parse_sse_delta`` — the streaming tap tracks the
    last non-null finish_reason so the JSONL record can carry it. Only called
    from the gated (logging-enabled) path, so the extra parse never touches the
    disabled hot path.
    """
    if not line.lstrip().startswith(b"data:"):
        return None
    payload = line.split(b"data:", 1)[1].strip()
    if payload == b"[DONE]":
        return None
    try:
        ev = json.loads(payload)
        return (ev.get("choices") or [{}])[0].get("finish_reason")
    except Exception:
        return None


def write_entry(
    settings,
    *,
    token_id,
    model_id,
    served_name,
    stream: bool,
    max_tokens,
    prompt_tokens,
    completion_tokens,
    finish_reason,
    prompt,
    completion,
    extra: dict | None = None,
) -> None:
    """Append one JSONL record. Best-effort — never raises into the caller.

    ``extra`` merges additional top-level keys into the record (used by the
    runaway detector to attach the trip signal and the tokens-in-think count);
    it never overrides the core fields above.
    """
    try:
        path = settings.content_log_path
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.time(),
            "token_id": token_id,
            "model_id": model_id,
            "served_name": served_name,
            "stream": stream,
            "max_tokens": max_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "finish_reason": finish_reason,
            "prompt": _cap(prompt, settings.content_log_max_chars),
            "completion": _cap(completion, settings.content_log_max_chars),
        }
        if extra:
            for k, v in extra.items():
                record.setdefault(k, v)
        line = json.dumps(record, ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        # Diagnostic logging must never break or delay the proxied response.
        log.warning("content_log: failed to write entry", exc_info=True)
