"""HF tokenizer cache used by the proxy for prompt/completion accounting.

The cache grows by one entry per distinct (hf_repo, trust_remote_code) tuple
the proxy has been asked to count tokens for. Each entry holds a fully
loaded ``AutoTokenizer`` (vocab + merges + special-tokens config), which
runs anywhere from ~1 MiB (small word-piece tokenizers) to ~20+ MiB
(sentencepiece + BPE merges for some Qwen/Llama variants) of process RSS.

S7 (#124) — code-review finding #5: the cache was previously **never**
flushed on model unload. A long-lived warden that loads-and-unloads a
rotating set of models (operator iterating on tuning, or a CI-style
"smoke every catalog entry" workflow) would accumulate one entry per
distinct repo ever seen — eventually OOMing the process. The fix is
``evict(hf_repo)``, called from ``unload`` in the supervisor's unload
hook (see app/models/routes_api.py::unload_model). Same hf_repo loaded
again later transparently re-fetches.

The cache is per-(hf_repo, trust_remote_code) on purpose: the same repo
with and without ``trust_remote_code`` produces different tokenizer
classes (custom code is loaded only with the flag set), so they must
not share an entry. Evict by repo only — both variants for the same
hf_repo go at the same time.
"""

import asyncio
import logging

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class TokenizerCache:
    """Lazy-loaded HF tokenizer cache keyed by (hf_repo, trust_remote_code). Used for accounting only."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, bool], object] = {}
        self._lock = asyncio.Lock()

    async def get(self, hf_repo: str, *, trust_remote_code: bool):
        key = (hf_repo, trust_remote_code)
        async with self._lock:
            if key not in self._cache:
                loop = asyncio.get_running_loop()
                self._cache[key] = await loop.run_in_executor(
                    None, lambda: AutoTokenizer.from_pretrained(hf_repo, trust_remote_code=trust_remote_code),
                )
            return self._cache[key]

    async def count(self, hf_repo: str, text: str, *, trust_remote_code: bool) -> int:
        if not text:
            return 0
        tok = await self.get(hf_repo, trust_remote_code=trust_remote_code)
        return len(tok.encode(text))

    async def evict(self, hf_repo: str) -> int:
        """Drop every cached tokenizer for ``hf_repo`` (both trust_remote_code
        variants). Returns the number of entries that were evicted. Idempotent
        — calling on an hf_repo never seen by the cache is a no-op that
        returns 0.

        Called from the model-unload path so the cache doesn't accumulate
        an entry per ever-loaded model over a long-lived warden's lifetime.
        See code-review finding #5 (S7, #124).
        """
        async with self._lock:
            to_drop = [k for k in self._cache if k[0] == hf_repo]
            for k in to_drop:
                self._cache.pop(k, None)
        if to_drop:
            logger.debug(
                "TokenizerCache.evict(%r) dropped %d entr%s",
                hf_repo, len(to_drop), "y" if len(to_drop) == 1 else "ies",
            )
        return len(to_drop)

    def size(self) -> int:
        """Number of cached tokenizer entries. For tests + observability."""
        return len(self._cache)
