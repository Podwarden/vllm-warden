from unittest.mock import MagicMock, patch


async def test_tokenizer_cache_returns_cached_instance():
    from app.proxy.tokenizers import TokenizerCache
    fake_tok = MagicMock()
    fake_tok.encode = lambda s: list(s.encode())
    with patch(
        "app.proxy.tokenizers.AutoTokenizer.from_pretrained",
        return_value=fake_tok,
    ) as load:
        cache = TokenizerCache()
        a = await cache.get("Qwen/Qwen3.5-9B", trust_remote_code=False)
        b = await cache.get("Qwen/Qwen3.5-9B", trust_remote_code=False)
    assert a is b
    assert load.call_count == 1
    load.assert_called_once_with("Qwen/Qwen3.5-9B", trust_remote_code=False)


async def test_count_tokens_uses_repo_tokenizer():
    from app.proxy.tokenizers import TokenizerCache
    fake_tok = MagicMock()
    fake_tok.encode = lambda s: list(s)
    with patch("app.proxy.tokenizers.AutoTokenizer.from_pretrained", return_value=fake_tok):
        cache = TokenizerCache()
        n = await cache.count("Qwen/Qwen3.5-9B", "hello", trust_remote_code=False)
    assert n == 5


async def test_count_handles_empty_string():
    from app.proxy.tokenizers import TokenizerCache
    fake_tok = MagicMock()
    fake_tok.encode = lambda s: []
    with patch("app.proxy.tokenizers.AutoTokenizer.from_pretrained", return_value=fake_tok):
        cache = TokenizerCache()
        assert await cache.count("Qwen/x", "", trust_remote_code=False) == 0


async def test_cache_key_separation_by_trust_flag():
    from app.proxy.tokenizers import TokenizerCache
    fake_tok_false = MagicMock(name="tok_false")
    fake_tok_true = MagicMock(name="tok_true")
    call_count = 0

    def _side_effect(repo, trust_remote_code):
        nonlocal call_count
        call_count += 1
        return fake_tok_false if not trust_remote_code else fake_tok_true

    with patch(
        "app.proxy.tokenizers.AutoTokenizer.from_pretrained",
        side_effect=_side_effect,
    ) as load:
        cache = TokenizerCache()
        a = await cache.get("some/model", trust_remote_code=False)
        b = await cache.get("some/model", trust_remote_code=True)

    assert load.call_count == 2
    assert a is not b
