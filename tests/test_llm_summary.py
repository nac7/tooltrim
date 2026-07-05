"""Tests for the llm-summary baseline (the 'why not just summarize?' baseline).

These run offline: the baseline is gated on an API key, so we exercise
registration, availability honesty, passthrough, caching, and the fail-loud path
without spending any tokens.
"""

import json

from eval.baselines import LLMSummary, get_baseline


def test_registered_and_available_is_honest(monkeypatch):
    b = get_baseline("llm-summary")
    assert b.name == "llm-summary"
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert LLMSummary(provider="claude").available() is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert LLMSummary(provider="claude").available() is True


def test_passthrough_under_budget_needs_no_api():
    b = LLMSummary()
    assert b.compress("already small", "q", 1000) == "already small"


def test_fails_loud_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    b = LLMSummary(provider="claude")
    try:
        b.compress("x " * 5000, "q", 100)
        assert False, "expected RuntimeError without an API key"
    except RuntimeError as e:
        assert "llm-summary unavailable" in str(e)


def test_cache_hit_skips_the_model(tmp_path, monkeypatch):
    # A pre-seeded cache entry must be returned without ever building a client,
    # so a rerun never re-spends tokens.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)  # no key => would fail if called
    cache = tmp_path / "llm_summary.json"
    b = LLMSummary(provider="claude", cache_path=str(cache))
    text = "y " * 5000

    import hashlib

    h = hashlib.sha1()
    h.update(f"{b.provider}:{b.model_id}\x00100\x00q\x00{text}".encode("utf-8", "replace"))
    cache.write_text(json.dumps({h.hexdigest(): "CACHED SUMMARY"}), encoding="utf-8")

    assert b.compress(text, "q", 100) == "CACHED SUMMARY"
