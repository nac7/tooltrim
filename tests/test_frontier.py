"""Tests for the frontier-model matrix driver (pure helpers + offline plumbing)."""

import pytest

import run_frontier as rf
from eval import default_baselines, evaluate_methods
from eval.models import KeywordModel


def test_parse_spec_provider_and_id():
    assert rf.parse_spec("claude:claude-haiku-4-5") == ("claude", "claude-haiku-4-5")
    assert rf.parse_spec("offline:") == ("offline", None)
    with pytest.raises(SystemExit):
        rf.parse_spec("bogus:x")


def test_preflight_flags_missing_keys(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    warns = rf.preflight([("claude", "x"), ("offline", None)])
    assert any("ANTHROPIC_API_KEY" in w for w in warns)
    # offline needs no key -> no warning for it
    assert not any("offline" in w for w in warns)


def test_significance_and_summary_render():
    model = KeywordModel()
    _, results = evaluate_methods(model, default_baselines(), budgets=(256,))
    row = rf.significance_row(results, "offline", 256)
    assert row.startswith("| offline | 256 |") and "|" in row


def test_write_summary_pareto(tmp_path):
    board = [{"model": "m1", "full_acc": 1.0, "full_tokens": 4000.0,
              "tt_acc": 1.0, "tt_tokens": 100.0, "retention": 1.0, "saved": 0.97,
              "tt_down": 0.9, "rag_down": 0.4}]
    out = tmp_path / "FRONTIER.md"
    rf.write_summary(str(out), 256, board, ["| m1 | 256 | +0.0pp | 1.000 | no |"])
    text = out.read_text(encoding="utf-8")
    assert "accuracy/token Pareto" in text and "m1" in text and "97.0%" in text
    assert "downstream" in text and "90%" in text and "40%" in text
