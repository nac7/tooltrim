"""Tests for the comparative baseline harness (tooltrim vs alternatives)."""

from eval import (
    DEFAULT_BASELINE_NAMES,
    default_baselines,
    evaluate_methods,
    get_baseline,
    mcnemar,
    methods_to_csv,
    methods_to_markdown,
)
from eval.baselines import _truncate_to_tokens
from eval.models import KeywordModel
from tooltrim import count_tokens


def test_default_baselines_present_and_available():
    names = [b.name for b in default_baselines()]
    assert names == list(DEFAULT_BASELINE_NAMES)
    assert "tooltrim" in names and "full" in names and "rag-topk" in names
    for b in default_baselines():
        assert b.available() is True


def test_optional_baselines_report_availability_honestly():
    # rag-embed / llmlingua-2 must only claim available() when their real
    # dependency is importable — never crash a run with a false positive.
    import importlib.util

    for name, dep in (("rag-embed", "sentence_transformers"),
                      ("llmlingua-2", "llmlingua")):
        b = get_baseline(name)
        installed = importlib.util.find_spec(dep) is not None
        # available() may run a deeper check than import (e.g. constructing a
        # model, which can fail on a CPU-only box or offline). The honest
        # invariant is one-directional: claiming available REQUIRES the
        # dependency be importable — never a false positive that crashes a run.
        if b.available():
            assert installed, f"{name} claims available() but {dep} is not importable"


def test_unavailable_baselines_are_skipped_not_run():
    # A method whose available() is False is dropped by evaluate_methods
    # rather than executed (which would raise).
    methods = [get_baseline("full"), get_baseline("rag-embed"), get_baseline("tooltrim")]
    _, results = evaluate_methods(KeywordModel(), methods, budgets=(256,))
    if not get_baseline("rag-embed").available():
        assert "rag-embed" not in results
    assert "tooltrim" in results and "full" in results


def test_get_baseline_unknown_raises():
    import pytest

    with pytest.raises(ValueError):
        get_baseline("does-not-exist")


def test_full_context_is_identity():
    full = get_baseline("full")
    text = "some long tool output " * 50
    assert full.compress(text, "query", 16) == text


def test_truncate_respects_budget():
    text = "\n".join(f"line {i} with several words here" for i in range(300))
    for budget in (32, 128, 256):
        head = get_baseline("truncate-head").compress(text, None, budget)
        tail = get_baseline("truncate-tail").compress(text, None, budget)
        assert count_tokens(head) <= budget
        assert count_tokens(tail) <= budget
        assert text.startswith(head)      # head slice is a real prefix
        assert text.endswith(tail)        # tail slice is a real suffix


def test_truncate_helper_edges():
    assert _truncate_to_tokens("", 100) == ""
    assert _truncate_to_tokens("short", 0) == ""
    # already within budget: returned untouched
    assert _truncate_to_tokens("tiny", 1000) == "tiny"


def test_rag_topk_keeps_query_relevant_region_within_budget():
    # Bury a needle in filler; a query overlapping the needle should retain it.
    filler = "\n".join(f"row {i} status ok latency low" for i in range(200))
    needle = "the deployment prefix is prod_9f2x for the eu-west region"
    text = filler + "\n" + needle + "\n" + filler
    out = get_baseline("rag-topk").compress(text, "what is the deployment prefix", 128)
    assert count_tokens(out) <= 128
    assert "prod_9f2x" in out


def test_rag_topk_falls_back_to_head_without_query():
    text = "\n".join(f"line {i} alpha beta gamma delta" for i in range(200))
    out = get_baseline("rag-topk").compress(text, "", 64)
    assert count_tokens(out) <= 64
    assert text.startswith(out)  # no query terms -> positional fallback


def test_evaluate_methods_grid_shape():
    methods = [get_baseline(n) for n in ("full", "truncate-head", "rag-topk", "tooltrim")]
    budgets = (128, 400)
    full, results = evaluate_methods(KeywordModel(), methods, budgets=budgets)
    assert set(results) == {"full", "truncate-head", "rag-topk", "tooltrim"}
    for name, rows in results.items():
        assert [r.budget for r in rows] == list(budgets)
        for r in rows:
            assert 0.0 <= r.accuracy <= 1.0
            assert 0.0 <= r.acc_lo <= r.accuracy <= r.acc_hi <= 1.0
            assert len(r.mask) == r.n
            assert sum(r.mask) == r.correct
    # "full" method should never save tokens and should match the full reference.
    for r in results["full"]:
        assert abs(r.accuracy - full.accuracy) < 1e-9
        assert r.saved_ratio <= 0.0 + 1e-9


def test_tooltrim_beats_naive_truncation_offline():
    # The whole thesis in one assertion: at a tight budget the query-aware
    # methods retain more of the needed facts than blind head truncation.
    methods = [get_baseline(n) for n in ("truncate-head", "rag-topk", "tooltrim")]
    _, results = evaluate_methods(KeywordModel(), methods, budgets=(128,))
    trunc = results["truncate-head"][0].accuracy
    tt = results["tooltrim"][0].accuracy
    assert tt >= trunc


def test_mcnemar_basic():
    # No discordant pairs -> p == 1.0
    _, _, p = mcnemar([True, True, False], [True, True, False])
    assert p == 1.0
    # Strongly discordant -> small p
    a = [True] * 10 + [True] * 10
    b = [False] * 10 + [False] * 10
    wins, losses, p = mcnemar(a, b)
    assert wins == 20 and losses == 0
    assert p < 0.001


def test_methods_markdown_and_csv_export():
    methods = [get_baseline(n) for n in ("full", "truncate-head", "tooltrim")]
    budgets = (256,)
    full, results = evaluate_methods(KeywordModel(), methods, budgets=budgets)
    md = methods_to_markdown("offline", full, results, budgets)
    assert "Comparative faithfulness" in md
    assert "Budget 256 tokens" in md
    assert "Significance" in md and "McNemar" in md
    assert "`tooltrim`" in md
    csv = methods_to_csv("offline", full, results)
    header = csv.splitlines()[0]
    assert header.startswith("model,method,budget,comp_tokens")
    assert "offline,tooltrim,256," in csv or "offline,tooltrim," in csv
