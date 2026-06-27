from eval import default_cases, evaluate, get_model, matches
from eval.models import KeywordModel


def test_dataset_shape():
    cases = default_cases()
    assert len(cases) >= 15
    types = {c.content_type for c in cases}
    assert types == {"text", "html", "json", "logs", "tabular"}
    for c in cases:
        assert c.tool_output and c.question and c.gold


def test_judge_matches():
    assert matches("The API rate limit is 5000 requests per hour.",
                   "5000 requests per hour")
    assert matches("the capital is Mateldorf, a small city", "Mateldorf")
    assert not matches("I could not find that information", "Mateldorf")


def test_offline_full_accuracy_is_perfect():
    # With the full output, the needle is always present and findable.
    full, _ = evaluate(KeywordModel(), budgets=(400,))
    assert full.accuracy == 1.0


def test_compression_retains_accuracy_at_reasonable_budget():
    full, results = evaluate(KeywordModel(), budgets=(400,))
    r = results[0]
    assert r.saved_ratio > 0.8          # big token savings
    assert r.retention >= 0.9           # but answers still recoverable


def test_retention_reported_across_budgets():
    full, results = evaluate(KeywordModel(), budgets=(128, 256, 400))
    assert [r.budget for r in results] == [128, 256, 400]
    for r in results:
        assert 0.0 <= r.retention <= 1.0


def test_get_model_offline():
    assert get_model("offline").name == "offline"


def test_get_model_unknown_raises():
    import pytest

    with pytest.raises(ValueError):
        get_model("nope")


def test_cached_model_avoids_recompute(tmp_path):
    from eval.models import CachedModel

    calls = {"n": 0}

    class Counting:
        name = "counting"

        def answer(self, question, context):
            calls["n"] += 1
            return "answer"

    cache_path = str(tmp_path / "c.json")
    m = CachedModel(Counting(), cache_path)
    assert m.answer("q", "ctx") == "answer"
    assert m.answer("q", "ctx") == "answer"  # served from cache
    assert calls["n"] == 1

    # a fresh wrapper reads the persisted cache — still no recompute
    m2 = CachedModel(Counting(), cache_path)
    assert m2.answer("q", "ctx") == "answer"
    assert calls["n"] == 1


def test_export_markdown_and_csv():
    from eval import to_csv, to_markdown
    from eval.models import KeywordModel

    full, results = evaluate(KeywordModel(), budgets=(256,))
    md = to_markdown("offline", full, results)
    csv = to_csv("offline", full, results)
    assert "Faithfulness under compression" in md
    assert "| budget |" in md
    assert csv.splitlines()[0].startswith("model,budget,comp_tokens")
    assert "offline,256," in csv
