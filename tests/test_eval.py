from eval import default_cases, evaluate, get_model, matches
from eval.models import KeywordModel


def test_dataset_shape():
    cases = default_cases()
    assert len(cases) >= 60
    types = {c.content_type for c in cases}
    assert types == {"text", "html", "json", "logs", "tabular"}
    cats = {c.category for c in cases}
    assert {"single", "multi", "distractor"} <= cats
    # ids unique
    assert len({c.id for c in cases}) == len(cases)
    for c in cases:
        assert c.tool_output and c.question and c.gold


def test_passed_scoring():
    from eval.judge import passed

    # multi-fact: all_of must also be present
    assert passed("rate is 5000 per hour, timeout 30 seconds",
                  "5000", all_of=("30 seconds",))
    assert not passed("rate is 5000 per hour", "5000", all_of=("30 seconds",))
    # distractor: must_not must be absent
    assert passed("the prefix is prod_9f2", "prod_9f2", must_not=("sand_1a0",))
    assert not passed("prod_9f2 replaces sand_1a0", "prod_9f2",
                      must_not=("sand_1a0",))


def test_wilson_ci():
    from eval import wilson_ci

    assert wilson_ci(0, 0) == (0.0, 0.0)
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0 and 0.0 < hi < 0.35
    lo, hi = wilson_ci(10, 10)
    assert hi == 1.0 and 0.6 < lo < 1.0
    lo, hi = wilson_ci(8, 16)
    assert lo < 0.5 < hi and 0.24 < lo < 0.30 and 0.70 < hi < 0.76


def test_results_carry_confidence_intervals():
    full, results = evaluate(KeywordModel(), budgets=(256,))
    assert 0.0 <= full.acc_lo <= full.accuracy <= full.acc_hi <= 1.0
    r = results[0]
    assert 0.0 <= r.acc_lo <= r.accuracy <= r.acc_hi <= 1.0


def test_judge_matches():
    assert matches("The API rate limit is 5000 requests per hour.",
                   "5000 requests per hour")
    assert matches("the capital is Mateldorf, a small city", "Mateldorf")
    assert not matches("I could not find that information", "Mateldorf")


def test_offline_full_accuracy_is_high():
    # The offline retriever is a strong control: with the full output it finds
    # the needed facts in nearly every case (multi-fact/distractor are harder).
    full, _ = evaluate(KeywordModel(), budgets=(400,))
    assert full.accuracy >= 0.85


def test_category_breakdown_offline():
    # offline retriever: perfect on retrieval-solvable (single+multi),
    # ~0 on distractors (those need reasoning) — by design.
    from eval import default_cases
    from eval.harness import category_breakdown, evaluate_detailed

    _, _, records = evaluate_detailed(KeywordModel(), budgets=(400,))
    bd = category_breakdown(records, (400,))
    assert bd["single"]["full_accuracy"] == 1.0
    assert bd["multi"]["full_accuracy"] == 1.0
    assert bd["distractor"]["full_accuracy"] == 0.0


def test_compression_retains_accuracy_at_reasonable_budget():
    full, results = evaluate(KeywordModel(), budgets=(400,))
    r = results[0]
    assert r.saved_ratio > 0.8          # big token savings
    assert r.retention >= 0.9           # but answers still recoverable


def test_retention_reported_across_budgets():
    full, results = evaluate(KeywordModel(), budgets=(128, 256, 400))
    assert [r.budget for r in results] == [128, 256, 400]
    for r in results:
        assert r.retention >= 0.0       # may exceed 1.0 (compression can help)


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
    assert "95% CI" in md
    header = csv.splitlines()[0]
    assert header.startswith("model,budget,comp_tokens")
    assert "acc_ci_lo" in header and "acc_ci_hi" in header
    assert "offline,256," in csv
