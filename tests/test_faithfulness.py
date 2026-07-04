"""Tests for the structural-faithfulness metrics."""

from eval.dataset import Case
from eval.faithfulness import (
    downstream_applies,
    downstream_extractable,
    downstream_rate,
    is_parseable,
    parseable_rate,
)


# --- is_parseable: whole-output grammar, not "contains a valid sub-object" ----

def test_valid_whole_json_parses():
    assert is_parseable('{"a": 1, "note": "refund to 4417"}', "json")
    assert is_parseable('[{"id": 1}, {"id": 2}]', "json")


def test_json_fragment_fails_even_if_it_contains_a_valid_object():
    # What RAG chunk selection produces on single-line JSON: starts mid-object.
    frag = '166, "note": "the answer"}, {"id": 167, "status": "ok"}'
    assert is_parseable(frag, "json") is False


def test_json_tolerates_trailing_footer_line():
    assert is_parseable('{"a": 1}\n[+412 chars omitted, ref=ab12]', "json")


def test_tabular_needs_header_and_matching_width():
    good = "id,region,note\n1,us,refund\n2,eu,ok"
    assert is_parseable(good, "tabular")
    # header-less rows (a fragment) are not a usable table
    assert is_parseable("1,us,refund\n2,eu,ok\n3", "tabular") is False


def test_logs_require_intact_prefix_on_every_line():
    good = ("2026-06-27 10:42:11 ERROR nfs mount timed out\n"
            "2026-06-27 10:00:00 INFO heartbeat ok")
    assert is_parseable(good, "logs")
    assert is_parseable("nfs mount timed out on /shared/cache", "logs") is False


def test_prose_and_html_are_lenient():
    assert is_parseable("The API rate limit is 5000 requests per hour.", "text")
    assert is_parseable("The capital of Auren is Mateldorf.", "html")
    assert is_parseable("<nav><li>menu</li></nav> stuff", "html") is False


# --- downstream_extractable: recoverable from a real parse -------------------

def _json_case(note="refund issued to customer 4417"):
    return Case("json-x", "json",
                '{"results": [{"id": 5, "note": "%s"}]}' % note,
                "which refund?", note)


def test_downstream_json_extractable_from_valid_parse():
    c = _json_case()
    assert downstream_extractable(c.tool_output, c) is True


def test_downstream_json_fails_on_fragment():
    c = _json_case()
    frag = '5, "note": "refund issued to customer 4417"}'  # unparseable whole
    assert downstream_extractable(frag, c) is False


def test_downstream_applies_only_to_structured_types():
    assert downstream_applies("json") and downstream_applies("tabular")
    assert not downstream_applies("logs") and not downstream_applies("text")


def test_tooltrim_json_parseable_where_rag_fragments():
    # Deterministic separation on a real JSON case: tooltrim emits valid JSON the
    # agent can json.loads; rag-topk returns a mid-object fragment that cannot.
    from eval.baselines import get_baseline
    from eval.dataset import default_cases

    jcase = next(c for c in default_cases() if c.content_type == "json")
    tt = get_baseline("tooltrim").compress(jcase.tool_output, jcase.question, 256)
    rag = get_baseline("rag-topk").compress(jcase.tool_output, jcase.question, 256)
    assert is_parseable(tt, "json") is True
    assert is_parseable(rag, "json") is False


def test_aggregate_cases_present_and_multi_gold():
    from eval.dataset import default_cases

    agg = [c for c in default_cases() if c.category == "aggregate"]
    assert len(agg) >= 4
    assert all(len(c.all_of) >= 3 for c in agg)  # each needs several records


def test_rates_aggregate_and_handle_inapplicable():
    pairs = [('{"a":1}', "json"), ("plain prose", "text")]
    assert parseable_rate(pairs) == 1.0
    # downstream applies only to the json case here
    items = [(p, Case("t", ct, p, "q", "1")) for p, ct in pairs]
    assert downstream_rate(items) == 1.0
    # no applicable cases -> None
    assert downstream_rate([("prose", Case("t", "text", "x", "q", "g"))]) is None
