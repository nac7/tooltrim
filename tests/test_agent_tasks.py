"""Tests for the end-to-end agent-task benchmark (code-consumer regime)."""

from eval.agent_tasks import default_agent_tasks, task_success_rate
from eval.baselines import get_baseline


def test_suite_is_idempotent_and_shaped():
    a = default_agent_tasks()
    b = default_agent_tasks()
    assert [t.tool_output for t in a] == [t.tool_output for t in b]  # reseeded
    assert len(a) >= 8
    assert {t.content_type for t in a} == {"json", "tabular"}
    assert {t.task_kind for t in a} <= {"count", "sum", "list", "lookup"}


def test_full_context_solves_every_task():
    tasks = default_agent_tasks()
    full = get_baseline("full")
    for t in tasks:
        assert t.succeeds(full.compress(t.tool_output, t.query, 256)), t.id


def test_unparseable_fragment_fails_the_task():
    # A JSON task: an arbitrary mid-object fragment must not "succeed" — consume
    # returns None because json.loads would raise.
    t = next(t for t in default_agent_tasks() if t.content_type == "json")
    frag = '7002, "amount": 340.0, "note": "failed payment for order 7002"}, {"id":'
    assert t.consume(frag) is None
    assert not t.succeeds(frag)


def test_tooltrim_beats_rag_on_json_code_consumption():
    # The headline: on serialized JSON, tooltrim's valid output lets the agent's
    # parse+compute step succeed where rag-topk's fragments cannot.
    json_tasks = [t for t in default_agent_tasks() if t.content_type == "json"]
    tt = get_baseline("tooltrim")
    rag = get_baseline("rag-topk")
    tt_rate = task_success_rate((t, tt.compress(t.tool_output, t.query, 256)) for t in json_tasks)
    rag_rate = task_success_rate((t, rag.compress(t.tool_output, t.query, 256)) for t in json_tasks)
    assert tt_rate == 1.0
    assert rag_rate == 0.0


def test_aggregation_needs_all_records():
    # A count/list task only succeeds if every target record survives; verify the
    # gold reflects the full planted set and tooltrim recovers it at a real budget.
    t = next(t for t in default_agent_tasks()
             if t.task_kind == "list" and t.content_type == "json")
    tt = get_baseline("tooltrim")
    assert t.succeeds(tt.compress(t.tool_output, t.query, 256))
    assert len(t.gold) >= 4
