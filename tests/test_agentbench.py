"""Tests for the multi-step agent-benchmark harness (compression in the loop)."""

from eval.agentbench import (
    CompressingMiddleware, Finish, NoCompression, Observation, ToolCall,
    run_episode, success_rate,
)
from eval.agentbench_suites import MockToolBenchmark, load_bfcl, load_taubench
from eval.baselines import get_baseline


def test_mock_tasks_reproducible_and_shaped():
    a = MockToolBenchmark().tasks()
    b = MockToolBenchmark().tasks()
    assert len(a) >= 8
    # Reseeded per task => bit-for-bit reproducible tool outputs.
    assert [t.tools[0].run({}) for t in a] == [t.tools[0].run({}) for t in b]
    for t in a:
        assert {tool.name for tool in t.tools} == {"get_order", "list_by_region"}


def test_reference_policy_is_two_hop():
    # The agent must call get_order, then list_by_region, then finish — proving the
    # task genuinely depends on two compressed tool outputs, not one.
    task = MockToolBenchmark().tasks()[0]
    pol = task.reference_policy
    obs = Observation(task_prompt=task.prompt)
    a1 = pol.act(obs, task.tools)
    assert isinstance(a1, ToolCall) and a1.name == "get_order"
    obs.history.append((a1.name, task.tools[0].run({})))     # full output
    a2 = pol.act(obs, task.tools)
    assert isinstance(a2, ToolCall) and a2.name == "list_by_region"
    obs.history.append((a2.name, task.tools[1].run({})))
    a3 = pol.act(obs, task.tools)
    assert isinstance(a3, Finish)


def test_full_context_solves_every_task():
    tasks = MockToolBenchmark().tasks()
    results = [run_episode(t, t.reference_policy, NoCompression()) for t in tasks]
    assert success_rate(results) == 1.0
    assert all(r.finished for r in results)


def test_tooltrim_preserves_success_truncation_breaks_it():
    tasks = MockToolBenchmark().tasks()
    tt = CompressingMiddleware(get_baseline("tooltrim"), 256)
    trunc = CompressingMiddleware(get_baseline("truncate-head"), 256)
    tt_results = [run_episode(t, t.reference_policy, tt) for t in tasks]
    tr_results = [run_episode(t, t.reference_policy, trunc) for t in tasks]
    assert success_rate(tt_results) == 1.0
    assert success_rate(tr_results) == 0.0
    # And tooltrim did so on a fraction of the tokens vs full context.
    assert all(r.context_tokens < r.raw_tokens for r in tt_results)


def test_real_adapters_report_unavailable_cleanly():
    for loader in (load_taubench, load_bfcl):
        adapter = loader()
        assert adapter.available() is False
        try:
            adapter.tasks()
            assert False, "unavailable adapter must raise from tasks()"
        except RuntimeError as e:
            assert "unavailable" in str(e)
