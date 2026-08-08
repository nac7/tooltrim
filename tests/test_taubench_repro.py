"""Tests for the publication-grade additions: trials, CIs, and raw persistence.

These cover the reporting/statistics layer and the reproducibility artifact
without needing tau-bench or any API calls.
"""

import json
import threading
import time

import run_taubench
from eval.repro import Checkpoint, build_manifest, resolve_model_snapshot, save_results
from eval.taubench_adapter import CompressionStats
from run_taubench import build_report, run_condition


def _stats():
    return CompressionStats(tool_calls=6, compressed=6, raw_tokens=6000,
                            context_tokens=600)


def test_report_has_ci_column_and_wilson_interval():
    ids = list(range(4))
    rewards = {"tooltrim": {i: 1.0 for i in ids}}
    md = build_report(rewards, {"tooltrim": _stats()}, budget=128)
    assert "95% CI" in md
    # 4/4 successes -> Wilson lower bound is well below 100% (not the naive 100%).
    tt_line = [ln for ln in md.splitlines() if ln.startswith("| `tooltrim`")][0]
    assert "[" in tt_line and "%]" in tt_line
    assert "100-100%" not in tt_line  # Wilson widens at the extreme


def test_report_accepts_per_trial_reward_lists():
    ids = list(range(3))
    # 3 tasks x 2 trials; tooltrim passes 5/6 observations, truncate passes 1/6.
    rewards = {
        "tooltrim": {0: [1.0, 1.0], 1: [1.0, 0.0], 2: [1.0, 1.0]},
        "truncate-head": {0: [0.0, 0.0], 1: [1.0, 0.0], 2: [0.0, 0.0]},
    }
    stats = {m: _stats() for m in rewards}
    md = build_report(rewards, stats, budget=128)
    assert "× 2 trials" in md
    # tooltrim majority-success on all 3 tasks; truncate on none -> discordant.
    assert "`tooltrim` **★**" in md


def test_report_mixed_float_and_list_backward_compatible():
    # A bare float must still be accepted (old callers / single-trial runs).
    rewards = {"tooltrim": {0: 1.0, 1: 0.0}}
    md = build_report(rewards, {"tooltrim": _stats()}, budget=128)
    assert "50%" in md


def test_resolve_model_snapshot_pins_alias():
    assert resolve_model_snapshot("gpt-4o-mini") == "gpt-4o-mini-2024-07-18"
    assert resolve_model_snapshot("some-custom-model") == "some-custom-model"


def test_manifest_pins_models_and_versions():
    args = {"agent_model": "gpt-4o-mini", "user_model": "gpt-4o-mini", "tasks": 20}
    m = build_manifest(args, seed=7)
    assert m["seed"] == 7
    assert m["models"]["agent"] == "gpt-4o-mini-2024-07-18"
    assert "python" in m["versions"] and "tau_bench_commit" in m["versions"]
    assert m["args"]["tasks"] == 20


def test_run_condition_concurrent_is_deterministic_and_accounts(monkeypatch):
    """Concurrency must not change results or lose stats vs. sequential.

    Patches the single-solve unit so no tau-bench/API is needed; reward encodes
    (task, trial) so we can assert the per-task lists are assembled in trial
    order regardless of completion order, and that every solve's stats merged.
    """
    max_seen = {"n": 0}
    cur = {"n": 0}
    lock = threading.Lock()

    def fake_solve(env_name, *, method, budget, min_tokens, task_index,
                   agent_model, agent_provider, user_model, user_provider,
                   **kw):
        with lock:
            cur["n"] += 1
            max_seen["n"] = max(max_seen["n"], cur["n"])
        time.sleep(0.01)  # widen the window so overlap actually happens
        with lock:
            cur["n"] -= 1
        # reward encodes identity; one tool_call per solve for accounting.
        return float(task_index * 10 + 1), CompressionStats(
            tool_calls=1, compressed=1, raw_tokens=100, context_tokens=10)

    monkeypatch.setattr(run_taubench, "_solve_once", fake_solve)
    rewards, stats = run_condition(
        "retail", method="tooltrim", budget=128, task_indices=[0, 1, 2],
        agent_model="m", agent_provider="openai",
        user_model="m", user_provider="openai", min_tokens=128,
        trials=2, max_concurrency=4)

    # Every (task, trial) present, in deterministic trial order per task.
    assert rewards == {0: [1.0, 1.0], 1: [11.0, 11.0], 2: [21.0, 21.0]}
    # 3 tasks x 2 trials = 6 solves, each merged once.
    assert stats.tool_calls == 6 and stats.raw_tokens == 600
    # Concurrency actually happened (more than one solve in flight at some point).
    assert max_seen["n"] > 1


def test_run_condition_serial_when_concurrency_one(monkeypatch):
    def fake_solve(env_name, *, method, budget, min_tokens, task_index, **kw):
        return 1.0, CompressionStats(tool_calls=1, raw_tokens=100, context_tokens=10)

    monkeypatch.setattr(run_taubench, "_solve_once", fake_solve)
    rewards, stats = run_condition(
        "retail", method="none", budget=128, task_indices=[0, 1],
        agent_model="m", agent_provider="openai",
        user_model="m", user_provider="openai", min_tokens=128,
        trials=1, max_concurrency=1)
    assert rewards == {0: [1.0], 1: [1.0]}
    assert stats.tool_calls == 2


def test_checkpoint_roundtrip_and_budget_independent_key(tmp_path):
    ck = Checkpoint(tmp_path / "checkpoint.jsonl", budget_independent={"none"})
    ck.record("tooltrim", 128, 3, 0, 1.0, _stats())
    # `none` is budget-independent: recorded and looked up under a sentinel budget,
    # so a lookup at any budget resolves to the same solve.
    ck.record("none", 64, 5, 0, 0.0, _stats())
    assert ck.key("none", 64, 5, 0) == ck.key("none", 256, 5, 0)

    reloaded = Checkpoint(tmp_path / "checkpoint.jsonl",
                          budget_independent={"none"}).load()
    assert len(reloaded) == 2
    assert reloaded.has(reloaded.key("tooltrim", 128, 3, 0))
    assert reloaded.has(reloaded.key("none", 256, 5, 0))  # any budget resolves
    assert reloaded.get(reloaded.key("tooltrim", 128, 3, 0))["reward"] == 1.0


def test_checkpoint_tolerates_torn_final_line(tmp_path):
    p = tmp_path / "checkpoint.jsonl"
    ck = Checkpoint(p, budget_independent={"none"})
    ck.record("tooltrim", 128, 0, 0, 1.0, _stats())
    # Simulate a hard kill mid-write: append a partial JSON line.
    with p.open("a", encoding="utf-8") as f:
        f.write('{"method": "tooltrim", "budget": 128, "task": 1')
    reloaded = Checkpoint(p, budget_independent={"none"}).load()
    assert len(reloaded) == 1  # the good line survives; the torn one is dropped


def test_run_condition_resumes_skipping_completed_solves(tmp_path, monkeypatch):
    """A resumed run must skip checkpointed solves (no re-spend) yet return the
    same rewards/stats as a full run, reloading skipped contributions."""
    calls = {"n": 0}

    def fake_solve(env_name, *, method, budget, min_tokens, task_index, **kw):
        calls["n"] += 1
        return 1.0, CompressionStats(tool_calls=1, raw_tokens=100, context_tokens=10)

    monkeypatch.setattr(run_taubench, "_solve_once", fake_solve)

    ck = Checkpoint(tmp_path / "checkpoint.jsonl")
    # First run over 3 tasks: 3 solves, all checkpointed.
    r1, s1 = run_condition("retail", method="tooltrim", budget=128,
                           task_indices=[0, 1, 2], agent_model="m",
                           agent_provider="openai", user_model="m",
                           user_provider="openai", min_tokens=128,
                           trials=1, max_concurrency=1, ckpt=ck)
    assert calls["n"] == 3

    # Reload checkpoint in a fresh process-equivalent; extend to 4 tasks.
    ck2 = Checkpoint(tmp_path / "checkpoint.jsonl").load()
    r2, s2 = run_condition("retail", method="tooltrim", budget=128,
                           task_indices=[0, 1, 2, 3], agent_model="m",
                           agent_provider="openai", user_model="m",
                           user_provider="openai", min_tokens=128,
                           trials=1, max_concurrency=1, ckpt=ck2)
    # Only the one new task hit the API; the first three were reloaded.
    assert calls["n"] == 4
    assert r2 == {0: [1.0], 1: [1.0], 2: [1.0], 3: [1.0]}
    # Stats for the reloaded solves were merged back, not lost.
    assert s2.tool_calls == 4 and s2.context_tokens == 40


def test_save_results_roundtrip(tmp_path):
    rewards = {"tooltrim": {0: [1.0, 0.0], 1: [1.0, 1.0]}}
    stats = {"tooltrim": _stats()}
    manifest = build_manifest({"agent_model": "gpt-4o-mini"})
    p = save_results(tmp_path / "budget_128.json", manifest=manifest,
                     rewards_by_method=rewards, stats_by_method=stats, budget=128)
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["budget"] == 128
    # Raw per-trial rewards survive verbatim (re-derivable offline, no re-spend).
    assert payload["rewards"]["tooltrim"]["0"] == [1.0, 0.0]
    assert payload["stats"]["tooltrim"]["context_tokens"] == 600
    assert payload["manifest"]["models"]["agent"] == "gpt-4o-mini-2024-07-18"
