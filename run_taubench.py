#!/usr/bin/env python3
"""Run tau-bench with tool observations compressed in-loop, across methods.

For each compression condition (none / truncate-head / rag-topk / tooltrim / ...),
this wraps tau-bench's env with ``CompressedEnv``, drives tau-bench's own agent
over a set of tasks, and records the task reward plus compression accounting. It
then writes a table with paired McNemar significance vs. tooltrim.

    # pilot: 20 retail tasks, tooltrim vs the baselines, at a 512-token budget
    python run_taubench.py --env retail --tasks 20 --budget 512 \
        --agent-model gpt-4o-mini --agent-provider openai \
        --user-model gpt-4o-mini --user-provider openai \
        --methods none,truncate-head,rag-topk,tooltrim

Needs tau-bench installed and API keys for BOTH the agent and the user simulator.
The reporting is factored out (``build_report``) and unit-tested without
tau-bench; only the run loop requires it.
"""
from __future__ import annotations

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from eval.metrics import fmt_ci, mcnemar, wilson_ci

OUT = Path("benchmarks/TAUBENCH.md")
SUCCESS_EPS = 1e-9


def _build_agent(env, *, model: str, provider: str):
    """Construct tau-bench's standard tool-calling agent for ``env``.

    Isolated here because the agent constructor differs across tau-bench
    versions. If your installed version differs, this is the ONE function to
    adjust — everything else is version-independent.
    """
    from tau_bench.agents.tool_calling_agent import ToolCallingAgent  # type: ignore

    return ToolCallingAgent(
        tools_info=env.tools_info,
        wiki=env.wiki,
        model=model,
        provider=provider,
    )


def _solve_once(env_name: str, *, method: str, budget: int, min_tokens: int,
                task_index: int, agent_model: str, agent_provider: str,
                user_model: str, user_provider: str, dialogue_turns: int):
    """Solve a single (task, trial): fresh isolated env + agent. Thread-safe.

    Returns ``(reward, stats)`` for just this solve; callers merge into totals.
    Each solve builds its own env, so nothing is shared across threads except
    read-only args — safe to run concurrently.
    """
    from eval.taubench_adapter import make_compressed_env

    env = make_compressed_env(
        env_name, method=method, budget=budget, min_tokens=min_tokens,
        dialogue_turns=dialogue_turns,
        user_strategy="llm", user_model=user_model, user_provider=user_provider,
        task_split="test",
        # Pin the task at construction. tau-bench's Env.__init__ otherwise picks
        # random.randint(0, len(tasks)) — inclusive, so it can index one past the
        # end and raise IndexError (~1/N per env). Passing task_index both avoids
        # that off-by-one and makes the run deterministic/reproducible.
        task_index=task_index,
    )
    agent = _build_agent(env, model=agent_model, provider=agent_provider)
    res = agent.solve(env=env, task_index=task_index)
    return float(getattr(res, "reward", 0.0)), env.stats


def run_condition(env_name: str, *, method: str, budget: int, task_indices: List[int],
                  agent_model: str, agent_provider: str,
                  user_model: str, user_provider: str, min_tokens: int,
                  dialogue_turns: int = 6,
                  trials: int = 1, max_concurrency: int = 1, ckpt=None):
    """Run one compression condition over ``task_indices``; return (rewards, stats).

    Matches tau-bench's harness: a fresh isolated env per (task, trial) so no
    state leaks, with compression stats accumulated across them. Each task is
    solved ``trials`` times to average over the stochastic user simulator; the
    reward list per task is returned so downstream stats see every observation.

    The work is API-bound, so up to ``max_concurrency`` (task, trial) solves run
    on a thread pool. Stats merging and reward collection are guarded by a lock;
    per-task reward order is deterministic (sorted by trial) regardless of the
    order solves complete, so runs are reproducible given fixed inputs.

    If ``ckpt`` (a ``Checkpoint``) is given, solves already recorded there are
    skipped and reloaded, and each new solve is appended as it completes — so an
    interrupted run resumes from exactly where it stopped instead of re-spending.
    """
    from eval.taubench_adapter import CompressionStats

    # (task_index, trial) -> reward, filled from the checkpoint and as futures complete.
    results: Dict[Tuple[int, int], float] = {}
    total = CompressionStats()
    lock = threading.Lock()
    all_jobs = [(i, t) for i in task_indices for t in range(trials)]

    # Reload any already-checkpointed solves; only the rest need the API.
    pending: List[Tuple[int, int]] = []
    for (i, t) in all_jobs:
        if ckpt is not None and ckpt.has(ckpt.key(method, budget, i, t)):
            rec = ckpt.get(ckpt.key(method, budget, i, t))
            results[(i, t)] = float(rec["reward"])
            total.merge(CompressionStats(**rec["stats"]))
        else:
            pending.append((i, t))
    resumed = len(all_jobs) - len(pending)
    if resumed:
        print(f"  [{method}] resumed {resumed}/{len(all_jobs)} solves from checkpoint",
              flush=True)
    done = resumed

    def work(job: Tuple[int, int]):
        i, _t = job
        return job, _solve_once(
            env_name, method=method, budget=budget, min_tokens=min_tokens,
            task_index=i, agent_model=agent_model, agent_provider=agent_provider,
            user_model=user_model, user_provider=user_provider,
            dialogue_turns=dialogue_turns)

    if pending:
        workers = max(1, min(max_concurrency, len(pending)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(work, job) for job in pending]
            for fut in as_completed(futs):
                (i, t), (r, stats) = fut.result()
                with lock:
                    results[(i, t)] = r
                    total.merge(stats)
                    if ckpt is not None:
                        ckpt.record(method, budget, i, t, r, stats)
                    done += 1
                    suffix = f" (trial {t + 1}/{trials})" if trials > 1 else ""
                    print(f"  [{method}] ({done}/{len(all_jobs)}) task {i}: "
                          f"reward={r:.2f}{suffix}", flush=True)

    # Assemble per-task reward lists in deterministic (task, trial) order.
    rewards: Dict[int, List[float]] = {
        i: [results[(i, t)] for t in range(trials)] for i in task_indices}
    return rewards, total


def _trial_rewards(v: object) -> List[float]:
    """Normalize a task's reward to a per-trial list (accepts a bare float)."""
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v]
    return [float(v)]


def build_report(rewards_by_method: Dict[str, Dict[int, object]],
                 stats_by_method: Dict[str, object], budget: int,
                 *, success_threshold: float = 1.0) -> str:
    """Pure: turn per-method rewards + stats into the TAUBENCH.md markdown.

    Success is ``reward >= success_threshold`` (tau-bench rewards are typically
    1.0 on full success). Each task's value may be a single float or a list of
    per-trial floats; the reported rate + Wilson 95% CI are computed over *all*
    (task, trial) observations, which is the honest denominator when the user
    simulator is stochastic. McNemar compares each method to tooltrim on the
    shared task ids (per-task success = majority over that task's trials).
    Testable without tau-bench.
    """
    methods = list(rewards_by_method)
    task_ids = sorted(set().union(*[set(r) for r in rewards_by_method.values()])) \
        if rewards_by_method else []
    thr = success_threshold - SUCCESS_EPS

    def obs_successes(method: str) -> List[bool]:
        """Flat pass/fail over every (task, trial) for a method."""
        out: List[bool] = []
        for i in task_ids:
            if i in rewards_by_method[method]:
                out += [r >= thr for r in _trial_rewards(rewards_by_method[method][i])]
        return out

    def succ(method: str, i: int) -> bool:
        """Per-task success: majority of trials pass (ties count as success)."""
        trials = _trial_rewards(rewards_by_method[method].get(i, 0.0))
        passes = sum(r >= thr for r in trials)
        return passes * 2 >= len(trials)

    trials_per = max(
        (len(_trial_rewards(v)) for r in rewards_by_method.values() for v in r.values()),
        default=1)
    lines = [
        "# tau-bench: task success with tool outputs compressed in-loop",
        "",
        f"Compression applied to every tool observation before it re-enters the "
        f"agent's context (budget {budget} tokens), tau-bench's reward and user "
        f"simulator unchanged. n={len(task_ids)} tasks"
        + (f" × {trials_per} trials" if trials_per > 1 else "")
        + ". Generated by `run_taubench.py`.",
        "",
        "| method | task success | 95% CI | avg ctx tokens | tokens saved | p vs tooltrim |",
        "|:--|--:|:--:|--:|--:|:--:|",
    ]
    tt = "tooltrim"
    for m in methods:
        succ_obs = obs_successes(m)
        n_obs = len(succ_obs) or 1
        k = sum(succ_obs)
        rate = k / n_obs
        lo, hi = wilson_ci(k, len(succ_obs))
        st = stats_by_method.get(m)
        ctx = getattr(st, "context_tokens", 0)
        calls = max(1, getattr(st, "tool_calls", 0))
        saved = getattr(st, "saved_ratio", 0.0)
        if m == tt or tt not in rewards_by_method:
            p_cell = "—"
        else:
            a = [succ(tt, i) for i in task_ids]
            b = [succ(m, i) for i in task_ids]
            _, _, p = mcnemar(a, b)
            p_cell = f"{p:.3f}" + ("*" if p < 0.05 else "")
        star = " **★**" if m == tt else ""
        lines.append(
            f"| `{m}`{star} | {rate:.0%} | {fmt_ci(lo, hi)} | {ctx / calls:.0f} "
            f"| {saved:.0%} | {p_cell} |")
    lines += [
        "",
        "\\* significant (paired McNemar, p<0.05). CI is the Wilson 95% interval "
        "over all (task, trial) observations. `none` is the uncompressed ceiling "
        "with identical accounting. A method that shreds a tool output the agent "
        "needs for a later turn shows up here as *lower* task success.",
    ]
    return "\n".join(lines) + "\n"


def _configure_llm_resilience(num_retries: int) -> None:
    """Make every litellm completion (agent AND user sim) retry on 429/5xx.

    litellm is the shared client under both tau-bench's agent and its LLM user
    simulator, so setting the module-level retry policy here covers calls we
    don't own. It uses exponential backoff and honors the ``Retry-After`` header,
    which turns a transient rate-limit (the run-killer under concurrency) into a
    short wait instead of a fatal exception.
    """
    try:
        import litellm

        litellm.num_retries = num_retries
        litellm.drop_params = True
        litellm.suppress_debug_info = True  # silence the per-call "Give Feedback" banner

        # litellm's built-in retry honors OpenAI's sub-second Retry-After on a
        # TPM 429 ("try again in 295ms"), so all num_retries attempts land inside
        # the same saturated minute and the run then dies with RateLimitError.
        # Wrap completion with an OUTER retry that enforces a real backoff
        # (>= _RL_MIN_SLEEP s, growing per attempt) so a retry actually waits for
        # the tokens-per-minute bucket to refill. tau-bench does
        # `from litellm import completion`, binding the function into each agent/
        # user module at import time, so we must rebind the wrapper in every
        # module that already imported it (and reassign litellm.completion so any
        # later import gets the wrapper too).
        import sys as _sys, time as _time, functools as _functools
        from litellm.exceptions import RateLimitError as _RateLimitError

        _orig_completion = litellm.completion
        _RL_MIN_SLEEP = 20.0

        @_functools.wraps(_orig_completion)
        def _resilient_completion(*a, **kw):
            attempts = max(1, num_retries)
            for i in range(attempts):
                try:
                    return _orig_completion(*a, **kw)
                except _RateLimitError:
                    if i == attempts - 1:
                        raise
                    wait = _RL_MIN_SLEEP * (i + 1)  # 20s, 40s, 60s, ...
                    print(f"  [rate-limit] TPM 429; waiting {wait:.0f}s for bucket "
                          f"refill (outer retry {i + 1}/{attempts - 1})", flush=True)
                    _time.sleep(wait)

        litellm.completion = _resilient_completion
        patched = 0
        for _mod in list(_sys.modules.values()):
            if getattr(_mod, "completion", None) is _orig_completion:
                _mod.completion = _resilient_completion
                patched += 1
        print(f"  llm resilience: wrapped completion in {patched} module(s); "
              f"outer 429 backoff {_RL_MIN_SLEEP:.0f}s x{num_retries}")
    except Exception as e:  # pragma: no cover - litellm always present with tau-bench
        print(f"  (warning: could not configure litellm retries: {e})")


# Methods whose result does not depend on the budget: they never compress, so a
# single run is valid for every budget. Running them once and reusing avoids
# re-spending on an identical (and, for `none`, the most expensive) condition.
BUDGET_INDEPENDENT = frozenset({"none"})


def _run_one_budget(args, budget: int, task_indices: List[int],
                    cache: Dict[str, Tuple[Dict[int, object], object]], ckpt=None):
    """Run every method at a single budget; return (rewards, stats) by method.

    Methods present in ``cache`` (budget-independent, e.g. ``none``) are reused
    from the single precomputed run instead of being solved again. ``ckpt`` is
    forwarded so per-solve resume applies to the per-budget conditions too.
    """
    rewards_by_method: Dict[str, Dict[int, object]] = {}
    stats_by_method: Dict[str, object] = {}
    for method in args.methods.split(","):
        if method in cache:
            print(f"== condition: {method} @ budget {budget} == "
                  f"(reused — budget-independent)")
            rewards, stats = cache[method]
        else:
            print(f"== condition: {method} @ budget {budget} ==")
            rewards, stats = run_condition(
                args.env, method=method, budget=budget, task_indices=task_indices,
                agent_model=args.agent_model, agent_provider=args.agent_provider,
                user_model=args.user_model, user_provider=args.user_provider,
                min_tokens=args.min_tokens, dialogue_turns=args.dialogue_turns,
                trials=args.trials,
                max_concurrency=args.max_concurrency, ckpt=ckpt)
        rewards_by_method[method] = rewards
        stats_by_method[method] = stats
    return rewards_by_method, stats_by_method


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="retail")
    ap.add_argument("--tasks", type=int, default=20, help="number of tasks (0..N-1)")
    ap.add_argument("--task-indices", default="",
                    help="explicit comma-separated task ids to run (overrides --tasks); "
                         "use to re-run only a subset, e.g. the lost tasks in an ablation")
    ap.add_argument("--budget", type=int, default=512, help="single budget (if --budgets unset)")
    ap.add_argument("--budgets", default="", help="comma-separated budget sweep, e.g. 64,128,256")
    ap.add_argument("--trials", type=int, default=1, help="solves per task (avg over user sim)")
    ap.add_argument("--max-concurrency", type=int, default=1,
                    help="max concurrent (task, trial) solves; API-bound, so >1 is a big speedup")
    ap.add_argument("--min-tokens", type=int, default=512)
    ap.add_argument("--dialogue-turns", type=int, default=6,
                    help="recent dialogue turns feeding the relevance query; "
                         "0 = instruction-only (the pre-2026-07 ablation arm)")
    ap.add_argument("--methods", default="none,truncate-head,rag-topk,tooltrim")
    ap.add_argument("--agent-model", default="gpt-4o-mini")
    ap.add_argument("--agent-provider", default="openai")
    ap.add_argument("--user-model", default="gpt-4o-mini")
    ap.add_argument("--user-provider", default="openai")
    ap.add_argument("--success-threshold", type=float, default=1.0)
    ap.add_argument("--num-retries", type=int, default=6,
                    help="litellm retries (exp. backoff, honors Retry-After) on 429/5xx")
    ap.add_argument("--results-dir", default="benchmarks/taubench_raw",
                    help="where per-budget raw JSON (manifest+rewards) is written")
    ap.add_argument("--resume", action="store_true",
                    help="resume from checkpoint.jsonl in --results-dir: skip "
                         "already-completed solves, re-spend only on the rest")
    args = ap.parse_args()

    _configure_llm_resilience(args.num_retries)
    from eval.repro import Checkpoint, build_manifest, save_results

    budgets = [int(b) for b in args.budgets.split(",") if b.strip()] or [args.budget]
    task_indices = ([int(x) for x in args.task_indices.split(",") if x.strip()]
                    or list(range(args.tasks)))
    manifest = build_manifest(vars(args))
    OUT.parent.mkdir(exist_ok=True)

    # Per-solve checkpoint (append-only): every completed solve is flushed here so
    # an interrupted run can be resumed with --resume, skipping done solves and
    # re-spending only on what's left. The file lives beside the raw results.
    ckpt = Checkpoint(Path(args.results_dir) / "checkpoint.jsonl",
                      budget_independent=BUDGET_INDEPENDENT)
    if args.resume:
        ckpt.load()
        n = len(ckpt)
        print(f"resume: loaded {n} completed solve(s) from {ckpt.path}"
              if n else f"resume: no prior checkpoint at {ckpt.path} (starting fresh)")
    elif ckpt.path.exists():
        raise SystemExit(
            f"refusing to start: {ckpt.path} already exists — a prior run's "
            f"checkpoint. Pass --resume to continue it, or use a fresh "
            f"--results-dir (or delete the file) to start over.")

    # Precompute any budget-independent condition (e.g. `none`) a single time and
    # reuse it across every budget, instead of re-solving an identical, expensive
    # condition once per budget.
    cache: Dict[str, Tuple[Dict[int, object], object]] = {}
    for method in args.methods.split(","):
        if method in BUDGET_INDEPENDENT and method not in cache:
            print(f"== condition: {method} (budget-independent — solving once, "
                  f"reused across {len(budgets)} budget(s)) ==")
            rewards, stats = run_condition(
                args.env, method=method, budget=budgets[0], task_indices=task_indices,
                agent_model=args.agent_model, agent_provider=args.agent_provider,
                user_model=args.user_model, user_provider=args.user_provider,
                min_tokens=args.min_tokens, dialogue_turns=args.dialogue_turns,
                trials=args.trials,
                max_concurrency=args.max_concurrency, ckpt=ckpt)
            cache[method] = (rewards, stats)

    for budget in budgets:
        rewards_by_method, stats_by_method = _run_one_budget(
            args, budget, task_indices, cache, ckpt=ckpt)

        # Persist raw results FIRST so a later report tweak never needs a re-spend.
        raw_path = save_results(
            Path(args.results_dir) / f"budget_{budget}.json",
            manifest=manifest, rewards_by_method=rewards_by_method,
            stats_by_method=stats_by_method, budget=budget)
        print(f"saved raw results -> {raw_path}")

        report = build_report(rewards_by_method, stats_by_method, budget,
                              success_threshold=args.success_threshold)
        out = OUT if len(budgets) == 1 else OUT.with_name(f"TAUBENCH_b{budget}.md")
        out.write_text(report, encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
