"""Trace specific tau-bench tasks with full trajectories, for error analysis.

The main sweep (`run_taubench.py`) records only reward + compression stats — it
discards ``SolveResult.messages``. To understand *why* compression breaks a task
we need the trajectory: what the agent saw (the compressed tool observations),
what it did (the actions), and, from the ``none`` run, what the full output
actually contained. This script re-runs a *small, pinpointed* set of tasks (the
ones where ``none`` succeeded but ``tooltrim`` failed) with message capture, and
dumps each trajectory to JSON plus a compact human-readable summary.

    python -m eval.trace_tasks --tasks 0,2,6,7,14,16,23,24 --budget 256 \
        --methods none,tooltrim --agent-model gpt-4o-mini --user-model gpt-4o-mini

Deliberately tiny (a dozen tasks × 2 methods) so it costs ~$1-2. Needs tau-bench
+ OPENAI_API_KEY, same as the sweep.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from run_taubench import _build_agent

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _tool_observations(messages: list) -> list:
    """Extract (tool_name, content) for each tool result the agent saw."""
    out = []
    for m in messages:
        if m.get("role") == "tool":
            out.append((m.get("name", "?"), m.get("content", "")))
    return out


def _actions(messages: list) -> list:
    """Extract the agent's tool calls / responses in order."""
    out = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function", {})
            out.append(f"CALL {fn.get('name')}({fn.get('arguments', '')})")
        if m.get("content") and not m.get("tool_calls"):
            out.append(f"SAY  {m['content'][:160]}")
    return out


def trace_one(env_name: str, *, method: str, budget: int, min_tokens: int,
              task_index: int, agent_model: str, agent_provider: str,
              user_model: str, user_provider: str) -> dict:
    from eval.taubench_adapter import make_compressed_env

    env = make_compressed_env(
        env_name, method=method, budget=budget, min_tokens=min_tokens,
        user_strategy="llm", user_model=user_model, user_provider=user_provider,
        task_split="test", task_index=task_index)
    agent = _build_agent(env, model=agent_model, provider=agent_provider)
    res = agent.solve(env=env, task_index=task_index)
    messages = list(getattr(res, "messages", []) or [])
    return {
        "method": method, "budget": budget, "task": task_index,
        "reward": float(getattr(res, "reward", 0.0)),
        "expand_calls": getattr(env.stats, "expand_calls", 0),
        "tool_observations": _tool_observations(messages),
        "actions": _actions(messages),
        "messages": messages,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", default="retail")
    ap.add_argument("--tasks", required=True, help="comma-separated task ids to trace")
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--min-tokens", type=int, default=512)
    ap.add_argument("--methods", default="none,tooltrim")
    ap.add_argument("--agent-model", default="gpt-4o-mini")
    ap.add_argument("--agent-provider", default="openai")
    ap.add_argument("--user-model", default="gpt-4o-mini")
    ap.add_argument("--user-provider", default="openai")
    ap.add_argument("--out", default="benchmarks/taubench_traces")
    args = ap.parse_args()

    from run_taubench import _configure_llm_resilience
    _configure_llm_resilience(6)

    tasks = [int(x) for x in args.tasks.split(",") if x.strip()]
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    for t in tasks:
        for method in args.methods.split(","):
            rec = trace_one(
                args.env, method=method, budget=args.budget,
                min_tokens=args.min_tokens, task_index=t,
                agent_model=args.agent_model, agent_provider=args.agent_provider,
                user_model=args.user_model, user_provider=args.user_provider)
            p = outdir / f"trace_{method}_b{args.budget}_t{t}.json"
            p.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
            n_obs = len(rec["tool_observations"])
            print(f"task {t:>2} [{method:<9}] reward={rec['reward']:.0f}  "
                  f"tool_obs={n_obs}  expand={rec['expand_calls']}  -> {p.name}",
                  flush=True)
    print(f"\nwrote {len(tasks) * len(args.methods.split(','))} traces to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
