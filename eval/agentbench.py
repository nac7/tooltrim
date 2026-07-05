"""Multi-step agent-benchmark harness: tooltrim as a tool-output preprocessor.

Everything else in ``eval/`` scores compression on a *single* tool output. Real
agents run a *loop*: call a tool, read its (large) output, decide the next call,
repeat, then submit. The place tooltrim lives is the seam between tool execution
and the model's context — so the honest end-to-end question is:

    when tool outputs are compressed before re-entering context, does the agent
    still complete the multi-step task?

This module makes that measurable and is designed to plug into a *real* benchmark
(tau-bench, BFCL) rather than only a bespoke suite:

  * ``ToolMiddleware`` is the compression seam. ``CompressingMiddleware`` wraps any
    baseline ``Compressor`` (``compress(text, query, budget)``) from ``eval.baselines``.
  * ``Policy`` is the agent's decision function (observation -> Action). A real run
    supplies an LLM tool-calling policy; the offline mock supplies a deterministic
    one, so the framework is model-agnostic and the loop itself is what's tested.
  * ``run_episode`` drives the loop and applies the middleware to every tool output.
  * ``MockToolBenchmark`` is a runnable, API-free multi-step benchmark (search ->
    read a value from the large result -> submit) that exercises the loop and
    shows compression mattering *across steps*.
  * ``load_taubench`` / ``load_bfcl`` are the real-benchmark adapter seams: they
    import the upstream package if present and otherwise report unavailable, so a
    real run is a matter of installing the benchmark + providing an LLM policy.

The design keeps the contribution — compression in the agent loop — decoupled
from which benchmark or model supplies the tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple, Union

from tooltrim.tokens import count_tokens

# --- actions & tools ----------------------------------------------------------


@dataclass
class ToolCall:
    name: str
    args: dict
    # The relevance query used to compress this call's output (the agent's intent).
    query: Optional[str] = None


@dataclass
class Finish:
    answer: object


Action = Union[ToolCall, Finish]


@dataclass
class Tool:
    name: str
    run: Callable[[dict], str]   # args -> raw (possibly huge) tool output
    description: str = ""


# --- compression seam ---------------------------------------------------------


class ToolMiddleware(Protocol):
    name: str

    def process(self, tool_name: str, query: Optional[str], output: str) -> str: ...


class NoCompression:
    """Baseline: full tool output re-enters context unchanged."""

    name = "none"

    def process(self, tool_name: str, query: Optional[str], output: str) -> str:
        return output


class CompressingMiddleware:
    """Compress every tool output with a baseline ``Compressor`` before it re-enters
    context. This is exactly where tooltrim sits in a production agent."""

    def __init__(self, compressor, budget: int):
        self.compressor = compressor
        self.budget = budget
        self.name = getattr(compressor, "name", compressor.__class__.__name__)

    def process(self, tool_name: str, query: Optional[str], output: str) -> str:
        return self.compressor.compress(output, query, self.budget)


# --- policy (the agent's brain) -----------------------------------------------


@dataclass
class Observation:
    task_prompt: str
    # (tool_name, compressed_output) for each step so far — what the model can see.
    history: List[Tuple[str, str]] = field(default_factory=list)


class Policy(Protocol):
    def act(self, obs: Observation, tools: Sequence[Tool]) -> Action: ...


# --- episode loop -------------------------------------------------------------


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    context_tokens: int          # total tokens the model had to read (post-compression)
    raw_tokens: int              # total tokens without compression (the counterfactual)
    finished: bool               # did the agent submit (vs hit the step cap)?


def run_episode(task: "BenchTask", policy: Policy, middleware: ToolMiddleware,
                *, max_steps: int = 8) -> EpisodeResult:
    tools = {t.name: t for t in task.tools}
    obs = Observation(task_prompt=task.prompt)
    context_tokens = 0
    raw_tokens = 0
    for step in range(1, max_steps + 1):
        action = policy.act(obs, task.tools)
        if isinstance(action, Finish):
            return EpisodeResult(
                success=bool(task.check(action.answer)), steps=step,
                context_tokens=context_tokens, raw_tokens=raw_tokens, finished=True)
        tool = tools.get(action.name)
        if tool is None:
            # Agent called an unknown tool — treat as a failed step, keep going.
            obs.history.append((action.name, f"[error: no tool named {action.name}]"))
            continue
        raw = tool.run(action.args)
        compressed = middleware.process(action.name, action.query, raw)
        raw_tokens += count_tokens(raw)
        context_tokens += count_tokens(compressed)
        obs.history.append((action.name, compressed))
    # Ran out of steps without finishing.
    return EpisodeResult(success=False, steps=max_steps,
                         context_tokens=context_tokens, raw_tokens=raw_tokens,
                         finished=False)


# --- benchmark protocol -------------------------------------------------------


@dataclass
class BenchTask:
    id: str
    prompt: str
    tools: List[Tool]
    check: Callable[[object], bool]   # final answer -> correct?
    # A deterministic reference policy so the suite is runnable/testable without an
    # LLM. A real benchmark run replaces this with an LLM tool-calling policy.
    reference_policy: "Policy"


def success_rate(results: Sequence[EpisodeResult]) -> float:
    results = list(results)
    return sum(r.success for r in results) / len(results) if results else 0.0
