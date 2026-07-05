"""tau-bench adapter: compress tool observations inside tau-bench's own loop.

Design choice (see paper Section on multi-step): we do *not* port tau-bench tasks
into our ``BenchTask`` framework. tau-bench ships its own agent loop, LLM user
simulator, and reward function; reimplementing those would be a lot of work and
would make the numbers non-comparable to published tau-bench results. Instead we
insert the single compression hook where it belongs — the tool observation that
re-enters the agent's context — and leave everything else (reward, user
simulator, agent) untouched.

The seam is an ``Env`` wrapper: ``CompressedEnv.step(action)`` delegates to the
real env, then compresses the returned observation when the action was a tool
call and the output is large. Because the env computes reward from actual
backend/DB state — not from the (now compressed) observation — compression only
changes what the *agent* sees. If a compressor shreds an output the agent needs
for a later step, the agent takes a wrong action and the reward drops. That is
exactly the effect we want to measure.

The wrapper is duck-typed so it runs against the real ``tau_bench`` ``Env`` and
against the mock in the tests without importing tau_bench. Wiring a real run is
then just ``make_compressed_env(...)`` + tau-bench's standard agent + API keys.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import Any, Callable, Optional

from tooltrim.tokens import count_tokens

# tau-bench's sentinel for "talk to the user" rather than call a tool.
RESPOND_ACTION_NAME = "respond"


@dataclass
class CompressionStats:
    """Accumulated compression accounting for one run/condition."""

    tool_calls: int = 0          # tool-call steps seen (excludes 'respond')
    compressed: int = 0          # of those, how many were actually compressed
    raw_tokens: int = 0          # observation tokens without compression
    context_tokens: int = 0      # observation tokens the agent actually read

    @property
    def saved_ratio(self) -> float:
        return 1 - (self.context_tokens / self.raw_tokens) if self.raw_tokens else 0.0

    def merge(self, other: "CompressionStats") -> None:
        """Accumulate another env's stats (tau-bench uses a fresh env per task)."""
        self.tool_calls += other.tool_calls
        self.compressed += other.compressed
        self.raw_tokens += other.raw_tokens
        self.context_tokens += other.context_tokens


# --- duck-typed access to Action / EnvResponse -------------------------------


def _attr(obj: Any, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _action_name(action: Any) -> Optional[str]:
    return _attr(action, "name")


def _action_kwargs(action: Any) -> dict:
    return _attr(action, "kwargs", {}) or {}


def _get_observation(resp: Any) -> Optional[str]:
    return _attr(resp, "observation")


def _replace_observation(resp: Any, new_obs: str) -> Any:
    """Return ``resp`` with its observation replaced, preserving everything else.

    Handles pydantic models (``model_copy``), dataclasses (``replace``), dicts,
    and plain objects (shallow copy + setattr) so we don't depend on tau-bench's
    concrete response type.
    """
    # pydantic v2
    if hasattr(resp, "model_copy"):
        try:
            return resp.model_copy(update={"observation": new_obs})
        except Exception:
            pass
    # dataclass
    if dataclasses.is_dataclass(resp) and not isinstance(resp, type):
        try:
            return dataclasses.replace(resp, observation=new_obs)
        except Exception:
            pass
    # dict
    if isinstance(resp, dict):
        out = dict(resp)
        out["observation"] = new_obs
        return out
    # plain object: shallow-copy and set the attribute
    import copy

    out = copy.copy(resp)
    setattr(out, "observation", new_obs)
    return out


def _default_query(instruction: str, action: Any) -> str:
    """Relevance query for compressing a tool result: the agent's immediate intent.

    We combine the standing task instruction (topical grounding) with the tool
    name and its arguments (the specific request), which is the best proxy for
    "what the agent is trying to accomplish" available at the env boundary.
    """
    name = _action_name(action) or ""
    try:
        kw = json.dumps(_action_kwargs(action), default=str)
    except Exception:
        kw = str(_action_kwargs(action))
    return f"{instruction} {name} {kw}".strip()


# --- the wrapper --------------------------------------------------------------


class CompressedEnv:
    """Wrap a tau-bench ``Env`` so tool observations are compressed in-loop.

    Args:
        inner: the real tau-bench env (or a duck-typed mock).
        compressor: a baseline ``Compressor`` (``compress(text, query, budget)``).
        budget: token budget for each compressed observation.
        min_tokens: only compress observations larger than this (matches
            production passthrough — small outputs are left alone).
        respond_action_name: the action name that talks to the user (not a tool).
        query_fn: override how the relevance query is built from
            ``(instruction, action)``.
    """

    def __init__(self, inner: Any, *, compressor: Any, budget: int,
                 min_tokens: int = 512,
                 respond_action_name: str = RESPOND_ACTION_NAME,
                 query_fn: Optional[Callable[[str, Any], str]] = None):
        self.inner = inner
        self.compressor = compressor
        self.budget = budget
        self.min_tokens = min_tokens
        self.respond_action_name = respond_action_name
        self.query_fn = query_fn or _default_query
        self.stats = CompressionStats()
        self._instruction = ""

    # Pass through any attribute we don't override (tools, wiki, actions, ...).
    def __getattr__(self, item):
        return getattr(self.inner, item)

    def reset(self, *args, **kwargs):
        resp = self.inner.reset(*args, **kwargs)
        # Remember the standing instruction for query grounding, if exposed.
        obs = _get_observation(resp)
        info = _attr(resp, "info")
        instr = _attr(info, "instruction") if info is not None else None
        self._instruction = instr or (obs if isinstance(obs, str) else "") or ""
        return resp

    def step(self, action: Any):
        resp = self.inner.step(action)
        obs = _get_observation(resp)
        name = _action_name(action)
        is_tool_call = bool(name) and name != self.respond_action_name
        if not (is_tool_call and isinstance(obs, str) and obs):
            return resp

        raw = count_tokens(obs)
        self.stats.tool_calls += 1
        self.stats.raw_tokens += raw
        if raw <= self.min_tokens:
            self.stats.context_tokens += raw
            return resp

        query = self.query_fn(self._instruction, action)
        compressed = self.compressor.compress(obs, query, self.budget)
        self.stats.compressed += 1
        self.stats.context_tokens += count_tokens(compressed)
        return _replace_observation(resp, compressed)


def make_compressed_env(env_name: str, *, method: str, budget: int,
                        min_tokens: int = 512, **env_kwargs) -> CompressedEnv:
    """Build a real tau-bench env wrapped for compression.

    Raises a clear error if tau-bench isn't installed; otherwise returns a
    ``CompressedEnv`` around ``tau_bench.envs.get_env(env_name, **env_kwargs)``.
    ``method`` is any registered baseline name ('tooltrim', 'rag-topk',
    'truncate-head', ...). Use ``method='none'`` to measure the uncompressed
    ceiling with identical accounting.
    """
    import importlib

    from .baselines import get_baseline

    try:
        envs = importlib.import_module("tau_bench.envs")
    except Exception as e:  # pragma: no cover - exercised only without tau_bench
        raise RuntimeError(
            "tau-bench not installed: pip install tau-bench "
            "(github.com/sierra-research/tau-bench)") from e

    inner = envs.get_env(env_name, **env_kwargs)
    if method == "none":
        # A passthrough compressor so accounting is identical to the real path.
        compressor = _NoOpCompressor()
    else:
        compressor = get_baseline(method)
    return CompressedEnv(inner, compressor=compressor, budget=budget,
                         min_tokens=min_tokens)


class _NoOpCompressor:
    name = "none"

    def compress(self, text: str, query: Optional[str], budget: int) -> str:
        return text
