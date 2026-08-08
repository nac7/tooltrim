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
from typing import Any, Callable, List, Optional

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
    expand_calls: int = 0        # expand_tool_output calls the agent made (recovery)

    @property
    def saved_ratio(self) -> float:
        return 1 - (self.context_tokens / self.raw_tokens) if self.raw_tokens else 0.0

    def merge(self, other: "CompressionStats") -> None:
        """Accumulate another env's stats (tau-bench uses a fresh env per task)."""
        self.tool_calls += other.tool_calls
        self.compressed += other.compressed
        self.raw_tokens += other.raw_tokens
        self.context_tokens += other.context_tokens
        self.expand_calls += other.expand_calls


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


def _replace_fields(resp: Any, **fields: Any) -> Any:
    """Return ``resp`` with the given fields replaced, preserving type/everything else.

    Same duck-typed strategy as :func:`_replace_observation` but for arbitrary
    fields — used to synthesize an expand response (new observation, reward 0,
    not done) that is the *same concrete type* the agent's loop already handles.
    """
    if hasattr(resp, "model_copy"):
        try:
            return resp.model_copy(update=dict(fields))
        except Exception:
            pass
    if dataclasses.is_dataclass(resp) and not isinstance(resp, type):
        try:
            return dataclasses.replace(resp, **fields)
        except Exception:
            pass
    if isinstance(resp, dict):
        out = dict(resp)
        out.update(fields)
        return out
    import copy

    out = copy.copy(resp)
    for k, v in fields.items():
        setattr(out, k, v)
    return out


# How many recent dialogue turns feed the relevance query. The opening task
# instruction alone is a poor proxy for intent: in tau-bench retail the user only
# names the *specific* thing they want ("the 1000ml black one") several turns in,
# in reply to the agent, so a query built from the instruction plus the tool args
# scores zero against every variant of a product and selection silently degrades
# to positional (keep the first k of 18). Set to 0 to restore the old
# instruction-only behaviour — that is the ablation arm.
_DIALOGUE_TURNS = 6

# Per-turn character cap, so one long agent monologue cannot crowd the recent
# user reply (where the discriminating detail lives) out of the query.
_TURN_MAX_CHARS = 400


def _default_query(instruction: str, action: Any) -> str:
    """Relevance query for compressing a tool result: the agent's immediate intent.

    We combine the standing grounding text (the task instruction, plus recent
    dialogue turns when the wrapper supplies them) with the tool name and its
    arguments, which together are the best proxy for "what the agent is trying to
    accomplish" available at the env boundary.
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
            ``(grounding, action)``.
        dialogue_turns: how many recent dialogue turns to append to the grounding
            text passed to ``query_fn``. 0 reproduces the instruction-only
            behaviour (the ablation arm).
    """

    def __init__(self, inner: Any, *, compressor: Any, budget: int,
                 min_tokens: int = 512,
                 respond_action_name: str = RESPOND_ACTION_NAME,
                 query_fn: Optional[Callable[[str, Any], str]] = None,
                 dialogue_turns: int = _DIALOGUE_TURNS):
        self.inner = inner
        self.compressor = compressor
        self.budget = budget
        self.min_tokens = min_tokens
        self.respond_action_name = respond_action_name
        self.query_fn = query_fn or _default_query
        self.dialogue_turns = dialogue_turns
        self.stats = CompressionStats()
        self._instruction = ""
        self._dialogue: List[str] = []
        # A recovery-capable compressor (tooltrim-expand) lets the agent pull back
        # full outputs. When present, we advertise its tool and route calls to it
        # instead of the inner env — turning aggressive compression from an
        # irrecoverable gamble into a safe default, which is tooltrim's real design.
        self._expandable = (hasattr(compressor, "handle_expand")
                            and hasattr(compressor, "expand_tool_spec"))
        self._expand_name = getattr(compressor, "expand_tool_name",
                                    "expand_tool_output")
        self._last_resp: Any = None

    # Pass through any attribute we don't override (tools, wiki, actions, ...).
    def __getattr__(self, item):
        return getattr(self.inner, item)

    @property
    def tools_info(self):
        """Inner env's tools, plus the expand tool when the compressor supports it.

        The agent is built from ``env.tools_info``, so appending the spec here is
        what actually hands the model the recovery tool. A no-op for the naive
        baselines and ``none`` (they aren't expand-capable), keeping their runs an
        exact apples-to-apples comparison against the recoverable path.
        """
        base = list(self.inner.tools_info)
        if self._expandable:
            base = base + [self.compressor.expand_tool_spec(style="openai")]
        return base

    def _handle_expand_action(self, action: Any) -> Any:
        """Serve an ``expand_tool_output`` call from the store, not the inner env.

        The inner env has no such tool, so we answer it here and return a response
        of the same concrete type the agent already consumes (observation swapped
        for the retrieved page, reward 0, not done). Expand is side-effect-free on
        env/DB state, so final reward is still computed by the real env — expand
        only changes what the agent can *see*, never what counts as success.
        """
        kw = _action_kwargs(action)
        ref = kw.get("ref")
        try:
            start = int(kw.get("start", 0) or 0)
        except (TypeError, ValueError):
            start = 0
        length = kw.get("length")
        try:
            length = int(length) if length not in (None, "") else None
        except (TypeError, ValueError):
            length = None
        text = self.compressor.handle_expand(str(ref), start=start, length=length)
        self.stats.expand_calls += 1
        if self._last_resp is not None:
            return _replace_fields(self._last_resp, observation=text,
                                   reward=0.0, done=False)
        # No prior real response to clone from (expand can't legitimately be the
        # first action — there's no ref yet — but stay duck-typed just in case).
        return {"observation": text, "reward": 0.0, "done": False, "info": {}}

    def reset(self, *args, **kwargs):
        resp = self.inner.reset(*args, **kwargs)
        # Remember the standing instruction for query grounding, if exposed.
        obs = _get_observation(resp)
        info = _attr(resp, "info")
        instr = _attr(info, "instruction") if info is not None else None
        self._instruction = instr or (obs if isinstance(obs, str) else "") or ""
        self._dialogue = []
        return resp

    def _grounding(self) -> str:
        """Instruction plus the most recent dialogue turns, for query grounding.

        Read-only over things the agent has already seen — the utterances it sent
        and the user's replies. No reward, no ground-truth action, nothing the
        agent does not itself have in context, so this cannot leak the answer.
        """
        if not self.dialogue_turns or not self._dialogue:
            return self._instruction
        recent = self._dialogue[-self.dialogue_turns:]
        return " ".join([self._instruction, *recent]).strip()

    def _record_turn(self, text: Any) -> None:
        if self.dialogue_turns and isinstance(text, str) and text.strip():
            self._dialogue.append(text.strip()[:_TURN_MAX_CHARS])

    def step(self, action: Any):
        name = _action_name(action)
        # Expand calls are served from the store, not the inner env (which has no
        # such tool). Do this before delegating so the inner env never sees it.
        if self._expandable and name == self._expand_name:
            return self._handle_expand_action(action)

        resp = self.inner.step(action)
        self._last_resp = resp  # template for cloning an expand response
        obs = _get_observation(resp)
        is_tool_call = bool(name) and name != self.respond_action_name
        if not (is_tool_call and isinstance(obs, str) and obs):
            # A "respond" turn: remember what the agent said and how the user
            # replied. The user's reply is where the discriminating detail shows
            # up ("the 1000ml black one"), and it is the whole point of grounding
            # the relevance query in the conversation rather than the opening
            # instruction, which was written before any of it was said.
            if name == self.respond_action_name:
                self._record_turn(_action_kwargs(action).get("content"))
                self._record_turn(obs)
            return resp

        raw = count_tokens(obs)
        self.stats.tool_calls += 1
        self.stats.raw_tokens += raw
        if raw <= self.min_tokens:
            self.stats.context_tokens += raw
            return resp

        query = self.query_fn(self._grounding(), action)
        compressed = self.compressor.compress(obs, query, self.budget)
        self.stats.compressed += 1
        self.stats.context_tokens += count_tokens(compressed)
        return _replace_observation(resp, compressed)


def make_compressed_env(env_name: str, *, method: str, budget: int,
                        min_tokens: int = 512,
                        dialogue_turns: int = _DIALOGUE_TURNS,
                        **env_kwargs) -> CompressedEnv:
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
                         min_tokens=min_tokens, dialogue_turns=dialogue_turns)


class _NoOpCompressor:
    name = "none"

    def compress(self, text: str, query: Optional[str], budget: int) -> str:
        return text
