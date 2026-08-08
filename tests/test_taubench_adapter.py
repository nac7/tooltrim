"""Tests for the tau-bench compression adapter, against a mocked tau-bench Env.

No tau_bench install required: we mock its Action / EnvResponse / Env duck types
and verify the wrapper compresses the right things, leaves reward/done untouched,
and accounts correctly. The pure reporting function is tested separately.
"""

import json
from dataclasses import dataclass
from typing import Optional

import pytest

from eval.taubench_adapter import (
    CompressedEnv, CompressionStats, make_compressed_env,
)
from run_taubench import build_report


# --- mocked tau-bench duck types ---------------------------------------------

@dataclass
class MockAction:
    name: str
    kwargs: dict


@dataclass
class MockResp:
    observation: str
    reward: float = 0.0
    done: bool = False
    info: Optional[dict] = None


class MockEnv:
    """Minimal tau-bench-shaped env: reset + step returning EnvResponse-likes."""

    def __init__(self, tool_output: str, reward: float = 1.0):
        self.tool_output = tool_output
        self.reward = reward
        self.tools_info = [{"name": "get_order"}]
        self.wiki = "policy wiki"

    def reset(self, task_index=None):
        return MockResp(observation="You are helping a customer.",
                        info={"instruction": "refund order 42"})

    def step(self, action):
        if action.name == "respond":
            return MockResp(observation="ok, one moment", reward=0.0, done=False)
        # A tool call returns the (large) tool output and the env's own reward.
        return MockResp(observation=self.tool_output, reward=self.reward, done=True)


def _big_json(n=200):
    recs = [{"id": i, "status": "failed" if i == 42 else "ok",
             "amount": 100 + i, "note": f"order {i} record padding text here"}
            for i in range(n)]
    return json.dumps({"results": recs})


class _Compressor:
    name = "fake"

    def compress(self, text, query, budget):
        return f"[compressed for: {query}]"


def test_compresses_large_tool_observation_and_preserves_reward():
    env = CompressedEnv(MockEnv(_big_json()), compressor=_Compressor(),
                        budget=256, min_tokens=100)
    env.reset()
    resp = env.step(MockAction("get_order", {"order_id": 42}))
    # Observation was compressed ...
    assert resp.observation.startswith("[compressed for:")
    # ... the query carried the agent's intent (instruction + tool + args) ...
    assert "refund order 42" in resp.observation and "get_order" in resp.observation
    # ... and reward/done were passed through untouched.
    assert resp.reward == 1.0 and resp.done is True
    assert env.stats.compressed == 1 and env.stats.tool_calls == 1


def test_small_output_passes_through_uncompressed():
    env = CompressedEnv(MockEnv("tiny result"), compressor=_Compressor(),
                        budget=256, min_tokens=100)
    env.reset()
    resp = env.step(MockAction("get_order", {"order_id": 1}))
    assert resp.observation == "tiny result"
    assert env.stats.tool_calls == 1 and env.stats.compressed == 0


def test_respond_action_is_not_compressed():
    env = CompressedEnv(MockEnv(_big_json()), compressor=_Compressor(),
                        budget=256, min_tokens=10)
    env.reset()
    resp = env.step(MockAction("respond", {"content": "hi"}))
    assert resp.observation == "ok, one moment"
    assert env.stats.tool_calls == 0  # respond is not a tool call

class _RecordingEnv(MockEnv):
    """Env whose user simulator reveals the discriminating detail mid-dialogue."""

    def step(self, action):
        if action.name == "respond":
            return MockResp(observation="I want the 1000ml black plastic one.",
                            reward=0.0, done=False)
        return MockResp(observation=self.tool_output, reward=self.reward, done=True)


def _query_seen(dialogue_turns):
    seen = {}

    class _Spy:
        name = "spy"

        def compress(self, text, query, budget):
            seen["query"] = query
            return "x"

    env = CompressedEnv(_RecordingEnv(_big_json()), compressor=_Spy(),
                        budget=64, min_tokens=10, dialogue_turns=dialogue_turns)
    env.reset()
    env.step(MockAction(name="respond", kwargs={"content": "Which size would you like?"}))
    env.step(MockAction(name="get_order", kwargs={"product_id": "8310926033"}))
    return seen["query"]


def test_relevance_query_includes_recent_dialogue_not_just_the_instruction():
    # Regression: the query used to be the opening instruction + tool args only.
    # In tau-bench retail the user names the specific variant they want several
    # turns in, so that query scored 0.0 against every variant of a product and
    # selection silently degraded to positional (first k of 18). The discriminating
    # words must reach the compressor.
    q = _query_seen(dialogue_turns=6)
    assert "1000ml" in q and "black" in q
    assert "refund order 42" in q  # standing instruction still grounds it


def test_dialogue_turns_zero_restores_instruction_only_query():
    # The ablation arm has to stay reproducible: dialogue_turns=0 must reproduce
    # the old behaviour exactly, so the A/B is a single flag.
    q = _query_seen(dialogue_turns=0)
    assert "1000ml" not in q
    assert "refund order 42" in q


def test_stats_saved_ratio():
    st = CompressionStats(tool_calls=2, compressed=2, raw_tokens=1000, context_tokens=100)
    assert st.saved_ratio == pytest.approx(0.9)


def test_passthrough_attributes_reach_inner_env():
    env = CompressedEnv(MockEnv(_big_json()), compressor=_Compressor(), budget=256)
    assert env.wiki == "policy wiki"          # __getattr__ delegates
    assert env.tools_info[0]["name"] == "get_order"


def test_make_compressed_env_raises_without_taubench():
    import importlib.util

    if importlib.util.find_spec("tau_bench") is not None:
        pytest.skip("tau_bench installed; the not-installed error path isn't exercised here")
    with pytest.raises(RuntimeError, match="tau-bench not installed"):
        make_compressed_env("retail", method="tooltrim", budget=256)


# --- expand / recovery path ---------------------------------------------------

def _ref_from(observation: str) -> str:
    """Pull the ``ref=XXXX`` id out of a tooltrim footer."""
    import re

    m = re.search(r"ref=([A-Za-z0-9]+)", observation)
    assert m, f"no ref footer in observation: {observation!r}"
    return m.group(1)


def test_expandable_compressor_advertises_expand_tool():
    from eval.baselines import TooltrimExpand

    env = CompressedEnv(MockEnv(_big_json()), compressor=TooltrimExpand(),
                        budget=128, min_tokens=50)
    names = [t.get("function", {}).get("name") or t.get("name")
             for t in env.tools_info]
    assert "get_order" in names            # inner env's tool preserved
    assert "expand_tool_output" in names   # recovery tool advertised to the agent


def test_non_expandable_compressor_does_not_add_tool():
    env = CompressedEnv(MockEnv(_big_json()), compressor=_Compressor(),
                        budget=128, min_tokens=50)
    names = [t.get("name") for t in env.tools_info]
    assert names == ["get_order"]          # naive baselines get no expand tool


def test_expand_call_recovers_full_output_from_store():
    from eval.baselines import TooltrimExpand

    env = CompressedEnv(MockEnv(_big_json()), compressor=TooltrimExpand(),
                        budget=128, min_tokens=50)
    env.reset()
    # A real tool call: output is compressed and carries a recovery ref.
    resp = env.step(MockAction("get_order", {"order_id": 42}))
    ref = _ref_from(resp.observation)
    assert env.stats.compressed == 1 and env.stats.expand_calls == 0

    # The agent now calls expand: served from the store, not the inner env.
    exp = env.step(MockAction("expand_tool_output", {"ref": ref}))
    assert '"status":"failed"' in exp.observation.replace(" ", "")  # record 42 recovered
    assert exp.reward == 0.0 and exp.done is False   # side-effect-free on reward/done
    assert env.stats.expand_calls == 1
    assert env.stats.tool_calls == 1  # expand is not counted as an inner tool call


def test_expand_bad_ref_returns_clear_message_not_crash():
    from eval.baselines import TooltrimExpand

    env = CompressedEnv(MockEnv(_big_json()), compressor=TooltrimExpand(),
                        budget=128, min_tokens=50)
    env.reset()
    env.step(MockAction("get_order", {"order_id": 42}))
    exp = env.step(MockAction("expand_tool_output", {"ref": "deadbeef"}))
    assert "no stored output" in exp.observation
    assert env.stats.expand_calls == 1


# --- pure reporting -----------------------------------------------------------

def test_build_report_scores_and_runs_mcnemar():
    # tooltrim solves all 6; truncate-head solves none -> discordant, significant.
    ids = list(range(6))
    rewards = {
        "none": {i: 1.0 for i in ids},
        "truncate-head": {i: 0.0 for i in ids},
        "tooltrim": {i: 1.0 for i in ids},
    }
    stats = {m: CompressionStats(tool_calls=6, compressed=6,
                                 raw_tokens=6000, context_tokens=600)
             for m in rewards}
    md = build_report(rewards, stats, budget=512)
    assert "tau-bench" in md
    assert "`tooltrim` **★**" in md
    assert "100%" in md and "0%" in md
    # truncate-head vs tooltrim is 0/6 vs 6/6 -> McNemar p should be small + starred.
    trunc_line = [ln for ln in md.splitlines() if ln.startswith("| `truncate-head`")][0]
    assert "*" in trunc_line
