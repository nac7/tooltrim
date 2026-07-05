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

def test_stats_saved_ratio():
    st = CompressionStats(tool_calls=2, compressed=2, raw_tokens=1000, context_tokens=100)
    assert st.saved_ratio == pytest.approx(0.9)


def test_passthrough_attributes_reach_inner_env():
    env = CompressedEnv(MockEnv(_big_json()), compressor=_Compressor(), budget=256)
    assert env.wiki == "policy wiki"          # __getattr__ delegates
    assert env.tools_info[0]["name"] == "get_order"


def test_make_compressed_env_raises_without_taubench():
    with pytest.raises(RuntimeError, match="tau-bench not installed"):
        make_compressed_env("retail", method="tooltrim", budget=256)


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
