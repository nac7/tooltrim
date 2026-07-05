"""Concrete agent-benchmark suites: an offline mock + real adapter seams.

``MockToolBenchmark`` is a genuine 2-hop task (get_order -> list_by_region ->
submit) where *both* tool outputs are large and must survive compression for the
agent to finish. It runs with no API keys, so it exercises and tests the loop.

``load_taubench`` / ``load_bfcl`` are the seams for real benchmarks: they import
the upstream package when installed and otherwise report why they're unavailable.
Wiring a real run means (a) installing the benchmark, (b) mapping its tasks/tools
into ``BenchTask``, and (c) supplying an LLM tool-calling ``Policy`` in place of
the mock's deterministic reference policy.
"""

from __future__ import annotations

import json
import random
from typing import List, Optional

from .agentbench import (
    Action, BenchTask, Finish, Observation, Policy, Tool, ToolCall,
)

# --- deterministic data generation -------------------------------------------

_REGIONS = ["us-east", "us-west", "eu-west", "ap-south"]


def _noise_events(rng: random.Random, n: int) -> List[dict]:
    return [{"ts": 1_700_000_000 + rng.randint(0, 9_999_999),
             "level": rng.choice(["info", "debug", "warn"]),
             "msg": "background event %d handled by worker %d"
                    % (rng.randint(1000, 9999), rng.randint(0, 32))}
            for _ in range(n)]


def _order_blob(order_id: int, region: str, rng: random.Random) -> str:
    # One order object buried in a bloated payload of unrelated events.
    obj = {"order_id": order_id, "region": region, "status": "failed",
           "events": _noise_events(rng, 220),
           "note": "order %d routed to region %s for settlement" % (order_id, region)}
    return json.dumps(obj)


def _region_blob(region: str, failed_amounts: List[float], rng: random.Random) -> str:
    # A large list; the failed records in `region` are what the agent must sum.
    records = []
    rid = 8000
    for amt in failed_amounts:
        records.append({"id": rid, "region": region, "status": "failed",
                        "amount": amt, "note": "failed payment in %s" % region})
        rid += 1
    # Bloat with non-matching records (other regions / statuses).
    for _ in range(280):
        records.append({"id": rid, "region": rng.choice(_REGIONS),
                        "status": rng.choice(["ok", "pending"]),
                        "amount": round(rng.uniform(1, 999), 2),
                        "note": "unrelated transaction record"})
        rid += 1
    rng.shuffle(records)
    return json.dumps({"results": records})


# --- the mock benchmark -------------------------------------------------------


class _MockPolicy:
    """Deterministic 2-hop reference agent: get_order -> list_by_region -> submit.

    Consumes each compressed tool output exactly as a real agent's next
    programmatic step would (``json.loads`` + field access). If a compressed
    output does not parse, the agent cannot recover the value and submits a
    sentinel that fails the check — which is the whole point.
    """

    def __init__(self, order_id: int):
        self.order_id = order_id

    def _region_from(self, blob: str) -> Optional[str]:
        try:
            obj = json.loads(blob)
        except Exception:
            return None
        return obj.get("region") if isinstance(obj, dict) else None

    def _sum_failed(self, blob: str, region: str) -> Optional[float]:
        try:
            obj = json.loads(blob)
        except Exception:
            return None
        recs = obj.get("results") if isinstance(obj, dict) else obj
        if not isinstance(recs, list):
            return None
        total = 0.0
        for r in recs:
            if (isinstance(r, dict) and r.get("region") == region
                    and r.get("status") == "failed"):
                try:
                    total += float(r["amount"])
                except (KeyError, ValueError, TypeError):
                    return None
        return round(total, 2)

    def act(self, obs: Observation, tools) -> Action:
        if not obs.history:
            return ToolCall("get_order", {"order_id": self.order_id},
                            query="order %d region settlement" % self.order_id)
        if len(obs.history) == 1:
            region = self._region_from(obs.history[0][1])
            if region is None:
                return Finish(None)  # first hop's output didn't survive compression
            return ToolCall("list_by_region", {"region": region},
                            query="failed payment %s amount" % region)
        # Second hop returned: sum the failed amounts in the region and submit.
        region = self._region_from(obs.history[0][1])
        if region is None:
            return Finish(None)
        return Finish(self._sum_failed(obs.history[1][1], region))


class MockToolBenchmark:
    """A runnable, API-free multi-step agent benchmark for the compression loop."""

    name = "mock-tools"

    def tasks(self, n: int = 12, seed: int = 7) -> List[BenchTask]:
        out: List[BenchTask] = []
        for i in range(n):
            rng = random.Random(seed * 1000 + i)  # reseeded per task => reproducible
            order_id = 7000 + i
            region = _REGIONS[i % len(_REGIONS)]
            failed_amounts = [round(rng.uniform(50, 500), 2)
                              for _ in range(rng.randint(3, 6))]
            gold = round(sum(failed_amounts), 2)

            order_out = _order_blob(order_id, region, random.Random(rng.random()))
            region_out = _region_blob(region, failed_amounts, random.Random(rng.random()))

            tools = [
                Tool("get_order", (lambda o: lambda args: o)(order_out),
                     "Fetch an order by id (large payload)."),
                Tool("list_by_region", (lambda r: lambda args: r)(region_out),
                     "List transactions in a region (large payload)."),
            ]
            out.append(BenchTask(
                id="mock-%02d" % i,
                prompt=("What is the total amount of failed payments in the "
                        "region of order %d?" % order_id),
                tools=tools,
                check=(lambda g: lambda ans: ans == g)(gold),
                reference_policy=_MockPolicy(order_id),
            ))
        return out


# --- real-benchmark adapter seams --------------------------------------------


class _Unavailable:
    """A benchmark adapter that isn't installed; explains how to enable it."""

    def __init__(self, name: str, reason: str):
        self.name = name
        self._reason = reason

    def available(self) -> bool:
        return False

    def tasks(self, **_):
        raise RuntimeError(f"{self.name} unavailable: {self._reason}")


def load_taubench():
    """tau-bench availability check.

    The real integration lives in ``eval.taubench_adapter`` and does *not* port
    tau-bench tasks into ``BenchTask`` — it wraps tau-bench's own ``Env`` so tool
    observations are compressed in-loop (``make_compressed_env``), keeping
    tau-bench's reward and user simulator intact. This function just reports
    whether the upstream package is importable.
    """
    import importlib.util

    if importlib.util.find_spec("tau_bench") is None:
        return _Unavailable(
            "tau-bench",
            "pip install tau-bench (github.com/sierra-research/tau-bench) + set an LLM key; "
            "then use eval.taubench_adapter.make_compressed_env")
    from . import taubench_adapter  # noqa: F401  (import side-checks availability)

    return _Unavailable(
        "tau-bench",
        "installed — run via eval.taubench_adapter.make_compressed_env + run_taubench.py")


def load_bfcl():
    """BFCL (Berkeley Function Calling Leaderboard) adapter seam."""
    import importlib.util

    if importlib.util.find_spec("bfcl") is None:
        return _Unavailable(
            "bfcl",
            "install the Berkeley Function Calling Leaderboard harness + set an LLM key")
    return _Unavailable("bfcl", "installed but adapter mapping not yet wired")
