"""End-to-end agent-task benchmark: does the compressed output survive being
*consumed by code*?

The faithfulness harness answers "did the fact survive, and can a person/LLM read
it out?". Real agents do more than read — they feed one tool's output into the
*next* step programmatically: ``json.loads(result)``, then filter/count/sum over
the records. A compressor that returns a mid-object JSON fragment (what RAG chunk
selection produces on single-line JSON) passes a lenient LLM read but *crashes the
agent's next `json.loads`*. Recall parity (Sec. results) cannot see this; this
benchmark can.

Each :class:`AgentTask` plants a small set of *target* records (matching the
agent's query) inside a large, type-appropriate tool output, and defines a task
whose answer is a deterministic function of those records — a count, a sum, the
list of ids, or a single-record lookup. ``AgentTask.consume`` models the agent's
next step exactly: parse the compressed output *as code would*, and if that
succeeds, compute the answer over the parsed records. If the output does not parse
as a whole (the agent's ``json.loads``/CSV read would raise), the task fails —
which is the honest bar, and the regime where content-type structure pays off.

This is deterministic and API-free: success is decided by real parsing + real
arithmetic, not an LLM judge, so the benchmark costs nothing and is bit-for-bit
reproducible. It measures the *code-consumer* regime specifically; the
LLM-reader regime (where a strong model reads around broken syntax) is the
recall-parity result reported elsewhere.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from .faithfulness import _csv_table, _load_json_whole

_RNG = random.Random(19)

_FILLER_WORDS = (
    "system user account order region latency cache retry queue worker session "
    "cluster shard policy invoice balance ledger metric trace request handler"
).split()

# filler records use only these non-matching statuses (never the target status),
# so the *only* records matching a task predicate are the planted targets.
_FILLER_STATUS = ("ok", "pending", "processing")


def _filler_note() -> str:
    return " ".join(_RNG.choice(_FILLER_WORDS) for _ in range(6)).capitalize() + "."


# --- blob builders: plant target records spread far apart among filler ---------

def _plant_positions(n_targets: int, n: int, base: int) -> List[int]:
    step = max(1, n // (n_targets + 1))
    return [(base + (k + 1) * step) % n for k in range(n_targets)]


def _task_blob_json(targets: List[dict], n: int, base: int) -> str:
    pos = _plant_positions(len(targets), n, base)
    at = dict(zip(pos, targets))
    records = []
    for i in range(n):
        if i in at:
            records.append({"id": at[i]["id"], "status": at[i]["status"],
                            "amount": at[i]["amount"], "note": at[i]["note"]})
        else:
            records.append({"id": 10_000 + i, "status": _RNG.choice(_FILLER_STATUS),
                            "amount": round(_RNG.uniform(1, 9999), 2),
                            "note": _filler_note()})
    return json.dumps({"page": 1, "total": n, "results": records})


def _task_blob_csv(targets: List[dict], n: int, base: int) -> str:
    pos = _plant_positions(len(targets), n, base)
    at = dict(zip(pos, targets))
    rows = ["id,region,status,amount,note"]
    for i in range(n):
        if i in at:
            t = at[i]
            rows.append(f"{t['id']},{t['region']},{t['status']},{t['amount']},{t['note']}")
        else:
            rows.append(f"{10_000 + i},{_RNG.choice(['us','eu','apac'])},"
                        f"{_RNG.choice(_FILLER_STATUS)},{round(_RNG.uniform(1,9999),2)},"
                        f"{_filler_note()}")
    return "\n".join(rows)


# --- the task -----------------------------------------------------------------

@dataclass(frozen=True)
class AgentTask:
    id: str
    content_type: str            # "json" | "tabular"
    tool_output: str             # full (bloated) blob
    query: str                   # agent goal — drives compression relevance
    task_kind: str               # "count" | "sum" | "list" | "lookup"
    marker: str                  # phrase that identifies a target record's note
    gold: object                 # deterministic expected answer
    lookup_id: Optional[int] = None   # for "lookup": which record's amount

    def _records(self, compressed: str) -> Optional[List[dict]]:
        """Parse the compressed output the way the agent's next step would.

        Returns the list of record dicts, or ``None`` if the whole output does not
        parse as its content type (a fragment — the agent's parse would raise).
        """
        if self.content_type == "json":
            val = _load_json_whole(compressed)
            if val is None:
                return None
            recs = val.get("results") if isinstance(val, dict) else val
            if not isinstance(recs, list):
                return None
            return [r for r in recs if isinstance(r, dict)]
        if self.content_type == "tabular":
            table = _csv_table(compressed)
            if table is None:
                return None
            header, rows = table
            out = []
            for r in rows:
                d = dict(zip(header, r))
                if "note" in d:
                    out.append(d)
            return out
        return None

    def _matches(self, rec: dict) -> bool:
        return self.marker.lower() in str(rec.get("note", "")).lower()

    def consume(self, compressed: str):
        """Model the agent's programmatic next step; return the computed answer,
        or ``None`` if the output could not be parsed/consumed."""
        recs = self._records(compressed)
        if recs is None:
            return None
        hits = [r for r in recs if self._matches(r)]
        if self.task_kind == "count":
            return len(hits)
        if self.task_kind == "sum":
            try:
                return round(sum(float(r["amount"]) for r in hits), 2)
            except (KeyError, ValueError, TypeError):
                return None
        if self.task_kind == "list":
            try:
                return tuple(sorted(int(r["id"]) for r in hits))
            except (KeyError, ValueError, TypeError):
                return None
        if self.task_kind == "lookup":
            for r in recs:
                try:
                    if int(r["id"]) == self.lookup_id:
                        return round(float(r["amount"]), 2)
                except (KeyError, ValueError, TypeError):
                    continue
            return None
        return None

    def succeeds(self, compressed: str) -> bool:
        return self.consume(compressed) == self.gold


# --- declarative task suite ----------------------------------------------------
# Each entry plants target records with a shared marker phrase in their note, then
# asks a count/sum/list/lookup question answerable only from those records.

def _targets(marker: str, ids: List[int], amounts: List[float],
             status: str = "failed", region: str = "us") -> List[dict]:
    return [{"id": i, "status": status, "amount": a, "region": region,
             "note": f"{marker} for order {i} processor stripe amount {a}"}
            for i, a in zip(ids, amounts)]


# (id, content_type, marker, ids, amounts, task_kind, query, [lookup_id])
_TASK_SPECS = [
    ("failed-count-json", "json", "failed payment",
     [7001, 7002, 7003, 7004], [120.50, 340.00, 55.25, 900.75], "count",
     "How many payments failed?", None),
    ("failed-sum-json", "json", "failed payment",
     [7001, 7002, 7003, 7004], [120.50, 340.00, 55.25, 900.75], "sum",
     "What is the total amount of the failed payments?", None),
    ("failed-list-json", "json", "failed payment",
     [7001, 7002, 7003, 7004, 7005], [120.5, 340.0, 55.25, 900.75, 12.0], "list",
     "List the order ids of every failed payment.", None),
    ("refund-count-json", "json", "refund requested",
     [8100, 8200, 8300], [50.0, 75.5, 20.0], "count",
     "How many refunds were requested?", None),
    ("refund-sum-json", "json", "refund requested",
     [8100, 8200, 8300], [50.0, 75.5, 20.0], "sum",
     "Total amount across the refund requests?", None),
    ("lookup-json", "json", "failed payment",
     [7001, 7002, 7003], [120.5, 340.0, 55.25], "lookup",
     "What is the amount for order 7002?", 7002),

    ("failed-count-csv", "tabular", "failed payment",
     [7001, 7002, 7003, 7004], [120.50, 340.00, 55.25, 900.75], "count",
     "How many payments failed?", None),
    ("failed-sum-csv", "tabular", "failed payment",
     [7001, 7002, 7003, 7004], [120.50, 340.00, 55.25, 900.75], "sum",
     "What is the total amount of the failed payments?", None),
    ("failed-list-csv", "tabular", "failed payment",
     [7001, 7002, 7003, 7004, 7005], [120.5, 340.0, 55.25, 900.75, 12.0], "list",
     "List the order ids of every failed payment.", None),
    ("refund-sum-csv", "tabular", "refund requested",
     [8100, 8200, 8300], [50.0, 75.5, 20.0], "sum",
     "Total amount across the refund requests?", None),
]


def _gold(kind: str, ids: List[int], amounts: List[float],
          lookup_id: Optional[int]) -> object:
    if kind == "count":
        return len(ids)
    if kind == "sum":
        return round(sum(amounts), 2)
    if kind == "list":
        return tuple(sorted(ids))
    if kind == "lookup":
        return round(dict(zip(ids, amounts))[lookup_id], 2)
    raise ValueError(kind)


def default_agent_tasks(n_filler: int = 300) -> List[AgentTask]:
    """The deterministic agent-task suite (idempotent: RNG reseeded on entry)."""
    _RNG.seed(19)
    tasks: List[AgentTask] = []
    for j, (tid, ct, marker, ids, amounts, kind, query, lookup_id) in enumerate(_TASK_SPECS):
        targets = _targets(marker, ids, amounts)
        build = _task_blob_json if ct == "json" else _task_blob_csv
        blob = build(targets, n_filler, base=17 + j * 23)
        tasks.append(AgentTask(
            id=tid, content_type=ct, tool_output=blob, query=query,
            task_kind=kind, marker=marker,
            gold=_gold(kind, ids, amounts, lookup_id), lookup_id=lookup_id))
    return tasks


def task_success_rate(tasks_and_outputs) -> float:
    """Mean success over ``(task, compressed_text)`` pairs."""
    items = list(tasks_and_outputs)
    if not items:
        return 0.0
    return sum(t.succeeds(c) for t, c in items) / len(items)
