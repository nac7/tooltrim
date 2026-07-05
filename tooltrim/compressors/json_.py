"""JSON compressor: structure-preserving pruning + sampling.

Strategy (deterministic):
  * Long strings are truncated with an elision marker.
  * Large arrays keep a sample of items plus a "(+N more items)" note; for arrays
    of objects the key schema is preserved via the sampled items.
  * With a query, array items are sampled by BM25 relevance instead of position.
  * The whole thing is rendered compactly and the sampling budget is tightened
    until it fits ``max_tokens``; as a last resort it degrades to text fitting.
"""

from __future__ import annotations

import json
from typing import Any, List

from .._config import relevance_floor_override
from ..relevance import score_chunks
from ..tokens import count_tokens
from ._budget import fit_chunks

# A kept record must score at least this fraction of the best-matching record's
# score. This turns array selection into a *relevance cliff* rather than a fill-to-k
# quota: single-needle queries keep just the needle (everything else falls off the
# cliff), while genuine multi-record matches (aggregation) keep the whole cluster
# because those records score similarly high. A fixed prior — deliberately not tuned
# per benchmark — that stops a larger budget from padding in keyword-stuffed noise.
_RELEVANCE_FLOOR = 0.5


def _truncate_str(s: str, max_str_len: int) -> str:
    if len(s) <= max_str_len:
        return s
    return s[:max_str_len] + "…(+%d chars)" % (len(s) - max_str_len)


def _sample_indices(items: List[Any], k: int, query: str | None) -> List[int]:
    # Query-aware selection treats the budget as a *cap*, not a target: keep only
    # the relevance-positive records (top-k of them), never padding out to k with
    # zero-score noise. Padding is what let a larger budget reintroduce distractors
    # and cost accuracy — a bigger array kept *more* wrong records, not more signal.
    # (Plain top-k-by-position, below, is the no-query fallback.)
    if query:
        rendered = [json.dumps(it, ensure_ascii=False, default=str) for it in items]
        scores = score_chunks(rendered, query)
        positive = [i for i in range(len(items)) if scores[i] > 0]
        if positive:
            positive.sort(key=lambda i: scores[i], reverse=True)
            # An ablation can lower the cliff to 0 (fill-to-k quota, the pre-cliff
            # behavior) to isolate the relevance-cliff's contribution.
            floor_frac = relevance_floor_override()
            if floor_frac is None:
                floor_frac = _RELEVANCE_FLOOR
            floor = scores[positive[0]] * floor_frac
            kept = [i for i in positive if scores[i] >= floor]
            return sorted(kept[:k])
    if len(items) <= k:
        return list(range(len(items)))
    return list(range(k))


def _prune(obj: Any, depth: int, *, max_list_items: int, max_str_len: int,
           max_depth: int, query: str | None) -> Any:
    if depth >= max_depth and isinstance(obj, (dict, list)):
        if isinstance(obj, dict):
            return {"…": "(%d keys elided)" % len(obj)}
        return ["(%d items elided)" % len(obj)]

    if isinstance(obj, str):
        return _truncate_str(obj, max_str_len)

    if isinstance(obj, dict):
        return {
            k: _prune(v, depth + 1, max_list_items=max_list_items,
                      max_str_len=max_str_len, max_depth=max_depth, query=query)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        idxs = _sample_indices(obj, max_list_items, query)
        out: List[Any] = [
            _prune(obj[i], depth + 1, max_list_items=max_list_items,
                   max_str_len=max_str_len, max_depth=max_depth, query=query)
            for i in idxs
        ]
        if len(obj) > len(idxs):
            out.append("(+%d more items)" % (len(obj) - len(idxs)))
        return out

    return obj


def compress(text: str, query: str | None, max_tokens: int) -> str:
    try:
        data = json.loads(text)
    except Exception:
        # Not actually parseable JSON: treat as text.
        return fit_chunks([p for p in text.split("\n") if p.strip()], query, max_tokens)

    # Progressively tighter sampling until it fits the budget.
    for max_list_items, max_str_len, max_depth in (
        (8, 200, 6), (5, 120, 5), (3, 80, 4), (2, 50, 3), (1, 30, 3),
    ):
        pruned = _prune(data, 0, max_list_items=max_list_items,
                        max_str_len=max_str_len, max_depth=max_depth, query=query)
        rendered = json.dumps(pruned, ensure_ascii=False, indent=None,
                              separators=(",", ":"), default=str)
        if count_tokens(rendered) <= max_tokens:
            return rendered

    # Still too big (e.g. enormous single object): degrade to text fitting.
    return fit_chunks(rendered.split(","), query, max_tokens)
