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

from ..relevance import score_chunks
from ..tokens import count_tokens
from ._budget import fit_chunks


def _truncate_str(s: str, max_str_len: int) -> str:
    if len(s) <= max_str_len:
        return s
    return s[:max_str_len] + "…(+%d chars)" % (len(s) - max_str_len)


def _sample_indices(items: List[Any], k: int, query: str | None) -> List[int]:
    if len(items) <= k:
        return list(range(len(items)))
    if query:
        rendered = [json.dumps(it, ensure_ascii=False, default=str) for it in items]
        scores = score_chunks(rendered, query)
        if any(s > 0 for s in scores):
            top = sorted(range(len(items)), key=lambda i: scores[i], reverse=True)[:k]
            return sorted(top)
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
