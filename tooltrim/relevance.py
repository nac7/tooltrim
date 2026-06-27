"""Query-aware relevance scoring (pure-Python BM25-lite).

When the agent's current goal / query is known, tooltrim keeps the *relevant*
parts of a tool output rather than blindly truncating. Scoring is lexical
(BM25) by design: zero dependencies, deterministic, and fast enough to run on
every tool call. An embedding-based scorer can be layered on later behind the
same interface.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Sequence

_WORD = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(text)]


def score_chunks(
    chunks: Sequence[str],
    query: str,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[float]:
    """Return a BM25 relevance score for each chunk against ``query``.

    Returns all-zero scores when there is no query or no query terms overlap,
    letting callers fall back to positional (head/tail) selection.
    """
    q_terms = set(tokenize(query or ""))
    n = len(chunks)
    if not q_terms or n == 0:
        return [0.0] * n

    docs = [tokenize(c) for c in chunks]
    avgdl = (sum(len(d) for d in docs) / n) or 1.0

    df: Counter = Counter()
    for d in docs:
        for t in set(d):
            if t in q_terms:
                df[t] += 1

    scores: List[float] = []
    for d in docs:
        tf = Counter(d)
        dl = len(d) or 1
        s = 0.0
        for t in q_terms:
            if df[t] == 0:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            f = tf[t]
            s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        scores.append(s)
    return scores
