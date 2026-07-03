"""Query-aware relevance scoring, with a pluggable scorer.

When the agent's current goal / query is known, tooltrim keeps the *relevant*
parts of a tool output rather than blindly truncating. The default scorer is
lexical (BM25): zero dependencies, deterministic, and fast enough to run on every
tool call.

Scoring is pluggable behind one tiny interface — ``scorer(chunks, query) ->
list[float]``. Swap in an embedding scorer for semantic matching (query "car"
finding a chunk about "automobiles") via :class:`~tooltrim.EmbeddingScorer`, or
set your own for the duration of a call with :func:`using_scorer`. Compressors
call :func:`score_chunks`, which dispatches to the active scorer, so a scorer set
on :class:`~tooltrim.ToolCompressor` threads through every content type.
"""

from __future__ import annotations

import contextvars
import math
import re
from collections import Counter
from contextlib import contextmanager
from typing import List, Sequence

_WORD = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(text)]


def bm25_scores(
    chunks: Sequence[str],
    query: str,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[float]:
    """BM25 relevance score for each chunk against ``query``.

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


class BM25Scorer:
    """The default lexical scorer (callable). Zero-dependency, deterministic."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def __call__(self, chunks: Sequence[str], query: str) -> List[float]:
        return bm25_scores(chunks, query, k1=self.k1, b=self.b)


_DEFAULT_SCORER = BM25Scorer()
_active_scorer: "contextvars.ContextVar" = contextvars.ContextVar(
    "tooltrim_scorer", default=None)


def score_chunks(chunks: Sequence[str], query: str) -> List[float]:
    """Score ``chunks`` against ``query`` using the active scorer (BM25 default)."""
    scorer = _active_scorer.get() or _DEFAULT_SCORER
    return scorer(chunks, query or "")


@contextmanager
def using_scorer(scorer):
    """Use ``scorer`` for scoring within the block (``None`` = keep current)."""
    if scorer is None:
        yield
        return
    token = _active_scorer.set(scorer)
    try:
        yield
    finally:
        _active_scorer.reset(token)
