"""Answer scoring: did the model recover the gold fact?

Deterministic, dependency-free. Normalizes punctuation/case, then accepts an
answer if the normalized gold string appears verbatim OR a sufficient fraction
of the gold's content tokens are present (handles models that paraphrase).
"""

from __future__ import annotations

import re

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize(s: str) -> str:
    return _NON_ALNUM.sub(" ", s.lower()).strip()


def matches(answer: str, gold: str, *, threshold: float = 0.7) -> bool:
    a = normalize(answer)
    g = normalize(gold)
    if not g:
        return False
    if g in a:
        return True
    gold_tokens = [t for t in g.split() if len(t) > 2] or g.split()
    answer_tokens = set(a.split())
    hits = sum(1 for t in gold_tokens if t in answer_tokens)
    return hits / len(gold_tokens) >= threshold


def passed(answer: str, gold: str, all_of=(), must_not=()) -> bool:
    """Full case verdict: gold present, every all_of present, no must_not present.

    Covers single-fact (gold only), multi-fact (all_of), and distractor
    (must_not) cases with one predicate.
    """
    if not matches(answer, gold):
        return False
    if not all(matches(answer, g) for g in all_of):
        return False
    if any(matches(answer, m) for m in must_not):
        return False
    return True
