"""Statistics for defensible rate reporting.

Wilson score interval for a binomial proportion — the right CI for accuracy on a
small number of cases (unlike the normal approximation, it stays inside [0,1] and
behaves well at the extremes, e.g. 16/16).
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% (default z=1.96) Wilson score interval for k successes out of n."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def fmt_ci(lo: float, hi: float) -> str:
    return f"[{lo*100:.0f}-{hi*100:.0f}%]"


def mcnemar(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> Tuple[int, int, float]:
    """McNemar's paired test comparing two methods on the *same* cases.

    Every case is scored by both methods, so the samples are paired — the right
    test for "is method A significantly more accurate than method B?" is McNemar
    on the discordant pairs, not a two-proportion z-test that assumes
    independence.

    Returns ``(b, c, p)`` where ``b`` = cases A-correct/B-wrong, ``c`` =
    A-wrong/B-correct, and ``p`` is the two-sided p-value (chi-square, 1 df, with
    Edwards' continuity correction). ``p == 1.0`` when there are no discordant
    pairs.
    """
    b = sum(1 for a, bb in zip(a_correct, b_correct) if a and not bb)
    c = sum(1 for a, bb in zip(a_correct, b_correct) if bb and not a)
    if b + c == 0:
        return b, c, 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)  # continuity-corrected
    stat = max(0.0, stat)
    p = math.erfc(math.sqrt(stat / 2.0))  # chi-square(1) survival function
    return b, c, min(1.0, p)
