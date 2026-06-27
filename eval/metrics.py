"""Statistics for defensible rate reporting.

Wilson score interval for a binomial proportion — the right CI for accuracy on a
small number of cases (unlike the normal approximation, it stays inside [0,1] and
behaves well at the extremes, e.g. 16/16).
"""

from __future__ import annotations

import math
from typing import Tuple


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
