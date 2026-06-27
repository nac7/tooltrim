"""Log compressor: dedup repetitive lines, keep errors + head/tail context."""

from __future__ import annotations

import re
from typing import List, Tuple

from ..relevance import score_chunks
from ..tokens import count_tokens

ELISION = "[…]"
_IMPORTANT = re.compile(
    r"\b(error|err|warn|warning|fatal|critical|crit|exception|traceback|"
    r"fail(ed|ure)?|panic|denied|refused|timeout|unhandled)\b",
    re.I,
)


def _dedup(lines: List[str]) -> List[Tuple[str, int]]:
    """Collapse runs of identical lines into ``(line, count)`` pairs."""
    out: List[Tuple[str, int]] = []
    for ln in lines:
        if out and out[-1][0] == ln:
            out[-1] = (ln, out[-1][1] + 1)
        else:
            out.append((ln, 1))
    return out


def _render(line: str, count: int) -> str:
    return line if count == 1 else "%s  (x%d)" % (line, count)


def compress(text: str, query: str | None, max_tokens: int) -> str:
    raw = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    collapsed = _dedup(raw)
    n = len(collapsed)
    if n == 0:
        return ""

    rel = score_chunks([c for c, _ in collapsed], query or "")
    has_rel = any(s > 0 for s in rel)

    keep = set()
    used = 0

    def try_add(i: int) -> bool:
        nonlocal used
        if i in keep:
            return True
        cost = count_tokens(_render(*collapsed[i])) + 1
        if used + cost > max_tokens and keep:
            return False
        keep.add(i)
        used += cost
        return True

    # 1) Importance order: error-ish lines (optionally boosted by query).
    def importance(i: int) -> float:
        base = 2.0 if _IMPORTANT.search(collapsed[i][0]) else 0.0
        return base + (rel[i] if has_rel else 0.0)

    for i in sorted(range(n), key=importance, reverse=True):
        if importance(i) <= 0:
            break
        if not try_add(i):
            break

    # 2) Fill remaining budget with head then tail for context.
    for i in list(range(n)) + list(range(n - 1, -1, -1)):
        if not try_add(i):
            break

    # Stitch in original order with elision markers.
    out: List[str] = []
    prev = -1
    for i in range(n):
        if i in keep:
            if prev != -1 and i - prev > 1:
                out.append(ELISION)
            out.append(_render(*collapsed[i]))
            prev = i
    if prev != -1 and prev < n - 1:
        out.append(ELISION)
    return "\n".join(out)
