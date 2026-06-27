"""Tabular (CSV/TSV/pipe) compressor: header + schema + sampled rows."""

from __future__ import annotations

from typing import List

from ..relevance import score_chunks
from ..tokens import count_tokens


def _delimiter(lines: List[str]) -> str:
    sample = lines[: min(10, len(lines))]
    best, best_count = ",", 0
    for delim in ("\t", "|", ","):
        counts = [ln.count(delim) for ln in sample]
        if counts and all(c == counts[0] for c in counts) and counts[0] > best_count:
            best, best_count = delim, counts[0]
    return best


def compress(text: str, query: str | None, max_tokens: int) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) <= 2:
        return text
    delim = _delimiter(lines)
    header, rows = lines[0], lines[1:]
    ncols = header.count(delim) + 1

    # Decide how many rows fit, after reserving budget for header + note line.
    keep_rows: List[str]
    if query:
        scores = score_chunks(rows, query)
        order = (sorted(range(len(rows)), key=lambda i: scores[i], reverse=True)
                 if any(s > 0 for s in scores) else list(range(len(rows))))
    else:
        order = list(range(len(rows)))

    selected: List[int] = []
    used = count_tokens(header) + 12  # header + note headroom
    for i in order:
        cost = count_tokens(rows[i]) + 1
        if used + cost > max_tokens and selected:
            break
        selected.append(i)
        used += cost

    selected.sort()
    keep_rows = [rows[i] for i in selected]
    omitted = len(rows) - len(keep_rows)
    out = [header] + keep_rows
    if omitted > 0:
        out.append("# (+%d more rows, %d columns total)" % (omitted, ncols))
    return "\n".join(out)
