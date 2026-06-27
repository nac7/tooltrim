"""Shared helpers for fitting text chunks into a token budget."""

from __future__ import annotations

from typing import List, Sequence

from ..relevance import score_chunks
from ..tokens import count_tokens

ELISION = "[…]"


def split_paragraphs(text: str) -> List[str]:
    """Split into paragraph-ish chunks on blank lines, then long lines."""
    raw = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n")]
    chunks = [p for p in raw if p]
    if len(chunks) <= 1:
        # No blank-line structure: fall back to single lines.
        chunks = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return chunks or ([text.strip()] if text.strip() else [])


def fit_chunks(
    chunks: Sequence[str],
    query: str | None,
    max_tokens: int,
) -> str:
    """Select chunks to fit ``max_tokens``.

    With a query: keep the highest-scoring chunks (BM25), then re-emit them in
    their original order with elision markers where content was dropped.
    Without a query (or no lexical overlap): keep a head+tail window.
    """
    chunks = list(chunks)
    if not chunks:
        return ""

    scores = score_chunks(chunks, query or "")
    has_signal = any(s > 0 for s in scores)

    if has_signal:
        # Keep only relevant chunks (positive score), best-first, within budget.
        # Don't pad with irrelevant content just to fill the budget.
        order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        keep: set = set()
        used = 0
        for i in order:
            if scores[i] <= 0:
                break
            cost = count_tokens(chunks[i]) + 2
            if used + cost > max_tokens and keep:
                break
            keep.add(i)
            used += cost
        if keep:
            return _stitch(chunks, keep)

    # Positional fallback: head + tail.
    return _head_tail(chunks, max_tokens)


def _stitch(chunks: Sequence[str], keep: set) -> str:
    out: List[str] = []
    prev = -1
    for i, ch in enumerate(chunks):
        if i in keep:
            if prev != -1 and i - prev > 1:
                out.append(ELISION)
            out.append(ch)
            prev = i
    if prev != -1 and prev < len(chunks) - 1:
        out.append(ELISION)
    return "\n\n".join(out)


def _head_tail(chunks: Sequence[str], max_tokens: int) -> str:
    head: List[str] = []
    tail: List[str] = []
    used = 0
    lo, hi = 0, len(chunks) - 1
    take_head = True
    while lo <= hi:
        idx = lo if take_head else hi
        cost = count_tokens(chunks[idx]) + 2
        if used + cost > max_tokens and (head or tail):
            break
        if take_head:
            head.append(chunks[idx])
            lo += 1
        else:
            tail.append(chunks[idx])
            hi -= 1
        used += cost
        take_head = not take_head
    parts = head
    if lo <= hi:
        parts = head + [ELISION] + list(reversed(tail))
    else:
        parts = head + list(reversed(tail))
    return "\n\n".join(parts)
