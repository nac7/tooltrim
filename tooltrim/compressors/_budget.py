"""Shared helpers for fitting text chunks into a token budget."""

from __future__ import annotations

import re
from typing import List, Sequence

from ..relevance import score_chunks
from ..tokens import count_tokens

ELISION = "[…]"

# Cap for a single chunk. Anything larger is sub-split so the budgeter can
# select *within* it — otherwise a newline-free blob (minified JSON-in-a-string,
# a single huge log line) would be one un-selectable chunk and pass through whole.
_MAX_CHUNK_TOKENS = 200
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _split_oversize(chunk: str, cap: int = _MAX_CHUNK_TOKENS) -> List[str]:
    """Break a too-large chunk into sentence-, then word-window-sized pieces."""
    if count_tokens(chunk) <= cap:
        return [chunk]
    pieces: List[str] = []
    for sent in _SENTENCE.split(chunk):
        sent = sent.strip()
        if not sent:
            continue
        if count_tokens(sent) <= cap:
            pieces.append(sent)
            continue
        # No usable punctuation (or one runaway sentence): fall back to words.
        words = sent.split()
        for i in range(0, len(words), 40):
            pieces.append(" ".join(words[i : i + 40]))
    return pieces or [chunk]


def split_paragraphs(text: str) -> List[str]:
    """Split into paragraph-ish chunks on blank lines, then long lines."""
    raw = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n")]
    chunks = [p for p in raw if p]
    if len(chunks) <= 1:
        # No blank-line structure: fall back to single lines.
        chunks = [ln.strip() for ln in text.splitlines() if ln.strip()]
    chunks = chunks or ([text.strip()] if text.strip() else [])
    out: List[str] = []
    for ch in chunks:
        out.extend(_split_oversize(ch))
    return out


def fit_chunks(
    chunks: Sequence[str],
    query: str | None,
    max_tokens: int,
    *,
    neighbor: int = 1,
) -> str:
    """Select chunks to fit ``max_tokens``.

    With a query: keep the highest-scoring chunks (BM25), then — budget
    permitting — pull in up to ``neighbor`` adjacent chunks on each side of the
    best matches so the model gets context, not just the bare matching line.
    Re-emit in original order with elision markers where content was dropped.
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
            used = _add_neighbors(chunks, keep, order, used, max_tokens, neighbor)
            return _stitch(chunks, keep)

    # Positional fallback: head + tail.
    return _head_tail(chunks, max_tokens)


def _add_neighbors(chunks: Sequence[str], keep: set, order: List[int],
                   used: int, max_tokens: int, neighbor: int) -> int:
    """Add up to ``neighbor`` adjacent chunks around matches, best-match first."""
    if neighbor <= 0:
        return used
    matched = [i for i in order if i in keep]  # highest-scoring kept first
    for i in matched:
        for off in range(1, neighbor + 1):
            for j in (i - off, i + off):
                if 0 <= j < len(chunks) and j not in keep:
                    cost = count_tokens(chunks[j]) + 2
                    if used + cost > max_tokens:
                        continue
                    keep.add(j)
                    used += cost
    return used


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
