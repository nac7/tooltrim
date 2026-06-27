"""Generic text / markdown compressor (query-aware extractive)."""

from __future__ import annotations

from ._budget import fit_chunks, split_paragraphs


def compress(text: str, query: str | None, max_tokens: int) -> str:
    chunks = split_paragraphs(text)
    return fit_chunks(chunks, query, max_tokens)
