"""Token counting.

Uses ``tiktoken`` (``cl100k_base``) when available for exact counts; otherwise
falls back to a fast character-based heuristic (~4 chars/token) that is good
enough for budgeting and savings estimates. The core library has no hard
dependency on ``tiktoken`` — install ``tooltrim[tokens]`` for exact counts.
"""

from __future__ import annotations

_ENCODER = None  # None = untried, False = unavailable, else an encoder


def _encoder():
    global _ENCODER
    if _ENCODER is not None:
        return _ENCODER
    try:
        import tiktoken

        _ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - depends on optional dep / network
        _ENCODER = False
    return _ENCODER


def count_tokens(text: str) -> int:
    """Return the (approximate) token count for ``text``."""
    if not text:
        return 0
    enc = _encoder()
    if enc:
        return len(enc.encode(text))
    # Heuristic: ~4 characters per token for English-like text. Conservative
    # rounding up so budgets are never silently exceeded.
    return max(1, (len(text) + 3) // 4)


def using_exact_counts() -> bool:
    """True if exact (tiktoken) counting is active, False if heuristic."""
    return bool(_encoder())
