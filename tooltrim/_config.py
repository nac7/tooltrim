"""Context-scoped overrides for compressor internals.

These exist so an evaluation can *ablate* individual mechanisms — the neighbor
window and the JSON relevance cliff — without threading knobs through every
content-type compressor. They mirror :func:`tooltrim.relevance.using_scorer`:
default is ``None`` (use the built-in behavior), and a value set for the duration
of a block overrides it. Not part of the public API; production code never sets
these, so default behavior is unchanged.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Iterator, Optional

# None => callers use their own default (fit_chunks neighbor=1).
_neighbor: "contextvars.ContextVar[Optional[int]]" = contextvars.ContextVar(
    "tooltrim_neighbor", default=None)
# None => json_ uses its module default (_RELEVANCE_FLOOR).
_relevance_floor: "contextvars.ContextVar[Optional[float]]" = contextvars.ContextVar(
    "tooltrim_relevance_floor", default=None)


def neighbor_override() -> Optional[int]:
    return _neighbor.get()


def relevance_floor_override() -> Optional[float]:
    return _relevance_floor.get()


@contextmanager
def using_config(*, neighbor: Optional[int] = None,
                 relevance_floor: Optional[float] = None) -> Iterator[None]:
    """Override compressor internals within the block. ``None`` leaves a knob as-is."""
    tokens = []
    if neighbor is not None:
        tokens.append((_neighbor, _neighbor.set(neighbor)))
    if relevance_floor is not None:
        tokens.append((_relevance_floor, _relevance_floor.set(relevance_floor)))
    try:
        yield
    finally:
        for var, tok in reversed(tokens):
            var.reset(tok)
