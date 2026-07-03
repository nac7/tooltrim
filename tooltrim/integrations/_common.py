"""Shared helpers for framework adapters (LangChain, LlamaIndex, ...)."""

from __future__ import annotations

from typing import Any, Callable, Optional

from ..core import ToolCompressor
from ..decorators import current_query


def compressed_output(output: Any, compressor: ToolCompressor,
                      query: Optional[str]) -> Any:
    """Compress a tool result if it's a string; pass anything else through.

    Pure (no framework import) so it can be unit-tested without LangChain/etc.
    """
    if isinstance(output, str) and output:
        return compressor.compress(output, query=query).text
    return output


def resolve_query(query_from: Optional[Callable[..., str]],
                  kwargs: dict) -> Optional[str]:
    """Query from ``query_from(**kwargs)`` if given, else the ambient query."""
    if query_from is not None:
        try:
            return query_from(**kwargs)
        except Exception:
            return None
    return current_query()
