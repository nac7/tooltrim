"""Decorators and helpers for wrapping tool functions.

The fastest way to adopt tooltrim: wrap the function your agent calls. The
return value (typically a string fed back into the model) is compressed
automatically.

The relevance ``query`` can come from three places, in priority order:
  1. an explicit ``query_from(*args, **kwargs)`` callable on the decorator,
  2. the ambient query set via :func:`set_query` / :func:`query_scope`,
  3. nothing (positional head/tail compression).
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Optional

from .core import CompressionResult, ToolCompressor

_current_query: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "tooltrim_query", default=None
)


def set_query(query: Optional[str]) -> None:
    """Set the ambient relevance query for subsequent compressed tool calls."""
    _current_query.set(query)


@contextmanager
def query_scope(query: Optional[str]):
    """Context manager: set the ambient query for the duration of the block."""
    token = _current_query.set(query)
    try:
        yield
    finally:
        _current_query.reset(token)


def current_query() -> Optional[str]:
    return _current_query.get()


def compressed_tool(
    max_tokens: int = 512,
    *,
    query_from: Optional[Callable[..., str]] = None,
    compressor: Optional[ToolCompressor] = None,
    return_result: bool = False,
):
    """Decorator that compresses a tool function's (string) output.

    Args:
        max_tokens: Budget (used to build a default compressor if none given).
        query_from: Optional ``f(*args, **kwargs) -> str`` to derive the query
            from the call arguments (e.g. a search tool's ``query`` param).
        compressor: Reuse a shared :class:`ToolCompressor` (recommended so the
            expand store is shared across tools).
        return_result: If True, return the full :class:`CompressionResult`
            instead of the compressed string.
    """
    tc = compressor or ToolCompressor(max_tokens=max_tokens)

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any):
            out = fn(*args, **kwargs)
            query = None
            if query_from is not None:
                try:
                    query = query_from(*args, **kwargs)
                except Exception:
                    query = None
            if query is None:
                query = current_query()
            result = tc.compress(out, query=query)
            return result if return_result else result.text

        wrapper.tool_compressor = tc  # type: ignore[attr-defined]
        return wrapper

    return decorate


def wrap_tool(
    fn: Callable[..., Any],
    *,
    max_tokens: int = 512,
    query_from: Optional[Callable[..., str]] = None,
    compressor: Optional[ToolCompressor] = None,
) -> Callable[..., Any]:
    """Functional form of :func:`compressed_tool` for wrapping an existing callable."""
    return compressed_tool(
        max_tokens=max_tokens, query_from=query_from, compressor=compressor
    )(fn)
