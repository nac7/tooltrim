"""LangChain adapter — compress a tool's output before it re-enters context.

Wrap any LangChain tool and the (string) result the agent gets back is
tooltrim-compressed, query-aware. No change to the tool's name, description, or
argument schema, so the agent calls it exactly as before::

    from langchain_core.tools import tool
    from tooltrim.integrations import compress_langchain_tool

    @tool
    def web_fetch(url: str) -> str:
        ...  # returns a huge HTML page

    fetch = compress_langchain_tool(web_fetch, max_tokens=400,
                                    query_from=lambda url: url)
    agent = create_react_agent(llm, [fetch])

The relevance query is taken from ``query_from(**tool_args)`` if given, else the
ambient query set via :func:`tooltrim.set_query` / :func:`tooltrim.query_scope`.

``langchain_core`` is imported lazily, so this module is import-safe even when
LangChain isn't installed; only :func:`compress_langchain_tool` needs it.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..core import ToolCompressor
from ..decorators import current_query


def _compressed_output(output: Any, compressor: ToolCompressor,
                       query: Optional[str]) -> Any:
    """Compress a tool result if it's a string; pass anything else through.

    Pure (no LangChain) so it can be unit-tested without the framework.
    """
    if isinstance(output, str) and output:
        return compressor.compress(output, query=query).text
    return output


def _resolve_query(query_from: Optional[Callable[..., str]],
                   kwargs: dict) -> Optional[str]:
    if query_from is not None:
        try:
            return query_from(**kwargs)
        except Exception:
            return None
    return current_query()


def compress_langchain_tool(
    tool: Any,
    *,
    max_tokens: int = 512,
    compressor: Optional[ToolCompressor] = None,
    query_from: Optional[Callable[..., str]] = None,
):
    """Return a new LangChain tool that compresses ``tool``'s string output.

    Args:
        tool: Any ``BaseTool`` (e.g. from the ``@tool`` decorator).
        max_tokens: Budget used to build a default compressor if none given.
        compressor: Reuse a shared :class:`~tooltrim.ToolCompressor` (recommended
            so the expand store is shared across tools).
        query_from: ``f(**tool_args) -> str`` to derive the relevance query from
            the call arguments (e.g. a search tool's ``query`` arg). Falls back
            to the ambient :func:`~tooltrim.query_scope` query.
    """
    from langchain_core.tools import StructuredTool

    tc = compressor or ToolCompressor(max_tokens=max_tokens)

    def func(**kwargs: Any) -> Any:
        out = tool.invoke(kwargs)
        return _compressed_output(out, tc, _resolve_query(query_from, kwargs))

    async def afunc(**kwargs: Any) -> Any:
        out = await tool.ainvoke(kwargs)
        return _compressed_output(out, tc, _resolve_query(query_from, kwargs))

    return StructuredTool.from_function(
        func=func,
        coroutine=afunc,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        return_direct=getattr(tool, "return_direct", False),
    )


def compress_langchain_tools(
    tools: List[Any],
    *,
    max_tokens: int = 512,
    compressor: Optional[ToolCompressor] = None,
    query_from: Optional[Callable[..., str]] = None,
) -> List[Any]:
    """Wrap a list of tools, sharing one compressor (and expand store) across them."""
    tc = compressor or ToolCompressor(max_tokens=max_tokens)
    return [
        compress_langchain_tool(t, compressor=tc, query_from=query_from)
        for t in tools
    ]
