"""LlamaIndex adapter — compress a tool's output before it re-enters context.

Wrap any LlamaIndex tool and the text the agent gets back is tooltrim-compressed,
query-aware. Name, description, and argument schema are unchanged, so the agent
calls it exactly as before::

    from llama_index.core.tools import FunctionTool
    from tooltrim.integrations import compress_llamaindex_tool

    def web_fetch(url: str) -> str:
        ...  # returns a huge HTML page

    tool = FunctionTool.from_defaults(fn=web_fetch)
    fetch = compress_llamaindex_tool(tool, max_tokens=400,
                                     query_from=lambda url: url)
    agent = FunctionAgent(tools=[fetch], llm=llm)

Unlike LangChain (whose tools return a string), a LlamaIndex tool returns a
``ToolOutput``. Only its ``content`` (the text the LLM reads) is compressed; the
structured ``raw_output`` / ``raw_input`` / ``is_error`` are preserved, and the
full text stays retrievable via the expand-store ref in the footer.

``llama_index.core`` is imported lazily, so this module is import-safe even when
LlamaIndex isn't installed.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..core import ToolCompressor
from ._common import compressed_output, resolve_query


def compress_llamaindex_tool(
    tool: Any,
    *,
    max_tokens: int = 512,
    compressor: Optional[ToolCompressor] = None,
    query_from: Optional[Callable[..., str]] = None,
):
    """Return a LlamaIndex tool that compresses ``tool``'s ``ToolOutput.content``.

    Args:
        tool: Any LlamaIndex ``BaseTool`` (e.g. from ``FunctionTool.from_defaults``).
        max_tokens: Budget used to build a default compressor if none given.
        compressor: Reuse a shared :class:`~tooltrim.ToolCompressor` (recommended
            so the expand store is shared across tools).
        query_from: ``f(**tool_args) -> str`` to derive the relevance query from
            the call arguments. Falls back to the ambient
            :func:`~tooltrim.query_scope` query.
    """
    from llama_index.core.tools import AsyncBaseTool  # lazy

    tc = compressor or ToolCompressor(max_tokens=max_tokens)

    def _compress(out: Any, kwargs: dict) -> Any:
        content = getattr(out, "content", None)
        if isinstance(content, str) and content:
            out.content = compressed_output(content, tc,
                                            resolve_query(query_from, kwargs))
        return out

    class _CompressedTool(AsyncBaseTool):
        @property
        def metadata(self):  # preserve name / description / fn_schema
            return tool.metadata

        def call(self, *args: Any, **kwargs: Any):
            return _compress(tool.call(*args, **kwargs), kwargs)

        async def acall(self, *args: Any, **kwargs: Any):
            return _compress(await tool.acall(*args, **kwargs), kwargs)

    return _CompressedTool()


def compress_llamaindex_tools(
    tools: List[Any],
    *,
    max_tokens: int = 512,
    compressor: Optional[ToolCompressor] = None,
    query_from: Optional[Callable[..., str]] = None,
) -> List[Any]:
    """Wrap a list of tools, sharing one compressor (and expand store) across them."""
    tc = compressor or ToolCompressor(max_tokens=max_tokens)
    return [
        compress_llamaindex_tool(t, compressor=tc, query_from=query_from)
        for t in tools
    ]
