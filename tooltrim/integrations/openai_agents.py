"""OpenAI Agents SDK adapter — compress a tool's output before it re-enters context.

Wrap any Agents SDK ``FunctionTool`` and the string it returns to the model is
tooltrim-compressed, query-aware. Name, description, JSON schema, guardrails, and
every other field are preserved (only ``on_invoke_tool`` is wrapped), so the
agent calls it exactly as before::

    from agents import function_tool
    from tooltrim.integrations import compress_openai_agents_tool

    @function_tool
    def web_fetch(url: str) -> str:
        ...  # returns a huge HTML page

    fetch = compress_openai_agents_tool(web_fetch, max_tokens=400,
                                        query_from=lambda url: url)
    agent = Agent(name="a", tools=[fetch])

The relevance query is derived from the tool's own JSON arguments via
``query_from(**args)``, else the ambient :func:`~tooltrim.query_scope` query.

``agents`` is imported lazily, so this module is import-safe without the SDK.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Callable, List, Optional

from ..core import ToolCompressor
from ._common import compressed_output, resolve_query


def _parse_args(input_str: str) -> dict:
    try:
        args = json.loads(input_str)
        return args if isinstance(args, dict) else {}
    except Exception:
        return {}


def compress_openai_agents_tool(
    tool: Any,
    *,
    max_tokens: int = 512,
    compressor: Optional[ToolCompressor] = None,
    query_from: Optional[Callable[..., str]] = None,
):
    """Return an Agents SDK ``FunctionTool`` that compresses ``tool``'s output.

    Args:
        tool: An Agents SDK ``FunctionTool`` (e.g. from ``@function_tool``).
        max_tokens: Budget used to build a default compressor if none given.
        compressor: Reuse a shared :class:`~tooltrim.ToolCompressor` (recommended
            so the expand store is shared across tools).
        query_from: ``f(**tool_args) -> str`` to derive the relevance query from
            the call's JSON arguments. Falls back to the ambient query.
    """
    tc = compressor or ToolCompressor(max_tokens=max_tokens)
    inner = tool.on_invoke_tool

    async def on_invoke_tool(ctx: Any, input_str: str) -> Any:
        out = await inner(ctx, input_str)
        query = resolve_query(query_from, _parse_args(input_str))
        return compressed_output(out, tc, query)

    return dataclasses.replace(tool, on_invoke_tool=on_invoke_tool)


def compress_openai_agents_tools(
    tools: List[Any],
    *,
    max_tokens: int = 512,
    compressor: Optional[ToolCompressor] = None,
    query_from: Optional[Callable[..., str]] = None,
) -> List[Any]:
    """Wrap a list of tools, sharing one compressor (and expand store) across them."""
    tc = compressor or ToolCompressor(max_tokens=max_tokens)
    return [
        compress_openai_agents_tool(t, compressor=tc, query_from=query_from)
        for t in tools
    ]
