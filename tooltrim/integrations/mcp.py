"""Model Context Protocol (MCP) integration — compress tool results at the
protocol boundary.

MCP is the emerging open standard for connecting LLMs to tools/data via servers.
Its ``tools/call`` responses (``CallToolResult``) are exactly the bloated
tool outputs tooltrim targets — often thousands of tokens of HTML/JSON/logs that
re-enter the model's context verbatim.

Three pieces, smallest to largest:

  - :func:`compress_tool_result` — pure: compress the text blocks of a single
    ``CallToolResult`` (skips errors and non-text content; preserves
    ``structuredContent``/``isError``). Testable without any server.
  - :func:`compressing_call_tool` — wrap an upstream ``call_tool`` coroutine so
    every result is compressed. Drop this into any MCP server/gateway.
  - :func:`run_stdio_gateway` — a ready gateway: sits in front of an upstream MCP
    server over stdio, re-exposes its tools unchanged, and compresses every
    result. Point any MCP client (Claude Desktop, an IDE, an agent) at it.

MCP has no per-call "user query", so relevance defaults to the tool's own
arguments (a search tool's ``query`` arg, a path, ...). Override with
``query_from(name, arguments)``.

``mcp`` is imported lazily (only the gateway needs it); ``pip install tooltrim[mcp]``.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional

from ..core import ToolCompressor

CallToolFn = Callable[[str, Optional[dict]], Awaitable[Any]]


def _query_from_args(arguments: Optional[dict]) -> Optional[str]:
    """Derive a relevance query from the tool-call arguments (string values)."""
    if not isinstance(arguments, dict):
        return None
    parts = [v for v in arguments.values() if isinstance(v, str) and v.strip()]
    return " ".join(parts) if parts else None


def compress_tool_result(result: Any, *, compressor: ToolCompressor,
                         query: Optional[str] = None) -> Any:
    """Return a ``CallToolResult`` with its text blocks compressed.

    Error results and non-text content (images/audio/structured) pass through
    untouched. Operates by duck-typing + pydantic ``model_copy``, so it needs no
    import of ``mcp`` and can be unit-tested with any compatible object.
    """
    if getattr(result, "isError", False):
        return result
    content = getattr(result, "content", None) or []
    new_blocks = []
    changed = False
    for block in content:
        text = getattr(block, "text", None)
        if getattr(block, "type", None) == "text" and isinstance(text, str) and text:
            res = compressor.compress(text, query=query)
            if res.compressed:
                new_blocks.append(block.model_copy(update={"text": res.text}))
                changed = True
                continue
        new_blocks.append(block)
    if not changed:
        return result
    return result.model_copy(update={"content": new_blocks})


def compressing_call_tool(
    upstream_call: CallToolFn,
    *,
    max_tokens: int = 512,
    compressor: Optional[ToolCompressor] = None,
    query_from: Optional[Callable[[str, Optional[dict]], Optional[str]]] = None,
) -> CallToolFn:
    """Wrap an upstream ``call_tool(name, arguments) -> CallToolResult`` coroutine
    so every result is tooltrim-compressed.

    ``query_from(name, arguments)`` supplies the relevance query; by default the
    tool-call arguments are used.
    """
    tc = compressor or ToolCompressor(max_tokens=max_tokens)

    async def call_tool(name: str, arguments: Optional[dict] = None) -> Any:
        result = await upstream_call(name, arguments)
        query = (query_from(name, arguments) if query_from
                 else _query_from_args(arguments))
        return compress_tool_result(result, compressor=tc, query=query)

    return call_tool


async def run_stdio_gateway(
    upstream: Any,
    *,
    max_tokens: int = 512,
    compressor: Optional[ToolCompressor] = None,
    query_from: Optional[Callable[[str, Optional[dict]], Optional[str]]] = None,
    name: str = "tooltrim-gateway",
) -> None:
    """Run a compressing MCP gateway over stdio in front of ``upstream``.

    ``upstream`` is a ``mcp.client.stdio.StdioServerParameters`` describing the
    real MCP server to wrap. This gateway speaks MCP to a client on one side and
    to the upstream server on the other, compressing every tool result in flight.
    """
    from mcp.client.session import ClientSession
    from mcp.client.stdio import stdio_client
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server

    tc = compressor or ToolCompressor(max_tokens=max_tokens)

    async with stdio_client(upstream) as (up_read, up_write):
        async with ClientSession(up_read, up_write) as session:
            await session.initialize()
            wrapped = compressing_call_tool(
                session.call_tool, compressor=tc, query_from=query_from)

            server = Server(name)

            @server.list_tools()
            async def _list_tools():  # pass through the upstream tool catalog
                return (await session.list_tools()).tools

            @server.call_tool()
            async def _call_tool(tool_name: str, arguments: dict):
                return await wrapped(tool_name, arguments)

            async with stdio_server() as (read, write):
                await server.run(read, write,
                                 server.create_initialization_options())
