"""Compress tool results at the MCP (Model Context Protocol) boundary.

Two ways to use tooltrim with MCP:

1. Run the gateway from the CLI in front of any MCP server — point your MCP
   client (Claude Desktop, an IDE, an agent) at it and every tool result is
   compressed in flight, no code change:

       tooltrim mcp -- npx -y @modelcontextprotocol/server-filesystem /path

2. Wrap the result-handling in your own server with `compressing_call_tool`, or
   compress a single `CallToolResult` with `compress_tool_result`. This example
   shows (2) with a fake upstream, so it runs without an MCP server binary.

Run (needs `mcp`):  python examples/08_mcp_gateway.py
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tooltrim import count_tokens


def _big_json():
    return json.dumps([{"id": i, "status": "shipped"} for i in range(400)]
                      + [{"id": 999, "status": "REFUNDED", "note": "customer 4417"}])


def main():
    try:
        from mcp.types import CallToolResult, TextContent
    except ImportError:
        print("Install MCP support for this demo:  pip install tooltrim[mcp]")
        return

    from tooltrim.integrations import compressing_call_tool

    # Pretend this is the upstream MCP server's tool handler.
    async def upstream_call(name, arguments):
        return CallToolResult(content=[TextContent(type="text", text=_big_json())])

    # One wrap: every result compressed, query taken from the call arguments.
    call = compressing_call_tool(upstream_call, max_tokens=200)

    raw = _big_json()
    out = asyncio.run(call("list_orders", {"query": "which customer was refunded?"}))
    compressed = out.content[0].text

    print(f"raw tool result:   {count_tokens(raw):>6,} tokens")
    print(f"compressed result: {count_tokens(compressed):>6,} tokens "
          f"({(1 - count_tokens(compressed) / count_tokens(raw)) * 100:.0f}% smaller)\n")
    print("=== compressed result the model sees ===")
    print(compressed)


if __name__ == "__main__":
    main()
