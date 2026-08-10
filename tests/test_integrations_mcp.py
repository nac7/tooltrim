import asyncio
import json

import pytest

from tooltrim import ToolCompressor
from tooltrim.integrations.mcp import (
    _query_from_args,
    compress_tool_result,
    compressing_call_tool,
)


def _big_json():
    return json.dumps([{"id": i, "note": f"row {i}"} for i in range(400)]
                      + [{"id": 999, "note": "refund to customer 4417"}])


def test_query_from_args_joins_string_values():
    assert _query_from_args({"query": "refund", "id": 4417, "path": "/x"}) == "refund /x"
    assert _query_from_args(None) is None
    assert _query_from_args({"id": 1}) is None


def test_compress_tool_result_compresses_text_blocks():
    pytest.importorskip("mcp")
    from mcp.types import CallToolResult, TextContent

    tc = ToolCompressor(max_tokens=80, add_footer=False)
    result = CallToolResult(content=[TextContent(type="text", text=_big_json())])
    out = compress_tool_result(result, compressor=tc, query="refund 4417")

    assert "4417" in out.content[0].text
    assert len(out.content[0].text) < len(_big_json())
    assert out.isError is False


def test_compress_tool_result_skips_errors():
    pytest.importorskip("mcp")
    from mcp.types import CallToolResult, TextContent

    tc = ToolCompressor(max_tokens=80, add_footer=False)
    err = CallToolResult(content=[TextContent(type="text", text=_big_json())],
                         isError=True)
    out = compress_tool_result(err, compressor=tc, query="refund")
    assert out is err                     # untouched


def test_compress_tool_result_preserves_structured_content():
    pytest.importorskip("mcp")
    from mcp.types import CallToolResult, TextContent

    tc = ToolCompressor(max_tokens=80, add_footer=False)
    result = CallToolResult(
        content=[TextContent(type="text", text=_big_json())],
        structuredContent={"kept": True})
    out = compress_tool_result(result, compressor=tc, query="refund 4417")
    assert out.structuredContent == {"kept": True}
    assert "4417" in out.content[0].text


def test_compress_tool_result_passes_through_small_text():
    pytest.importorskip("mcp")
    from mcp.types import CallToolResult, TextContent

    tc = ToolCompressor(max_tokens=512, add_footer=False)
    result = CallToolResult(content=[TextContent(type="text", text="short")])
    out = compress_tool_result(result, compressor=tc, query="x")
    assert out is result                  # nothing compressed -> same object


def test_compressing_call_tool_wraps_upstream():
    pytest.importorskip("mcp")
    from mcp.types import CallToolResult, TextContent

    async def upstream(name, arguments):
        return CallToolResult(content=[TextContent(type="text", text=_big_json())])

    call = compressing_call_tool(upstream, max_tokens=80)
    # query defaults to the tool-call arguments
    out = asyncio.run(call("lookup", {"query": "refund 4417"}))
    assert "4417" in out.content[0].text
    assert len(out.content[0].text) < len(_big_json())


def test_serve_standalone_server_compress_and_expand():
    """The `tooltrim serve` standalone server: list tools, compress with a
    retained fact + ref, then expand the ref back to the full original."""
    pytest.importorskip("mcp")
    import re

    from mcp.shared.memory import create_connected_server_and_client_session

    from tooltrim.integrations.mcp import build_tool_server

    async def run():
        server = build_tool_server(max_tokens=80)
        async with create_connected_server_and_client_session(server) as client:
            listed = await client.list_tools()
            assert {t.name for t in listed.tools} == {"compress", "expand_tool_output"}

            big = _big_json()
            res = await client.call_tool(
                "compress", {"text": big, "query": "refund 4417"})
            text = res.content[0].text
            assert "4417" in text                 # on-topic fact retained
            assert len(text) < len(big)           # actually compressed

            m = re.search(r"ref=(\w+)", text)     # footer exposes an expand ref
            assert m, text
            expanded = await client.call_tool(
                "expand_tool_output", {"ref": m.group(1)})
            full = expanded.content[0].text
            assert len(full) > len(text)          # got the full original back
            assert "row 200" in full              # a row dropped from the extract

    asyncio.run(run())


def test_serve_compress_passes_through_small_text():
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session

    from tooltrim.integrations.mcp import build_tool_server

    async def run():
        server = build_tool_server(max_tokens=512)
        async with create_connected_server_and_client_session(server) as client:
            res = await client.call_tool("compress", {"text": "just a short note"})
            assert res.content[0].text == "just a short note"   # untouched

    asyncio.run(run())
