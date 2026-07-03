import asyncio
import json

import pytest


def _big_output():
    return json.dumps([{"id": i, "note": f"row {i}"} for i in range(400)]
                      + [{"id": 999, "note": "the refund went to customer 4417"}])


def _make_base_tool():
    # Build a FunctionTool whose on_invoke_tool ignores ctx, so we can drive it
    # in a test without constructing a full SDK ToolContext.
    from agents import FunctionTool

    async def invoke(ctx, input_str):
        return _big_output()

    return FunctionTool(
        name="lookup",
        description="Look up records.",
        params_json_schema={"type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"]},
        on_invoke_tool=invoke,
    )


def test_compress_openai_agents_tool_wraps_and_compresses():
    pytest.importorskip("agents")
    from tooltrim.integrations import compress_openai_agents_tool

    base = _make_base_tool()
    wrapped = compress_openai_agents_tool(
        base, max_tokens=80, query_from=lambda query: query)

    # identity + config preserved (dataclasses.replace keeps every other field)
    assert wrapped.name == "lookup"
    assert wrapped.description == "Look up records."
    assert wrapped.params_json_schema == base.params_json_schema
    assert wrapped.strict_json_schema == base.strict_json_schema

    out = asyncio.run(wrapped.on_invoke_tool(None, '{"query": "refund customer 4417"}'))
    assert "4417" in out                    # needle kept
    assert len(out) < len(_big_output())    # compressed


def test_compress_openai_agents_tool_passes_through_non_string():
    pytest.importorskip("agents")
    from agents import FunctionTool

    from tooltrim.integrations import compress_openai_agents_tool

    async def invoke(ctx, input_str):
        return {"structured": True}          # non-string output

    base = FunctionTool(name="t", description="d",
                        params_json_schema={"type": "object", "properties": {},
                                            "required": []},
                        on_invoke_tool=invoke)
    wrapped = compress_openai_agents_tool(base, max_tokens=80)
    out = asyncio.run(wrapped.on_invoke_tool(None, "{}"))
    assert out == {"structured": True}       # untouched
