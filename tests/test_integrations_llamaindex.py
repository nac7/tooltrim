import json

import pytest


def _big_output():
    return json.dumps([{"id": i, "note": f"row {i}"} for i in range(400)]
                      + [{"id": 999, "note": "the refund went to customer 4417"}])


def test_compress_llamaindex_tool_wraps_and_compresses():
    pytest.importorskip("llama_index.core")
    from llama_index.core.tools import FunctionTool

    from tooltrim.integrations import compress_llamaindex_tool

    def lookup(query: str) -> str:
        """Look up records."""
        return _big_output()

    base = FunctionTool.from_defaults(fn=lookup, name="lookup",
                                      description="Look up records.")
    wrapped = compress_llamaindex_tool(
        base, max_tokens=80, query_from=lambda query: query)

    # identity preserved so the agent calls it the same way
    assert wrapped.metadata.name == "lookup"
    assert "Look up records." in wrapped.metadata.description

    out = wrapped.call(query="refund customer 4417")
    # compressed content, needle kept
    assert "4417" in out.content
    assert len(out.content) < len(_big_output())
    # structured raw_output preserved (the full original)
    assert out.raw_output == _big_output()
    assert out.is_error is False


def test_compress_llamaindex_tools_shares_compressor():
    pytest.importorskip("llama_index.core")
    from llama_index.core.tools import FunctionTool

    from tooltrim.integrations import compress_llamaindex_tools

    def a(query: str) -> str:
        """A."""
        return _big_output()

    def b(query: str) -> str:
        """B."""
        return _big_output()

    tools = compress_llamaindex_tools(
        [FunctionTool.from_defaults(fn=a), FunctionTool.from_defaults(fn=b)],
        max_tokens=80, query_from=lambda query: query)
    assert len(tools) == 2
    out = tools[0].call(query="refund 4417")
    assert "4417" in out.content
    assert out.raw_output == _big_output()  # full original preserved
