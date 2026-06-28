import json

import pytest

from tooltrim import ToolCompressor, query_scope
from tooltrim.integrations.langchain import (
    _compressed_output,
    _resolve_query,
)


def _big_output():
    return json.dumps([{"id": i, "note": f"row {i}"} for i in range(400)]
                      + [{"id": 999, "note": "the refund went to customer 4417"}])


def test_compressed_output_shrinks_strings_keeps_fact():
    tc = ToolCompressor(max_tokens=80, add_footer=False)
    out = _compressed_output(_big_output(), tc, "refund customer 4417")
    assert isinstance(out, str)
    assert "4417" in out
    assert len(out) < len(_big_output())


def test_compressed_output_passes_through_non_strings():
    tc = ToolCompressor(max_tokens=80, add_footer=False)
    payload = {"rows": list(range(1000))}
    assert _compressed_output(payload, tc, "anything") is payload


def test_resolve_query_prefers_query_from():
    assert _resolve_query(lambda url: f"about {url}", {"url": "x"}) == "about x"


def test_resolve_query_falls_back_to_ambient():
    with query_scope("ambient q"):
        assert _resolve_query(None, {}) == "ambient q"


def test_resolve_query_swallows_query_from_errors():
    def boom(**kwargs):
        raise ValueError("nope")

    # falls back to ambient (None here) instead of raising
    assert _resolve_query(boom, {"x": 1}) is None


def test_compress_langchain_tool_wraps_and_compresses():
    pytest.importorskip("langchain_core")
    from langchain_core.tools import tool

    from tooltrim.integrations import compress_langchain_tool

    @tool
    def lookup(query: str) -> str:
        """Look up records."""
        return _big_output()

    wrapped = compress_langchain_tool(
        lookup, max_tokens=80, query_from=lambda query: query)

    # identity preserved so the agent calls it the same way
    assert wrapped.name == "lookup"
    assert "Look up records." in wrapped.description

    out = wrapped.invoke({"query": "refund customer 4417"})
    assert "4417" in out
    assert len(out) < len(_big_output())
