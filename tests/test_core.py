import json

from tooltrim import CompressionResult, ToolCompressor


def _big_json(n=400):
    return json.dumps([{"id": i, "v": "x" * 40} for i in range(n)])


def test_small_output_passes_through():
    tc = ToolCompressor(max_tokens=512)
    res = tc.compress("short and sweet")
    assert res.compressed is False
    assert res.text == "short and sweet"
    assert res.ref is None
    assert res.saved_tokens == 0


def test_large_output_is_compressed_within_budget():
    tc = ToolCompressor(max_tokens=200, add_footer=False)
    res = tc.compress(_big_json())
    assert res.compressed is True
    assert res.compressed_tokens <= 200
    assert res.saved_tokens > 0
    assert 0.0 < res.saved_ratio <= 1.0


def test_expand_returns_full_output():
    tc = ToolCompressor(max_tokens=120)
    original = _big_json()
    res = tc.compress(original)
    assert res.ref is not None
    assert tc.expand(res.ref) == original


def test_expand_slice():
    tc = ToolCompressor(max_tokens=120)
    original = _big_json()
    res = tc.compress(original)
    assert tc.expand(res.ref, start=0, length=10) == original[:10]


def test_footer_present_and_contains_ref():
    tc = ToolCompressor(max_tokens=120, add_footer=True)
    res = tc.compress(_big_json())
    assert "tooltrim" in res.text
    assert res.ref in res.text


def test_str_dunder_is_text():
    tc = ToolCompressor(max_tokens=120)
    res = tc.compress(_big_json())
    assert str(res) == res.text


def test_non_string_output_coerced():
    tc = ToolCompressor(max_tokens=120)
    res = tc.compress({"a": [1, 2, 3] * 500})
    assert res.compressed is True
    assert res.compressed_tokens <= 120 + 24  # body budget + footer headroom


def test_store_disabled_yields_no_ref():
    tc = ToolCompressor(max_tokens=120, store=None, add_footer=True)
    res = tc.compress(_big_json())
    assert res.ref is None
    # footer is skipped when there's no ref
    assert "tooltrim" not in res.text


def test_returns_compression_result_type():
    tc = ToolCompressor()
    assert isinstance(tc.compress("hi"), CompressionResult)


def test_expand_tool_spec_styles():
    tc = ToolCompressor()
    openai = tc.expand_tool_spec(style="openai")
    assert openai["type"] == "function"
    assert openai["function"]["name"] == tc.EXPAND_TOOL_NAME
    assert "ref" in openai["function"]["parameters"]["properties"]

    anthro = tc.expand_tool_spec(style="anthropic")
    assert anthro["name"] == tc.EXPAND_TOOL_NAME
    assert "input_schema" in anthro

    raw = tc.expand_tool_spec(style="raw")
    assert raw["name"] == tc.EXPAND_TOOL_NAME and "schema" in raw


def test_handle_expand_returns_full_output():
    tc = ToolCompressor(max_tokens=120)
    original = _big_json(80)  # > budget (so it compresses) but < one page
    res = tc.compress(original)
    assert len(original) < 8000
    assert tc.handle_expand(res.ref) == original


def test_handle_expand_paging():
    tc = ToolCompressor(max_tokens=120)
    original = _big_json(2000)
    res = tc.compress(original)
    page = tc.handle_expand(res.ref, page_chars=500)
    assert page.startswith(original[:500])
    assert tc.EXPAND_TOOL_NAME in page and "start=500" in page


def test_handle_expand_unknown_ref():
    tc = ToolCompressor(max_tokens=120)
    msg = tc.handle_expand("deadbeef")
    assert "no stored output" in msg and "deadbeef" in msg
