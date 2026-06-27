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
