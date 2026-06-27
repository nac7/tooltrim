import json

from tooltrim import ToolCompressor, compressed_tool, query_scope, set_query, wrap_tool


def _big_text(needle="NEEDLE-9000"):
    paras = [f"Filler paragraph {i} about unrelated topics." for i in range(200)]
    paras[88] = f"Important: the access token is {needle}."
    return "\n\n".join(paras)


def test_compressed_tool_returns_string_and_compresses():
    @compressed_tool(max_tokens=60)
    def fetch():
        return _big_text()

    out = fetch()
    assert isinstance(out, str)
    assert len(out) < len(_big_text())


def test_query_from_drives_relevance():
    @compressed_tool(max_tokens=50, query_from=lambda q: q)
    def search(q):
        return _big_text()

    out = search("access token")
    assert "NEEDLE-9000" in out


def test_query_scope_sets_ambient_query():
    @compressed_tool(max_tokens=50)
    def fetch():
        return _big_text()

    with query_scope("access token"):
        out = fetch()
    assert "NEEDLE-9000" in out


def test_set_query_global():
    @compressed_tool(max_tokens=50)
    def fetch():
        return _big_text()

    set_query("access token")
    try:
        assert "NEEDLE-9000" in fetch()
    finally:
        set_query(None)


def test_return_result_gives_metrics():
    @compressed_tool(max_tokens=60, return_result=True)
    def fetch():
        return _big_text()

    res = fetch()
    assert res.saved_tokens > 0
    assert res.ref is not None


def test_shared_compressor_shares_store():
    tc = ToolCompressor(max_tokens=60)

    @compressed_tool(compressor=tc, return_result=True)
    def fetch():
        return _big_text()

    res = fetch()
    assert tc.expand(res.ref) == _big_text()


def test_wrap_tool_functional_form():
    def raw():
        return json.dumps([{"i": i} for i in range(500)])

    wrapped = wrap_tool(raw, max_tokens=80)
    assert len(wrapped()) < len(raw())
