import json

from tooltrim.compressors import html, json_, logs, tabular, text
from tooltrim.tokens import count_tokens


def test_json_array_of_objects_keeps_schema_and_notes_omission():
    data = [{"id": i, "name": f"user{i}", "email": f"u{i}@x.com"} for i in range(500)]
    out = json_.compress(json.dumps(data), query=None, max_tokens=200)
    assert count_tokens(out) <= 200
    assert "more items" in out
    # schema preserved
    assert "name" in out and "email" in out


def test_json_query_prioritizes_relevant_items():
    data = [{"id": i, "note": f"row {i}"} for i in range(200)]
    data[137]["note"] = "the special needle marker"
    out = json_.compress(json.dumps(data), query="needle marker", max_tokens=120)
    assert "needle" in out


def test_html_extracts_text_drops_script_style():
    page = (
        "<html><head><style>.a{color:red}</style>"
        "<script>var x=1;evil()</script></head>"
        "<body><nav>menu menu menu</nav>"
        "<article><p>The capital of France is Paris.</p>"
        "<p>Unrelated filler paragraph about cats.</p></article>"
        "<footer>copyright</footer></body></html>"
    )
    out = html.compress(page, query="capital of France", max_tokens=40)
    assert "Paris" in out
    assert "evil" not in out and "color:red" not in out


def test_tabular_keeps_header_and_limits_rows():
    rows = ["name,age,city"] + [f"user{i},{20+i},city{i}" for i in range(1000)]
    out = tabular.compress("\n".join(rows), query=None, max_tokens=100)
    assert out.startswith("name,age,city")
    assert "more rows" in out
    assert count_tokens(out) <= 110  # header reserved a little headroom


def test_logs_dedup_and_keep_errors():
    lines = ["2026-06-27 INFO heartbeat"] * 200
    lines.insert(100, "2026-06-27 ERROR disk full on /data")
    out = logs.compress("\n".join(lines), query=None, max_tokens=80)
    assert "ERROR disk full" in out
    assert "(x" in out  # dedup marker for the repeated heartbeat


def test_text_query_extraction():
    paras = [f"Paragraph number {i} about gardening." for i in range(100)]
    paras[42] = "The launch code is 1234-ALPHA."
    out = text.compress("\n\n".join(paras), query="launch code", max_tokens=40)
    assert "1234-ALPHA" in out


def test_neighbor_context_included():
    from tooltrim.compressors._budget import fit_chunks

    chunks = [f"filler clause about gardening number {i}" for i in range(60)]
    chunks[30] = "the access code is ALPHA7"
    chunks[31] = "this value expires at NEIGHBORWORD midnight"
    # with neighbor context, the adjacent line is pulled in
    out = fit_chunks(chunks, "access code", max_tokens=60, neighbor=1)
    assert "ALPHA7" in out and "NEIGHBORWORD" in out
    # with neighbor disabled, only the matching line is kept
    out0 = fit_chunks(chunks, "access code", max_tokens=60, neighbor=0)
    assert "ALPHA7" in out0 and "NEIGHBORWORD" not in out0


def test_logs_keeps_context_around_error():
    lines = [f"2026-06-27 INFO step {i} completed ok" for i in range(200)]
    lines[100] = "2026-06-27 INFO opening file payments.dat for write"
    lines[101] = "2026-06-27 ERROR disk full on /data write aborted"
    out = logs.compress("\n".join(lines), query=None, max_tokens=120)
    assert "ERROR disk full" in out
    assert "opening file payments.dat" in out  # the preceding context line
