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


def test_json_budget_is_a_cap_not_a_target():
    # One genuinely-relevant record among keyword-stuffed distractors that repeat
    # the query term. A larger budget must NOT pad the output with those near-zero
    # noise records: selection follows the relevance cliff, so the compressed size
    # stays flat as the budget grows (the fix that stopped accuracy decaying with
    # budget). Regression for the budget-as-target padding bug.
    data = [{"id": i, "note": "account account order ledger account"} for i in range(300)]
    data[7]["note"] = "approved limit for the premium account is 25000 dollars"
    blob = json.dumps({"total": 300, "results": data})
    outs = {b: json_.compress(blob, query="approved limit premium account", max_tokens=b)
            for b in (128, 256, 800)}
    for b, out in outs.items():
        assert "25000 dollars" in out, f"lost the needle at budget {b}"
    # bigger budget did not reintroduce noise: output size is stable, not inflated
    assert count_tokens(outs[800]) <= count_tokens(outs[128]) + 5


def test_json_aggregation_keeps_the_whole_relevant_cluster():
    # When several records are all genuinely relevant (they share the query terms),
    # the relevance cliff keeps the whole cluster, not just the single best — so
    # aggregation queries stay answerable.
    data = [{"id": i, "note": f"unrelated filler row {i} about latency"} for i in range(200)]
    for oid in (7001, 7002, 7003, 7004):
        data.append({"id": oid, "note": f"failed payment for order {oid} processor stripe"})
    blob = json.dumps({"results": data})
    out = json_.compress(blob, query="failed payment order", max_tokens=256)
    for oid in (7001, 7002, 7003, 7004):
        assert str(oid) in out, f"dropped a relevant record {oid}"


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


def test_text_compresses_single_line_blob():
    # Regression: a newline-free blob (minified JSON-in-a-string, one huge log
    # line) used to be one un-selectable chunk and passed through uncompressed.
    blob = ("routine heartbeat ok, nothing to see here. " * 1500
            + "the launch code is 1234-ALPHA.")
    assert count_tokens(blob) > 5000
    out = text.compress(blob, query="launch code", max_tokens=200)
    assert count_tokens(out) <= 200
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
