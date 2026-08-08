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


def test_json_small_entity_array_keeps_every_item_at_tight_budget():
    # A multi-item order is an enumerable entity: at a tight budget the compressor
    # must NOT decimate it to one item (the retail failure mode). It keeps all
    # line items and finds the budget by shortening free-text prose instead.
    order = {"order_id": "W12345", "status": "pending",
             "items": [{"item_id": f"item_{i:04d}", "qty": 1,
                        "note": "a fairly long free-text description that can be trimmed " * 2}
                       for i in range(6)]}
    out = json_.compress(json.dumps(order), query="cancel order W12345", max_tokens=256)
    assert count_tokens(out) <= 256
    # every item_id survives — none of the 6 line items were dropped
    for i in range(6):
        assert f"item_{i:04d}" in out


def test_json_does_not_truncate_identifier_strings():
    # Whitespace-free identifiers are load-bearing and cheap; they must survive
    # even when the budget forces short max_str_len on prose fields.
    rec = {"item_id": "item_1008292230", "sku": "ABC-999-XYZ-01234567",
           "desc": "this is a long human readable description " * 8}
    out = json_.compress(json.dumps(rec), query=None, max_tokens=48)
    assert "item_1008292230" in out          # id kept whole, no elision marker
    assert "ABC-999-XYZ-01234567" in out


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


def test_json_tight_budget_on_nested_object_stays_valid_and_keeps_scalars():
    # Regression: a deeply-nested single object (e.g. a tau-bench retail order)
    # at a tight budget must degrade to still-valid JSON that keeps top-level
    # scalar fields the agent needs (like "status"), NOT cliff to the text
    # fallback that shatters the object on commas and drops those fields.
    # Previously this collapsed to ~21 tokens of just the id fields.
    order = {
        "order_id": "#W2611340", "user_id": "james_li_5688",
        "address": {"address1": "215 River Road", "city": "New York",
                    "state": "NY", "zip": "10083", "country": "USA"},
        "items": [{"name": "Water Bottle", "product_id": "8310926033",
                   "price": 47.84, "options": {"capacity": "1000ml", "color": "blue"}},
                  {"name": "Office Chair", "product_id": "4794339885",
                   "price": 488.81, "options": {"material": "fabric", "color": "black"}}],
        "status": "processed",
        "payment_history": [{"transaction_type": "payment", "amount": 536.65}],
    }
    out = json_.compress(json.dumps(order), query="cancel pending order status",
                         max_tokens=128)
    assert count_tokens(out) <= 128
    # Still parses as JSON (the fallback would have shattered it):
    json.loads(out)
    # Keeps the fields an agent gates decisions on, not just the ids:
    assert '"status"' in out and "processed" in out
    # And actually fills a reasonable share of the budget (no 21-token collapse):
    assert count_tokens(out) >= 64


def test_json_id_keyed_record_map_keeps_whole_records_not_elided_shells():
    # Regression (tau-bench retail `get_product_details`): a dict used as a
    # collection *keyed by item_id* has no sampling lever of its own, so the
    # ladder used to shrink it via depth elision — collapsing every variant to
    # {"…": "(4 keys elided)"} and wiping all 20 item_ids, `available` flags,
    # options and prices at once. The agent then cannot name a variant to
    # exchange into, and the task fails deterministically (0/5 across trials).
    # A *subset of whole records* must win over *every record, gutted*.
    variants = {
        f"{9690244451 + i}": {
            "item_id": f"{9690244451 + i}",
            "options": {"switch type": "clicky" if i % 2 else "tactile",
                        "backlight": "RGB", "size": "full size"},
            "available": i == 1,
            "price": 236.51 + i,
        }
        for i in range(20)
    }
    product = {"name": "Mechanical Keyboard", "product_id": "1656367028",
               "variants": variants}
    out = json_.compress(json.dumps(product), query=None, max_tokens=128)
    assert count_tokens(out) <= 128
    parsed = json.loads(out)  # must stay valid JSON

    kept = [v for k, v in parsed["variants"].items() if isinstance(v, dict)]
    assert kept, "every variant was elided to a shell"
    # At least one variant survives *complete* — id, options and availability are
    # all load-bearing for the exchange step.
    assert any("item_id" in v and "available" in v and "options" in v for v in kept)
    # And the elision is accounted for rather than silent.
    assert "more entries" in out


def test_json_never_emits_unparseable_output_at_any_budget():
    # The old last resort split the rendered JSON on commas and rejoined the
    # fragments, producing output that no longer parsed at all. Whatever the
    # budget, a JSON input must yield JSON output.
    order = {
        "order_id": "#W2378156", "user_id": "yusuf_rossi_9620",
        "status": "exchange requested",
        "address": {"address1": "763 Broadway", "city": "Philadelphia",
                    "state": "PA", "zip": "19122", "country": "USA"},
        "items": [{"name": f"Item {i}", "item_id": f"{4202497723 + i}",
                   "price": 342.81, "options": {"color": "blue", "size": "L"}}
                  for i in range(5)],
        "fulfillments": [{"tracking_id": ["843453391169"], "item_ids": ["4202497723"]}],
        "payment_history": [{"transaction_type": "payment", "amount": 1449.03}],
        "exchange_payment_method_id": "credit_card_9513926",
        "exchange_price_difference": -48.31,
    }
    raw = json.dumps(order)
    for budget in (8, 16, 32, 64, 128, 256, 512):
        out = json_.compress(raw, query=None, max_tokens=budget)
        json.loads(out)  # raises if the output was shattered
        assert count_tokens(out) <= budget


def test_json_flat_lookup_map_is_sampled_by_relevance_not_alphabetical_order():
    # Regression (tau-bench retail `list_all_product_types`): a 50-entry
    # name -> id lookup table is a *collection*, not a record. It used to keep
    # neither branch's protection: every key was retained, the budget was blown,
    # and the last-resort skeleton kept whichever entries came first — so a task
    # about a "water bottle" saw a catalogue that stopped at "Electric Toothbrush"
    # and concluded the store had none.
    catalogue = {name: str(1000000000 + i) for i, name in enumerate([
        "Action Camera", "Air Purifier", "Backpack", "Bicycle", "Bookshelf",
        "Coffee Maker", "Cycling Helmet", "Desk Lamp", "Digital Camera",
        "Dumbbell Set", "E-Reader", "Electric Kettle", "Electric Toothbrush",
        "Fleece Jacket", "Garden Hose", "Grill", "Headphones", "Hiking Boots",
        "Indoor Security Camera", "Jigsaw Puzzle", "Laptop", "LED Light Bulb",
        "Luggage Set", "Makeup Kit", "Mechanical Keyboard", "Notebook",
        "Office Chair", "Patio Umbrella", "Perfume", "Pet Bed", "Portable Charger",
        "Running Shoes", "Skateboard", "Smart Thermostat", "Smart Watch",
        "Smartphone", "Sneakers", "Espresso Machine", "Sunglasses", "Tablet",
        "Tea Kettle", "Yoga Mat", "Wall Clock", "Wireless Earbuds", "Water Bottle",
        "Vacuum Cleaner", "T-Shirt", "Bluetooth Speaker", "Air Fryer", "Bath Mat",
    ])}
    raw = json.dumps(catalogue)
    out = json_.compress(raw, query="swap out my water bottle and desk lamp",
                         max_tokens=128)
    assert count_tokens(out) <= 128
    kept = json.loads(out)
    # The two products the task is actually about survive, though they sit at
    # opposite ends of the alphabet and "Water Bottle" is 45th.
    assert "Water Bottle" in kept and "Desk Lamp" in kept


def test_json_truncation_is_never_silent():
    # A truncated result that does not *say* it was truncated is worse than a
    # small one: the agent reads it as the complete answer and never re-queries.
    # The elision marker must be reserved for, not appended only if it happens to
    # fit after the budget is already spent.
    catalogue = {f"Product Type Number {i}": str(1000000000 + i) for i in range(60)}
    raw = json.dumps(catalogue)
    for budget in (16, 24, 32, 64, 128, 256):
        out = json_.compress(raw, query=None, max_tokens=budget)
        assert count_tokens(out) <= budget
        parsed = json.loads(out)
        if isinstance(parsed, dict) and len(parsed) >= len(catalogue):
            continue  # nothing was dropped, so no marker is required
        assert "…" in out, f"silent truncation at budget {budget}: {out!r}"


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
