from tooltrim.metrics import Metrics


def test_records_and_renders_prometheus():
    m = Metrics()
    m.record(messages=1, tokens_in=14415, tokens_out=26, latency_s=0.2)
    m.record(messages=2, tokens_in=1000, tokens_out=100, latency_s=0.1)

    snap = m.snapshot()
    assert snap["tooltrim_requests_total"] == 2
    assert snap["tooltrim_messages_compressed_total"] == 3
    assert snap["tooltrim_tokens_in_total"] == 15415
    assert snap["tooltrim_tokens_out_total"] == 126
    assert snap["tooltrim_tokens_saved_total"] == 15289
    assert snap["tooltrim_request_latency_seconds_count"] == 2

    text = m.render()
    assert "# TYPE tooltrim_requests_total counter" in text
    assert "tooltrim_tokens_saved_total 15289" in text
    assert "# TYPE tooltrim_request_latency_seconds summary" in text
    assert "tooltrim_request_latency_seconds_count 2" in text


def test_fail_open_and_upstream_error_counters():
    m = Metrics()
    m.record(fail_open=True)
    m.record(upstream_error=True)
    snap = m.snapshot()
    assert snap["tooltrim_fail_open_total"] == 1
    assert snap["tooltrim_upstream_errors_total"] == 1


def test_proxy_metrics_endpoint_serves_prometheus():
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from tooltrim import ToolCompressor
    from tooltrim.metrics import Metrics
    from tooltrim.proxy import make_handler

    metrics = Metrics()
    metrics.record(messages=1, tokens_in=100, tokens_out=10)
    handler = make_handler(ToolCompressor(), "https://example.invalid/v1",
                           verbose=False, metrics=metrics)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        port = server.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics") as r:
            body = r.read().decode()
        assert "tooltrim_tokens_saved_total 90" in body
    finally:
        server.shutdown()
