from tooltrim import StreamingCompressor, compress_stream, count_tokens


def _log_lines(n=5000):
    for i in range(n):
        if i == 3210:
            yield "2026-06-28 ERROR disk full on /data; refund job 4417 aborted\n"
        else:
            yield f"2026-06-28 INFO heartbeat {i} ok\n"


def test_compress_stream_keeps_needle_within_budget():
    out = compress_stream(_log_lines(), max_tokens=120, query="refund 4417 disk")
    assert "4417" in out and "disk full" in out
    assert count_tokens(out) <= 120


def test_streaming_is_memory_bounded():
    # a tiny retention buffer must still surface the query-matching line out of
    # a huge stream — proving we don't hold the whole thing.
    sc = StreamingCompressor(max_tokens=120, query="refund 4417",
                             keep_chunks=16, head=4, tail=4, max_important=8)
    for line in _log_lines(20000):
        sc.feed(line)
    # buffers never exceeded their caps
    assert len(sc._topk) <= 16
    assert len(sc._head) <= 4
    assert len(sc._tail) <= 4
    out = sc.close()
    assert "4417" in out


def test_feed_handles_partial_lines_across_calls():
    sc = StreamingCompressor(max_tokens=80, query="secret code")
    blob = ("noise line one\n" * 50
            + "the secret code is ALPHA-9\n"
            + "noise line two\n" * 50)
    # feed one character at a time — partial lines must be reassembled
    for ch in blob:
        sc.feed(ch)
    out = sc.close()
    assert "ALPHA-9" in out


def test_stream_no_query_falls_back_to_head_tail():
    lines = [f"line number {i}\n" for i in range(500)]
    out = compress_stream(lines, max_tokens=60)
    assert "line number 0" in out          # head kept
    assert count_tokens(out) <= 60


def test_stream_accepts_bytes():
    chunks = [b"info ok\n", b"ERROR something broke\n", b"info ok\n"]
    out = compress_stream(chunks, max_tokens=60, query="error")
    assert "ERROR something broke" in out


def test_original_tokens_tracked():
    sc = StreamingCompressor(max_tokens=80, query="x")
    sc.feed("aaaa bbbb\ncccc dddd\n")
    sc.close()
    assert sc.original_tokens > 0
