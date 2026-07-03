"""Compress a huge stream with bounded memory.

`compress_stream` consumes an iterable (file, subprocess stdout, HTTP stream)
line by line and never holds the whole thing — only small head/tail/top-K/
important-line buffers — then fits the survivors to the budget.

Run:  python examples/07_streaming.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tooltrim import compress_stream, count_tokens


def huge_log(n=500_000):
    """A half-million-line log we never materialize in memory."""
    for i in range(n):
        if i == 371_402:
            yield "2026-06-28 12:04:11 ERROR disk full on /data; refund job 4417 aborted\n"
        else:
            yield f"2026-06-28 12:00:00 INFO worker heartbeat {i} ok\n"


def main():
    out = compress_stream(huge_log(), max_tokens=200, query="disk error refund 4417")
    approx_in = 500_000 * 9  # ~9 tokens/line, never counted in full
    print(f"streamed ~{approx_in:,} tokens of logs with bounded memory")
    print(f"compressed result: {count_tokens(out)} tokens\n")
    print("=== what the agent sees ===")
    print(out)


if __name__ == "__main__":
    main()
