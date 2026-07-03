"""Drop tooltrim into an existing LlamaIndex agent — one line per tool.

``compress_llamaindex_tool`` wraps any LlamaIndex tool and returns one with the
*same* name / description / args schema, so the agent calls it unchanged — but
the ``ToolOutput.content`` the LLM reads is tooltrim-compressed first. The
structured ``raw_output`` is preserved, and the full text stays retrievable via
the expand-store ref.

Run (needs `llama-index-core`):  python examples/05_llamaindex_tool.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tooltrim import count_tokens


def _long_docs(topic: str) -> str:
    """Imagine this scrapes a long documentation page."""
    sections = [f"## Section {i}\n" + ("filler. " * 60) for i in range(40)]
    sections[17] = ("## Rate limits\nThe API allows 5000 requests/hour per key; "
                    "bursts up to 100/sec are throttled with HTTP 429.")
    return "\n\n".join(sections)


def main():
    try:
        from llama_index.core.tools import FunctionTool
    except ImportError:
        print("Install `llama-index-core` for this demo:  pip install tooltrim[llamaindex]")
        return

    from tooltrim.integrations import compress_llamaindex_tool

    # An ordinary LlamaIndex tool you already have.
    base = FunctionTool.from_defaults(
        fn=_long_docs, name="fetch_documentation",
        description="Fetch documentation about a topic.")

    # One line: same tool, compressed + query-aware off its `topic` argument.
    fetch = compress_llamaindex_tool(
        base, max_tokens=300, query_from=lambda topic: topic)

    raw = base.call(topic="rate limits")
    out = fetch.call(topic="rate limits")

    print(f"tool name preserved: {fetch.metadata.name!r}")
    print(f"raw content:        {count_tokens(raw.content):>6,} tokens")
    print(f"compressed content: {count_tokens(out.content):>6,} tokens "
          f"({(1 - count_tokens(out.content) / count_tokens(raw.content)) * 100:.0f}% smaller)")
    print(f"raw_output preserved: {out.raw_output == raw.raw_output}\n")
    print("=== compressed content the agent sees ===")
    print(out.content)


if __name__ == "__main__":
    main()
