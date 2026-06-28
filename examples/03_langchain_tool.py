"""Drop tooltrim into an existing LangChain agent — one line per tool.

You already have LangChain tools. ``compress_langchain_tool`` wraps any of them
and returns a tool with the *same* name / description / args schema, so the agent
calls it unchanged — but the (string) result is tooltrim-compressed before it
ever lands in the scratchpad. The relevance query is taken from the tool's own
arguments via ``query_from``.

Run (needs `langchain-core`):  python examples/03_langchain_tool.py
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
        from langchain_core.tools import tool
    except ImportError:
        print("Install `langchain-core` for this demo:  pip install tooltrim[langchain]")
        return

    from tooltrim.integrations import compress_langchain_tool

    # An ordinary LangChain tool you already have.
    @tool
    def fetch_documentation(topic: str) -> str:
        """Fetch documentation about a topic."""
        return _long_docs(topic)

    # One line: same tool, compressed + query-aware off its `topic` argument.
    fetch = compress_langchain_tool(
        fetch_documentation, max_tokens=300, query_from=lambda topic: topic)

    raw = fetch_documentation.invoke({"topic": "rate limits"})
    out = fetch.invoke({"topic": "rate limits"})

    print(f"tool name preserved: {fetch.name!r}")
    print(f"raw output:        {count_tokens(raw):>6,} tokens")
    print(f"compressed output: {count_tokens(out):>6,} tokens "
          f"({(1 - count_tokens(out) / count_tokens(raw)) * 100:.0f}% smaller)\n")
    print("=== compressed output the agent sees ===")
    print(out)


if __name__ == "__main__":
    main()
