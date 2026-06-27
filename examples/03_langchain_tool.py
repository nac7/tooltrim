"""Use tooltrim with a LangChain tool.

Because tooltrim wraps the underlying function, the compressed output flows
through whatever framework you use. Here we compress before LangChain's @tool
sees the result, so the agent's scratchpad stays small.

Run (needs `langchain-core`):  python examples/03_langchain_tool.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tooltrim import ToolCompressor, wrap_tool

TC = ToolCompressor(max_tokens=300)


def _raw_fetch_docs(topic: str) -> str:
    """Imagine this scrapes a long documentation page."""
    sections = [f"## Section {i}\n" + ("filler. " * 60) for i in range(40)]
    sections[17] = ("## Rate limits\nThe API allows 5000 requests/hour per key; "
                    "bursts up to 100/sec are throttled with HTTP 429.")
    return "\n\n".join(sections)


# Compress + make it query-aware off the tool's own argument.
fetch_docs = wrap_tool(_raw_fetch_docs, compressor=TC,
                       query_from=lambda topic: topic)


def main():
    try:
        from langchain_core.tools import tool
    except ImportError:
        print("Install `langchain-core` for the full demo. Compressed output:\n")
        print(fetch_docs("rate limits"))
        return

    @tool
    def fetch_documentation(topic: str) -> str:
        """Fetch documentation about a topic (auto-compressed by tooltrim)."""
        return fetch_docs(topic)

    out = fetch_documentation.invoke({"topic": "rate limits"})
    print("=== LangChain tool output (compressed) ===")
    print(out)


if __name__ == "__main__":
    main()
