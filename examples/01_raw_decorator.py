"""Simplest possible adoption: decorate a tool, get compact output.

Run:  python examples/01_raw_decorator.py
No API keys, no network — pure local demo.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tooltrim import compressed_tool


@compressed_tool(max_tokens=120, query_from=lambda query, **_: query)
def fake_web_search(query: str) -> str:
    """Pretend this fetched a big, noisy web page."""
    boilerplate = "\n\n".join(
        f"Navigation / ad / cookie banner block #{i}. Subscribe now!"
        for i in range(60)
    )
    answer = "The Eiffel Tower is 330 metres tall (including antennas)."
    return boilerplate + "\n\n" + answer + "\n\n" + boilerplate


if __name__ == "__main__":
    out = fake_web_search("how tall is the Eiffel Tower")
    print("=== what the agent receives ===")
    print(out)
