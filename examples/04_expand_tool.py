"""Give the agent an `expand` tool so it can recover full output on demand.

tooltrim compresses aggressively; the expand tool makes that safe — when the
compact extract is missing a detail, the agent calls expand(ref) to page through
the full original. This makes compression a safe default, not a lossy gamble.

Run:  python examples/04_expand_tool.py   (offline, no API key)
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tooltrim import ToolCompressor

# One shared compressor so the tool and its expand store line up.
TC = ToolCompressor(max_tokens=200)


def get_orders(customer_id: str) -> str:
    orders = [{"id": i, "status": "shipped", "total": i * 10} for i in range(300)]
    orders[42]["status"] = "REFUNDED"
    return json.dumps({"customer": customer_id, "orders": orders})


if __name__ == "__main__":
    # 1) The tool returns a compressed result (with a ref in the footer).
    res = TC.compress(get_orders("C-1007"), query="which order was refunded?")
    print("=== compressed tool result given to the model ===")
    print(res.text[-300:])
    print(f"\n[{res.original_tokens} -> {res.compressed_tokens} tokens]")

    # 2) Register this tool spec alongside your real tools (OpenAI shown).
    print("\n=== expand tool the agent can call ===")
    print(json.dumps(TC.expand_tool_spec(style="openai"), indent=2))

    # 3) When the model emits expand_tool_output(ref=...), run the handler:
    print("\n=== handler output for the model's expand call ===")
    print(TC.handle_expand(res.ref, page_chars=400))

    # Anthropic Messages API: TC.expand_tool_spec(style="anthropic")
    # Cheap models / SDK tool-runners: wrap TC.handle_expand in your tool fn.
