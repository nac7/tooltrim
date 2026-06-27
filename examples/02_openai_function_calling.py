"""Use tooltrim with OpenAI-style tool/function calling.

tooltrim is provider-agnostic — it just compresses the *string* your tool
returns before you hand it back to the model as a tool message. The same shape
works for the Anthropic Messages API (compress the tool_result content).

Run (needs `openai` + OPENAI_API_KEY):  python examples/02_openai_function_calling.py
The tool itself runs offline; only the model call needs a key.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tooltrim import ToolCompressor, query_scope

# One shared compressor so every tool shares the same expand store.
TC = ToolCompressor(max_tokens=400)


def get_orders(customer_id: str) -> str:
    """A tool that returns a big paginated JSON blob."""
    orders = [
        {"id": i, "customer": customer_id, "status": "shipped",
         "total": round(i * 12.5, 2), "items": [f"sku-{i}-{j}" for j in range(5)]}
        for i in range(300)
    ]
    orders[42]["status"] = "REFUNDED"
    return json.dumps({"customer": customer_id, "count": len(orders), "orders": orders})


def run_tool(name: str, args: dict, user_goal: str) -> str:
    raw = {"get_orders": get_orders}[name](**args)
    # Compress relative to what the user is trying to do.
    with query_scope(user_goal):
        res = TC.compress(raw, query=user_goal)
    print(f"[tooltrim] {name}: {res.original_tokens} -> {res.compressed_tokens} "
          f"tokens ({res.saved_ratio*100:.0f}% saved), full output ref={res.ref}")
    return res.text


def main():
    try:
        from openai import OpenAI
    except ImportError:
        print("Install `openai` to run the live model loop. Showing the tool result only:\n")
        print(run_tool("get_orders", {"customer_id": "C-1007"},
                       "did this customer get any refunds?"))
        return

    client = OpenAI()
    tools = [{
        "type": "function",
        "function": {
            "name": "get_orders",
            "description": "Get all orders for a customer.",
            "parameters": {
                "type": "object",
                "properties": {"customer_id": {"type": "string"}},
                "required": ["customer_id"],
            },
        },
    }]
    user_goal = "Did customer C-1007 get any refunds?"
    messages = [{"role": "user", "content": user_goal}]

    first = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, tools=tools
    )
    call = first.choices[0].message.tool_calls[0]
    messages.append(first.choices[0].message)

    # >>> tooltrim sits exactly here: compress the tool result before returning it <<<
    tool_result = run_tool(call.function.name,
                           json.loads(call.function.arguments), user_goal)
    messages.append({"role": "tool", "tool_call_id": call.id, "content": tool_result})

    final = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    print("\nAssistant:", final.choices[0].message.content)


if __name__ == "__main__":
    main()
