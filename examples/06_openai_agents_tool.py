"""Drop tooltrim into an OpenAI Agents SDK agent — one line per tool.

``compress_openai_agents_tool`` wraps any Agents SDK ``FunctionTool`` and returns
one with the *same* name / description / JSON schema / guardrails — only the
string it returns to the model is tooltrim-compressed, query-aware.

Run (needs `openai-agents`):  python examples/06_openai_agents_tool.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tooltrim import count_tokens


def _long_docs(topic: str) -> str:
    sections = [f"## Section {i}\n" + ("filler. " * 60) for i in range(40)]
    sections[17] = ("## Rate limits\nThe API allows 5000 requests/hour per key; "
                    "bursts up to 100/sec are throttled with HTTP 429.")
    return "\n\n".join(sections)


def main():
    try:
        from agents import FunctionTool
    except ImportError:
        print("Install `openai-agents` for this demo:  pip install tooltrim[openai-agents]")
        return

    from tooltrim.integrations import compress_openai_agents_tool

    # A FunctionTool (here with a plain invoker so the demo runs without the
    # agent runtime; @function_tool-decorated tools wrap identically).
    async def invoke(ctx, input_str):
        import json
        return _long_docs(json.loads(input_str)["topic"])

    base = FunctionTool(
        name="fetch_documentation", description="Fetch documentation about a topic.",
        params_json_schema={"type": "object",
                            "properties": {"topic": {"type": "string"}},
                            "required": ["topic"]},
        on_invoke_tool=invoke)

    fetch = compress_openai_agents_tool(
        base, max_tokens=300, query_from=lambda topic: topic)

    args = '{"topic": "rate limits"}'
    raw = asyncio.run(base.on_invoke_tool(None, args))
    out = asyncio.run(fetch.on_invoke_tool(None, args))

    print(f"tool name preserved: {fetch.name!r}")
    print(f"raw output:        {count_tokens(raw):>6,} tokens")
    print(f"compressed output: {count_tokens(out):>6,} tokens "
          f"({(1 - count_tokens(out) / count_tokens(raw)) * 100:.0f}% smaller)\n")
    print("=== compressed output the agent sees ===")
    print(out)


if __name__ == "__main__":
    main()
