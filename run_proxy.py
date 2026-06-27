"""Run the tooltrim OpenAI-compatible compression proxy.

    python run_proxy.py                          # -> https://api.openai.com/v1
    python run_proxy.py --upstream https://api.groq.com/openai/v1
    python run_proxy.py --port 8800 --max-tokens 400

Then point any OpenAI-compatible client at it — no app changes:

    from openai import OpenAI
    client = OpenAI(base_url="http://127.0.0.1:8800/v1", api_key="<upstream key>")

Every role:"tool" / role:"function" message is compressed before forwarding.
"""

from __future__ import annotations

import argparse

from tooltrim.proxy import serve


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8800)
    p.add_argument("--max-tokens", type=int, default=512,
                   help="token budget per tool result")
    p.add_argument("--upstream", default=None,
                   help="upstream base url (default: $TOOLTRIM_UPSTREAM_BASE_URL "
                        "or https://api.openai.com/v1)")
    args = p.parse_args()
    serve(host=args.host, port=args.port,
          max_tokens=args.max_tokens, upstream_base=args.upstream)


if __name__ == "__main__":
    main()
