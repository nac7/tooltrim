"""Run the tooltrim faithfulness-under-compression benchmark.

Offline (default, no keys):
    python run_faithfulness.py

Real LLM (needs the provider SDK + key):
    python run_faithfulness.py --model claude            # ANTHROPIC_API_KEY
    python run_faithfulness.py --model claude --model-id claude-sonnet-4-6
    python run_faithfulness.py --model openai            # OPENAI_API_KEY
    python run_faithfulness.py --model groq              # GROQ_API_KEY (free tier)
    python run_faithfulness.py --model ollama            # local, no key

Cache answers (so reruns don't re-spend tokens) and export results:
    python run_faithfulness.py --model claude --cache .cache/claude.json --out results
    python run_faithfulness.py --budgets 64,128,256,400,800
"""

from __future__ import annotations

import argparse
import os

from eval import (
    CachedModel,
    evaluate,
    format_report,
    get_model,
    to_csv,
    to_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="offline",
                        help="offline | claude | openai | groq | ollama")
    parser.add_argument("--model-id", default=None,
                        help="override the provider's model id")
    parser.add_argument("--budgets", default="128,256,400,800",
                        help="comma-separated token budgets")
    parser.add_argument("--cache", default=None,
                        help="path to a JSON answer cache (avoids re-spending)")
    parser.add_argument("--out", default=None,
                        help="directory to write <model>.md and <model>.csv")
    args = parser.parse_args()

    budgets = tuple(int(b) for b in args.budgets.split(",") if b.strip())
    model = get_model(args.model, model_id=args.model_id)
    if args.cache:
        model = CachedModel(model, args.cache)

    name = getattr(model, "name", args.model)
    full, results = evaluate(model, budgets=budgets)
    print(format_report(name, full, results))

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        safe = name.replace("/", "_").replace(":", "_")
        md_path = os.path.join(args.out, f"{safe}.md")
        csv_path = os.path.join(args.out, f"{safe}.csv")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(to_markdown(name, full, results))
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            f.write(to_csv(name, full, results))
        print(f"\nwrote {md_path} and {csv_path}")


if __name__ == "__main__":
    main()
