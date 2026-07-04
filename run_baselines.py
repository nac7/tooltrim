"""Comparative faithfulness benchmark: tooltrim vs the obvious baselines.

Where ``run_faithfulness.py`` scores tooltrim against full-context, this scores
tooltrim against the *alternatives* — naive truncation, query-aware RAG top-k,
and (optionally) LLMLingua-2 — on identical cases and budgets, with a paired
McNemar significance test. This is the table a paper's baseline/Pareto section
is built from.

Offline (default, no keys, no model downloads):
    python run_baselines.py

Add optional baselines (installed separately):
    python run_baselines.py --methods full,truncate-head,rag-topk,rag-embed,tooltrim
    python run_baselines.py --with-llmlingua        # needs: pip install llmlingua

Real LLM judge + export:
    python run_baselines.py --model groq --cache .cache/groq.json --out benchmarks/runs
    python run_baselines.py --budgets 128,256,400,800
"""

from __future__ import annotations

import argparse
import os

from eval import (
    CachedModel,
    default_baselines,
    evaluate_methods,
    format_methods_report,
    get_baseline,
    get_model,
    methods_to_csv,
    methods_to_markdown,
)


def _build_methods(spec: str | None, with_llmlingua: bool):
    if spec:
        names = [s.strip() for s in spec.split(",") if s.strip()]
        methods = [get_baseline(n) for n in names]
    else:
        methods = default_baselines()
    if with_llmlingua and not any(m.name == "llmlingua-2" for m in methods):
        methods.insert(-1, get_baseline("llmlingua-2"))  # before tooltrim
    # Warn (don't crash) on anything unavailable — it will be skipped.
    for m in methods:
        if hasattr(m, "available") and not m.available():
            print(f"[skip] baseline '{m.name}' is not installed - omitting it.")
    return methods


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="offline",
                        help="offline | claude | openai | groq | ollama")
    parser.add_argument("--model-id", default=None,
                        help="override the provider's model id")
    parser.add_argument("--methods", default=None,
                        help="comma-separated baselines (default: full,"
                             "truncate-head,truncate-tail,rag-topk,tooltrim)")
    parser.add_argument("--with-llmlingua", action="store_true",
                        help="also run LLMLingua-2 (needs: pip install llmlingua)")
    parser.add_argument("--budgets", default="128,256,400,800",
                        help="comma-separated token budgets")
    parser.add_argument("--limit", type=int, default=None,
                        help="score only the first N cases (keeps a real-LLM run "
                             "inside free-tier token caps; full-context pass is "
                             "the token hog)")
    parser.add_argument("--reference", default="tooltrim",
                        help="method to run significance tests against")
    parser.add_argument("--cache", default=None,
                        help="path to a JSON answer cache (avoids re-spending)")
    parser.add_argument("--out", default=None,
                        help="directory to write comparison.md and comparison.csv")
    args = parser.parse_args()

    budgets = tuple(int(b) for b in args.budgets.split(",") if b.strip())
    model = get_model(args.model, model_id=args.model_id)
    if args.cache:
        model = CachedModel(model, args.cache)
    name = getattr(model, "name", args.model)

    methods = _build_methods(args.methods, args.with_llmlingua)
    cases = None
    if args.limit is not None:
        from eval import default_cases
        cases = default_cases()[:args.limit]
        print(f"[limit] scoring the first {len(cases)} cases")
    full, results = evaluate_methods(model, methods, cases=cases, budgets=budgets)

    print(format_methods_report(name, full, results, budgets,
                                reference=args.reference))

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        safe = name.replace("/", "_").replace(":", "_")
        md_path = os.path.join(args.out, f"comparison_{safe}.md")
        csv_path = os.path.join(args.out, f"comparison_{safe}.csv")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(methods_to_markdown(name, full, results, budgets,
                                        reference=args.reference))
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            f.write(methods_to_csv(name, full, results))
        print(f"\nwrote {md_path} and {csv_path}")


if __name__ == "__main__":
    main()
