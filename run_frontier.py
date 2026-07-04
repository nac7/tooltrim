"""Frontier-model matrix run: the accuracy/token Pareto for the paper.

Runs the comparative baseline grid (tooltrim vs full / truncation / RAG top-k /
...) across a matrix of >=4 frontier models on the *same* cases and budgets, then
aggregates a single cross-model artifact: a Pareto (accuracy vs tokens) and a
leaderboard, plus the paired tooltrim-vs-rag-topk significance test per model
(the paper's open question: does content-type structure separate tooltrim from
plain RAG selection under a real LLM judge?).

Each model is specified as ``provider:model_id`` where provider is one of
``claude | openai | groq | ollama``. Answers are cached per model under
``.cache/frontier/`` so a crashed or resumed run never re-spends tokens.

Examples
--------
Ready-to-fire default matrix (needs the matching API keys set):

    export ANTHROPIC_API_KEY=...  OPENAI_API_KEY=...  GROQ_API_KEY=...
    python run_frontier.py --limit 30 --budgets 128,256,800

Custom matrix + a single budget for a quick Pareto point:

    python run_frontier.py \
        --models claude:claude-haiku-4-5,openai:gpt-4o-mini,groq:llama-3.3-70b-versatile \
        --budgets 256

Dry run (prints the plan and the per-model key each provider needs, runs nothing):

    python run_frontier.py --dry-run
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

from eval import (
    CachedModel,
    default_baselines,
    evaluate_methods,
    get_baseline,
    get_model,
    mcnemar,
    methods_to_csv,
    methods_to_markdown,
)

# A sensible >=4-model frontier default spanning three providers. Override with
# --models. (Model ids are provider-current; swap as the frontier moves.)
DEFAULT_MATRIX = (
    "claude:claude-haiku-4-5",
    "claude:claude-sonnet-5",
    "openai:gpt-4o-mini",
    "groq:llama-3.3-70b-versatile",
)

# Which env var each provider reads (for --dry-run guidance / preflight).
PROVIDER_KEY = {
    "claude": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "ollama": None,  # local, no key
    "offline": None,  # deterministic BM25 control — no key, for smoke tests / a baseline row
}


def parse_spec(spec: str) -> Tuple[str, str]:
    provider, _, model_id = spec.partition(":")
    provider = provider.strip().lower()
    if provider not in PROVIDER_KEY:
        raise SystemExit(f"unknown provider in '{spec}' "
                         f"(use {'|'.join(PROVIDER_KEY)})")
    return provider, model_id.strip() or None  # type: ignore[return-value]


def preflight(specs: List[Tuple[str, str]]) -> List[str]:
    """Return human-readable warnings for any missing API keys (never raises)."""
    warns = []
    for provider, _ in specs:
        env = PROVIDER_KEY[provider]
        if env and not os.environ.get(env):
            warns.append(f"  [!] {provider}: ${env} is not set")
    return warns


def build_methods(with_llmlingua: bool):
    methods = default_baselines()
    if with_llmlingua and not any(m.name == "llmlingua-2" for m in methods):
        methods.insert(-1, get_baseline("llmlingua-2"))
    return [m for m in methods
            if not (hasattr(m, "available") and not m.available())]


def significance_row(results, model_name: str, budget: int,
                     a: str = "tooltrim", b: str = "rag-topk") -> str:
    """One line: is tooltrim != rag-topk at this budget (paired McNemar)?"""
    ra = next((r for r in results.get(a, []) if r.budget == budget), None)
    rb = next((r for r in results.get(b, []) if r.budget == budget), None)
    if not ra or not rb or not ra.mask or not rb.mask:
        return f"| {model_name} | {budget} | n/a | n/a | n/a |"
    _, _, p = mcnemar(list(ra.mask), list(rb.mask))  # (b, c, p_value)
    delta = (ra.accuracy - rb.accuracy) * 100
    sig = "**yes**" if p < 0.05 else "no"
    return (f"| {model_name} | {budget} | {delta:+.1f}pp | {p:.3f} | {sig} |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(DEFAULT_MATRIX),
                    help="comma-separated provider:model_id specs (>=4 for the paper)")
    ap.add_argument("--budgets", default="128,256,800",
                    help="comma-separated token budgets")
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N cases (controls real-LLM spend)")
    ap.add_argument("--with-llmlingua", action="store_true")
    ap.add_argument("--cache-dir", default=".cache/frontier")
    ap.add_argument("--out", default="benchmarks/runs")
    ap.add_argument("--summary", default="benchmarks/FRONTIER.md")
    ap.add_argument("--pareto-budget", type=int, default=256,
                    help="budget used for the cross-model Pareto/leaderboard")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan + key requirements and exit")
    args = ap.parse_args()

    specs = [parse_spec(s) for s in args.models.split(",") if s.strip()]
    budgets = tuple(int(b) for b in args.budgets.split(",") if b.strip())
    if len(specs) < 4:
        print(f"[note] {len(specs)} models specified; the paper's Pareto wants >=4.")

    warns = preflight(specs)
    print("Frontier matrix run")
    print(f"  models : {', '.join(f'{p}:{m}' for p, m in specs)}")
    print(f"  budgets: {budgets}   limit: {args.limit}   pareto@ {args.pareto_budget}")
    if warns:
        print("Missing keys (these models will fail unless set):")
        print("\n".join(warns))
    if args.dry_run:
        print("\n(dry run — nothing executed)")
        return

    from eval import default_cases
    cases = default_cases()[:args.limit] if args.limit is not None else None

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # rows for the cross-model leaderboard at the Pareto budget
    board: List[dict] = []
    sig_lines: List[str] = []

    for provider, model_id in specs:
        model = get_model(provider, model_id=model_id)
        name = getattr(model, "name", f"{provider}:{model_id}")
        safe = name.replace("/", "_").replace(":", "_")
        cache = os.path.join(args.cache_dir, f"{safe}.json")
        model = CachedModel(model, cache)

        methods = build_methods(args.with_llmlingua)
        print(f"\n=== {name} ===")
        full, results = evaluate_methods(model, methods, cases=cases, budgets=budgets)

        # per-model artifacts (reuse the existing formatters)
        md = os.path.join(args.out, f"comparison_{safe}.md")
        csv = os.path.join(args.out, f"comparison_{safe}.csv")
        with open(md, "w", encoding="utf-8") as f:
            f.write(methods_to_markdown(name, full, results, budgets))
        with open(csv, "w", encoding="utf-8", newline="") as f:
            f.write(methods_to_csv(name, full, results))
        print(f"  wrote {md}")

        tt = next((r for r in results.get("tooltrim", [])
                   if r.budget == args.pareto_budget), None)
        if tt:
            board.append({
                "model": name, "full_acc": full.accuracy,
                "full_tokens": full.avg_tokens, "tt_acc": tt.accuracy,
                "tt_tokens": tt.avg_tokens, "retention": tt.retention,
                "saved": tt.saved_ratio,
            })
        for b in budgets:
            sig_lines.append(significance_row(results, name, b))

    write_summary(args.summary, args.pareto_budget, board, sig_lines)
    print(f"\nwrote cross-model summary -> {args.summary}")


def write_summary(path: str, pareto_budget: int, board: List[dict],
                  sig_lines: List[str]) -> None:
    lines = ["# Frontier-model matrix: accuracy/token Pareto\n",
             f"Cross-model comparison at a **{pareto_budget}-token** budget. "
             "`full` is uncompressed tool output; `tooltrim` is query-aware "
             "compression to the budget. Retention = tooltrim accuracy / "
             "full-context accuracy.\n",
             "| model | full acc | full tokens | tooltrim acc | tooltrim tokens "
             "| retention | tokens saved |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(board, key=lambda x: -x["tt_acc"]):
        lines.append(
            f"| {r['model']} | {r['full_acc']*100:.0f}% | {r['full_tokens']:,.0f} "
            f"| {r['tt_acc']*100:.0f}% | {r['tt_tokens']:,.0f} "
            f"| {r['retention']*100:.0f}% | {r['saved']*100:.1f}% |")
    lines += ["\n## tooltrim vs RAG top-k (paired McNemar)\n",
              "Does content-type structure separate tooltrim from plain "
              "query-aware RAG selection under a real LLM judge?\n",
              "| model | budget | Δ acc (tooltrim − rag-topk) | p-value | "
              "significant (p<0.05) |",
              "|---|---:|---:|---:|---:|"]
    lines += sig_lines
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
