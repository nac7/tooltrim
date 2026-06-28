"""Combine multiple saved faithfulness runs into one model-comparison table.

    python compare_runs.py benchmarks/runs/*/run.json --out benchmarks/COMPARISON.md

Each run.json (written by save_benchmark.py) contributes one row. The token
savings are identical across models (same deterministic compressor + dataset),
so they're reported once in the caption; the table compares accuracy + CIs.
"""

from __future__ import annotations

import argparse
import glob
import json
from typing import List


def _ci(d: dict) -> str:
    lo, hi = d.get("accuracy_ci95", [0, 0])
    return f"{d['accuracy']*100:.0f}% [{lo*100:.0f}-{hi*100:.0f}%]"


def build_table(runs: List[dict]) -> str:
    budgets = sorted({b["budget"] for r in runs for b in r["budget_results"]})
    lines = ["# tooltrim faithfulness — model comparison", ""]

    # savings caption (model-independent)
    saved = {}
    for r in runs:
        for b in r["budget_results"]:
            saved.setdefault(b["budget"], b["saved_ratio"])
    cap = ", ".join(f"@{b}: {saved[b]*100:.1f}%" for b in budgets)
    n = runs[0]["dataset"]["n"] if runs else 0
    lines.append(f"Dataset: {n} cases. Token savings (same for all models): {cap}. "
                 f"Accuracy shown as point estimate with 95% Wilson CI.")
    lines.append("")

    header = "| model | full | " + " | ".join(f"@{b}" for b in budgets) + " |"
    lines.append(header)
    lines.append("|---|---:|" + "---:|" * len(budgets))
    for r in runs:
        name = r["model"]["name"]
        full = _ci(r["full"])
        by_b = {b["budget"]: _ci(b) for b in r["budget_results"]}
        row = f"| `{name}` | {full} | " + \
              " | ".join(by_b.get(b, "—") for b in budgets) + " |"
        lines.append(row)

    # per-category full accuracy, if present
    if all("category_breakdown" in r for r in runs):
        cats = sorted({c for r in runs for c in r["category_breakdown"]})
        lines += ["", "### Full-context accuracy by category", "",
                  "| model | " + " | ".join(cats) + " |",
                  "|---|" + "---:|" * len(cats)]
        for r in runs:
            bd = r["category_breakdown"]
            row = f"| `{r['model']['name']}` | " + " | ".join(
                f"{bd[c]['full_accuracy']*100:.0f}%" if c in bd else "—"
                for c in cats) + " |"
            lines.append(row)
        lines.append("")
        lines.append("*Distractor cases need reasoning (current vs deprecated "
                     "value); they separate strong models from retrieval.*")

    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+", help="run.json paths (globs ok)")
    p.add_argument("--out", default=None, help="write the table to this file")
    args = p.parse_args()

    files: List[str] = []
    for pat in args.paths:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        raise SystemExit("no run.json files matched")

    runs = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            runs.append(json.load(fh))
    # stronger models tend to have higher full accuracy — sort ascending so the
    # table reads small -> strong
    runs.sort(key=lambda r: r["full"]["accuracy"])

    table = build_table(runs)
    print(table)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(table)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
