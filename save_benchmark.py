"""Persist a faithfulness run as a citable artifact (tracked under benchmarks/).

Writes, into benchmarks/runs/<date>_<label>/:
  - report.md     human-readable Pareto table
  - results.csv   one row per budget
  - run.json      full provenance + per-case detail (for a paper appendix)

Use the same --cache you used for the run so this is instant and free:

    python save_benchmark.py --model ollama --model-id llama3.1:8b \
        --cache .cache/ollama_llama31.json --budgets 128,256,400 --label llama3.1-8b
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import subprocess
import sys

import tooltrim
from eval import (
    CachedModel,
    default_cases,
    evaluate_detailed,
    get_model,
    to_csv,
    to_markdown,
)
from eval.harness import BudgetResult, FullResult
from tooltrim import using_exact_counts


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="offline")
    p.add_argument("--model-id", default=None)
    p.add_argument("--budgets", default="128,256,400")
    p.add_argument("--cache", default=None)
    p.add_argument("--label", default=None, help="folder label (default: model name)")
    p.add_argument("--note", default="", help="free-text note recorded in run.json")
    args = p.parse_args()

    budgets = tuple(int(b) for b in args.budgets.split(",") if b.strip())
    model = get_model(args.model, model_id=args.model_id)
    if args.cache:
        model = CachedModel(model, args.cache)
    name = getattr(model, "name", args.model)

    cases = default_cases()
    full, results, records = evaluate_detailed(model, cases=cases, budgets=budgets)

    label = args.label or name.replace("/", "_").replace(":", "_")
    date = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    out_dir = os.path.join("benchmarks", "runs", f"{date}_{label}")
    os.makedirs(out_dir, exist_ok=True)

    type_counts: dict = {}
    for c in cases:
        type_counts[c.content_type] = type_counts.get(c.content_type, 0) + 1

    record = {
        "tool": "tooltrim",
        "tooltrim_version": tooltrim.__version__,
        "run_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "exact_tiktoken": using_exact_counts(),
        "note": args.note,
        "model": {"name": name, "provider": args.model, "model_id": args.model_id},
        "dataset": {
            "n": len(cases),
            "types": type_counts,
            "case_ids": [c.id for c in cases],
        },
        "budgets": list(budgets),
        "full": _full_dict(full),
        "budget_results": [_budget_dict(r) for r in results],
        "cases": [_case_dict(r) for r in records],
    }

    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(_report_md(record, full, results))
    with open(os.path.join(out_dir, "results.csv"), "w", encoding="utf-8",
              newline="") as f:
        f.write(to_csv(name, full, results))
    with open(os.path.join(out_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    print(f"saved run to {out_dir}/  (report.md, results.csv, run.json)")
    print(f"  model={name}  full={full.correct}/{full.n}  "
          f"commit={record['git_commit']}  dirty={record['git_dirty']}")


def _full_dict(f: FullResult) -> dict:
    return {"accuracy": f.accuracy, "correct": f.correct, "n": f.n,
            "avg_tokens": f.avg_tokens}


def _budget_dict(r: BudgetResult) -> dict:
    return {"budget": r.budget, "avg_tokens": r.avg_tokens,
            "saved_ratio": r.saved_ratio, "accuracy": r.accuracy,
            "correct": r.correct, "n": r.n, "retention": r.retention}


def _case_dict(rec) -> dict:
    return {
        "id": rec.id,
        "content_type": rec.content_type,
        "question": rec.question,
        "gold": rec.gold,
        "full": {"correct": rec.full_correct, "tokens": rec.full_tokens,
                 "answer": rec.full_answer},
        "per_budget": {str(b): v for b, v in rec.per_budget.items()},
    }


def _report_md(record: dict, full: FullResult, results) -> str:
    m = record["model"]["name"]
    head = (
        f"# tooltrim faithfulness run — `{m}`\n\n"
        f"- **Run (UTC):** {record['run_utc']}\n"
        f"- **tooltrim:** v{record['tooltrim_version']}  |  "
        f"**commit:** `{record['git_commit']}`"
        f"{' (dirty)' if record['git_dirty'] else ''}\n"
        f"- **Python:** {record['python']}  |  **Platform:** {record['platform']}\n"
        f"- **Token counts:** {'exact (tiktoken cl100k_base)' if record['exact_tiktoken'] else 'heuristic ~4 chars/token'}\n"
        f"- **Dataset:** {record['dataset']['n']} curated cases "
        f"({', '.join(f'{k}:{v}' for k, v in record['dataset']['types'].items())})\n"
    )
    if record["note"]:
        head += f"- **Note:** {record['note']}\n"
    head += "\nReproduce: `python run_faithfulness.py --model "
    head += f"{record['model']['provider']}"
    if record["model"]["model_id"]:
        head += f" --model-id {record['model']['model_id']}"
    head += f" --budgets {','.join(map(str, record['budgets']))}`\n\n"
    return head + to_markdown(m, full, results)


if __name__ == "__main__":
    main()
