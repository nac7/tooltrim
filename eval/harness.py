"""Run full-vs-compressed and report accuracy retained vs tokens saved."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from tooltrim import ToolCompressor, count_tokens, using_exact_counts

from .dataset import Case, default_cases
from .judge import matches
from .models import QAModel


@dataclass
class FullResult:
    accuracy: float
    correct: int
    n: int
    avg_tokens: float


@dataclass
class BudgetResult:
    budget: int
    avg_tokens: float
    saved_ratio: float
    accuracy: float
    correct: int
    n: int
    retention: float  # accuracy_compressed / accuracy_full


def _full_pass(cases: Sequence[Case], model: QAModel) -> FullResult:
    correct = 0
    tokens = 0
    for c in cases:
        tokens += count_tokens(c.tool_output)
        if matches(model.answer(c.question, c.tool_output), c.gold):
            correct += 1
    n = len(cases)
    return FullResult(correct / n if n else 0.0, correct, n, tokens / n if n else 0.0)


def _budget_pass(cases: Sequence[Case], model: QAModel, budget: int,
                 full_acc: float) -> BudgetResult:
    tc = ToolCompressor(max_tokens=budget, add_footer=False)
    correct = 0
    tokens = 0
    for c in cases:
        res = tc.compress(c.tool_output, query=c.question)
        tokens += res.compressed_tokens
        if matches(model.answer(c.question, res.text), c.gold):
            correct += 1
    n = len(cases)
    acc = correct / n if n else 0.0
    full_tokens = sum(count_tokens(c.tool_output) for c in cases) or 1
    return BudgetResult(
        budget=budget,
        avg_tokens=tokens / n if n else 0.0,
        saved_ratio=1 - (tokens / full_tokens),
        accuracy=acc,
        correct=correct,
        n=n,
        retention=(acc / full_acc) if full_acc else 0.0,
    )


def evaluate(model: QAModel, *, cases: Sequence[Case] | None = None,
             budgets: Sequence[int] = (128, 256, 400, 800)):
    """Return (FullResult, [BudgetResult]) for the model across budgets."""
    cases = list(cases) if cases is not None else default_cases()
    full = _full_pass(cases, model)
    results = [_budget_pass(cases, model, b, full.accuracy) for b in budgets]
    return full, results


def format_report(model_name: str, full: FullResult,
                  results: List[BudgetResult]) -> str:
    lines: List[str] = []
    lines.append(
        f"faithfulness under compression  |  model={model_name}  |  "
        f"cases={full.n}  |  exact tiktoken: {using_exact_counts()}"
    )
    lines.append("")
    lines.append(
        f"full-context accuracy: {full.correct}/{full.n} = "
        f"{full.accuracy*100:.1f}%   (avg {full.avg_tokens:,.0f} tokens/case)"
    )
    lines.append("")
    header = (f"{'budget':>7} {'comp_tok':>9} {'saved':>8} "
              f"{'accuracy':>10} {'retention':>10}")
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        lines.append(
            f"{r.budget:>7} {r.avg_tokens:>9,.0f} {r.saved_ratio*100:>7.1f}% "
            f"{r.correct:>3}/{r.n} {r.accuracy*100:>4.0f}% {r.retention*100:>9.1f}%"
        )
    return "\n".join(lines)


def to_markdown(model_name: str, full: FullResult,
                results: List[BudgetResult]) -> str:
    """Render results as a Markdown table — paste straight into a README/paper."""
    lines: List[str] = []
    lines.append(f"### Faithfulness under compression — `{model_name}`")
    lines.append("")
    lines.append(
        f"Full-context accuracy: **{full.correct}/{full.n} "
        f"({full.accuracy*100:.1f}%)**, avg {full.avg_tokens:,.0f} tokens/case.")
    lines.append("")
    lines.append("| budget | comp tokens | tokens saved | accuracy | retention |")
    lines.append("|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r.budget} | {r.avg_tokens:,.0f} | {r.saved_ratio*100:.1f}% | "
            f"{r.correct}/{r.n} ({r.accuracy*100:.0f}%) | {r.retention*100:.1f}% |")
    return "\n".join(lines) + "\n"


def to_csv(model_name: str, full: FullResult,
           results: List[BudgetResult]) -> str:
    import csv
    import io

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["model", "budget", "comp_tokens", "saved_ratio",
                "accuracy", "retention", "full_accuracy", "full_avg_tokens", "n"])
    for r in results:
        w.writerow([model_name, r.budget, f"{r.avg_tokens:.1f}",
                    f"{r.saved_ratio:.4f}", f"{r.accuracy:.4f}",
                    f"{r.retention:.4f}", f"{full.accuracy:.4f}",
                    f"{full.avg_tokens:.1f}", r.n])
    return buf.getvalue()
