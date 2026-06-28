"""Run full-vs-compressed and report accuracy retained vs tokens saved."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from tooltrim import ToolCompressor, count_tokens, using_exact_counts

from .dataset import Case, default_cases
from .judge import passed
from .metrics import fmt_ci, wilson_ci
from .models import QAModel


@dataclass
class FullResult:
    accuracy: float
    correct: int
    n: int
    avg_tokens: float
    acc_lo: float = 0.0
    acc_hi: float = 0.0


@dataclass
class BudgetResult:
    budget: int
    avg_tokens: float
    saved_ratio: float
    accuracy: float
    correct: int
    n: int
    retention: float  # accuracy_compressed / accuracy_full
    acc_lo: float = 0.0
    acc_hi: float = 0.0


def _full_pass(cases: Sequence[Case], model: QAModel) -> FullResult:
    correct = 0
    tokens = 0
    for c in cases:
        tokens += count_tokens(c.tool_output)
        if passed(model.answer(c.question, c.tool_output),
                  c.gold, c.all_of, c.must_not):
            correct += 1
    n = len(cases)
    lo, hi = wilson_ci(correct, n)
    return FullResult(correct / n if n else 0.0, correct, n,
                      tokens / n if n else 0.0, lo, hi)


def _budget_pass(cases: Sequence[Case], model: QAModel, budget: int,
                 full_acc: float) -> BudgetResult:
    tc = ToolCompressor(max_tokens=budget, add_footer=False)
    correct = 0
    tokens = 0
    for c in cases:
        res = tc.compress(c.tool_output, query=c.question)
        tokens += res.compressed_tokens
        if passed(model.answer(c.question, res.text),
                  c.gold, c.all_of, c.must_not):
            correct += 1
    n = len(cases)
    acc = correct / n if n else 0.0
    lo, hi = wilson_ci(correct, n)
    full_tokens = sum(count_tokens(c.tool_output) for c in cases) or 1
    return BudgetResult(
        budget=budget,
        avg_tokens=tokens / n if n else 0.0,
        saved_ratio=1 - (tokens / full_tokens),
        accuracy=acc,
        correct=correct,
        n=n,
        retention=(acc / full_acc) if full_acc else 0.0,
        acc_lo=lo,
        acc_hi=hi,
    )


def evaluate(model: QAModel, *, cases: Sequence[Case] | None = None,
             budgets: Sequence[int] = (128, 256, 400, 800)):
    """Return (FullResult, [BudgetResult]) for the model across budgets."""
    cases = list(cases) if cases is not None else default_cases()
    full = _full_pass(cases, model)
    results = [_budget_pass(cases, model, b, full.accuracy) for b in budgets]
    return full, results


@dataclass
class CaseRecord:
    """Per-case detail for auditability / a paper appendix."""

    id: str
    content_type: str
    category: str
    question: str
    gold: str
    full_correct: bool
    full_tokens: int
    full_answer: str
    per_budget: Dict[int, Dict[str, Any]] = field(default_factory=dict)


def evaluate_detailed(model: QAModel, *, cases: Sequence[Case] | None = None,
                      budgets: Sequence[int] = (128, 256, 400, 800)):
    """Like evaluate(), but also returns per-case records (full + per-budget).

    Aggregate math is identical to evaluate(); use this when persisting a run.
    """
    cases = list(cases) if cases is not None else default_cases()
    n = len(cases)

    records: List[CaseRecord] = []
    full_correct = 0
    full_tokens = 0
    for c in cases:
        ans = model.answer(c.question, c.tool_output)
        tok = count_tokens(c.tool_output)
        ok = passed(ans, c.gold, c.all_of, c.must_not)
        full_correct += int(ok)
        full_tokens += tok
        records.append(CaseRecord(c.id, c.content_type, c.category, c.question,
                                  c.gold, ok, tok, ans))
    f_lo, f_hi = wilson_ci(full_correct, n)
    full = FullResult(full_correct / n if n else 0.0, full_correct, n,
                      full_tokens / n if n else 0.0, f_lo, f_hi)

    by_id = {r.id: r for r in records}
    results: List[BudgetResult] = []
    for b in budgets:
        tc = ToolCompressor(max_tokens=b, add_footer=False)
        correct = 0
        tokens = 0
        for c in cases:
            res = tc.compress(c.tool_output, query=c.question)
            ans = model.answer(c.question, res.text)
            ok = passed(ans, c.gold, c.all_of, c.must_not)
            correct += int(ok)
            tokens += res.compressed_tokens
            by_id[c.id].per_budget[b] = {
                "correct": ok, "tokens": res.compressed_tokens, "answer": ans}
        acc = correct / n if n else 0.0
        lo, hi = wilson_ci(correct, n)
        results.append(BudgetResult(
            budget=b, avg_tokens=tokens / n if n else 0.0,
            saved_ratio=1 - (tokens / (full_tokens or 1)),
            accuracy=acc, correct=correct, n=n,
            retention=(acc / full.accuracy) if full.accuracy else 0.0,
            acc_lo=lo, acc_hi=hi))
    return full, results, records


def category_breakdown(records: List[CaseRecord],
                       budgets: Sequence[int]) -> Dict[str, Any]:
    """Per-category full + per-budget accuracy (single / multi / distractor)."""
    out: Dict[str, Any] = {}
    for cat in sorted({r.category for r in records}):
        rs = [r for r in records if r.category == cat]
        n = len(rs)
        full_c = sum(int(r.full_correct) for r in rs)
        per = {}
        for b in budgets:
            c = sum(1 for r in rs if r.per_budget.get(b, {}).get("correct"))
            per[b] = {"correct": c, "n": n, "accuracy": c / n if n else 0.0}
        out[cat] = {"n": n, "full_correct": full_c,
                    "full_accuracy": full_c / n if n else 0.0,
                    "per_budget": per}
    return out


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
        f"{full.accuracy*100:.1f}% {fmt_ci(full.acc_lo, full.acc_hi)}   "
        f"(avg {full.avg_tokens:,.0f} tokens/case)"
    )
    lines.append("")
    header = (f"{'budget':>7} {'comp_tok':>9} {'saved':>8} "
              f"{'accuracy':>11} {'95% CI':>11} {'retention':>10}")
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        lines.append(
            f"{r.budget:>7} {r.avg_tokens:>9,.0f} {r.saved_ratio*100:>7.1f}% "
            f"{r.correct:>3}/{r.n} {r.accuracy*100:>4.0f}% "
            f"{fmt_ci(r.acc_lo, r.acc_hi):>11} {r.retention*100:>9.1f}%"
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
        f"({full.accuracy*100:.1f}%)** {fmt_ci(full.acc_lo, full.acc_hi)}, "
        f"avg {full.avg_tokens:,.0f} tokens/case. CIs are 95% Wilson.")
    lines.append("")
    lines.append("| budget | comp tokens | tokens saved | accuracy | 95% CI | retention |")
    lines.append("|---:|---:|---:|---:|:--:|---:|")
    for r in results:
        lines.append(
            f"| {r.budget} | {r.avg_tokens:,.0f} | {r.saved_ratio*100:.1f}% | "
            f"{r.correct}/{r.n} ({r.accuracy*100:.0f}%) | "
            f"{fmt_ci(r.acc_lo, r.acc_hi)} | {r.retention*100:.1f}% |")
    return "\n".join(lines) + "\n"


def to_csv(model_name: str, full: FullResult,
           results: List[BudgetResult]) -> str:
    import csv
    import io

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["model", "budget", "comp_tokens", "saved_ratio",
                "accuracy", "acc_ci_lo", "acc_ci_hi", "retention",
                "full_accuracy", "full_acc_ci_lo", "full_acc_ci_hi",
                "full_avg_tokens", "n"])
    for r in results:
        w.writerow([model_name, r.budget, f"{r.avg_tokens:.1f}",
                    f"{r.saved_ratio:.4f}", f"{r.accuracy:.4f}",
                    f"{r.acc_lo:.4f}", f"{r.acc_hi:.4f}", f"{r.retention:.4f}",
                    f"{full.accuracy:.4f}", f"{full.acc_lo:.4f}",
                    f"{full.acc_hi:.4f}", f"{full.avg_tokens:.1f}", r.n])
    return buf.getvalue()
