"""Run full-vs-compressed and report accuracy retained vs tokens saved."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from tooltrim import ToolCompressor, count_tokens, using_exact_counts

from .dataset import Case, default_cases
from .judge import passed
from .metrics import fmt_ci, mcnemar, wilson_ci
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


# --- comparative benchmark (tooltrim vs baselines) ---------------------------
# Everything above scores tooltrim against full-context. A publishable claim
# needs tooltrim scored against the *alternatives* (truncation, RAG top-k,
# LLMLingua-2) on identical cases and budgets. These functions do that, plus a
# paired significance test (McNemar) on the discordant cases.

@dataclass
class MethodBudget:
    """One (method, budget) cell of the comparison grid."""

    method: str
    budget: int
    avg_tokens: float
    saved_ratio: float
    accuracy: float
    correct: int
    n: int
    retention: float          # accuracy / full-context accuracy
    acc_lo: float = 0.0
    acc_hi: float = 0.0
    mask: tuple = ()          # per-case correctness, for paired significance
    valid_structure: float = 1.0        # frac of outputs still parseable as their type
    downstream: Optional[float] = None  # frac of json/tabular cases extractable in code


def _method_budget_pass(cases: Sequence[Case], model: QAModel, comp,
                        budget: int, full_acc: float,
                        full_tokens: int) -> MethodBudget:
    from tooltrim import count_tokens as _ct

    from .faithfulness import downstream_rate, parseable_rate

    correct = 0
    tokens = 0
    mask: List[bool] = []
    structure_pairs: List[tuple] = []   # (text, content_type)
    downstream_items: List[tuple] = []  # (text, case)
    for c in cases:
        text = comp.compress(c.tool_output, c.question, budget)
        tokens += _ct(text)
        structure_pairs.append((text, c.content_type))
        downstream_items.append((text, c))
        ok = passed(model.answer(c.question, text), c.gold, c.all_of, c.must_not)
        mask.append(ok)
        correct += int(ok)
    n = len(cases)
    acc = correct / n if n else 0.0
    lo, hi = wilson_ci(correct, n)
    return MethodBudget(
        method=getattr(comp, "name", comp.__class__.__name__),
        budget=budget,
        avg_tokens=tokens / n if n else 0.0,
        saved_ratio=1 - (tokens / (full_tokens or 1)),
        accuracy=acc,
        correct=correct,
        n=n,
        retention=(acc / full_acc) if full_acc else 0.0,
        acc_lo=lo,
        acc_hi=hi,
        mask=tuple(mask),
        valid_structure=parseable_rate(structure_pairs),
        downstream=downstream_rate(downstream_items),
    )


def evaluate_methods(model: QAModel, compressors: Sequence[Any], *,
                     cases: Sequence[Case] | None = None,
                     budgets: Sequence[int] = (128, 256, 400, 800)):
    """Score several compressors on the same cases/budgets.

    Returns ``(full, results)`` where ``full`` is the full-context reference and
    ``results`` maps each method name to its list of :class:`MethodBudget` rows.
    A compressor advertising ``available() is False`` is skipped.
    """
    cases = list(cases) if cases is not None else default_cases()
    full = _full_pass(cases, model)
    full_tokens = sum(count_tokens(c.tool_output) for c in cases) or 1
    results: Dict[str, List[MethodBudget]] = {}
    for comp in compressors:
        if hasattr(comp, "available") and not comp.available():
            continue
        results[getattr(comp, "name", comp.__class__.__name__)] = [
            _method_budget_pass(cases, model, comp, b, full.accuracy, full_tokens)
            for b in budgets
        ]
    return full, results


def _rows_by_budget(results: Dict[str, List[MethodBudget]], budget: int
                    ) -> List[MethodBudget]:
    rows = [r for rs in results.values() for r in rs if r.budget == budget]
    # Highest accuracy first, then fewest tokens as the tie-break.
    return sorted(rows, key=lambda r: (-r.accuracy, r.avg_tokens))


def methods_to_markdown(model_name: str, full: FullResult,
                        results: Dict[str, List[MethodBudget]],
                        budgets: Sequence[int], *,
                        reference: str = "tooltrim") -> str:
    """Comparison tables (one per budget) + a paired-significance section."""
    lines: List[str] = []
    lines.append(f"## Comparative faithfulness — `{model_name}`")
    lines.append("")
    lines.append(
        f"Full-context accuracy: **{full.correct}/{full.n} "
        f"({full.accuracy*100:.1f}%)** {fmt_ci(full.acc_lo, full.acc_hi)}, "
        f"avg {full.avg_tokens:,.0f} tokens/case. CIs are 95% Wilson; each "
        "method is scored on the *same* cases so accuracies are directly "
        "comparable.")
    lines.append("")
    for b in budgets:
        rows = _rows_by_budget(results, b)
        if not rows:
            continue
        lines.append(f"### Budget {b} tokens")
        lines.append("")
        lines.append("| method | comp tokens | tokens saved | accuracy | 95% CI "
                     "| retention | parseable | downstream |")
        lines.append("|:--|---:|---:|---:|:--:|---:|---:|---:|")
        for r in rows:
            star = " **★**" if r.method == reference else ""
            down = "—" if r.downstream is None else f"{r.downstream*100:.0f}%"
            lines.append(
                f"| `{r.method}`{star} | {r.avg_tokens:,.0f} | "
                f"{r.saved_ratio*100:.1f}% | {r.correct}/{r.n} "
                f"({r.accuracy*100:.0f}%) | {fmt_ci(r.acc_lo, r.acc_hi)} | "
                f"{r.retention*100:.1f}% | {r.valid_structure*100:.0f}% | {down} |")
        lines.append("")
    lines.append("*parseable* = fraction of compressed outputs that still parse as "
                 "their content type (what the agent's next `json.loads`/CSV read "
                 "does). *downstream* = fraction of json/tabular cases whose gold "
                 "fact is recoverable from a valid parse in code.")
    lines.append("")

    sig = _significance_section(results, budgets, reference=reference)
    if sig:
        lines.append(sig)
    return "\n".join(lines) + "\n"


def _significance_section(results: Dict[str, List[MethodBudget]],
                          budgets: Sequence[int], *,
                          reference: str = "tooltrim") -> str:
    if reference not in results:
        return ""
    others = [m for m in results if m != reference]
    if not others:
        return ""
    ref_by_budget = {r.budget: r for r in results[reference]}
    lines: List[str] = []
    lines.append(f"### Significance — `{reference}` vs each baseline (McNemar, paired)")
    lines.append("")
    lines.append("| budget | baseline | Δ accuracy | wins/losses | p-value | significant (p<0.05) |")
    lines.append("|---:|:--|---:|:--:|---:|:--:|")
    for b in budgets:
        ref = ref_by_budget.get(b)
        if ref is None:
            continue
        for m in others:
            other = next((r for r in results[m] if r.budget == b), None)
            if other is None:
                continue
            # b = ref-correct/other-wrong (ref wins), c = ref-wrong/other-correct
            wins, losses, p = mcnemar(ref.mask, other.mask)
            delta = (ref.accuracy - other.accuracy) * 100
            sig = "yes" if p < 0.05 else "no"
            lines.append(
                f"| {b} | `{m}` | {delta:+.1f}% | {wins}/{losses} | "
                f"{p:.3f} | {sig} |")
    return "\n".join(lines)


def methods_to_csv(model_name: str, full: FullResult,
                   results: Dict[str, List[MethodBudget]]) -> str:
    import csv
    import io

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["model", "method", "budget", "comp_tokens", "saved_ratio",
                "accuracy", "acc_ci_lo", "acc_ci_hi", "retention",
                "valid_structure", "downstream", "full_accuracy", "n"])
    for rows in results.values():
        for r in rows:
            down = "" if r.downstream is None else f"{r.downstream:.4f}"
            w.writerow([model_name, r.method, r.budget, f"{r.avg_tokens:.1f}",
                        f"{r.saved_ratio:.4f}", f"{r.accuracy:.4f}",
                        f"{r.acc_lo:.4f}", f"{r.acc_hi:.4f}",
                        f"{r.retention:.4f}", f"{r.valid_structure:.4f}", down,
                        f"{full.accuracy:.4f}", r.n])
    return buf.getvalue()


def format_methods_report(model_name: str, full: FullResult,
                          results: Dict[str, List[MethodBudget]],
                          budgets: Sequence[int], *,
                          reference: str = "tooltrim") -> str:
    """Plain-text console version of the comparison (no Markdown)."""
    lines: List[str] = []
    lines.append(
        f"comparative faithfulness  |  model={model_name}  |  "
        f"cases={full.n}  |  exact tiktoken: {using_exact_counts()}")
    lines.append(
        f"full-context: {full.correct}/{full.n} = {full.accuracy*100:.1f}% "
        f"(avg {full.avg_tokens:,.0f} tokens/case)")
    for b in budgets:
        rows = _rows_by_budget(results, b)
        if not rows:
            continue
        lines.append("")
        lines.append(f"budget={b} tokens")
        header = (f"  {'method':<15} {'comp_tok':>9} {'saved':>7} "
                  f"{'accuracy':>11} {'retention':>10} {'parse':>6} {'downstr':>8}")
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for r in rows:
            mark = " *" if r.method == reference else "  "
            down = "    —" if r.downstream is None else f"{r.downstream*100:>6.0f}%"
            lines.append(
                f"  {r.method:<15} {r.avg_tokens:>9,.0f} "
                f"{r.saved_ratio*100:>6.1f}% {r.correct:>3}/{r.n} "
                f"{r.accuracy*100:>4.0f}% {r.retention*100:>9.1f}% "
                f"{r.valid_structure*100:>5.0f}% {down}{mark}")
    return "\n".join(lines)
