"""Faithfulness-under-compression evaluation for tooltrim.

The deterministic benchmark (`benchmark.py`) proves *how much* tooltrim shrinks
tool output. This package proves the part that actually matters: whether a model
still answers correctly when fed the *compressed* output instead of the full
one.

For each (tool_output, question, gold_answer) case we ask a model twice — once
with the full output, once with the tooltrim-compressed output — and report
**accuracy retained vs tokens saved** across a sweep of budgets.

Runs offline by default via a deterministic retrieval model (correct iff the
needed fact survived compression); plug in Claude / OpenAI / Groq / Ollama for a
real-LLM run via `eval.models.get_model`.
"""

from .baselines import (
    Compressor,
    DEFAULT_BASELINE_NAMES,
    default_baselines,
    get_baseline,
)
from .dataset import Case, default_cases
from .harness import (
    BudgetResult,
    CaseRecord,
    MethodBudget,
    evaluate,
    evaluate_detailed,
    evaluate_methods,
    format_methods_report,
    format_report,
    methods_to_csv,
    methods_to_markdown,
    to_csv,
    to_markdown,
)
from .judge import matches
from .metrics import mcnemar, wilson_ci
from .models import CachedModel, QAModel, get_model

__all__ = [
    "Case",
    "default_cases",
    "QAModel",
    "CachedModel",
    "get_model",
    "matches",
    "wilson_ci",
    "mcnemar",
    "BudgetResult",
    "CaseRecord",
    "MethodBudget",
    "evaluate",
    "evaluate_detailed",
    "evaluate_methods",
    "format_report",
    "format_methods_report",
    "to_markdown",
    "to_csv",
    "methods_to_markdown",
    "methods_to_csv",
    "Compressor",
    "default_baselines",
    "get_baseline",
    "DEFAULT_BASELINE_NAMES",
]
