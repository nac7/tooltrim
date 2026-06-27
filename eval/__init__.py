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

from .dataset import Case, default_cases
from .harness import (
    BudgetResult,
    CaseRecord,
    evaluate,
    evaluate_detailed,
    format_report,
    to_csv,
    to_markdown,
)
from .judge import matches
from .models import CachedModel, QAModel, get_model

__all__ = [
    "Case",
    "default_cases",
    "QAModel",
    "CachedModel",
    "get_model",
    "matches",
    "BudgetResult",
    "CaseRecord",
    "evaluate",
    "evaluate_detailed",
    "format_report",
    "to_markdown",
    "to_csv",
]
