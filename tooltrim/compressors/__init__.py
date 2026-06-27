"""Per-content-type compressors.

Each module exposes ``compress(text, query, max_tokens) -> str``. Compressors are
deterministic and dependency-free; they aim to preserve the information an agent
is most likely to need while fitting a token budget.
"""

from . import html, json_, logs, tabular, text  # noqa: F401

__all__ = ["html", "json_", "logs", "tabular", "text"]
