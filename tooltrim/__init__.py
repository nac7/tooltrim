"""tooltrim — drop-in compression for LLM agent tool outputs.

Shrink bloated tool results (HTML, JSON, logs, tables, long text) before they
re-enter an agent's context window. Deterministic and dependency-free by
default; full outputs stay retrievable via short refs.

Quickstart::

    from tooltrim import compressed_tool

    @compressed_tool(max_tokens=400)
    def web_fetch(url: str) -> str:
        ...  # returns a huge HTML page
    # the agent now receives a compact, on-topic extract instead

Or imperatively::

    from tooltrim import ToolCompressor
    tc = ToolCompressor(max_tokens=400)
    result = tc.compress(huge_output, query="what is the user's email?")
    print(result.text, result.saved_tokens)
    full = tc.expand(result.ref)  # get the original back on demand
"""

from __future__ import annotations

from .core import CompressionResult, ToolCompressor
from .decorators import (
    compressed_tool,
    current_query,
    query_scope,
    set_query,
    wrap_tool,
)
from .detect import detect_type
from .llm import LLMDistiller
from .metrics import Metrics
from .store import BaseStore, FileStore, OutputStore, RedisStore, S3Store
from .tokens import count_tokens, using_exact_counts

__version__ = "0.1.0"

__all__ = [
    "ToolCompressor",
    "CompressionResult",
    "compressed_tool",
    "wrap_tool",
    "set_query",
    "query_scope",
    "current_query",
    "OutputStore",
    "BaseStore",
    "FileStore",
    "RedisStore",
    "S3Store",
    "Metrics",
    "LLMDistiller",
    "detect_type",
    "count_tokens",
    "using_exact_counts",
    "__version__",
]
