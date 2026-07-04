"""Baseline compressors for a *comparative* faithfulness benchmark.

The faithfulness harness on its own answers "does tooltrim keep the model
correct while cutting tokens?". A paper needs the harder question: "does it do
so *better than the obvious alternatives*?". This module supplies those
alternatives behind one tiny interface so the harness can score any of them on
the same cases, at the same budgets, with the same judge:

    class Compressor(Protocol):
        name: str
        def compress(self, text: str, query: str | None, budget: int) -> str: ...

Baselines shipped here (all query-aware where it makes sense, none needing a
network):

  - ``full``            no compression (the reference/ceiling).
  - ``truncate-head``   keep the first ``budget`` tokens (the naive default).
  - ``truncate-tail``   keep the last ``budget`` tokens.
  - ``rag-topk``        chunk the output, BM25-rank chunks against the query,
                        greedily keep the best until the budget fills, restored
                        to original order — the canonical RAG selection baseline,
                        deliberately *content-type-agnostic* to isolate the value
                        of tooltrim's per-type structure awareness.
  - ``rag-embed``       same, but semantic (needs ``tooltrim[embeddings]``).
  - ``llmlingua-2``     Microsoft LLMLingua-2 (needs ``pip install llmlingua``).
  - ``tooltrim``        the real thing (content-type router + relevance).

Optional baselines advertise availability via ``available()`` so a run can skip
what isn't installed instead of crashing.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Sequence, runtime_checkable

from tooltrim import ToolCompressor, count_tokens
from tooltrim.relevance import bm25_scores


@runtime_checkable
class Compressor(Protocol):
    name: str

    def compress(self, text: str, query: Optional[str], budget: int) -> str: ...

    def available(self) -> bool: ...


# --- helpers -----------------------------------------------------------------

def _truncate_to_tokens(text: str, budget: int, *, tail: bool = False) -> str:
    """Return the longest head (or tail) slice of ``text`` within ``budget`` tokens.

    Binary-searches on character length so it is exact under tiktoken and only
    O(log n) token counts — the compressor never silently exceeds the budget.
    """
    if budget <= 0 or not text:
        return ""
    if count_tokens(text) <= budget:
        return text
    lo, hi = 0, len(text)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        slice_ = text[-mid:] if tail else text[:mid]
        if count_tokens(slice_) <= budget:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return text[-best:] if tail else text[:best]


def _chunks(text: str, size: int = 40, overlap: int = 8) -> List[str]:
    """Split ``text`` into overlapping word windows (line-aware).

    Kept intentionally simple and content-type-agnostic: a RAG baseline that
    knows nothing about HTML/JSON/log structure, which is precisely the point of
    comparison.
    """
    out: List[str] = []
    step = max(1, size - overlap)
    for line in text.splitlines() or [text]:
        words = line.split()
        if not words:
            continue
        if len(words) <= size:
            out.append(line.strip())
            continue
        for i in range(0, len(words), step):
            out.append(" ".join(words[i : i + size]))
    return out or [text]


def _select_by_score(text: str, scores: Sequence[float], chunks: Sequence[str],
                     budget: int) -> str:
    """Greedily keep the highest-scoring chunks within ``budget``, original order.

    Falls back to a head slice when no chunk scores (e.g. empty/irrelevant
    query), matching how a real RAG system degrades.
    """
    if not any(s > 0 for s in scores):
        return _truncate_to_tokens(text, budget)
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    kept: List[int] = []
    used = 0
    for i in ranked:
        if scores[i] <= 0:
            break
        cost = count_tokens(chunks[i]) + 1  # +1 for the joining separator
        if used + cost > budget:
            continue
        kept.append(i)
        used += cost
        if used >= budget:
            break
    if not kept:  # even the single best chunk overflows — hard-truncate it
        return _truncate_to_tokens(chunks[ranked[0]], budget)
    return "\n".join(chunks[i] for i in sorted(kept))


# --- baselines ---------------------------------------------------------------

class FullContext:
    """No compression — the accuracy ceiling and token upper bound."""

    name = "full"

    def compress(self, text: str, query: Optional[str], budget: int) -> str:
        return text

    def available(self) -> bool:
        return True


class TruncateHead:
    name = "truncate-head"

    def compress(self, text: str, query: Optional[str], budget: int) -> str:
        return _truncate_to_tokens(text, budget)

    def available(self) -> bool:
        return True


class TruncateTail:
    name = "truncate-tail"

    def compress(self, text: str, query: Optional[str], budget: int) -> str:
        return _truncate_to_tokens(text, budget, tail=True)

    def available(self) -> bool:
        return True


class RagTopK:
    """Query-aware chunk selection (BM25), content-type-agnostic."""

    name = "rag-topk"

    def __init__(self, chunk_size: int = 40, overlap: int = 8):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def compress(self, text: str, query: Optional[str], budget: int) -> str:
        if count_tokens(text) <= budget:
            return text
        chunks = _chunks(text, self.chunk_size, self.overlap)
        scores = bm25_scores(chunks, query or "")
        return _select_by_score(text, scores, chunks, budget)

    def available(self) -> bool:
        return True


class RagEmbed:
    """Semantic chunk selection — needs ``tooltrim[embeddings]``."""

    name = "rag-embed"

    def __init__(self, chunk_size: int = 40, overlap: int = 8, model: Optional[str] = None):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._scorer = None
        # EmbeddingScorer imports from tooltrim regardless, but it loads the
        # actual model lazily — so probe the real dependency, not just the class.
        import importlib.util

        if importlib.util.find_spec("sentence_transformers") is not None:
            try:
                from tooltrim import EmbeddingScorer  # type: ignore

                self._scorer = EmbeddingScorer(model=model) if model else EmbeddingScorer()
            except Exception:
                self._scorer = None

    def available(self) -> bool:
        return self._scorer is not None

    def compress(self, text: str, query: Optional[str], budget: int) -> str:
        if self._scorer is None:
            raise RuntimeError("rag-embed unavailable: install tooltrim[embeddings]")
        if count_tokens(text) <= budget:
            return text
        chunks = _chunks(text, self.chunk_size, self.overlap)
        scores = self._scorer(chunks, query or "")
        return _select_by_score(text, scores, chunks, budget)


class LLMLingua2:
    """Microsoft LLMLingua-2 prompt compression — needs ``pip install llmlingua``.

    LLMLingua targets a token count rather than a hard cap; we pass ``budget`` as
    the target and let the harness measure the tokens it actually emits (as it
    does for every method), so the comparison stays apples-to-apples.
    """

    name = "llmlingua-2"

    def __init__(self, model: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                 device_map: str = "cpu"):
        self._model = model
        self._compressor = None
        try:
            from llmlingua import PromptCompressor  # type: ignore

            # Default to CPU so the baseline is runnable on machines without a
            # CUDA GPU; callers with a GPU can pass device_map="cuda".
            self._compressor = PromptCompressor(
                model_name=model, use_llmlingua2=True, device_map=device_map)
        except Exception:
            self._compressor = None

    def available(self) -> bool:
        return self._compressor is not None

    def compress(self, text: str, query: Optional[str], budget: int) -> str:
        if self._compressor is None:
            raise RuntimeError("llmlingua-2 unavailable: pip install llmlingua")
        if count_tokens(text) <= budget:
            return text
        out = self._compressor.compress_prompt(
            text, target_token=budget, question=query or "")
        return out.get("compressed_prompt", text)


class Tooltrim:
    """tooltrim itself — content-type router + query-aware relevance."""

    name = "tooltrim"

    def __init__(self, add_footer: bool = False, scorer=None):
        self.add_footer = add_footer
        self.scorer = scorer

    def compress(self, text: str, query: Optional[str], budget: int) -> str:
        tc = ToolCompressor(max_tokens=budget, add_footer=self.add_footer,
                            store=None, scorer=self.scorer)
        return tc.compress(text, query=query).text

    def available(self) -> bool:
        return True


# Names of the always-available offline baselines, in a sensible report order.
DEFAULT_BASELINE_NAMES = (
    "full", "truncate-head", "truncate-tail", "rag-topk", "tooltrim")

_REGISTRY = {
    "full": FullContext,
    "truncate-head": TruncateHead,
    "truncate-tail": TruncateTail,
    "rag-topk": RagTopK,
    "rag-embed": RagEmbed,
    "llmlingua-2": LLMLingua2,
    "tooltrim": Tooltrim,
}


def get_baseline(name: str) -> Compressor:
    """Instantiate a baseline by name (see ``_REGISTRY`` for the full set)."""
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"unknown baseline '{name}' (choose from {', '.join(_REGISTRY)})")
    return _REGISTRY[key]()


def default_baselines() -> List[Compressor]:
    """The offline, always-available comparison set (full + naive + RAG + tooltrim)."""
    return [get_baseline(n) for n in DEFAULT_BASELINE_NAMES]
