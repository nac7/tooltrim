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
  - ``llm-summary``     ask a real LLM to summarize the output to the budget (the
                        "why not just summarize?" baseline; needs an API key).
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
        self._model = model
        self._scorer = None
        self._probed = False  # has _probe() run yet?

    def _probe(self):
        """Build the embedding scorer once, caching success/failure.

        Constructing a sentence-transformers model is the expensive part, so we
        defer it out of ``__init__`` — merely listing or naming this baseline no
        longer loads a model. ``available()`` still does the honest *deep* check
        (dependency importable AND model constructible) on first call, so it
        never false-positives into a mid-run crash; it just pays for it lazily.
        """
        if self._probed:
            return
        self._probed = True
        import importlib.util

        # EmbeddingScorer imports from tooltrim regardless, but it loads the
        # actual model lazily — so probe the real dependency, not just the class.
        if importlib.util.find_spec("sentence_transformers") is not None:
            try:
                from tooltrim import EmbeddingScorer  # type: ignore

                self._scorer = (
                    EmbeddingScorer(model=self._model) if self._model else EmbeddingScorer()
                )
            except Exception:
                self._scorer = None

    def available(self) -> bool:
        self._probe()
        return self._scorer is not None

    def compress(self, text: str, query: Optional[str], budget: int) -> str:
        self._probe()
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
        self._device_map = device_map
        self._compressor = None
        self._probed = False  # has _probe() run yet?

    def _probe(self):
        """Load the LLMLingua-2 compressor once, caching success/failure.

        Deferred out of ``__init__`` so naming/listing this baseline is cheap;
        the (large) model only loads on first ``available()`` or ``compress()``.
        """
        if self._probed:
            return
        self._probed = True
        try:
            from llmlingua import PromptCompressor  # type: ignore

            # Default to CPU so the baseline is runnable on machines without a
            # CUDA GPU; callers with a GPU can pass device_map="cuda".
            self._compressor = PromptCompressor(
                model_name=self._model, use_llmlingua2=True, device_map=self._device_map)
        except Exception:
            self._compressor = None

    def available(self) -> bool:
        self._probe()
        return self._compressor is not None

    def compress(self, text: str, query: Optional[str], budget: int) -> str:
        self._probe()
        if self._compressor is None:
            raise RuntimeError("llmlingua-2 unavailable: pip install llmlingua")
        if count_tokens(text) <= budget:
            return text
        out = self._compressor.compress_prompt(
            text, target_token=budget, question=query or "")
        return out.get("compressed_prompt", text)


class LLMSummary:
    """Ask a real LLM to summarize the tool output to the token budget.

    This is the "why not just summarize with a cheap model?" baseline every
    reviewer asks for. It wraps the library's :class:`~tooltrim.LLMDistiller`
    with a real completion function and a disk cache (so reruns never re-spend
    tokens). The point of the comparison is not answer-recall alone but
    *downstream-extractability*: a fluent LLM summary of a JSON payload is prose,
    so it no longer parses as JSON — exactly the structural property tooltrim
    preserves and a summarizer discards.

    Providers: ``claude`` (ANTHROPIC_API_KEY), ``openai`` (OPENAI_API_KEY),
    ``groq`` (GROQ_API_KEY). Availability is a cheap env-var check; the client is
    built lazily on first use.
    """

    name = "llm-summary"

    _KEY_ENV = {
        "claude": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "groq": "GROQ_API_KEY",
    }

    def __init__(self, provider: str = "claude", model_id: Optional[str] = None,
                 cache_path: str = ".cache/llm_summary.json"):
        self.provider = provider.lower()
        self.model_id = model_id
        self.cache_path = cache_path
        self._complete = None
        self._probed = False
        self._cache: Optional[dict] = None

    def available(self) -> bool:
        import os

        env = self._KEY_ENV.get(self.provider)
        return bool(env and os.environ.get(env))

    def _load_cache(self) -> dict:
        if self._cache is None:
            import json
            import os

            self._cache = {}
            if os.path.exists(self.cache_path):
                try:
                    with open(self.cache_path, "r", encoding="utf-8") as f:
                        self._cache = json.load(f)
                except Exception:
                    self._cache = {}
        return self._cache

    def _flush_cache(self) -> None:
        import json
        import os

        os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)), exist_ok=True)
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._cache, f)
        os.replace(tmp, self.cache_path)

    def _build_complete(self):
        """Construct a ``complete(prompt)->str`` for the chosen provider (lazy)."""
        if self._probed:
            return self._complete
        self._probed = True
        if not self.available():
            return None
        try:
            if self.provider == "claude":
                import anthropic

                client = anthropic.Anthropic()
                model = self.model_id or "claude-haiku-4-5"

                def complete(prompt: str) -> str:
                    resp = client.messages.create(
                        model=model, max_tokens=1024,
                        messages=[{"role": "user", "content": prompt}])
                    return "".join(b.text for b in resp.content if b.type == "text")
            else:
                import os

                from openai import OpenAI

                base_url = ("https://api.groq.com/openai/v1"
                            if self.provider == "groq" else None)
                client = OpenAI(base_url=base_url,
                                api_key=os.environ.get(self._KEY_ENV[self.provider], ""))
                model = self.model_id or (
                    "llama-3.3-70b-versatile" if self.provider == "groq" else "gpt-4o-mini")

                def complete(prompt: str) -> str:
                    resp = client.chat.completions.create(
                        model=model, max_tokens=1024,
                        messages=[{"role": "user", "content": prompt}])
                    return resp.choices[0].message.content or ""

            self._complete = complete
        except Exception:
            self._complete = None
        return self._complete

    def compress(self, text: str, query: Optional[str], budget: int) -> str:
        if count_tokens(text) <= budget:
            return text
        import hashlib

        cache = self._load_cache()
        h = hashlib.sha1()
        h.update(f"{self.provider}:{self.model_id}\x00{budget}\x00{query}\x00{text}"
                 .encode("utf-8", "replace"))
        key = h.hexdigest()
        if key in cache:
            return cache[key]

        complete = self._build_complete()
        if complete is None:
            raise RuntimeError(
                f"llm-summary unavailable: set {self._KEY_ENV.get(self.provider)}")
        from tooltrim import LLMDistiller

        out = LLMDistiller(complete, max_tokens=budget).compress(
            text, query=query, max_tokens=budget)
        cache[key] = out
        self._flush_cache()
        return out


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


class TooltrimExpand:
    """tooltrim with its recovery path *enabled*: compressed outputs carry a ref
    footer and the agent is handed an ``expand_tool_output`` tool to pull back the
    full content on demand.

    This is tooltrim's actual design — aggressive compression made safe by
    expand-on-demand — which the pure ``tooltrim`` baseline (``store=None``, no
    footer, no tool) deliberately switches off to isolate lossy compression. The
    naive baselines (truncate/rag) *cannot* offer this: they discard the content,
    so there is nothing to expand. Wiring it here is what lets the tau-bench run
    measure recoverable vs. irrecoverable compression rather than lossy-vs-lossy.

    The env adapter checks for ``expand_tool_spec``/``handle_expand`` (duck-typed)
    to advertise the tool and route calls back here. Each instance owns its own
    store, so refs never leak across tasks (tau-bench uses a fresh env per task).
    """

    name = "tooltrim-expand"

    # A directive footer, not just an FYI. The pilot showed the agent almost
    # never invoked recovery (~0.1 calls/episode), so the mechanism was barely
    # exercised. This nudges it in-observation — where the model is actually
    # looking — to consult the ref before acting when a needed field is missing.
    _FOOTER = ("\n\n[tooltrim: {saved} tokens omitted from this output "
               "(ref={ref}). If a detail you need — an id, amount, list item, or "
               "status — is not shown above, call expand_tool_output(ref={ref}) "
               "to read the full output before you answer or act.]")

    def __init__(self, page_chars: int = 8000, scorer=None):
        from tooltrim.store import OutputStore

        self.page_chars = page_chars
        self._tc = ToolCompressor(max_tokens=512, add_footer=True,
                                  store=OutputStore(), scorer=scorer,
                                  footer_template=self._FOOTER)

    @property
    def expand_tool_name(self) -> str:
        return self._tc.EXPAND_TOOL_NAME

    def compress(self, text: str, query: Optional[str], budget: int) -> str:
        return self._tc.compress(text, query=query, max_tokens=budget).text

    def expand_tool_spec(self, style: str = "openai") -> dict:
        return self._tc.expand_tool_spec(style=style)

    def handle_expand(self, ref: str, *, start: int = 0,
                      length: Optional[int] = None) -> str:
        return self._tc.handle_expand(ref, start=start, length=length,
                                      page_chars=self.page_chars)

    def available(self) -> bool:
        return True


class AblatedTooltrim:
    """Shipped tooltrim with the relevance cliff and/or neighbor window overridden.

    Confirming ablation for the retail tau-bench negative result: the shipped
    cliff (relevance_floor=0.5) refuses to pad the budget, so at a *larger* budget
    it can leave the record the agent needs unselected (a "don't-pad" under-fill).
    Lowering the floor to 0.0 restores fill-to-k (keep the top-k relevance-positive
    records, backfilling the budget); widening ``neighbor`` restores more
    surrounding context on the text path. If flipping these knobs recovers the
    tasks shipped tooltrim lost, the diagnosis is confirmed and the fix direction
    (hybrid retention) is validated. Uses the same context-scoped overrides as the
    component ablation ladder, applied *inside* ``compress`` so it is thread-safe.
    """

    def __init__(self, name: str, *, relevance_floor: float, neighbor: int):
        self.name = name
        self.relevance_floor = relevance_floor
        self.neighbor = neighbor

    def compress(self, text: str, query: Optional[str], budget: int) -> str:
        from tooltrim._config import using_config

        tc = ToolCompressor(max_tokens=budget, add_footer=False, store=None)
        with using_config(neighbor=self.neighbor,
                          relevance_floor=self.relevance_floor):
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
    "llm-summary": LLMSummary,
    "tooltrim": Tooltrim,
    "tooltrim-expand": TooltrimExpand,
    # Confirming-ablation variants (floor=0 restores fill-to-k; wide widens the
    # neighbor window). Not part of the shipped comparison set.
    "tt-floor0": lambda: AblatedTooltrim("tt-floor0", relevance_floor=0.0, neighbor=1),
    "tt-floor0-wide": lambda: AblatedTooltrim(
        "tt-floor0-wide", relevance_floor=0.0, neighbor=3),
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
