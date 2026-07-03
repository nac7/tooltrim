"""Embedding-based relevance scorer (optional, semantic).

BM25 is lexical: the query "car" won't match a chunk that only says "automobile".
An :class:`EmbeddingScorer` ranks chunks by cosine similarity to the query in
embedding space, so semantically related content survives compression even
without shared words.

It is provider-agnostic: pass any ``embed(list[str]) -> list[vector]`` callable
(OpenAI, Cohere, a local model, ...). With no callable it lazily loads
``sentence-transformers`` (``pip install tooltrim[embeddings]``)::

    from tooltrim import ToolCompressor, EmbeddingScorer

    # bring your own embeddings
    scorer = EmbeddingScorer(embed=lambda texts: my_client.embed(texts))
    tc = ToolCompressor(max_tokens=400, scorer=scorer)

    # or let it load a local sentence-transformers model
    tc = ToolCompressor(scorer=EmbeddingScorer())
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence

EmbedFn = Callable[[List[str]], Sequence[Sequence[float]]]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class EmbeddingScorer:
    """Semantic scorer: cosine similarity of chunk embeddings to the query.

    Args:
        embed: ``f(list[str]) -> list[vector]``. If ``None``, a
            ``sentence-transformers`` model (``model``) is loaded lazily.
        model: sentence-transformers model name used when ``embed`` is ``None``.
        floor: scores at/below this cosine are treated as no-signal (clamped to
            0.0), so unrelated chunks don't crowd out the budget and callers can
            still fall back to positional selection when nothing is relevant.
    """

    def __init__(self, embed: Optional[EmbedFn] = None, *,
                 model: str = "all-MiniLM-L6-v2", floor: float = 0.0):
        self._embed = embed
        self._model_name = model
        self.floor = floor

    def _ensure_embed(self) -> EmbedFn:
        if self._embed is None:
            from sentence_transformers import SentenceTransformer  # lazy

            m = SentenceTransformer(self._model_name)

            def _embed(texts: List[str]):
                return m.encode(list(texts), normalize_embeddings=True).tolist()

            self._embed = _embed
        return self._embed

    def __call__(self, chunks: Sequence[str], query: str) -> List[float]:
        chunks = list(chunks)
        if not query or not chunks:
            return [0.0] * len(chunks)
        embed = self._ensure_embed()
        vecs = list(embed([query] + chunks))
        qv, cvs = vecs[0], vecs[1:]
        out: List[float] = []
        for cv in cvs:
            s = _cosine(qv, cv)
            out.append(s if s > self.floor else 0.0)
        return out
