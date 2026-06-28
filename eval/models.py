"""Provider-agnostic QA models for the faithfulness harness.

A model answers a question using only a provided context (the tool output, full
or compressed). The interface is deliberately tiny:

    class QAModel(Protocol):
        name: str
        def answer(self, question: str, context: str) -> str: ...

`offline` (default) is a deterministic BM25 retrieval model: it returns the
context window most relevant to the question, so it answers correctly *iff the
needed fact survived compression*. That isolates exactly what we want to measure
— faithfulness of the compressor — with no API keys.

Real adapters (`claude`, `openai`, `groq`, `ollama`) send the same prompt to a
live model. They import their SDK lazily so the core eval has no dependencies.
"""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from tooltrim.relevance import score_chunks

PROMPT = (
    "Answer the question using ONLY the tool output below. "
    "Quote the exact relevant value. If the answer is not present, say "
    "\"not found\".\n\n"
    "Question: {question}\n\n"
    "Tool output:\n{context}\n\n"
    "Answer concisely:"
)


@runtime_checkable
class QAModel(Protocol):
    name: str

    def answer(self, question: str, context: str) -> str: ...


def _windows(text: str, size: int = 30, overlap: int = 10) -> List[str]:
    """Split context into overlapping word windows for retrieval granularity."""
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


class KeywordModel:
    """Offline, deterministic retrieval 'model' — no API key required."""

    name = "offline"

    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    def answer(self, question: str, context: str) -> str:
        windows = _windows(context)
        scores = score_chunks(windows, question)
        if not any(s > 0 for s in scores):
            return windows[0]
        # Return the top-k relevant regions (each with a neighbor for context),
        # merged in original order. top-k lets multi-fact answers surface facts
        # that live in different parts of the document.
        ranked = sorted(range(len(windows)), key=lambda i: scores[i], reverse=True)
        picked = [i for i in ranked if scores[i] > 0][: self.top_k]
        keep: set = set()
        for i in picked:
            for j in range(max(0, i - 1), min(len(windows), i + 2)):
                keep.add(j)
        return " ".join(windows[i] for i in sorted(keep))


class ClaudeModel:
    """Anthropic Claude. Defaults to Haiku 4.5 (cheap extraction)."""

    def __init__(self, model: str = "claude-haiku-4-5", max_tokens: int = 256):
        import anthropic

        self.name = model
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic()

    def answer(self, question: str, context: str) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user",
                       "content": PROMPT.format(question=question, context=context)}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")


class OpenAICompatibleModel:
    """Any OpenAI-compatible endpoint (OpenAI, Groq, Ollama, ...)."""

    def __init__(self, model: str, *, base_url: str | None = None,
                 api_key_env: str = "OPENAI_API_KEY", max_tokens: int = 256,
                 name: str | None = None):
        import os

        from openai import OpenAI

        self.name = name or model
        self.model = model
        self.max_tokens = max_tokens
        self._client = OpenAI(
            base_url=base_url,
            api_key=os.environ.get(api_key_env, "not-needed"),
        )

    def answer(self, question: str, context: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user",
                       "content": PROMPT.format(question=question, context=context)}],
        )
        return resp.choices[0].message.content or ""


class CachedModel:
    """Wrap a QAModel with a JSON disk cache so reruns don't re-spend tokens.

    Keyed by (model name, question, context) — identical calls return the cached
    answer. The cache is flushed on every miss so a crashed run loses nothing.
    """

    def __init__(self, inner: "QAModel", path: str):
        import json
        import os

        self.inner = inner
        self.name = getattr(inner, "name", "model")
        self.path = path
        self._cache: dict = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}

    def _key(self, question: str, context: str) -> str:
        import hashlib

        h = hashlib.sha1()
        h.update(f"{self.name}\x00{question}\x00{context}".encode("utf-8", "replace"))
        return h.hexdigest()

    def _flush(self) -> None:
        import json
        import os

        tmp = self.path + ".tmp"
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._cache, f)
        os.replace(tmp, self.path)

    def answer(self, question: str, context: str) -> str:
        key = self._key(question, context)
        if key in self._cache:
            return self._cache[key]
        ans = self.inner.answer(question, context)
        self._cache[key] = ans
        self._flush()
        return ans


def get_model(name: str, *, model_id: str | None = None) -> QAModel:
    """Factory: 'offline' | 'claude' | 'openai' | 'groq' | 'ollama'."""
    name = name.lower()
    if name == "offline":
        return KeywordModel()
    if name == "claude":
        return ClaudeModel(model=model_id or "claude-haiku-4-5")
    if name == "openai":
        return OpenAICompatibleModel(model=model_id or "gpt-4o-mini")
    if name == "groq":
        return OpenAICompatibleModel(
            model=model_id or "llama-3.3-70b-versatile",
            base_url="https://api.groq.com/openai/v1",
            api_key_env="GROQ_API_KEY",
        )
    if name == "ollama":
        return OpenAICompatibleModel(
            model=model_id or "llama3.1",
            base_url="http://localhost:11434/v1",
            api_key_env="OLLAMA_API_KEY",
        )
    raise ValueError(f"unknown model '{name}' (use offline|claude|openai|groq|ollama)")
