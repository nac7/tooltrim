"""Optional LLM-based distiller (provider-agnostic).

The deterministic compressors run with zero LLM calls. When you want maximal
compression — or summarization rather than extraction — you can plug in *any*
LLM via a simple callable. tooltrim stays provider-agnostic: you supply a
function ``complete(prompt: str) -> str`` wrapping OpenAI, Anthropic, a local
model, anything.

Use a small/cheap model here: distilling a 12k-token tool result down to a few
hundred tokens with a cheap model still saves the *expensive* model from
re-reading the full blob on every subsequent turn.
"""

from __future__ import annotations

from typing import Callable, Optional

from .tokens import count_tokens

CompleteFn = Callable[[str], str]

_DEFAULT_PROMPT = (
    "You are compressing a tool result so an AI agent can keep working with a "
    "small context. Keep every fact, value, id, number, name, and error the "
    "agent will likely need; drop boilerplate, navigation, and repetition. "
    "Target about {budget} tokens.{focus}\n\n"
    "--- TOOL RESULT ---\n{content}\n--- END ---\n\n"
    "Compressed result:"
)


class LLMDistiller:
    """Distill text with a user-supplied completion function.

    Args:
        complete: ``f(prompt) -> str``; wraps any LLM provider.
        max_tokens: Target budget for the distilled output.
        prompt_template: Override the distillation instructions. Receives
            ``budget``, ``focus`` and ``content`` format keys.
    """

    def __init__(
        self,
        complete: CompleteFn,
        *,
        max_tokens: int = 512,
        prompt_template: Optional[str] = None,
    ) -> None:
        self.complete = complete
        self.max_tokens = max_tokens
        self.prompt_template = prompt_template or _DEFAULT_PROMPT

    def compress(
        self,
        text: str,
        query: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        budget = max_tokens if max_tokens is not None else self.max_tokens
        if count_tokens(text) <= budget:
            return text
        focus = (
            ""
            if not query
            else " The agent is specifically trying to: %s" % query.strip()
        )
        prompt = self.prompt_template.format(budget=budget, focus=focus, content=text)
        try:
            return self.complete(prompt).strip()
        except Exception:
            # Never let distillation failure break the tool; caller can chain a
            # deterministic compressor as fallback.
            return text
