"""Core API: :class:`ToolCompressor` and :class:`CompressionResult`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from .detect import detect_type
from .store import BaseStore, OutputStore
from .tokens import count_tokens
from .compressors import html as _html
from .compressors import json_ as _json
from .compressors import logs as _logs
from .compressors import tabular as _tab
from .compressors import text as _text

_COMPRESSORS = {
    "html": _html.compress,
    "json": _json.compress,
    "tabular": _tab.compress,
    "logs": _logs.compress,
    "text": _text.compress,
}


@dataclass
class CompressionResult:
    """Outcome of compressing a single tool output."""

    text: str
    content_type: str
    original_tokens: int
    compressed_tokens: int
    ref: Optional[str] = None
    compressed: bool = True

    @property
    def saved_tokens(self) -> int:
        return max(0, self.original_tokens - self.compressed_tokens)

    @property
    def saved_ratio(self) -> float:
        if self.original_tokens <= 0:
            return 0.0
        return self.saved_tokens / self.original_tokens

    def __str__(self) -> str:  # what an agent actually consumes
        return self.text


def _coerce(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, bytes):
        return output.decode("utf-8", "replace")
    try:
        import json

        return json.dumps(output, ensure_ascii=False, default=str)
    except Exception:
        return str(output)


class ToolCompressor:
    """Compress tool outputs to fit a token budget, keeping the full output retrievable.

    Args:
        max_tokens: Target budget for the compressed output.
        store: Where full outputs are stashed for expand-on-demand. Pass ``None``
            to disable stashing (no refs, lower memory).
        add_footer: Append a one-line note with savings + the expand ref.
        footer_template: Customize the footer. Receives ``saved``, ``original``,
            ``compressed``, ``ref`` as ``str.format`` keys.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        *,
        store: Optional[BaseStore] = "default",  # type: ignore[assignment]
        add_footer: bool = True,
        footer_template: Optional[str] = None,
    ) -> None:
        self.max_tokens = max_tokens
        if store == "default":
            store = OutputStore()
        self.store: Optional[BaseStore] = store
        self.add_footer = add_footer
        self.footer_template = footer_template or (
            "\n\n[tooltrim: compressed {original}->{compressed} tokens "
            "(saved {saved}); full output ref={ref}]"
        )

    def compress(
        self,
        output: Any,
        *,
        query: Optional[str] = None,
        content_type: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> CompressionResult:
        text = _coerce(output)
        original = count_tokens(text)
        budget = max_tokens if max_tokens is not None else self.max_tokens

        # Already within budget: pass through untouched (no ref, no footer).
        if original <= budget:
            ctype = content_type or detect_type(text)
            return CompressionResult(text, ctype, original, original,
                                     ref=None, compressed=False)

        ctype = content_type or detect_type(text)
        compressor = _COMPRESSORS.get(ctype, _text.compress)
        # Leave headroom for the footer line.
        body_budget = max(16, budget - (24 if self.add_footer else 0))
        body = compressor(text, query, body_budget)

        ref = self.store.put(text) if self.store is not None else None

        body_tokens = count_tokens(body)
        final = body
        if self.add_footer and ref is not None:
            final = body + self.footer_template.format(
                original=original,
                compressed=body_tokens,
                saved=max(0, original - body_tokens),
                ref=ref,
            )

        return CompressionResult(
            text=final,
            content_type=ctype,
            original_tokens=original,
            compressed_tokens=count_tokens(final),
            ref=ref,
            compressed=True,
        )

    def expand(self, ref: str, *, start: int = 0,
               length: Optional[int] = None) -> Optional[str]:
        """Retrieve the full (or sliced) original output behind ``ref``."""
        if self.store is None:
            return None
        return self.store.expand(ref, start=start, length=length)

    # --- expand-as-a-tool ----------------------------------------------------
    # Register this with your agent so it can pull back the full output when a
    # compressed extract isn't enough — turning aggressive compression into a
    # safe default rather than a lossy gamble.

    EXPAND_TOOL_NAME = "expand_tool_output"

    def expand_tool_spec(self, *, style: str = "openai") -> dict:
        """Return a tool/function definition for the expand tool.

        ``style``: "openai" (chat/completions function), "anthropic" (Messages
        tool), or "raw" (name/description/schema).
        """
        description = (
            "Retrieve the full, uncompressed output behind a tooltrim reference "
            "(shown in compressed results as 'full output ref=XXXX'). Call this "
            "when the compressed extract is missing a detail you need. Returns a "
            "page of characters; use start/length to read more."
        )
        schema = {
            "type": "object",
            "properties": {
                "ref": {"type": "string",
                        "description": "the ref id, e.g. a1b2c3d4"},
                "start": {"type": "integer",
                          "description": "character offset to start from (default 0)"},
                "length": {"type": "integer",
                           "description": "max characters to return (default: one page)"},
            },
            "required": ["ref"],
        }
        if style == "openai":
            return {"type": "function", "function": {
                "name": self.EXPAND_TOOL_NAME,
                "description": description, "parameters": schema}}
        if style == "anthropic":
            return {"name": self.EXPAND_TOOL_NAME,
                    "description": description, "input_schema": schema}
        return {"name": self.EXPAND_TOOL_NAME,
                "description": description, "schema": schema}

    def handle_expand(self, ref: str, *, start: int = 0,
                      length: Optional[int] = None, page_chars: int = 8000) -> str:
        """Execute an expand call; returns text (paged) or a clear error string.

        Wire this to the tool the agent calls. When ``length`` is omitted, a
        single page (``page_chars``) is returned with a continuation hint so the
        model can fetch the next page rather than flooding its own context.
        """
        text = self.expand(ref, start=start, length=length)
        if text is None:
            return (f"[tooltrim: no stored output for ref={ref!r}. It may have "
                    f"expired or the store is disabled.]")
        if length is None and len(text) > page_chars:
            nxt = start + page_chars
            return (text[:page_chars] +
                    f"\n[tooltrim: truncated to {page_chars} chars; call "
                    f"{self.EXPAND_TOOL_NAME}(ref={ref!r}, start={nxt}) for more]")
        return text
