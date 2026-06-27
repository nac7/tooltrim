"""HTML compressor: extract readable text, then fit the budget.

Uses the stdlib ``html.parser`` (no bs4/lxml dependency). Drops non-content
elements (script/style/nav/header/footer/etc.), keeps block structure as
paragraph breaks, and then applies query-aware extraction to fit the budget.
"""

from __future__ import annotations

from html.parser import HTMLParser
from typing import List

from ._budget import fit_chunks

_SKIP_TAGS = {
    "script", "style", "noscript", "template", "svg", "canvas",
    "head", "nav", "header", "footer", "aside", "form", "button",
}
_BLOCK_TAGS = {
    "p", "div", "section", "article", "li", "tr", "br", "h1", "h2",
    "h3", "h4", "h5", "h6", "pre", "blockquote", "td", "th",
}


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._parts: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0 and data.strip():
            self._parts.append(data.strip())

    def text(self) -> str:
        return " ".join(self._parts)


def extract_text(html: str) -> str:
    parser = _Extractor()
    try:
        parser.feed(html)
    except Exception:
        # Malformed HTML: fall back to whatever we extracted before the error.
        pass
    raw = parser.text()
    # Normalise whitespace into paragraph blocks.
    paragraphs = [seg.strip() for seg in raw.split("\n")]
    paragraphs = [" ".join(p.split()) for p in paragraphs if p.strip()]
    return "\n\n".join(paragraphs)


def compress(text: str, query: str | None, max_tokens: int) -> str:
    extracted = extract_text(text)
    chunks = [p for p in extracted.split("\n\n") if p.strip()]
    return fit_chunks(chunks, query, max_tokens)
