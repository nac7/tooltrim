"""Content-type detection for tool outputs.

Cheap, dependency-free heuristics to route a blob of text to the right
compressor. Detection is best-effort; callers can always override with an
explicit ``content_type``.
"""

from __future__ import annotations

import json
import re

ContentType = str  # one of: "json", "html", "tabular", "logs", "text"

_HTML_HINT = re.compile(r"<!doctype html|<html[\s>]|<body[\s>]|<div[\s>]|<p[\s>]", re.I)
_TAG = re.compile(r"<[a-zA-Z/][^>]{0,200}>")
_LOG_LINE = re.compile(
    r"""(?xim)
    ^\s*(
        \d{4}-\d{2}-\d{2}[ t]\d{2}:\d{2}        # ISO timestamp
        | \[\d{2}:\d{2}:\d{2}\]                  # [hh:mm:ss]
        | (trace|debug|info|warn|warning|error|err|fatal|critical)\b   # level word
        | (traceback \(most recent call last\))  # python traceback
    )
    """
)


def _looks_json(text: str) -> bool:
    s = text.lstrip()
    if not s or s[0] not in "{[":
        return False
    try:
        json.loads(text)
        return True
    except Exception:
        return False


def _looks_tabular(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()][:20]
    if len(lines) < 3:
        return False
    for delim in ("\t", ",", "|"):
        counts = [ln.count(delim) for ln in lines]
        first = counts[0]
        # Consistent, non-trivial delimiter count across rows => table.
        if first >= 1 and all(c == first for c in counts):
            return True
    return False


def _looks_logs(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()][:40]
    if len(lines) < 4:
        return False
    hits = sum(1 for ln in lines if _LOG_LINE.search(ln))
    return hits >= max(3, len(lines) // 3)


def detect_type(text: str) -> ContentType:
    """Classify ``text`` into a coarse content type for compressor routing."""
    if not text or not text.strip():
        return "text"

    if _looks_json(text):
        return "json"

    if _HTML_HINT.search(text) or len(_TAG.findall(text)) >= 5:
        return "html"

    if _looks_logs(text):
        return "logs"

    if _looks_tabular(text):
        return "tabular"

    return "text"
