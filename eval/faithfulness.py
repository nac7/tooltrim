"""Structural-faithfulness metrics: is the compressed output still *usable*?

Recall (did the fact survive?) is only half the story. An agent typically has to
*act on* a tool result — parse the JSON, read the table, follow the schema — with
its next tool call. Query-aware selectors that return disconnected chunks
(RAG top-k, truncation) can shred a serialized JSON object into an unparseable
fragment even when the needed fact is inside it. These metrics measure that,
deterministically and with no API calls, so they cost nothing to add to a run.

Two measures, both computed from the compressed string + its content type:

- ``is_parseable(text, content_type)`` — does the *whole* output parse as its
  declared type? This is exactly what the agent's next ``json.loads`` /
  ``csv.DictReader`` does. A fragment that merely *contains* a valid sub-object
  is not parseable as a whole and fails, which is the honest bar.
- ``downstream_extractable(text, case)`` — for json/tabular, can the gold fact be
  recovered *programmatically* from a valid parse (not just spotted in prose)?

The definitions are content-type-defined, not tuned to any one compressor:
line-oriented outputs (logs, CSV rows, prose) are preserved by chunk selection
too, so those types legitimately do not separate — only serialized/structured
outputs (single-line JSON, aggregation over many records) do.
"""

from __future__ import annotations

import csv
import io
import json
import re
from typing import Optional

from .judge import matches, normalize

_LOG_PREFIX = re.compile(
    r"^\s*\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\s+"
    r"(?:INFO|WARN|WARNING|ERROR|DEBUG|TRACE|FATAL)\b",
    re.IGNORECASE,
)
_MARKUP = re.compile(r"</?(?:script|style|div|span|nav|header|footer|li|ul|a|p)\b", re.I)

# Types where "parse it and pull the field out in code" is a meaningful next step.
_DOWNSTREAM_TYPES = ("json", "tabular")


def _load_json_whole(text: str):
    """Parse ``text`` as JSON if the *whole* string is one JSON value.

    Tolerates surrounding whitespace and a single trailing footer line (some
    compressors append a short ``[+N chars omitted, ref=…]`` note). Returns the
    parsed value, or ``None`` if the whole output is not valid JSON — a mid-object
    fragment (what chunk selection produces on single-line JSON) returns ``None``.
    """
    s = text.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # allow one trailing footer line
    if "\n" in s:
        head = s.rsplit("\n", 1)[0].strip()
        try:
            return json.loads(head)
        except json.JSONDecodeError:
            return None
    return None


def _csv_table(text: str):
    """Return (fieldnames, rows) if ``text`` is a CSV table with a header and at
    least one data row of matching width; else ``None``."""
    s = text.strip()
    if not s or "\n" not in s:
        return None
    reader = csv.reader(io.StringIO(s))
    try:
        rows = [r for r in reader if r]
    except csv.Error:
        return None
    if len(rows) < 2:
        return None
    header = rows[0]
    if len(header) < 2:
        return None
    # a header row's cells should not be purely numeric (that would be data)
    if all(re.fullmatch(r"-?\d+(?:\.\d+)?", c.strip() or "") for c in header):
        return None
    # every row must have the header's width: a ragged/truncated row means the
    # output was cut mid-record (a fragment), which is not a usable table.
    width = len(header)
    if any(len(r) != width for r in rows):
        return None
    return header, rows[1:]


def is_parseable(text: str, content_type: str) -> bool:
    """Does the whole output parse as its declared content type?

    ``text`` and ``html`` are treated as prose (no machine grammar to break), so
    they always pass — the metric never penalizes a type it can't fairly judge.
    """
    ct = (content_type or "").lower()
    if ct == "json":
        return _load_json_whole(text) is not None
    if ct == "tabular":
        return _csv_table(text) is not None
    if ct == "logs":
        lines = [ln for ln in text.splitlines() if ln.strip()]
        return bool(lines) and all(_LOG_PREFIX.match(ln) for ln in lines)
    if ct == "html":
        return not _MARKUP.search(text)
    return True  # prose / unknown: nothing to break


def downstream_applies(content_type: str) -> bool:
    return (content_type or "").lower() in _DOWNSTREAM_TYPES


def downstream_extractable(text: str, case) -> bool:
    """Can the gold fact be pulled out of a *valid parse* by code (json/tabular)?

    Mirrors the agent's next step: parse the tool output, then find the field.
    A fragment that fails to parse fails here even if the fact is textually
    present — that is the whole point.
    """
    ct = (case.content_type or "").lower()
    golds = [case.gold, *case.all_of]
    if ct == "json":
        val = _load_json_whole(text)
        if val is None:
            return False
        blob = json.dumps(val, ensure_ascii=False)
        return all(matches(blob, g) for g in golds)
    if ct == "tabular":
        table = _csv_table(text)
        if table is None:
            return False
        _, data = table
        blob = "\n".join(",".join(r) for r in data)
        return all(matches(blob, g) for g in golds)
    return False


def parseable_rate(pairs) -> float:
    """Mean of ``is_parseable`` over ``(text, content_type)`` pairs."""
    pairs = list(pairs)
    if not pairs:
        return 1.0
    return sum(is_parseable(t, ct) for t, ct in pairs) / len(pairs)


def downstream_rate(items) -> Optional[float]:
    """Mean of ``downstream_extractable`` over the applicable cases in ``items``
    (each an ``(text, case)`` pair). Returns ``None`` if none apply."""
    applicable = [(t, c) for t, c in items if downstream_applies(c.content_type)]
    if not applicable:
        return None
    return sum(downstream_extractable(t, c) for t, c in applicable) / len(applicable)
