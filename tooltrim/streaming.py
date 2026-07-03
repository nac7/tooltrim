"""Streaming compression — bound memory on outputs too big to hold at once.

Some tool outputs arrive as a stream and don't fit in memory: a multi-GB log
file, a subprocess's stdout, an HTTP streaming response. :class:`StreamingCompressor`
consumes such a stream incrementally and keeps memory **constant** regardless of
total size, then emits a compressed, query-relevant result at the end.

How it stays bounded: as lines arrive it retains only
  - a small **head** window and a **tail** ring (structure / fallback),
  - the top-K lines by query-term overlap (relevance), and
  - always-kept **important** lines (errors/warnings), up to a cap.

At :meth:`close` it runs the normal budget fit (with the active scorer and
neighbor context) over just those survivors — so the final selection reuses the
same machinery as in-memory compression. Because the bounding stage is lexical,
a purely semantic match with no shared query words may not survive; for that,
compress in memory with an :class:`~tooltrim.EmbeddingScorer`.

    from tooltrim import compress_stream
    text = compress_stream(open("huge.log"), max_tokens=400, query="disk error")
"""

from __future__ import annotations

import re
from collections import deque
from heapq import heappush, heapreplace
from typing import Iterable, List, Optional, Tuple, Union

from .compressors._budget import _split_oversize, fit_chunks
from .relevance import tokenize, using_scorer
from .tokens import count_tokens

_IMPORTANT = re.compile(r"error|warn|critical|exception|traceback|fatal|fail",
                        re.IGNORECASE)


def _overlap(chunk: str, q_terms: set) -> int:
    if not q_terms:
        return 0
    return len(q_terms & set(tokenize(chunk)))


class StreamingCompressor:
    """Compress an incrementally-fed stream with bounded memory.

    Feed text/bytes with :meth:`feed` (partial lines across calls are handled),
    then call :meth:`close` for the compressed result.
    """

    def __init__(self, max_tokens: int = 512, query: Optional[str] = None, *,
                 scorer=None, keep_chunks: int = 256, head: int = 8,
                 tail: int = 8, max_important: int = 64):
        self.max_tokens = max_tokens
        self.query = query
        self.scorer = scorer
        self.keep_chunks = keep_chunks
        self.head_cap = head
        self._q_terms = set(tokenize(query or ""))
        self._residual = ""
        self._order = 0
        self.original_tokens = 0
        self._head: List[Tuple[int, str]] = []
        self._tail: "deque[Tuple[int, str]]" = deque(maxlen=tail)
        self._topk: List[Tuple[int, int, str]] = []  # min-heap (overlap, order, chunk)
        self._important: List[Tuple[int, str]] = []
        self._max_important = max_important

    def feed(self, data: Union[str, bytes]) -> "StreamingCompressor":
        if isinstance(data, (bytes, bytearray)):
            data = data.decode("utf-8", "replace")
        text = self._residual + data
        lines = text.split("\n")
        self._residual = lines.pop()  # trailing partial line, kept for next feed
        for ln in lines:
            self._ingest(ln)
        return self

    def _ingest(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        for chunk in _split_oversize(line):
            self._add(chunk)

    def _add(self, chunk: str) -> None:
        i = self._order
        self._order += 1
        self.original_tokens += count_tokens(chunk)
        if len(self._head) < self.head_cap:
            self._head.append((i, chunk))
        self._tail.append((i, chunk))
        if _IMPORTANT.search(chunk) and len(self._important) < self._max_important:
            self._important.append((i, chunk))
        if self._q_terms:
            ov = _overlap(chunk, self._q_terms)
            if ov > 0:
                if len(self._topk) < self.keep_chunks:
                    heappush(self._topk, (ov, i, chunk))
                elif ov > self._topk[0][0]:
                    heapreplace(self._topk, (ov, i, chunk))

    def close(self) -> str:
        if self._residual.strip():
            self._ingest(self._residual)
            self._residual = ""
        # Union all retained lines, keyed by original order.
        retained = {}
        for i, ch in self._head:
            retained[i] = ch
        for i, ch in self._tail:
            retained[i] = ch
        for i, ch in self._important:
            retained[i] = ch
        for _ov, i, ch in self._topk:
            retained[i] = ch
        chunks = [retained[i] for i in sorted(retained)]
        with using_scorer(self.scorer):
            return fit_chunks(chunks, self.query, self.max_tokens)


def compress_stream(chunks: Iterable[Union[str, bytes]], max_tokens: int = 512,
                    query: Optional[str] = None, **kwargs) -> str:
    """Feed an iterable of str/bytes through a :class:`StreamingCompressor`.

    ``chunks`` can be a file object, a generator of lines, response chunks, etc.
    """
    sc = StreamingCompressor(max_tokens, query, **kwargs)
    for c in chunks:
        sc.feed(c)
    return sc.close()
