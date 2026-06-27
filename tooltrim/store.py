"""Full-output stash with stable references.

Compression is lossy by design, so tooltrim keeps the *full* original output
addressable behind a short reference id. The agent can call ``expand(ref)`` (or
a configured expand tool) to retrieve the complete output, or a specific slice
of it, when the compressed view is not enough. This turns compression into
"compression + retrieval" rather than irreversible truncation.

The default store is in-process and bounded (LRU). Swap in your own by
implementing ``put``/``get`` and passing it to :class:`~tooltrim.ToolCompressor`.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Optional


class OutputStore:
    """A bounded, thread-safe, content-addressed store for full tool outputs."""

    def __init__(self, max_entries: int = 256):
        self.max_entries = max_entries
        self._data: "OrderedDict[str, str]" = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _make_ref(text: str) -> str:
        # Short, stable, content-addressed id. Same content -> same ref.
        return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:8]

    def put(self, text: str) -> str:
        ref = self._make_ref(text)
        with self._lock:
            if ref in self._data:
                self._data.move_to_end(ref)
            else:
                self._data[ref] = text
                while len(self._data) > self.max_entries:
                    self._data.popitem(last=False)
        return ref

    def get(self, ref: str) -> Optional[str]:
        with self._lock:
            text = self._data.get(ref)
            if text is not None:
                self._data.move_to_end(ref)
            return text

    def expand(
        self,
        ref: str,
        *,
        start: int = 0,
        length: Optional[int] = None,
    ) -> Optional[str]:
        """Return the full output for ``ref``, or a ``[start:start+length]`` slice."""
        text = self.get(ref)
        if text is None:
            return None
        if length is None:
            return text[start:]
        return text[start : start + length]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._data)
