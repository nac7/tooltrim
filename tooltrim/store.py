"""Full-output stash with stable references.

Compression is lossy by design, so tooltrim keeps the *full* original output
addressable behind a short reference id. The agent can call ``expand(ref)`` (or
a configured expand tool) to retrieve the complete output, or a specific slice
of it, when the compressed view is not enough. This turns compression into
"compression + retrieval" rather than irreversible truncation.

The default store (:class:`OutputStore`) is in-process and bounded (LRU) — fine
for a single worker. To scale horizontally (multiple workers/replicas behind a
load balancer) the store must be **shared**, or a ``ref`` minted by one worker
can't be expanded by another. Swap in a shared backend:

  - :class:`FileStore`  — zero-dependency, a shared directory (NFS/EFS/EBS).
  - :class:`RedisStore` — hot, with TTL (``pip install tooltrim[redis]``).
  - :class:`S3Store`    — durable/cold object storage (``pip install tooltrim[s3]``).

All backends are content-addressed: identical content yields the same ref, so
writes are idempotent and dedup for free. A custom backend only needs ``put``
and ``get``; inherit :class:`BaseStore` to get ``expand`` slicing.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Optional


def make_ref(text: str) -> str:
    """Short, stable, content-addressed id. Same content -> same ref."""
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:8]


class BaseStore:
    """Base for stores: backends implement ``put``/``get``; ``expand`` is shared."""

    def put(self, text: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def get(self, ref: str) -> Optional[str]:  # pragma: no cover - interface
        raise NotImplementedError

    def expand(self, ref: str, *, start: int = 0,
               length: Optional[int] = None) -> Optional[str]:
        """Return the full output for ``ref``, or a ``[start:start+length]`` slice."""
        text = self.get(ref)
        if text is None:
            return None
        if length is None:
            return text[start:]
        return text[start : start + length]


class OutputStore(BaseStore):
    """A bounded, thread-safe, content-addressed store for full tool outputs.

    In-process LRU. The default; not shared across processes (see module docs).
    """

    def __init__(self, max_entries: int = 256):
        self.max_entries = max_entries
        self._data: "OrderedDict[str, str]" = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _make_ref(text: str) -> str:  # kept for backwards compatibility
        return make_ref(text)

    def put(self, text: str) -> str:
        ref = make_ref(text)
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

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._data)


class FileStore(BaseStore):
    """Zero-dependency shared store: one file per ref under ``root``.

    Point several workers at the same directory (a shared volume — NFS, EFS, a
    mounted PVC) and refs minted anywhere are expandable everywhere. Durable
    across restarts. Writes are atomic (temp file + rename).
    """

    def __init__(self, root: str):
        import os

        self.root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, ref: str) -> str:
        import os

        return os.path.join(self.root, f"{ref}.txt")

    def put(self, text: str) -> str:
        import os

        ref = make_ref(text)
        path = self._path(ref)
        if not os.path.exists(path):
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, path)
        return ref

    def get(self, ref: str) -> Optional[str]:
        try:
            with open(self._path(ref), "r", encoding="utf-8") as f:
                return f.read()
        except (OSError, ValueError):
            return None


class RedisStore(BaseStore):
    """Shared, hot store backed by Redis, with an optional TTL.

    ``pip install tooltrim[redis]``. Pass a ``redis`` client or a ``url``::

        RedisStore(url="redis://localhost:6379/0", ttl_seconds=86400)
    """

    def __init__(self, client=None, *, url: Optional[str] = None,
                 prefix: str = "tooltrim:", ttl_seconds: Optional[int] = None):
        if client is None:
            import redis  # lazy: only needed for this backend

            client = redis.Redis.from_url(url or "redis://localhost:6379/0")
        self._r = client
        self.prefix = prefix
        self.ttl = ttl_seconds

    def put(self, text: str) -> str:
        ref = make_ref(text)
        self._r.set(self.prefix + ref, text.encode("utf-8"), ex=self.ttl)
        return ref

    def get(self, ref: str) -> Optional[str]:
        val = self._r.get(self.prefix + ref)
        if val is None:
            return None
        return val.decode("utf-8") if isinstance(val, (bytes, bytearray)) else str(val)


class S3Store(BaseStore):
    """Durable, shared store backed by S3 (or any S3-compatible object store).

    ``pip install tooltrim[s3]``. Expiry is best handled with an S3 lifecycle
    rule on ``prefix``. Pass a boto3 client or let it build a default one::

        S3Store(bucket="my-bucket", prefix="tooltrim/")
    """

    def __init__(self, bucket: str, *, client=None, prefix: str = "tooltrim/"):
        if client is None:
            import boto3  # lazy: only needed for this backend

            client = boto3.client("s3")
        self._s3 = client
        self.bucket = bucket
        self.prefix = prefix

    def _key(self, ref: str) -> str:
        return f"{self.prefix}{ref}"

    def put(self, text: str) -> str:
        ref = make_ref(text)
        self._s3.put_object(Bucket=self.bucket, Key=self._key(ref),
                            Body=text.encode("utf-8"))
        return ref

    def get(self, ref: str) -> Optional[str]:
        try:
            obj = self._s3.get_object(Bucket=self.bucket, Key=self._key(ref))
        except Exception:
            return None
        return obj["Body"].read().decode("utf-8")
