"""Container entrypoint: tooltrim compression proxy with a shared Redis store.

Configured entirely from the environment so the same image scales horizontally
behind a Service; every replica shares one Redis store, so an ``expand`` for a
compressed output resolves no matter which replica served the original request
(content-addressed refs -> no coordination needed).

Env:
    TOOLTRIM_HOST         bind host (default 0.0.0.0)
    TOOLTRIM_PORT         bind port (default 8800)
    TOOLTRIM_MAX_TOKENS   per-tool-output budget (default 512)
    TOOLTRIM_UPSTREAM     upstream base URL (e.g. the mock upstream or a real API)
    TOOLTRIM_REDIS_URL    redis://host:6379/0 — enables the shared expand store
"""

from __future__ import annotations

import os

from tooltrim.metrics import Metrics
from tooltrim.proxy import serve


def _store():
    url = os.environ.get("TOOLTRIM_REDIS_URL")
    if not url:
        return None
    from tooltrim.store import RedisStore  # requires tooltrim[redis]

    return RedisStore(url=url)


def main() -> None:
    serve(
        host=os.environ.get("TOOLTRIM_HOST", "0.0.0.0"),
        port=int(os.environ.get("TOOLTRIM_PORT", "8800")),
        max_tokens=int(os.environ.get("TOOLTRIM_MAX_TOKENS", "512")),
        upstream_base=os.environ.get("TOOLTRIM_UPSTREAM"),
        store=_store(),
        metrics=Metrics(),
    )


if __name__ == "__main__":
    main()
