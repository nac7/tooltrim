"""Lightweight metrics for the proxy — Prometheus text format, zero dependencies.

Enterprises won't run a proxy they can't observe. This emits the standard
Prometheus exposition format (scrape it at ``GET /metrics``) without requiring
``prometheus_client``: a small thread-safe registry of counters plus a couple of
sum/count pairs for averages.

    from tooltrim.metrics import Metrics
    m = Metrics()
    m.record(messages=1, tokens_in=14415, tokens_out=26, latency_s=0.21)
    print(m.render())          # Prometheus text
"""

from __future__ import annotations

import threading
from typing import Dict


class Metrics:
    """Thread-safe counters for proxy activity."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._c: Dict[str, float] = {
            "tooltrim_requests_total": 0.0,
            "tooltrim_messages_compressed_total": 0.0,
            "tooltrim_tokens_in_total": 0.0,
            "tooltrim_tokens_out_total": 0.0,
            "tooltrim_tokens_saved_total": 0.0,
            "tooltrim_fail_open_total": 0.0,
            "tooltrim_upstream_errors_total": 0.0,
            "tooltrim_request_latency_seconds_sum": 0.0,
            "tooltrim_request_latency_seconds_count": 0.0,
        }

    def record(self, *, messages: int = 0, tokens_in: int = 0,
               tokens_out: int = 0, latency_s: float | None = None,
               fail_open: bool = False, upstream_error: bool = False) -> None:
        """Record one proxied request."""
        with self._lock:
            self._c["tooltrim_requests_total"] += 1
            self._c["tooltrim_messages_compressed_total"] += messages
            self._c["tooltrim_tokens_in_total"] += tokens_in
            self._c["tooltrim_tokens_out_total"] += tokens_out
            self._c["tooltrim_tokens_saved_total"] += max(0, tokens_in - tokens_out)
            if fail_open:
                self._c["tooltrim_fail_open_total"] += 1
            if upstream_error:
                self._c["tooltrim_upstream_errors_total"] += 1
            if latency_s is not None:
                self._c["tooltrim_request_latency_seconds_sum"] += latency_s
                self._c["tooltrim_request_latency_seconds_count"] += 1

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._c)

    def render(self) -> str:
        """Render the registry as Prometheus text exposition format."""
        helps = {
            "tooltrim_requests_total": ("counter", "Proxied requests seen."),
            "tooltrim_messages_compressed_total": ("counter", "Tool messages compressed."),
            "tooltrim_tokens_in_total": ("counter", "Tokens in compressed tool messages (pre)."),
            "tooltrim_tokens_out_total": ("counter", "Tokens after compression (post)."),
            "tooltrim_tokens_saved_total": ("counter", "Tokens saved by compression."),
            "tooltrim_fail_open_total": ("counter", "Requests forwarded unmodified after an error."),
            "tooltrim_upstream_errors_total": ("counter", "Upstream/transport failures."),
            "tooltrim_request_latency_seconds": ("summary", "Proxy round-trip latency."),
        }
        snap = self.snapshot()
        lines = []
        for base, (mtype, help_text) in helps.items():
            lines.append(f"# HELP {base} {help_text}")
            lines.append(f"# TYPE {base} {mtype}")
            if mtype == "summary":
                lines.append(f"{base}_sum {snap[base + '_sum']:g}")
                lines.append(f"{base}_count {snap[base + '_count']:g}")
            else:
                lines.append(f"{base} {snap[base]:g}")
        return "\n".join(lines) + "\n"


# Process-wide default registry (the proxy uses this unless given another).
default_metrics = Metrics()
