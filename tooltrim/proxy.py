"""Drop-in compression proxy — zero-code-change adoption.

Point any client at this proxy and every tool result in the request is
compressed by tooltrim *before* being forwarded upstream. The agent's most
recent user message is used as the relevance query, so each bloated tool result
is trimmed to what the model actually needs — across any language or framework,
no app changes.

Two wire formats are understood, routed by request path:
  - **OpenAI** (`/v1/chat/completions`): `role:"tool"` / `role:"function"`
    messages whose ``content`` is the tool-result string.
  - **Anthropic** (`/v1/messages`): ``tool_result`` blocks nested inside a
    ``role:"user"`` message's ``content`` list (content may be a string or a
    list of ``{"type":"text",...}`` blocks).

Stdlib-only (http.server + urllib) to keep the dependency-free promise. The
request body is rewritten; the upstream response is passed through unchanged
(streaming responses are forwarded as-is). Every transform **fails open**: any
unexpected shape forwards the original bytes so a production call never breaks.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .core import ToolCompressor


@dataclass
class CompressionStats:
    original_tokens: int = 0
    compressed_tokens: int = 0
    messages_compressed: int = 0
    failed_open: bool = False  # an error forced an untouched passthrough

    @property
    def saved_tokens(self) -> int:
        return max(0, self.original_tokens - self.compressed_tokens)


def _latest_user_query(messages: List[Dict[str, Any]]) -> Optional[str]:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):  # OpenAI content-parts form
                parts = [p.get("text", "") for p in content
                         if isinstance(p, dict) and p.get("type") == "text"]
                if parts:
                    return " ".join(parts)
    return None


def _compress_str(text: str, compressor: ToolCompressor, query: Optional[str],
                  stats: CompressionStats) -> Optional[str]:
    """Compress one tool-result string, updating ``stats``.

    Returns the compressed text, or ``None`` if it was already within budget
    (so callers can leave the original block untouched).
    """
    if not isinstance(text, str) or not text:
        return None
    res = compressor.compress(text, query=query)
    if not res.compressed:
        return None
    stats.original_tokens += res.original_tokens
    stats.compressed_tokens += res.compressed_tokens
    stats.messages_compressed += 1
    return res.text


def compress_messages(
    messages: List[Dict[str, Any]],
    compressor: ToolCompressor,
    *,
    query: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], CompressionStats]:
    """Return a new messages list with OpenAI tool/function results compressed.

    Pure function — no I/O — so it can be unit-tested without a network.
    """
    if query is None:
        query = _latest_user_query(messages)

    stats = CompressionStats()
    out: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role in ("tool", "function") and isinstance(content, str):
            compressed = _compress_str(content, compressor, query, stats)
            if compressed is not None:
                new_msg = dict(msg)
                new_msg["content"] = compressed
                out.append(new_msg)
                continue
        out.append(msg)
    return out, stats


def _compress_tool_result_block(
    block: Dict[str, Any], compressor: ToolCompressor,
    query: Optional[str], stats: CompressionStats,
) -> Tuple[Dict[str, Any], bool]:
    """Compress one Anthropic ``tool_result`` block (content: str or block list)."""
    inner = block.get("content")
    if isinstance(inner, str):
        compressed = _compress_str(inner, compressor, query, stats)
        if compressed is not None:
            nb = dict(block)
            nb["content"] = compressed
            return nb, True
        return block, False
    if isinstance(inner, list):
        new_inner: List[Any] = []
        changed = False
        for part in inner:
            if isinstance(part, dict) and part.get("type") == "text":
                compressed = _compress_str(part.get("text", ""), compressor,
                                           query, stats)
                if compressed is not None:
                    np = dict(part)
                    np["text"] = compressed
                    new_inner.append(np)
                    changed = True
                    continue
            new_inner.append(part)
        if changed:
            nb = dict(block)
            nb["content"] = new_inner
            return nb, True
    return block, False


def compress_anthropic_messages(
    messages: List[Dict[str, Any]],
    compressor: ToolCompressor,
    *,
    query: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], CompressionStats]:
    """Return a new messages list with Anthropic ``tool_result`` blocks compressed.

    In the Messages API a tool result is a block inside a ``role:"user"``
    message's ``content`` list, not a standalone message. Pure function — no I/O.
    """
    if query is None:
        query = _latest_user_query(messages)

    stats = CompressionStats()
    out: List[Dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if msg.get("role") == "user" and isinstance(content, list):
            new_content: List[Any] = []
            changed = False
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    nb, used = _compress_tool_result_block(
                        block, compressor, query, stats)
                    new_content.append(nb)
                    changed = changed or used
                else:
                    new_content.append(block)
            if changed:
                new_msg = dict(msg)
                new_msg["content"] = new_content
                out.append(new_msg)
                continue
        out.append(msg)
    return out, stats


def transform_request_body(raw: bytes, compressor: ToolCompressor,
                           *, api: str = "openai") -> Tuple[bytes, CompressionStats]:
    """Compress tool results inside a raw JSON request body.

    ``api`` selects the wire format: ``"openai"`` (chat/completions) or
    ``"anthropic"`` (messages). Fails open: any error (non-JSON body, unexpected
    shape, compressor failure) returns the original bytes untouched so a
    production call is never broken.
    """
    try:
        payload = json.loads(raw)
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return raw, CompressionStats()
        if api == "anthropic":
            new_messages, stats = compress_anthropic_messages(messages, compressor)
        else:
            new_messages, stats = compress_messages(messages, compressor)
        payload["messages"] = new_messages
        return json.dumps(payload).encode("utf-8"), stats
    except Exception:
        return raw, CompressionStats(failed_open=True)


# --- HTTP server ---------------------------------------------------------------

def make_handler(compressor: ToolCompressor, upstream_base: str, verbose: bool = True,
                 metrics=None):
    import time
    import urllib.error
    import urllib.request
    from http.server import BaseHTTPRequestHandler

    from .metrics import default_metrics

    metrics = metrics if metrics is not None else default_metrics

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # silence default noisy logging
            pass

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/").endswith("/metrics"):
                self._send(200, metrics.render().encode("utf-8"),
                           "text/plain; version=0.0.4")
            else:
                self._send(404, b'{"error":"not found"}', "application/json")

        def _proxy(self) -> None:
            t0 = time.monotonic()
            upstream_error = False
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length else b""

            body, stats = (raw, CompressionStats())
            path = self.path.rstrip("/")
            if path.endswith("/chat/completions"):
                body, stats = transform_request_body(raw, compressor, api="openai")
            elif path.endswith("/messages"):
                body, stats = transform_request_body(raw, compressor, api="anthropic")

            upstream = upstream_base.rstrip("/") + self.path[len("/v1"):] \
                if self.path.startswith("/v1") else upstream_base.rstrip("/") + self.path

            req = urllib.request.Request(upstream, data=body, method="POST")
            # Forward the client's headers, minus hop-by-hop / length ones we
            # recompute. Crucially this preserves User-Agent and Accept: many
            # provider edges (Groq/OpenAI/Anthropic sit behind Cloudflare) reject
            # the default "Python-urllib" UA with an error 1010.
            _skip = {"host", "content-length", "connection", "transfer-encoding",
                     "accept-encoding", "keep-alive", "proxy-connection"}
            for h in self.headers:
                if h.lower() not in _skip:
                    req.add_header(h, self.headers[h])
            if not req.has_header("User-agent"):
                req.add_header("User-Agent", "tooltrim-proxy")
            req.add_header("Content-Length", str(len(body)))

            try:
                with urllib.request.urlopen(req) as resp:
                    data = resp.read()
                    self.send_response(resp.status)
                    for k, v in resp.headers.items():
                        if k.lower() in ("transfer-encoding", "content-length", "connection"):
                            continue
                        self.send_header(k, v)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
            except urllib.error.HTTPError as e:
                data = e.read()
                self.send_response(e.code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:  # upstream unreachable
                upstream_error = True
                msg = json.dumps({"error": {"message": f"tooltrim proxy: {e}"}}).encode()
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)

            metrics.record(
                messages=stats.messages_compressed,
                tokens_in=stats.original_tokens,
                tokens_out=stats.compressed_tokens,
                latency_s=time.monotonic() - t0,
                fail_open=stats.failed_open,
                upstream_error=upstream_error,
            )
            if verbose and stats.messages_compressed:
                print(f"[tooltrim] {stats.messages_compressed} tool msg(s): "
                      f"{stats.original_tokens}->{stats.compressed_tokens} tokens "
                      f"(saved {stats.saved_tokens})", flush=True)

        def do_POST(self) -> None:  # noqa: N802
            self._proxy()

    return _Handler


def serve(host: str = "127.0.0.1", port: int = 8800, *,
          max_tokens: int = 512, upstream_base: Optional[str] = None,
          store=None, add_footer: bool = False, metrics=None) -> None:
    from http.server import ThreadingHTTPServer

    upstream = upstream_base or os.environ.get(
        "TOOLTRIM_UPSTREAM_BASE_URL", "https://api.openai.com/v1")
    compressor = ToolCompressor(max_tokens=max_tokens, add_footer=add_footer,
                                store=store)
    handler = make_handler(compressor, upstream, metrics=metrics)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"tooltrim proxy on http://{host}:{port}/v1  ->  {upstream}  "
          f"(budget={max_tokens} tok/tool-result)\n"
          f"  metrics: http://{host}:{port}/metrics (Prometheus)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
