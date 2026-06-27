"""OpenAI-compatible compression proxy — zero-code-change adoption.

Point any OpenAI-compatible client at this proxy (`base_url=http://host:8800/v1`)
and every `role:"tool"` / `role:"function"` message in the request is compressed
by tooltrim *before* being forwarded upstream. The agent's most recent user
message is used as the relevance query, so each bloated tool result is trimmed to
what the model actually needs — across any language or framework, no app changes.

Stdlib-only (http.server + urllib) to keep the dependency-free promise. The
request body is rewritten; the upstream response is passed through unchanged
(streaming responses are forwarded as-is).
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


def compress_messages(
    messages: List[Dict[str, Any]],
    compressor: ToolCompressor,
    *,
    query: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], CompressionStats]:
    """Return a new messages list with tool/function results compressed.

    Pure function — no I/O — so it can be unit-tested without a network.
    """
    if query is None:
        query = _latest_user_query(messages)

    stats = CompressionStats()
    out: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role in ("tool", "function") and isinstance(content, str) and content:
            res = compressor.compress(content, query=query)
            if res.compressed:
                stats.original_tokens += res.original_tokens
                stats.compressed_tokens += res.compressed_tokens
                stats.messages_compressed += 1
                new_msg = dict(msg)
                new_msg["content"] = res.text
                out.append(new_msg)
                continue
        out.append(msg)
    return out, stats


def transform_request_body(raw: bytes, compressor: ToolCompressor) -> Tuple[bytes, CompressionStats]:
    """Compress tool messages inside a raw chat-completions JSON body.

    Fails open: any error (non-JSON body, unexpected shape, compressor failure)
    returns the original bytes untouched so a production call is never broken.
    """
    try:
        payload = json.loads(raw)
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return raw, CompressionStats()
        new_messages, stats = compress_messages(messages, compressor)
        payload["messages"] = new_messages
        return json.dumps(payload).encode("utf-8"), stats
    except Exception:
        return raw, CompressionStats()


# --- HTTP server ---------------------------------------------------------------

def make_handler(compressor: ToolCompressor, upstream_base: str, verbose: bool = True):
    import urllib.error
    import urllib.request
    from http.server import BaseHTTPRequestHandler

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # silence default noisy logging
            pass

        def _proxy(self) -> None:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length else b""

            body, stats = (raw, CompressionStats())
            if self.path.rstrip("/").endswith("/chat/completions"):
                body, stats = transform_request_body(raw, compressor)

            upstream = upstream_base.rstrip("/") + self.path[len("/v1"):] \
                if self.path.startswith("/v1") else upstream_base.rstrip("/") + self.path

            req = urllib.request.Request(upstream, data=body, method="POST")
            for h in ("Authorization", "Content-Type", "OpenAI-Organization",
                      "anthropic-version", "x-api-key"):
                if h in self.headers:
                    req.add_header(h, self.headers[h])
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
                msg = json.dumps({"error": {"message": f"tooltrim proxy: {e}"}}).encode()
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)

            if verbose and stats.messages_compressed:
                print(f"[tooltrim] {stats.messages_compressed} tool msg(s): "
                      f"{stats.original_tokens}->{stats.compressed_tokens} tokens "
                      f"(saved {stats.saved_tokens})", flush=True)

        def do_POST(self) -> None:  # noqa: N802
            self._proxy()

    return _Handler


def serve(host: str = "127.0.0.1", port: int = 8800, *,
          max_tokens: int = 512, upstream_base: Optional[str] = None) -> None:
    from http.server import ThreadingHTTPServer

    upstream = upstream_base or os.environ.get(
        "TOOLTRIM_UPSTREAM_BASE_URL", "https://api.openai.com/v1")
    compressor = ToolCompressor(max_tokens=max_tokens, add_footer=False)
    handler = make_handler(compressor, upstream)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"tooltrim proxy on http://{host}:{port}/v1  ->  {upstream}  "
          f"(budget={max_tokens} tok/tool-result)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
