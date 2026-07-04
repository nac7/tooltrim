"""A zero-dependency mock LLM upstream for load testing the proxy in isolation.

Returns a canned OpenAI/Anthropic-shaped response for any POST, so a load test
measures the *proxy's* compression throughput and overhead without spending money
or hitting a real provider's rate limits. Latency is optionally simulated via
UPSTREAM_DELAY_MS to model a real backend.
"""

from __future__ import annotations

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DELAY_MS = int(os.environ.get("UPSTREAM_DELAY_MS", "0"))
_OPENAI = {
    "id": "chatcmpl-mock", "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                 "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 0, "completion_tokens": 1, "total_tokens": 1},
}
_ANTHROPIC = {
    "id": "msg-mock", "type": "message", "role": "assistant",
    "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
    "usage": {"input_tokens": 0, "output_tokens": 1},
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, obj):
        if DELAY_MS:
            time.sleep(DELAY_MS / 1000.0)
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        self._send(_ANTHROPIC if "messages" in self.path else _OPENAI)

    def do_GET(self):  # health
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")


def main():
    port = int(os.environ.get("PORT", "9000"))
    print(f"mock upstream on :{port} (delay {DELAY_MS}ms)", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
