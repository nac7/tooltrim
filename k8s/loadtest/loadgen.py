"""Concurrent load generator for the tooltrim proxy — RPS, p50/p99, tokens saved.

Zero external deps (threads + urllib). Fires realistic, bloated chat-completion
requests (each carries a large tool result) at the proxy and reports throughput,
latency percentiles, and---by diffing the proxy's Prometheus /metrics before and
after---the tokens and dollars saved by compression.

Runnable anywhere the proxy URL is reachable (local, port-forward, or ingress):

    # terminal 1: mock upstream ; terminal 2: proxy ; terminal 3:
    python k8s/loadtest/loadgen.py --url http://localhost:8800 \
        --requests 2000 --concurrency 50 --price-per-mtok 3.0
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

# A bloated tool result (paginated JSON) with one buried fact — the realistic
# shape tooltrim targets. ~large payload so compression has something to do.
_RECORDS = [{"id": i, "status": "ok", "amount": i * 3.5,
             "note": "routine entry no useful content"} for i in range(600)]
_RECORDS[411]["note"] = "refund issued to customer 4417 for amount 250"
_TOOL_OUTPUT = json.dumps({"page": 1, "total": 600, "results": _RECORDS})

_BODY = json.dumps({
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "user", "content": "Which customer got a refund?"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "list_orders", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": _TOOL_OUTPUT},
    ],
}).encode()


def _fetch_metrics(url: str) -> Dict[str, float]:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=10) as r:
            out = {}
            for line in r.read().decode().splitlines():
                if line and not line.startswith("#"):
                    k, _, v = line.partition(" ")
                    try:
                        out[k] = float(v)
                    except ValueError:
                        pass
            return out
    except Exception:
        return {}


def _one(url: str) -> Optional[float]:
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=_BODY, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer test"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()
        return time.perf_counter() - t0
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8800")
    ap.add_argument("--requests", type=int, default=1000)
    ap.add_argument("--concurrency", type=int, default=50)
    ap.add_argument("--price-per-mtok", type=float, default=3.0,
                    help="input $/1M tokens for the cost-saved estimate")
    args = ap.parse_args()

    before = _fetch_metrics(args.url)
    lats: List[float] = []
    errors = 0
    lock = threading.Lock()

    def worker(_):
        nonlocal errors
        dt = _one(args.url)
        with lock:
            if dt is None:
                errors += 1
            else:
                lats.append(dt)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(worker, range(args.requests)))
    wall = time.perf_counter() - t0
    after = _fetch_metrics(args.url)

    ok = len(lats)
    rps = ok / wall if wall else 0.0
    lats.sort()

    def pct(p):
        return lats[min(len(lats) - 1, int(p / 100 * len(lats)))] * 1000 if lats else 0.0

    saved = after.get("tooltrim_tokens_saved_total", 0) - \
        before.get("tooltrim_tokens_saved_total", 0)
    tin = after.get("tooltrim_tokens_in_total", 0) - before.get("tooltrim_tokens_in_total", 0)
    tout = after.get("tooltrim_tokens_out_total", 0) - before.get("tooltrim_tokens_out_total", 0)
    dollars = saved / 1_000_000 * args.price_per_mtok

    print("\n=== tooltrim proxy load test ===")
    print(f"requests   : {args.requests} (ok {ok}, errors {errors}) @ "
          f"concurrency {args.concurrency}")
    print(f"wall       : {wall:.2f}s")
    print(f"throughput : {rps:,.0f} req/s")
    print(f"latency ms : p50 {pct(50):.1f}  p90 {pct(90):.1f}  p99 {pct(99):.1f}  "
          f"mean {statistics.mean(lats)*1000:.1f}" if lats else "latency ms : n/a")
    if tin:
        print(f"tokens     : in {tin:,.0f} -> out {tout:,.0f}  "
              f"(saved {saved:,.0f}, {saved/tin*100:.1f}%)")
        print(f"cost saved : ${dollars:,.4f} on this run "
              f"(@ ${args.price_per_mtok}/1M input tokens)")
        if ok:
            print(f"proj @1M req: ${dollars/ok*1_000_000:,.0f} saved per 1M requests")
    else:
        print("tokens     : /metrics unavailable (is this the proxy URL?)")


if __name__ == "__main__":
    main()
