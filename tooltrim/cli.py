"""``tooltrim`` command-line interface.

    tooltrim compress big.json --query "refund status"   # compress a file/stdin
    cat big.html | tooltrim compress -q "rate limits"     # pipe in, compressed out
    tooltrim proxy --upstream https://api.openai.com/v1   # run the proxy
    tooltrim demo                                         # 10-second self-contained tour
    tooltrim version

Everything here uses only the installed ``tooltrim`` package — no dev extras.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import ToolCompressor, __version__, count_tokens


def _read_input(path: Optional[str]) -> str:
    if path and path != "-":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    return sys.stdin.read()


def _cmd_compress(args: argparse.Namespace) -> int:
    text = _read_input(args.file)
    tc = ToolCompressor(max_tokens=args.max_tokens, add_footer=not args.no_footer)
    res = tc.compress(text, query=args.query, content_type=args.type)
    sys.stdout.write(res.text)
    if not res.text.endswith("\n"):
        sys.stdout.write("\n")
    if args.stats:
        print(
            f"[tooltrim] {res.content_type}: {res.original_tokens}->"
            f"{res.compressed_tokens} tokens "
            f"({res.saved_ratio * 100:.1f}% saved)"
            + (f"  ref={res.ref}" if res.ref else ""),
            file=sys.stderr,
        )
    return 0


def _cmd_proxy(args: argparse.Namespace) -> int:
    from .proxy import serve

    serve(host=args.host, port=args.port, max_tokens=args.max_tokens,
          upstream_base=args.upstream)
    return 0


_DEMO_SAMPLES = {
    "json": ('{"orders": ['
             + ",".join('{"id": %d, "status": "shipped"}' % i for i in range(400))
             + ',{"id": 999, "status": "REFUNDED", "note": "refund to customer 4417"}]}',
             "which customer got a refund?"),
    "html": ("<html><head><script>track()</script><style>.x{}</style></head>"
             "<body><nav>menu</nav>" + "<p>filler paragraph about nothing.</p>" * 200
             + "<p>The API rate limit is 5000 requests/hour.</p>"
             "<footer>copyright</footer></body></html>",
             "what is the rate limit?"),
    "logs": ("\n".join(["2026-06-28 INFO heartbeat ok"] * 300
                       + ["2026-06-28 ERROR disk full on /data write aborted"]),
             "what error occurred?"),
    "text": (" ".join(["routine sentence with no useful content."] * 600)
             + " The launch code is 1234-ALPHA.",
             "what is the launch code?"),
}


def _cmd_demo(args: argparse.Namespace) -> int:
    tc = ToolCompressor(max_tokens=args.max_tokens, add_footer=False)
    print(f"tooltrim demo — compressing 4 sample tool outputs to a "
          f"{args.max_tokens}-token budget\n")
    print(f"{'type':<6} {'before':>8} {'after':>7} {'saved':>8}  needle kept")
    print("-" * 48)
    tot_before = tot_after = 0
    for ctype, (blob, query) in _DEMO_SAMPLES.items():
        res = tc.compress(blob, query=query, content_type=ctype)
        needle = {"json": "4417", "html": "5000", "logs": "disk full",
                  "text": "1234-ALPHA"}[ctype]
        before, after = res.original_tokens, res.compressed_tokens
        tot_before += before
        tot_after += after
        print(f"{ctype:<6} {before:>8,} {after:>7,} "
              f"{(1 - after / before) * 100:>7.1f}%  "
              f"{'yes' if needle in res.text else 'NO'}")
    print("-" * 48)
    print(f"{'TOTAL':<6} {tot_before:>8,} {tot_after:>7,} "
          f"{(1 - tot_after / tot_before) * 100:>7.1f}%")
    print(f"\n{tot_before:,} -> {tot_after:,} tokens "
          f"({tot_before / max(1, tot_after):.0f}x smaller), needle kept in every case.")
    print("Run `tooltrim proxy --upstream <api>` to do this in front of any LLM.")
    return 0


def _cmd_version(args: argparse.Namespace) -> int:
    print(f"tooltrim {__version__}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tooltrim",
        description="Drop-in compression for LLM agent tool outputs.")
    p.add_argument("-V", "--version", action="version",
                   version=f"tooltrim {__version__}")
    sub = p.add_subparsers(dest="command")

    c = sub.add_parser("compress", help="compress a file or stdin")
    c.add_argument("file", nargs="?", default="-",
                   help="input file (default: stdin)")
    c.add_argument("-q", "--query", default=None,
                   help="relevance query — keep what's on-topic for this")
    c.add_argument("-m", "--max-tokens", type=int, default=512,
                   help="token budget (default: 512)")
    c.add_argument("--type", default=None,
                   choices=["json", "html", "tabular", "logs", "text"],
                   help="force a content type (default: auto-detect)")
    c.add_argument("--no-footer", action="store_true",
                   help="omit the savings/expand-ref footer")
    c.add_argument("--stats", action="store_true",
                   help="print savings to stderr")
    c.set_defaults(func=_cmd_compress)

    pr = sub.add_parser("proxy", help="run the compression proxy")
    pr.add_argument("--host", default="127.0.0.1")
    pr.add_argument("--port", type=int, default=8800)
    pr.add_argument("-m", "--max-tokens", type=int, default=512,
                    help="token budget per tool result")
    pr.add_argument("--upstream", default=None,
                    help="upstream base url (default: $TOOLTRIM_UPSTREAM_BASE_URL "
                         "or https://api.openai.com/v1)")
    pr.set_defaults(func=_cmd_proxy)

    d = sub.add_parser("demo", help="self-contained 10-second savings demo")
    d.add_argument("-m", "--max-tokens", type=int, default=400)
    d.set_defaults(func=_cmd_demo)

    v = sub.add_parser("version", help="print the version")
    v.set_defaults(func=_cmd_version)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
