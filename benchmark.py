"""Benchmark tooltrim on realistic, bloated tool outputs.

Generates representative tool results an LLM agent commonly ingests — a fetched
web page, a paginated REST/JSON response, a server log dump, a CSV export, and
a long document — then reports token savings at a typical 400-token budget.

Run:  python benchmark.py
Numbers use exact tiktoken (cl100k_base) counts when ``tooltrim[tokens]`` is
installed, otherwise a ~4 chars/token heuristic.
"""

from __future__ import annotations

import json
import random

from tooltrim import ToolCompressor, count_tokens, using_exact_counts

random.seed(7)

WORDS = (
    "system user account order payment region latency cache retry queue worker "
    "session cluster shard token policy invoice balance ledger metric trace "
    "request response handler service gateway upstream downstream timeout"
).split()


def _sentence(n=12):
    return " ".join(random.choice(WORDS) for _ in range(n)).capitalize() + "."


def sample_html(needle: str) -> str:
    body = []
    for i in range(120):
        if i == 73:
            body.append(f"<p>{needle}</p>")
        else:
            body.append(f"<p>{_sentence()}</p>")
    nav = "".join(f"<li><a href='/p/{i}'>link {i}</a></li>" for i in range(40))
    return (
        "<!doctype html><html><head><title>Docs</title>"
        "<style>body{font:14px}</style><script>analytics(123)</script></head>"
        f"<body><header>Site header</header><nav><ul>{nav}</ul></nav>"
        f"<main><article>{''.join(body)}</article></main>"
        "<footer>(c) 2026 Example Inc. Privacy Terms Contact</footer></body></html>"
    )


def sample_json(needle: str) -> str:
    items = []
    for i in range(300):
        items.append(
            {
                "id": i,
                "status": random.choice(["ok", "pending", "failed"]),
                "amount": round(random.uniform(1, 9999), 2),
                "note": needle if i == 211 else _sentence(8),
                "meta": {"region": random.choice(["us", "eu", "apac"]), "retries": i % 4},
            }
        )
    return json.dumps({"page": 1, "total": 300, "results": items})


def sample_logs(needle: str) -> str:
    lines = []
    for i in range(400):
        if i == 250:
            lines.append(f"2026-06-27 10:42:11 ERROR {needle}")
        else:
            lines.append("2026-06-27 10:%02d:%02d INFO heartbeat ok worker=3"
                         % (i % 60, (i * 7) % 60))
    return "\n".join(lines)


def sample_csv(needle: str) -> str:
    rows = ["id,region,status,amount,note"]
    for i in range(500):
        note = needle if i == 333 else _sentence(4)
        rows.append(f"{i},{random.choice(['us','eu','apac'])},ok,{i*3}.50,{note}")
    return "\n".join(rows)


def sample_text(needle: str) -> str:
    paras = [_sentence(40) for _ in range(150)]
    paras[97] = needle
    return "\n\n".join(paras)


def main():
    budget = 400
    tc = ToolCompressor(max_tokens=budget, add_footer=False)
    cases = [
        ("Web page (HTML)", sample_html, "The API rate limit is 5000 requests per hour."),
        ("REST response (JSON)", sample_json, "refund issued to customer #4417"),
        ("Server logs", sample_logs, "disk usage 98% on /var/data, writes failing"),
        ("CSV export", sample_csv, "flagged for manual compliance review"),
        ("Long document (text)", sample_text, "The deployment key rotates every 24 hours."),
    ]

    print(f"tooltrim benchmark  |  budget={budget} tokens  |  "
          f"exact tiktoken counts: {using_exact_counts()}\n")
    header = f"{'tool output':24} {'before':>9} {'after':>8} {'saved':>9}  needle"
    print(header)
    print("-" * len(header))

    total_before = total_after = 0
    for name, gen, needle in cases:
        raw = gen(needle)
        res = tc.compress(raw, query=needle)
        total_before += res.original_tokens
        total_after += res.compressed_tokens
        kept = "yes" if needle.split()[0].lower() in res.text.lower() else "NO"
        print(f"{name:24} {res.original_tokens:>9} {res.compressed_tokens:>8} "
              f"{res.saved_ratio*100:>7.1f}%  needle kept: {kept}")

    print("-" * len(header))
    overall = (total_before - total_after) / total_before * 100
    print(f"{'TOTAL':24} {total_before:>9} {total_after:>8} {overall:>7.1f}%")
    print(f"\n{total_before} -> {total_after} tokens "
          f"({total_before / max(1, total_after):.1f}x smaller context)")


if __name__ == "__main__":
    main()
