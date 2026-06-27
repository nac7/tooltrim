"""Curated faithfulness cases: a planted fact buried in realistic, bloated output.

Each case embeds one distinctive fact ("needle") inside a large, type-appropriate
tool output, paired with a question whose wording overlaps the needle (the agent's
goal becomes tooltrim's relevance query). A case is "passed" if the model recovers
the gold fact from the output it was given.

Filler is generated deterministically (seeded) so runs are reproducible.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import List

_RNG = random.Random(7)

_WORDS = (
    "system user account order region latency cache retry queue worker session "
    "cluster shard policy invoice balance ledger metric trace request response "
    "handler service gateway upstream downstream timeout pipeline scheduler"
).split()


@dataclass(frozen=True)
class Case:
    id: str
    content_type: str
    tool_output: str
    question: str
    gold: str


def _sentence(n: int = 12) -> str:
    return " ".join(_RNG.choice(_WORDS) for _ in range(n)).capitalize() + "."


# --- per-type embedding helpers ------------------------------------------------

def _text_blob(needle: str, n: int = 140, at: int = 97) -> str:
    paras = [_sentence(28) for _ in range(n)]
    paras[at % n] = needle
    return "\n\n".join(paras)


def _html_blob(needle: str, n: int = 120, at: int = 73) -> str:
    body = []
    for i in range(n):
        body.append(f"<p>{needle if i == at % n else _sentence()}</p>")
    nav = "".join(f"<li><a href='/p/{i}'>link {i}</a></li>" for i in range(40))
    return (
        "<!doctype html><html><head><title>Docs</title>"
        "<style>body{font:14px}</style><script>analytics(1)</script></head>"
        f"<body><header>Site header</header><nav><ul>{nav}</ul></nav>"
        f"<main><article>{''.join(body)}</article></main>"
        "<footer>(c) 2026 Example Inc.</footer></body></html>"
    )


def _json_blob(note: str, n: int = 300, at: int = 211) -> str:
    items = []
    for i in range(n):
        items.append({
            "id": i,
            "status": _RNG.choice(["ok", "pending", "failed"]),
            "amount": round(_RNG.uniform(1, 9999), 2),
            "note": note if i == at % n else _sentence(8),
        })
    return json.dumps({"page": 1, "total": n, "results": items})


def _logs_blob(needle: str, n: int = 400, at: int = 250) -> str:
    lines = []
    for i in range(n):
        if i == at % n:
            lines.append(f"2026-06-27 10:42:11 ERROR {needle}")
        else:
            lines.append("2026-06-27 10:%02d:%02d INFO heartbeat ok worker=3"
                         % (i % 60, (i * 7) % 60))
    return "\n".join(lines)


def _csv_blob(cell: str, n: int = 500, at: int = 333) -> str:
    rows = ["id,region,status,amount,note"]
    for i in range(n):
        note = cell if i == at % n else _sentence(4)
        rows.append(f"{i},{_RNG.choice(['us','eu','apac'])},ok,{i*3}.50,{note}")
    return "\n".join(rows)


# --- the cases -----------------------------------------------------------------

def default_cases() -> List[Case]:
    """Return the curated faithfulness cases (16 across 5 content types)."""
    cases: List[Case] = []

    # Text
    cases += [
        Case("text-ratelimit", "text",
             _text_blob("The API rate limit is 5000 requests per hour per key."),
             "What is the API rate limit per hour?", "5000 requests per hour"),
        Case("text-deploykey", "text",
             _text_blob("The deployment key rotates automatically every 24 hours."),
             "How often does the deployment key rotate?", "every 24 hours"),
        Case("text-port", "text",
             _text_blob("The metrics exporter listens on port 9187 by default."),
             "Which port does the metrics exporter listen on?", "port 9187"),
    ]

    # HTML
    cases += [
        Case("html-capital", "html",
             _html_blob("The capital of the Republic of Auren is Mateldorf."),
             "What is the capital of the Republic of Auren?", "Mateldorf"),
        Case("html-ceo", "html",
             _html_blob("Acme Corp's chief executive officer is Dana Whitlock."),
             "Who is Acme Corp's chief executive officer?", "Dana Whitlock"),
        Case("html-version", "html",
             _html_blob("The current stable release is version 12.4.1."),
             "What is the current stable release version?", "12.4.1"),
    ]

    # JSON
    cases += [
        Case("json-refund", "json",
             _json_blob("refund issued to customer 4417 for amount 250"),
             "Which note mentions a refund issued to a customer?",
             "refund issued to customer 4417"),
        Case("json-fraud", "json",
             _json_blob("flagged for manual fraud review by analyst kim"),
             "Which note was flagged for manual fraud review?",
             "flagged for manual fraud review"),
        Case("json-chargeback", "json",
             _json_blob("chargeback disputed under reason code 4853"),
             "Which note mentions a chargeback dispute reason code?",
             "chargeback disputed under reason code 4853"),
    ]

    # Logs
    cases += [
        Case("logs-disk", "logs",
             _logs_blob("disk usage 98 percent on /var/data writes failing"),
             "What error mentions disk usage on /var/data?",
             "disk usage 98 percent on /var/data"),
        Case("logs-oom", "logs",
             _logs_blob("out of memory killing process pid 8123"),
             "What error mentions an out of memory condition?",
             "out of memory killing process pid 8123"),
        Case("logs-cert", "logs",
             _logs_blob("tls certificate expired for host api.example.com"),
             "What error mentions an expired tls certificate?",
             "tls certificate expired for host api.example.com"),
    ]

    # Tabular
    cases += [
        Case("csv-compliance", "tabular",
             _csv_blob("flagged for compliance review under sanctions list"),
             "Which note was flagged for compliance review?",
             "flagged for compliance review under sanctions list"),
        Case("csv-vip", "tabular",
             _csv_blob("vip customer requires priority handling tier gold"),
             "Which note mentions a vip customer requiring priority handling?",
             "vip customer requires priority handling"),
        Case("csv-refund", "tabular",
             _csv_blob("partial refund approved by supervisor delgado"),
             "Which note mentions a partial refund approval?",
             "partial refund approved by supervisor delgado"),
        Case("csv-duplicate", "tabular",
             _csv_blob("duplicate transaction detected and voided automatically"),
             "Which note mentions a duplicate transaction?",
             "duplicate transaction detected and voided"),
    ]

    return cases
