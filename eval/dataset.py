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


# --- declarative case specs ----------------------------------------------------
# (content_type, needle, question, gold)
#   text/html: `needle` is the sentence embedded in the page; gold is the fact.
#   json/logs/tabular: `needle` is the field value / log line; gold is the match.

_SPECS = [
    # ---- text (prose) ----
    ("text", "The API rate limit is 5000 requests per hour per key.",
     "What is the API rate limit per hour?", "5000 requests per hour"),
    ("text", "The deployment key rotates automatically every 24 hours.",
     "How often does the deployment key rotate?", "every 24 hours"),
    ("text", "The metrics exporter listens on port 9187 by default.",
     "Which port does the metrics exporter listen on?", "port 9187"),
    ("text", "The maximum upload size is 250 megabytes per file.",
     "What is the maximum upload size per file?", "250 megabytes"),
    ("text", "The default session timeout is 30 minutes of inactivity.",
     "What is the default session timeout?", "30 minutes"),
    ("text", "Nightly backups run at 02:00 UTC every day.",
     "What time do the nightly backups run?", "02:00 UTC"),
    ("text", "The support contact address is help@auren.example for all tiers.",
     "What is the support contact address?", "help@auren.example"),
    ("text", "The cache time-to-live is configured to 900 seconds.",
     "What is the cache time-to-live in seconds?", "900 seconds"),
    ("text", "The service guarantees 99.95 percent uptime under the SLA.",
     "What uptime does the SLA guarantee?", "99.95 percent"),
    ("text", "Each webhook is retried up to 5 times before being dropped.",
     "How many times is a webhook retried?", "5 times"),

    # ---- html (web page) ----
    ("html", "The capital of the Republic of Auren is Mateldorf.",
     "What is the capital of the Republic of Auren?", "Mateldorf"),
    ("html", "Acme Corp's chief executive officer is Dana Whitlock.",
     "Who is Acme Corp's chief executive officer?", "Dana Whitlock"),
    ("html", "The current stable release is version 12.4.1.",
     "What is the current stable release version?", "12.4.1"),
    ("html", "The company headquarters is located in Brindale.",
     "Where is the company headquarters located?", "Brindale"),
    ("html", "The organization was founded in the year 1987.",
     "In what year was the organization founded?", "1987"),
    ("html", "The museum closes at 6 pm on Sundays.",
     "What time does the museum close on Sundays?", "6 pm"),
    ("html", "The keynote speaker is Professor Halvard Reyes.",
     "Who is the keynote speaker?", "Halvard Reyes"),
    ("html", "The tower building has 42 floors in total.",
     "How many floors does the tower building have?", "42 floors"),
    ("html", "The product warranty lasts 36 months from purchase.",
     "How long does the product warranty last?", "36 months"),
    ("html", "The ferry departs from Pier 9 every morning.",
     "Which pier does the ferry depart from?", "Pier 9"),

    # ---- json (api response notes) ----
    ("json", "refund issued to customer 4417 for amount 250",
     "Which note mentions a refund issued to a customer?",
     "refund issued to customer 4417"),
    ("json", "flagged for manual fraud review by analyst kim",
     "Which note was flagged for manual fraud review?",
     "flagged for manual fraud review"),
    ("json", "chargeback disputed under reason code 4853",
     "Which note mentions a chargeback dispute reason code?",
     "chargeback disputed under reason code 4853"),
    ("json", "account locked after 5 failed login attempts",
     "Which note mentions an account locked after failed logins?",
     "account locked after 5 failed login"),
    ("json", "shipment delayed at customs checkpoint port 7",
     "Which note mentions a shipment delayed at customs?",
     "shipment delayed at customs"),
    ("json", "coupon SAVE20 applied at checkout successfully",
     "Which note mentions a coupon applied at checkout?",
     "coupon SAVE20 applied at checkout"),
    ("json", "subscription downgraded to tier basic by request",
     "Which note mentions a subscription downgraded to basic tier?",
     "subscription downgraded to tier basic"),
    ("json", "payment retried via backup processor stripe",
     "Which note mentions a payment retried via a backup processor?",
     "payment retried via backup processor"),
    ("json", "address verification failed with avs code N",
     "Which note mentions an address verification failure?",
     "address verification failed"),
    ("json", "invoice exported to ledger batch 88 nightly",
     "Which note mentions an invoice exported to a ledger batch?",
     "invoice exported to ledger batch 88"),

    # ---- logs (server logs) ----
    ("logs", "disk usage 98 percent on /var/data writes failing",
     "What error mentions disk usage on /var/data?",
     "disk usage 98 percent on /var/data"),
    ("logs", "out of memory killing process pid 8123",
     "What error mentions an out of memory condition?",
     "out of memory killing process pid 8123"),
    ("logs", "tls certificate expired for host api.example.com",
     "What error mentions an expired tls certificate?",
     "tls certificate expired for host api.example.com"),
    ("logs", "connection refused to database replica 2",
     "What error mentions a refused database connection?",
     "connection refused to database replica 2"),
    ("logs", "deadlock detected on table orders during commit",
     "What error mentions a deadlock on a table?",
     "deadlock detected on table orders"),
    ("logs", "rate limit exceeded for api key 7af3 throttling",
     "What error mentions a rate limit exceeded for an api key?",
     "rate limit exceeded for api key 7af3"),
    ("logs", "failed to bind to port 8443 already in use",
     "What error mentions a failure to bind to a port?",
     "failed to bind to port 8443"),
    ("logs", "segmentation fault in worker thread 11 crashed",
     "What error mentions a segmentation fault in a worker thread?",
     "segmentation fault in worker thread 11"),
    ("logs", "kafka consumer lag exceeded 100000 messages",
     "What error mentions kafka consumer lag?",
     "kafka consumer lag exceeded 100000"),
    ("logs", "nfs mount timed out on /shared/cache retrying",
     "What error mentions an nfs mount timeout?",
     "nfs mount timed out on /shared/cache"),

    # ---- tabular (csv export notes) ----
    ("tabular", "flagged for compliance review under sanctions list",
     "Which note was flagged for compliance review?",
     "flagged for compliance review under sanctions list"),
    ("tabular", "vip customer requires priority handling tier gold",
     "Which note mentions a vip customer requiring priority handling?",
     "vip customer requires priority handling"),
    ("tabular", "partial refund approved by supervisor delgado",
     "Which note mentions a partial refund approval?",
     "partial refund approved by supervisor delgado"),
    ("tabular", "duplicate transaction detected and voided automatically",
     "Which note mentions a duplicate transaction?",
     "duplicate transaction detected and voided"),
    ("tabular", "manual override applied by agent 0091 at desk",
     "Which note mentions a manual override applied by an agent?",
     "manual override applied by agent 0091"),
    ("tabular", "high risk score requires secondary approval review",
     "Which note mentions a high risk score requiring secondary approval?",
     "high risk score requires secondary approval"),
    ("tabular", "chargeback reversed after evidence submitted by merchant",
     "Which note mentions a chargeback reversed after evidence?",
     "chargeback reversed after evidence submitted"),
    ("tabular", "loyalty points credited for referral bonus program",
     "Which note mentions loyalty points credited for a referral?",
     "loyalty points credited for referral"),
    ("tabular", "shipment rerouted to alternate warehouse depot 3",
     "Which note mentions a shipment rerouted to an alternate warehouse?",
     "shipment rerouted to alternate warehouse"),
    ("tabular", "tax exemption applied for nonprofit account holder",
     "Which note mentions a tax exemption for a nonprofit account?",
     "tax exemption applied for nonprofit account"),
]

_BUILDERS = {
    "text": _text_blob,
    "html": _html_blob,
    "json": _json_blob,
    "logs": _logs_blob,
    "tabular": _csv_blob,
}


def default_cases() -> List[Case]:
    """Return the curated faithfulness cases (50 across 5 content types)."""
    counters: dict = {}
    cases: List[Case] = []
    for i, (ctype, needle, question, gold) in enumerate(_SPECS):
        idx = counters.get(ctype, 0) + 1
        counters[ctype] = idx
        # vary the needle position across cases so it isn't always mid-document
        blob = _BUILDERS[ctype](needle, at=29 + i * 13)
        cases.append(Case(f"{ctype}-{idx:02d}", ctype, blob, question, gold))
    return cases
