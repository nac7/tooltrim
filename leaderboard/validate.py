#!/usr/bin/env python3
"""Validate leaderboard/results.json and render the standings table.

A new submission is a PR that adds entries to results.json. This script is the
gate: it checks every entry is well-formed and self-consistent, then prints the
Markdown standings so LEADERBOARD.md can be regenerated. Run in CI on any change
to results.json so the leaderboard can never merge malformed or unverifiable
numbers.

    python leaderboard/validate.py            # validate only
    python leaderboard/validate.py --render   # validate + print Markdown
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.json"

TRACKS = {"faithfulness", "downstream_extractability", "agent_tasks"}
REQUIRED = {"method", "track", "metric", "value", "n", "source"}


def validate(doc: dict) -> list[str]:
    errs: list[str] = []
    entries = doc.get("entries")
    if not isinstance(entries, list) or not entries:
        return ["'entries' must be a non-empty list"]
    for i, e in enumerate(entries):
        tag = f"entry[{i}] ({e.get('method', '?')}/{e.get('track', '?')})"
        missing = REQUIRED - e.keys()
        if missing:
            errs.append(f"{tag}: missing required fields {sorted(missing)}")
            continue
        if e["track"] not in TRACKS:
            errs.append(f"{tag}: unknown track '{e['track']}' (allowed: {sorted(TRACKS)})")
        v = e["value"]
        if not isinstance(v, (int, float)) or not (0.0 <= v <= 1.0):
            errs.append(f"{tag}: value {v!r} must be a rate in [0, 1]")
        if not isinstance(e["n"], int) or e["n"] <= 0:
            errs.append(f"{tag}: n must be a positive int")
        # A significance claim must carry the evidence that backs it.
        sig = e.get("vs_rag_topk")
        if sig is not None:
            if sig.get("significant") and "mcnemar_p" not in sig:
                errs.append(f"{tag}: claims significance but has no 'mcnemar_p'")
            p = sig.get("mcnemar_p")
            if p is not None and sig.get("significant") and p >= 0.05:
                errs.append(f"{tag}: significant=true but mcnemar_p={p} >= 0.05")
        # Every headline number must name a script that regenerates it.
        if e.get("verified") and not e.get("source"):
            errs.append(f"{tag}: verified=true requires a 'source' script")
    return errs


def render(doc: dict) -> str:
    lines = [f"# {doc['benchmark']} Leaderboard", "", doc.get("description", ""), ""]
    for track in ("agent_tasks", "downstream_extractability", "faithfulness"):
        rows = [e for e in doc["entries"] if e["track"] == track]
        if not rows:
            continue
        lines += [f"## {track}", "", doc.get("tracks", {}).get(track, ""), ""]
        lines.append("| method | model | budget | metric | value | n | p vs RAG | source |")
        lines.append("|:--|:--|--:|:--|--:|--:|:--:|:--|")
        for e in sorted(rows, key=lambda r: -r["value"]):
            sig = e.get("vs_rag_topk") or {}
            p = f"{sig['mcnemar_p']}" + ("*" if sig.get("significant") else "") if "mcnemar_p" in sig else "—"
            lines.append(
                f"| `{e['method']}` | {e.get('model', '—')} | {e.get('budget', '—')} "
                f"| {e['metric']} | {e['value']:.0%} | {e['n']} | {p} | `{e['source']}` |"
            )
        lines.append("")
    lines.append("\\* significant (paired McNemar, p<0.05). Regenerate: `python leaderboard/validate.py --render`.")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true", help="print Markdown standings after validating")
    args = ap.parse_args()

    doc = json.loads(RESULTS.read_text(encoding="utf-8"))
    errs = validate(doc)
    if errs:
        print("INVALID results.json:", file=sys.stderr)
        for e in errs:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print(f"OK: {len(doc['entries'])} entries valid", file=sys.stderr)
    if args.render:
        print(render(doc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
