"""Export the Tool-Output Faithfulness Benchmark to a HuggingFace-loadable form.

Emits ``data/tofb.jsonl`` — one JSON object per case, with the (large,
deterministically generated) tool output inlined so the dataset is fully
self-contained and reproducible. HuggingFace Datasets auto-loads the JSONL via
the ``json`` builder; the dataset card (README.md) declares the schema.

Run from the repo root:

    python hf_dataset/export.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Allow running from the repo root without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.dataset import default_cases  # noqa: E402
from tooltrim import count_tokens  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "data"


def to_record(case) -> dict:
    return {
        "id": case.id,
        "content_type": case.content_type,
        "category": case.category,
        "question": case.question,
        "gold": case.gold,
        "all_of": list(case.all_of),
        "must_not": list(case.must_not),
        "tool_output": case.tool_output,
        "tool_output_tokens": count_tokens(case.tool_output),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = default_cases()
    out_path = OUT_DIR / "tofb.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(to_record(case), ensure_ascii=False) + "\n")

    by_cat: dict = {}
    by_type: dict = {}
    total_tokens = 0
    for c in cases:
        by_cat[c.category] = by_cat.get(c.category, 0) + 1
        by_type[c.content_type] = by_type.get(c.content_type, 0) + 1
        total_tokens += count_tokens(c.tool_output)

    print(f"wrote {len(cases)} cases -> {out_path}")
    print(f"  by category: {dict(sorted(by_cat.items()))}")
    print(f"  by content_type: {dict(sorted(by_type.items()))}")
    print(f"  total tool-output tokens: {total_tokens:,} "
          f"(mean {total_tokens // len(cases):,}/case)")
    print(f"  file size: {os.path.getsize(out_path):,} bytes")


if __name__ == "__main__":
    main()
