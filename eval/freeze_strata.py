"""Freeze the exposure strata from the existing baseline, before the multi-trial run.

Exposure(task) = (tool observations compressed) / (tool calls), pooled over the
5 tooltrim trials of the 600-episode baseline in
``benchmarks/taubench_multitrial/checkpoint.jsonl``. This is trajectory-dependent,
so it is read from the already-paid baseline rather than recomputed; it uses the
tooltrim arm because that is the arm where compression actually happens.

Strata:
  * placebo  : exposure == 0 (tooltrim never fired -> byte-identical no-op)
  * low/high : median split of the remaining tasks; ties (>= median) go to high.

Run: ``python -m eval.freeze_strata`` from the repo root. Output is deterministic;
the values printed here are the ones frozen in paper/PREREGISTRATION_multitrial.md.
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

CHECKPOINT = Path("benchmarks/taubench_multitrial/checkpoint.jsonl")
DIAGNOSTIC = {0, 6, 7, 15, 35}  # tasks used to find the bugs; excluded from headline


def main() -> None:
    recs = [json.loads(l) for l in CHECKPOINT.read_text().splitlines() if l.strip()]
    tt = [r for r in recs if r["method"] == "tooltrim"]
    tc, cp = defaultdict(int), defaultdict(int)
    for r in tt:
        s, t = r["stats"], r["task"]
        tc[t] += s["tool_calls"]
        cp[t] += s["compressed"]
    tasks = sorted(tc)
    exp = {t: (cp[t] / tc[t] if tc[t] else 0.0) for t in tasks}

    placebo = [t for t in tasks if cp[t] == 0]
    nonzero = [t for t in tasks if cp[t] > 0]
    med = statistics.median(exp[t] for t in nonzero)
    low = [t for t in nonzero if exp[t] < med]
    high = [t for t in nonzero if exp[t] >= med]  # tie -> high

    print(f"tooltrim episodes: {len(tt)}")
    print(f"median exposure among nonzero tasks: {med:.4f}")
    print(f"placebo ({len(placebo)}): {placebo}")
    print(f"low     ({len(low)}): {low}")
    print(f"high    ({len(high)}): {high}")
    print(f"headline (excl. {sorted(DIAGNOSTIC)}):")
    for name, s in (("placebo", placebo), ("low", low), ("high", high)):
        keep = [t for t in s if t not in DIAGNOSTIC]
        print(f"  {name}: {len(keep)} {keep}")


if __name__ == "__main__":
    main()
