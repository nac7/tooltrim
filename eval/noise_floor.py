"""Noise floor + paired equivalence analysis from the multi-trial tau-bench run.

Two questions, both answered from data already on disk (no API calls, no GPU):

1. NOISE FLOOR. With model, prompt, method and task all held fixed, how often do
   two independent trials disagree on the outcome? This is the irreducible
   stochasticity of the system. No claim about a *change* between two model
   versions is interpretable unless the observed drift exceeds this floor.

2. EQUIVALENCE. Is `tooltrim` outcome-equivalent to `none`? Tested properly:
   paired by task, cluster-bootstrapped over tasks, and evaluated with TOST
   against an explicit margin rather than eyeballing two means.

Run from the tooltrim repo root, same convention as eval.pilot_analysis:

    python -m eval.noise_floor
    python -m eval.noise_floor --checkpoint benchmarks/taubench_multitrial/checkpoint.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from itertools import combinations
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DEFAULT_CK = Path(__file__).parent.parent / "benchmarks" / \
    "taubench_multitrial" / "checkpoint.jsonl"


def load(path: Path) -> dict[tuple[str, int], list[float]]:
    """-> {(method, task): [reward per trial]}"""
    by = collections.defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue                      # tolerate a half-written final line
        by[(r["method"], r["task"])][r["trial"]] = float(r["reward"])
    return {k: [v[t] for t in sorted(v)] for k, v in by.items()}


# --------------------------------------------------------------------------
# 1. noise floor
# --------------------------------------------------------------------------

def noise_floor(data: dict) -> None:
    print("=" * 74)
    print("1. NOISE FLOOR - same model, same prompt, same task, different trial")
    print("=" * 74)

    per_method = collections.defaultdict(lambda: [0, 0])   # [discordant, total] pairs
    unstable = collections.defaultdict(int)
    tasks_per_method = collections.defaultdict(int)

    for (method, task), rewards in sorted(data.items()):
        if len(rewards) < 2:
            continue
        tasks_per_method[method] += 1
        pairs = list(combinations(rewards, 2))
        disc = sum(1 for a, b in pairs if a != b)
        per_method[method][0] += disc
        per_method[method][1] += len(pairs)
        if len(set(rewards)) > 1:
            unstable[method] += 1

    print(f"\n{'method':<16} {'tasks':>6} {'flaky tasks':>12} {'pairwise disagreement':>23}")
    print("-" * 62)
    for method in sorted(per_method):
        disc, tot = per_method[method]
        n = tasks_per_method[method]
        print(f"{method:<16} {n:>6} {unstable[method]:>6} ({unstable[method]/n:>4.0%}) "
              f"{disc:>10}/{tot:<5} = {disc/tot:>6.1%}")

    print("\nInterpretation: two runs of an UNCHANGED system disagree this often.")
    print("Any migration-drift measurement below this level is indistinguishable")
    print("from noise, and a single-trial A/B comparison cannot resolve it.")


# --------------------------------------------------------------------------
# 2. paired equivalence
# --------------------------------------------------------------------------

def _paired(data: dict, a: str, b: str) -> list[tuple[int, float, float]]:
    tasks = sorted({t for (m, t) in data if m == a} & {t for (m, t) in data if m == b})
    out = []
    for t in tasks:
        ra, rb = data[(a, t)], data[(b, t)]
        if ra and rb:
            out.append((t, sum(ra) / len(ra), sum(rb) / len(rb)))
    return out


def _boot_ci(diffs: list[float], n_boot: int = 20000, alpha: float = 0.05,
             seed: int = 0) -> tuple[float, float, float]:
    """Cluster bootstrap over tasks (tasks are the independent unit, not trials)."""
    rng = random.Random(seed)
    n = len(diffs)
    point = sum(diffs) / n
    means = []
    for _ in range(n_boot):
        s = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    lo = means[int(n_boot * alpha / 2)]
    hi = means[int(n_boot * (1 - alpha / 2)) - 1]
    return point, lo, hi


def equivalence(data: dict, a: str, b: str, margins=(0.05, 0.10)) -> None:
    print("\n" + "=" * 74)
    print(f"2. EQUIVALENCE - '{b}' vs '{a}' (paired by task, {len(_paired(data,a,b))} tasks)")
    print("=" * 74)

    rows = _paired(data, a, b)
    if not rows:
        print("no overlapping tasks")
        return
    diffs = [rb - ra for _, ra, rb in rows]
    mean_a = sum(ra for _, ra, _ in rows) / len(rows)
    mean_b = sum(rb for _, _, rb in rows) / len(rows)
    point, lo, hi = _boot_ci(diffs)

    print(f"\n  mean reward  {a:<10} = {mean_a:.3f}")
    print(f"  mean reward  {b:<10} = {mean_b:.3f}")
    print(f"  paired diff ({b} - {a}) = {point:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")

    n_worse = sum(1 for d in diffs if d < 0)
    n_better = sum(1 for d in diffs if d > 0)
    print(f"  tasks worse/tied/better: {n_worse}/{len(diffs)-n_worse-n_better}/{n_better}")

    print("\n  TOST equivalence verdict:")
    for m in margins:
        verdict = ("EQUIVALENT" if lo > -m and hi < m else
                   "INCONCLUSIVE" if (lo < -m < hi) or (lo < m < hi) else
                   "DIFFERENT")
        print(f"    margin +/-{m:.0%}: {verdict:<13} "
              f"(CI [{lo:+.3f}, {hi:+.3f}] vs [{-m:+.2f}, {m:+.2f}])")


# --------------------------------------------------------------------------
# 3. how many trials would you actually need?
# --------------------------------------------------------------------------

def power(data: dict, a: str, b: str) -> None:
    print("\n" + "=" * 74)
    print("3. SAMPLE SIZE - trials needed to resolve a given effect")
    print("=" * 74)

    rows = _paired(data, a, b)
    diffs = [rb - ra for _, ra, rb in rows]
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    sd = var ** 0.5
    print(f"\n  per-task paired-difference SD = {sd:.3f}  (n={n} tasks)")
    print(f"\n  {'target margin':>14} {'tasks needed':>14}")
    print("  " + "-" * 30)
    for m in (0.15, 0.10, 0.05, 0.03):
        # width of 95% CI ~ 2*1.96*sd/sqrt(k) < 2*m  ->  k > (1.96*sd/m)^2
        k = (1.96 * sd / m) ** 2
        print(f"  {m:>13.0%} {k:>14.0f}")
    print("\n  (at the trial count already in this data; more trials per task")
    print("   shrink sd further, which is the cheaper axis when tasks are scarce)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CK)
    ap.add_argument("--baseline", default="none")
    ap.add_argument("--candidate", default="tooltrim")
    args = ap.parse_args()

    data = load(args.checkpoint)
    n_ep = sum(len(v) for v in data.values())
    methods = sorted({m for m, _ in data})
    print(f"loaded {n_ep} episodes | methods={methods} | "
          f"{len({t for _, t in data})} tasks\n")

    noise_floor(data)
    for cand in [m for m in methods if m != args.baseline]:
        equivalence(data, args.baseline, cand)
    power(data, args.baseline, args.candidate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
