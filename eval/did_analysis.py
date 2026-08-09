"""Pre-registered difference-in-differences headline for the multi-trial run.

Implements paper/PREREGISTRATION_multitrial.md sec 4 exactly:

  per-task gap  g_t = mean(tooltrim reward on t) - mean(none reward on t)
  affected gap    A = mean over the 16 compression-affected headline tasks of g_t
  placebo  gap    P = mean over the 19 byte-identical placebo tasks of g_t
  DiD = A - P                       (nets the run-to-run noise floor out of A)

The exposure strata are FROZEN from the paid baseline and imported from
eval.freeze_strata (single source of truth) rather than recomputed on the run
being analysed -- exposure is trajectory-dependent, so recomputing it on new data
would let the strata drift with the result. Diagnostic tasks {0,6,7,15,35} are
excluded from every headline set.

CI: stratified cluster bootstrap over tasks (the task is the unit of analysis),
20000 resamples, seed 0; reported on the DiD and on each component gap.

Run from the repo root:

    python -m eval.did_analysis
    python -m eval.did_analysis --checkpoint benchmarks/taubench_multitrial_v2/checkpoint.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

from eval.freeze_strata import compute_strata

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DEFAULT_CK = Path("benchmarks/taubench_multitrial_v2/checkpoint.jsonl")

# baseline anchors quoted in the pre-registration (sec 3), for a side-by-side read
BASE_PLACEBO = -0.053
BASE_AFFECTED = -0.013
BASE_DID = +0.040


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


def gap(data, t, a="none", b="tooltrim") -> float:
    rb, ra = data[(b, t)], data[(a, t)]
    return sum(rb) / len(rb) - sum(ra) / len(ra)


def ci(samples, alpha=0.05) -> tuple[float, float]:
    s = sorted(samples)
    n = len(s)
    return s[int(n * alpha / 2)], s[int(n * (1 - alpha / 2)) - 1]


def _boot_mean(gaps, tasks, rng) -> float:
    return sum(gaps[tasks[rng.randrange(len(tasks))]] for _ in tasks) / len(tasks)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CK)
    ap.add_argument("--strata-baseline", type=Path, default=None,
                    help="checkpoint the frozen strata are read from "
                         "(default: eval.freeze_strata's paid baseline)")
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    st = compute_strata(args.strata_baseline) if args.strata_baseline else compute_strata()
    affected, placebo, headline = st["affected"], st["headline_placebo"], st["headline"]

    data = load(args.checkpoint)
    missing = [t for t in headline
               if ("none", t) not in data or ("tooltrim", t) not in data]
    if missing:
        print(f"WARNING: {len(missing)} headline task(s) missing from checkpoint: "
              f"{missing} -- partial run?", file=sys.stderr)
        headline = [t for t in headline if t not in missing]
        affected = [t for t in affected if t not in missing]
        placebo = [t for t in placebo if t not in missing]

    gaps = {t: gap(data, t) for t in headline}
    A = sum(gaps[t] for t in affected) / len(affected)
    P = sum(gaps[t] for t in placebo) / len(placebo)
    DID = A - P
    H = sum(gaps[t] for t in headline) / len(headline)

    rng = random.Random(args.seed)
    A_bs, P_bs, D_bs = [], [], []
    for _ in range(args.n_boot):
        a = _boot_mean(gaps, affected, rng)
        p = _boot_mean(gaps, placebo, rng)
        A_bs.append(a); P_bs.append(p); D_bs.append(a - p)
    aL, aH = ci(A_bs); pL, pH = ci(P_bs); dL, dH = ci(D_bs)

    def pp(x): return f"{x * 100:+.1f}pp"

    print("=" * 72)
    print("PRE-REGISTERED HEADLINE - difference-in-differences")
    print(f"  checkpoint: {args.checkpoint}")
    print("=" * 72)
    print(f"tasks: affected={len(affected)}  placebo={len(placebo)}  "
          f"headline={len(headline)}  (bootstrap {args.n_boot}, seed {args.seed})\n")
    print(f"  affected gap (tooltrim-none)  = {pp(A)}   95% CI [{pp(aL)}, {pp(aH)}]   (baseline {pp(BASE_AFFECTED)})")
    print(f"  placebo  gap (noise floor)    = {pp(P)}   95% CI [{pp(pL)}, {pp(pH)}]   (baseline {pp(BASE_PLACEBO)})")
    print("  " + "-" * 66)
    print(f"  DiD = affected - placebo      = {pp(DID)}   95% CI [{pp(dL)}, {pp(dH)}]   (baseline {pp(BASE_DID)})")
    print(f"  (secondary) headline-35 gap   = {pp(H)}\n")

    print("PREDICTION VERDICTS (pre-registration sec 3)")
    p3 = pL <= BASE_PLACEBO <= pH
    print(f"  P3 placebo reproduces noise floor: baseline {pp(BASE_PLACEBO)} "
          f"{'INSIDE' if p3 else 'OUTSIDE'} new CI [{pp(pL)}, {pp(pH)}]  ->  "
          f"{'CORROBORATED' if p3 else 'REFUTED (harness drift -- investigate)'}")
    incl0 = dL <= 0 <= dH
    if not incl0:
        v = "REFUTED: CI excludes 0 on the " + ("harmful" if dH < 0 else "beneficial") + " side"
    elif dL >= -0.15:
        v = "CONFIRMED: no detectable harm, harmful bound within +/-15pp"
    else:
        v = ("NOT REFUTED: CI includes 0 (no detectable harm) but harmful bound "
             "wider than +/-15pp -> unresolved at this task count, not certified")
    print(f"  P4 DiD indistinguishable from 0:   CI [{pp(dL)}, {pp(dH)}] "
          f"{'includes' if incl0 else 'excludes'} 0  ->  {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
