"""Analyze the tau-bench *expand pilot* and apply a PRE-REGISTERED go/no-go rule.

Reads the per-solve ``checkpoint.jsonl`` a run writes (so it works mid-run on a
partial checkpoint and again at completion), and reports, per budget:

  * task-success rate + Wilson 95% CI for ``none`` / ``tooltrim`` /
    ``tooltrim-expand``;
  * the gap each compressor leaves to the uncompressed ``none`` ceiling;
  * whether the recovery path (tooltrim-expand) closes that gap; and
  * two sanity gates — did the agent actually *use* expand, and does the
    recoverable path still *save* tokens (else "recovery" is just re-reading).

Why a pre-registered rule: the Pilot A negative result was retracted once an
ablation showed it sat inside n=1 sampling noise. Fixing the decision threshold
*before* seeing these numbers is what stops the same post-hoc rationalization.
This pilot is single-trial: its job is a directional go/no-go on whether to fund
the multi-trial sweep, NOT to produce a citable effect. So the thresholds below
are deliberately coarse — "is there a signal worth confirming?", not "is it
significant?".

======================================================================
PRE-REGISTERED DECISION RULE  (fixed 2026-07-20, before any pilot results)
======================================================================
Let d(b) = success(tooltrim-expand @ b) - success(tooltrim @ b), over the three
budgets b in {64, 128, 256}. Then:

  GO  (fund the multi-trial sweep) iff
        d(b) >= +0.10 at >= 2 of the 3 budgets,  OR  d(b) >= +0.15 at any budget.
  NO-GO (stop; write the null/methods result) iff
        d(b) <= +0.05 at ALL three budgets.
  AMBIGUOUS otherwise -> report; consider a cheap targeted add-trial before the
        full sweep.

Two independent SANITY gates (a GO is only meaningful if BOTH hold):
  (S1) the agent actually called expand at least once (else the mechanism was
       never exercised — that is itself a finding: fix the tool affordance
       before spending on the sweep);
  (S2) tooltrim-expand still saves tokens vs the ceiling (context < raw), else
       recovery collapsed back to reading the full output and there is no
       compression benefit to defend.

A "STRONG GO" additionally requires that at some budget where tooltrim loses
>= 0.10 to the ceiling, tooltrim-expand comes within 0.05 of it — i.e. recovery
demonstrably restores nearly all the accuracy lost to lossy compression.
======================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from eval.metrics import wilson_ci

try:  # keep output ASCII-safe under a cp1252 console, but don't crash if it isn't
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SUCCESS_THRESHOLD = 1.0
SUCCESS_EPS = 1e-9
NONE, TT, TTE = "none", "tooltrim", "tooltrim-expand"
BUDGET_INDEPENDENT_KEY = -1  # `none` is stored under this sentinel budget

# --- pre-registered thresholds (do NOT tune to the observed data) ------------
GO_DELTA_MULTI = 0.10   # +10pp expand-over-tooltrim needed at >= 2 budgets
GO_MIN_BUDGETS = 2
GO_DELTA_SINGLE = 0.15  # OR +15pp at any single budget
NOGO_DELTA = 0.05       # <= +5pp at ALL budgets => stop
STRONG_CEILING_LOSS = 0.10   # budget where tooltrim loses this much to ceiling ...
STRONG_RECOVERY_GAP = 0.05   # ... and expand comes within this of the ceiling
EXPECTED_BUDGETS = [64, 128, 256]  # the pilot's budget sweep
EXPECTED_TASKS = 40                # solves per (method, budget) when complete


def load_solves(path: Path) -> List[dict]:
    """Read every complete solve record from a (possibly partial) checkpoint."""
    if not path.exists():
        raise SystemExit(f"no checkpoint at {path} (has the run written any solves yet?)")
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # torn final line from a hard kill; skip
    return out


def aggregate(solves: List[dict]):
    """Group solves into (method, budget) -> rewards + summed stats."""
    rewards: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    stats: Dict[Tuple[str, int], Dict[str, int]] = defaultdict(
        lambda: defaultdict(int))
    for r in solves:
        key = (r["method"], int(r["budget"]))
        rewards[key].append(float(r["reward"]))
        for k, v in (r.get("stats") or {}).items():
            if isinstance(v, (int, float)):
                stats[key][k] += v
    return rewards, stats


def rate(rewards: List[float]) -> Tuple[float, int, int]:
    thr = SUCCESS_THRESHOLD - SUCCESS_EPS
    k = sum(1 for x in rewards if x >= thr)
    n = len(rewards)
    return (k / n if n else 0.0), k, n


def _method_budget_key(rewards, method: str, budget: int):
    """`none` is stored budget-independent; everything else under its budget."""
    if method == NONE:
        return (NONE, BUDGET_INDEPENDENT_KEY)
    return (method, budget)


def analyze(results_dir: Path) -> int:
    solves = load_solves(results_dir / "checkpoint.jsonl")
    rewards, stats = aggregate(solves)

    budgets = sorted({b for (m, b) in rewards if m != NONE and b >= 0})
    none_key = (NONE, BUDGET_INDEPENDENT_KEY)
    have_none = none_key in rewards

    print(f"\n=== tau-bench expand pilot - {results_dir} ===")
    print(f"solves on disk: {len(solves)}  |  budgets seen: {budgets or '-'}")
    if have_none:
        cr, ck, cn = rate(rewards[none_key])
        clo, chi = wilson_ci(ck, cn)
        print(f"\nceiling  none: {cr:5.0%}  ({ck}/{cn})  95% CI [{clo:.0%},{chi:.0%}]")
    else:
        print("\nceiling  none: not yet available")

    # Per-budget table.
    print(f"\n{'budget':>6} | {'method':<16} | {'success':>8} | {'n':>5} | "
          f"{'gap2ceil':>8} | {'d_vs_tt':>7} | {'expand':>6} | {'saved':>6}")
    print("-" * 82)
    deltas: Dict[int, float] = {}
    completeness: List[str] = []
    for b in budgets:
        cr = rate(rewards[none_key])[0] if have_none else None
        row_rates = {}
        for m in (TT, TTE):
            key = (m, b)
            if key not in rewards:
                continue
            mr, mk, mn = rate(rewards[key])
            row_rates[m] = mr
            gap = (cr - mr) if cr is not None else None
            delta = (mr - row_rates[TT]) if (m == TTE and TT in row_rates) else None
            st = stats[key]
            exp = st.get("expand_calls", 0)
            raw = st.get("raw_tokens", 0) or 1
            saved = 1 - (st.get("context_tokens", 0) / raw)
            print(f"{b:>6} | {m:<16} | {mr:>7.0%} | {mn:>5} | "
                  f"{('%+.0f%%' % (gap*100)) if gap is not None else '   -':>8} | "
                  f"{('%+.0f%%' % (delta*100)) if delta is not None else '   -':>7} | "
                  f"{exp:>6} | {saved:>5.0%}")
        if TT in row_rates and TTE in row_rates:
            deltas[b] = row_rates[TTE] - row_rates[TT]
        # completeness note per budget
        for m in (TT, TTE):
            n = rate(rewards.get((m, b), []))[2]
            if n < 40:
                completeness.append(f"{m}@{b}: {n}/40")

    if completeness:
        print("\nincomplete (still running): " + ", ".join(completeness))

    # --- apply the pre-registered rule --------------------------------------
    print("\n--- PRE-REGISTERED go/no-go (thresholds fixed 2026-07-20) ---")
    expected = EXPECTED_BUDGETS
    none_n = rate(rewards[none_key])[2] if have_none else 0
    incomplete = (none_n < EXPECTED_TASKS) or any(
        rate(rewards.get((m, b), []))[2] < EXPECTED_TASKS
        for b in expected for m in (TT, TTE))
    if incomplete:
        print("VERDICT: PENDING - need none + both compressors complete at every "
              "budget (40 tasks each) before the rule is decisive. Numbers above are "
              "the live partial view.")
        return 0
    deltas = {b: rate(rewards[(TTE, b)])[0] - rate(rewards[(TT, b)])[0]
              for b in expected}

    n_multi = sum(1 for d in deltas.values() if d >= GO_DELTA_MULTI)
    max_delta = max(deltas.values())
    tte_expand_total = sum(stats[(TTE, b)].get("expand_calls", 0) for b in budgets)
    tte_saves = all(
        stats[(TTE, b)].get("context_tokens", 0)
        < (stats[(TTE, b)].get("raw_tokens", 0) or 1) for b in budgets)

    s1 = tte_expand_total > 0
    s2 = tte_saves
    go = (n_multi >= GO_MIN_BUDGETS) or (max_delta >= GO_DELTA_SINGLE)
    nogo = all(d <= NOGO_DELTA for d in deltas.values())

    cr = rate(rewards[none_key])[0]
    strong = any(
        (cr - rate(rewards[(TT, b)])[0]) >= STRONG_CEILING_LOSS
        and (cr - rate(rewards[(TTE, b)])[0]) <= STRONG_RECOVERY_GAP
        for b in budgets)

    print(f"d(expand-tt) by budget: " +
          ", ".join(f"{b}:{d:+.0%}" for b, d in sorted(deltas.items())))
    print(f"budgets with d>={GO_DELTA_MULTI:.0%}: {n_multi}  |  max d: {max_delta:+.0%}")
    print(f"sanity S1 (agent used expand): {'PASS' if s1 else 'FAIL'} "
          f"({tte_expand_total} calls)")
    print(f"sanity S2 (expand still saves tokens): {'PASS' if s2 else 'FAIL'}")

    if not (s1 and s2):
        print("\nVERDICT: INVALID — a sanity gate failed. Fix the mechanism "
              "(tool affordance / savings) before trusting any go/no-go.")
    elif go:
        print(f"\nVERDICT: {'STRONG ' if strong else ''}GO — fund the multi-trial "
              "sweep (--trials 3). The recovery path shows a directional gain worth "
              "confirming out of noise.")
    elif nogo:
        print("\nVERDICT: NO-GO — expand does not beat lossy tooltrim beyond the "
              "noise band. Stop; write the null/methods result rather than spending "
              "on the sweep.")
    else:
        print("\nVERDICT: AMBIGUOUS — signal present but under the GO bar. Consider "
              "a cheap targeted add-trial on the moved budgets before the full sweep.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="benchmarks/taubench_pilot_expand",
                    help="dir containing checkpoint.jsonl")
    args = ap.parse_args()
    return analyze(Path(args.results_dir))


if __name__ == "__main__":
    raise SystemExit(main())
