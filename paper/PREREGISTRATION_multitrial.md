# Pre-registration — multi-trial tau-bench sweep after the compressor fixes

**Finalised 2026-08-08, before the run — supersedes the 2026-07-26 draft.** The
predictions and analysis were re-anchored to the existing baseline's own numbers
(placebo −5.3pp, DiD +4.0pp, the ±14.8pp CI floor) using no new benchmark data;
this is the version frozen at launch. **Nothing below may be edited after
launch.** Amendments go in a dated "Amendments" section at the bottom, never by
rewriting a prediction.

This project has twice published an internal conclusion it later had to retract
(the "-18pp structural penalty", then the "task-success-neutral" thesis), both
times because the analysis was chosen after seeing the data. This document fixes
the predictions, the analysis, and the decision rule in advance.

## 1. What changed since the 600-episode baseline

Three defects were found by tracing five high-exposure tasks (0, 6, 7, 15, 35)
and reproducing their payloads offline. Two are fixed in the compressor, one in
the benchmark adapter.

| # | Defect | Fix | Status |
|---|--------|-----|--------|
| 1 | Dict-of-records (`variants`) had no sampling lever, so tight budgets used depth elision and wiped every `item_id`/`available`/`price` at once. Output was also invalid JSON via a comma-split fallback. | `_is_collection_map` + entry-wise sampling; decimate before eliding depth; `_scalar_skeleton` replaces the comma-split fallback. | Verified offline; **measured** on tasks 0/6/7/15/35 (tooltrim 0/25 → 4/25, all wins on task 0). |
| 2a | Flat lookup maps (50-entry `name -> id` catalogue) were not treated as collections; the catalogue silently ended mid-alphabet with no elision marker. | Flat-map branch (`_FLAT_MAP_MIN`); elision marker is now reserved, never dropped. | Verified offline only. **No benchmark evidence.** |
| 2b | Relevance query built from the opening instruction + tool args scored **0.0 against all 17 laptop variants**, so selection degraded to positional. | Query grounded in the last 6 dialogue turns (`--dialogue-turns`, 0 = old behaviour). | Verified offline only. **No benchmark evidence.** |

## 2. What the offline check actually showed (and its limits)

On task 35's real `get_product_details` payload (17 laptop variants, 1200 tok, budget 128):

| | old query | new query |
|---|---|---|
| variants with BM25 score > 0 | **0 / 17** | **17 / 17** |
| budget used | 77 / 128 tok | 128 / 128 tok |
| variants kept | 1 | 2 |
| the variant the user actually wants (`5052031638`) | absent | **still absent** |

So the fix restores the *signal* — decisively — but does not deliver the right
record. Two reasons, both worth stating before the run rather than after:

1. **BM25 cannot separate "what I have" from "what I want."** The user's turn
   names both ("change my *17-inch* laptop order to a *13-inch* version… silver
   or black, i5 over i7"), so the 17-inch black variant scores highly and
   consumes half the budget.
2. **Budget 128 is below the floor for this observation.** The output was budget-
   bound at 128/128 tokens with 2 of 17 variants kept. Even a perfect ranker gets
   ~2 draws from a 17-option space. This is an under-budgeting result, not a
   compressor bug.

Point 2 is the reason the predictions below are deliberately modest.

## 3. Predictions (the falsifiable part)

Stated before seeing any new benchmark data. "Recovers" means the per-task
success rate over 5 trials rises above 0. Each prediction is a pre-specified
directional test; the decision rule (§6) is holistic, so no family-wise
correction is applied — the predictions are read together, not as independent
significance tests. "Scope" marks whether a prediction is scored on the headline
35 tasks (§4), on the placebo control stratum, or on the excluded diagnostic
tasks, which are reported separately.

The headline claim is a **placebo-controlled no-detectable-harm result** (a
CI-bounded difference-in-differences, not a formal TOST equivalence certificate —
see §5): once run-to-run LLM noise is measured by a byte-identical no-op control,
compression causes no task-success harm distinguishable from that control on
non-diagnostic tasks. The estimator is a difference-in-differences (§4): the
tooltrim − none gap on compression-*affected* headline tasks, minus the same gap
on the placebo stratum. Baseline anchors from the existing 600-episode run are
quoted so each prediction is checkable.

| # | Prediction | Scope | Confirm / refute criterion | Confidence |
|---|-----------|-------|----------------------------|-----------|
| P1 | Task 6 recovers (its instruction contains "water bottle" and "desk lamp", which the flat-map fix can now select on). | Diagnostic (t6) | Confirmed if task-6 tooltrim success over 5 trials > 0 **and** ≥ its `none` success. | Moderate — its `none` ceiling is only 1–2/5, so this is weakly powered either way. |
| P2 | Tasks 7 and 35 do **not** reliably recover at budget 128. The query fix helps ranking but the budget cannot represent a 12–18 option space. | Diagnostic (t7, t35) | Confirmed if each of tasks 7 and 35 has tooltrim success ≤ 1/5; refuted if either reaches ≥ 3/5. | Moderate-high |
| P3 | The **placebo stratum reproduces the noise floor.** The 19 byte-identical no-op tasks (§4.1) had a tooltrim − none gap of **−5.3pp** in the baseline — a pure measurement of run-to-run LLM stochasticity, since inputs are identical. The new run is a second independent draw of that floor. | Headline control (19 placebo) | Corroborated if the new placebo gap has a 95% cluster-bootstrap CI overlapping the baseline −5.3pp. **Refuted (harness drift — stop and investigate) if the two placebo estimates are separated beyond CI overlap**, i.e. the no-op control itself moved. | High |
| P4 | **Diff-in-differences is indistinguishable from zero.** DiD = (tooltrim − none on the 16 affected headline tasks) − (same on the 19 placebo). Baseline DiD = **+4.0pp** (affected −1.3pp *minus* placebo −5.3pp), baseline 95% CI [−8.4, +16.4]: affected tasks already lose *less* than the no-op control. | Headline (35, DiD) | Confirmed if the new-run DiD 95% cluster-bootstrap CI **includes 0** and its harmful-side bound is no worse than ~−15pp. Refuted if the CI excludes 0 on the harmful side. **No formal TOST equivalence is claimed** — the DiD half-width floors at ~±8pp at 40 tasks (§5), so this is a CI-bounded "no detectable harm", not an equivalence certificate. | Moderate-high |
| P5 | The high-exposure collapse is a property of the **diagnostic tasks**, not the headline. At budget 128 tooltrim stays clearly below none on tasks 0/6/7/15/35 (budget-bound, not defect-bound). Reported as exploratory — the headline high-6 stratum is 0%/0% in baseline and carries no signal. | Diagnostic (0,6,7,15,35), exploratory | Descriptive: report the per-task diagnostic gaps and whether they remain negative after the fixes. Motivates a budget sweep, not a harm claim. | Moderate (exploratory) |

**The honest headline if these hold:** compression is task-success-neutral against
a byte-identical control — the apparent −9.0pp "penalty" on all 40 tasks is an
artifact of (a) not controlling for a ~5pp benchmark noise floor and (b) five
tasks used to *find* the bugs. The reusable contribution is the placebo stratum as
a noise-floor instrument, pre-registered and held out. The high-exposure collapse
is real but concentrated and budget-bound; it is an exploratory secondary that
motivates a follow-up budget sweep, not the headline.

## 4. Analysis plan (fixed in advance)

- **Headline estimator: difference-in-differences, unit of analysis the task.**
  DiD = (tooltrim − none gap on the 16 compression-*affected* headline tasks,
  i.e. low ∪ high in §4.1) − (tooltrim − none gap on the 19 placebo tasks). The
  placebo term is a byte-identical no-op, so it measures run-to-run noise; the DiD
  nets it out. Cluster bootstrap over the 35 headline tasks, 20 000 resamples,
  seed 0; report the 95% CI on the DiD **and** on each component gap.
  `noise_floor.py` already implements the task-cluster bootstrap.
- **The headline is the DiD point estimate and its 95% cluster-bootstrap CI**,
  not a formal equivalence certificate. TOST at ±5pp and ±10pp is computed and
  reported for completeness, but §5 shows neither margin is reachable at 40 tasks,
  so the claim rests on the CI (does it contain 0? how far does the harmful bound
  reach?), not on a TOST pass. The absolute tooltrim − none gap on the 35 tasks is
  reported too, but the DiD is the headline because it cancels the shared
  task-difficulty structure the placebo stratum exposes.
- **The placebo stratum is the control and a second noise-floor estimate.** Its
  gap and CI are reported next to the baseline's −5.3pp (P3).
- The `p` column in generated `TAUBENCH.md` is **not** to be quoted: it runs
  McNemar on per-task majority-of-5 votes while the same row reports a Wilson CI
  over 200 observations as if independent. Fixing that is a separate task.
- **Tasks 0, 6, 7, 15 and 35 are reported separately and excluded from the
  headline.** They were used to find the bugs; testing on them is fitting and
  testing on the same data. The high-exposure collapse (P5) is analysed only here,
  as an exploratory secondary.

### 4.1 Frozen strata (task IDs fixed before launch)

Exposure = tool observations compressed / tool calls, **pooled over the 5
tooltrim trials of the existing 600-episode baseline**
(`benchmarks/taubench_multitrial/checkpoint.jsonl`). Exposure is
trajectory-dependent, so it is read from the already-paid baseline, not
recomputed. Strata: placebo = exposure 0; the remaining 21 tasks are median-split
(median exposure 0.4878), with ties assigned to `high` (`>= median`). The
derivation is `eval/freeze_strata.py`; these exact IDs are frozen now to remove
any post-hoc boundary choice.

| Stratum | All 40 tasks | Headline (excl. 0, 6, 7, 15, 35) |
|---|---|---|
| placebo (exposure 0) | 3, 4, 10, 11, 12, 13, 14, 16, 17, 18, 21, 22, 25, 28, 31, 32, 33, 34, 38 — **19** | same 19 |
| low (0 < exp < 0.4878) | 1, 5, 8, 9, 23, 24, 26, 30, 36, 37 — **10** | same 10 |
| high (exp ≥ 0.4878) | 0, 2, 6, 7, 15, 19, 20, 27, 29, 35, 39 — **11** | 2, 19, 20, 27, 29, 39 — **6** |

All five diagnostic tasks are in the high stratum, so the headline high stratum
is only **6 tasks**, and in the baseline both arms score 0% there — no signal.
This is why the high-exposure collapse is analysed on the diagnostic tasks
(exploratory, P5) rather than as a headline stratum. The 16 affected headline
tasks (low ∪ high) drive the DiD. Task 3 is a degenerate placebo (0 tool calls in
every trial: tooltrim can never fire); it is retained under the definition but
contributes no discrimination.

## 5. Run specification

```
python run_taubench.py --env retail --tasks 40 --budget 128 --trials 5 \
  --methods none,tooltrim,truncate-head \
  --agent-model gpt-4o-mini --agent-provider openai \
  --user-model gpt-4o-mini --user-provider openai \
  --max-concurrency 1 --num-retries 6 --dialogue-turns 6 \
  --results-dir benchmarks/taubench_multitrial_v2 --resume
```

600 solves, ~$20–30, ~5–8h, `--max-concurrency 1` (TIER-1, 200k TPM). Directly
comparable to `benchmarks/taubench_multitrial/` — same tasks, trials, budget and
models; the only differences are the three fixes.

**Power note (decided before launch — 5 trials).** The DiD is clustered over
tasks, so its CI has a floor set by between-task spread across the 16 affected +
19 placebo tasks. From the baseline's own rates, the analytic 95% half-width of
the DiD is ±14.8pp at 5 trials, ±11.4pp at 10, ±9.3pp at 20, and **±8.0pp even at
40 trials** — the placebo term is pure within-task noise (shrinks with trials) but
the affected term carries real between-task variance that trials cannot touch.
Consequences, accepted before launch:

- **Neither ±5pp nor ±10pp TOST equivalence is reachable at 40 tasks.** ±5pp is
  below the infinite-trial floor; ±10pp needs ~20 trials and only borderline. So
  the paper does **not** claim a formal equivalence certificate.
- **More trials is the wrong lever.** 10 trials (~2×, ~$40–60) narrows the CI but
  crosses no threshold that changes the claim. The lever for a tight certificate
  is more *tasks* (more clusters) — e.g. the full ~115 retail tasks at 5 trials —
  recorded here as a possible follow-up, not this run.
- **5 trials is sufficient for this paper's contribution:** the noise-floor
  instrument (placebo stratum), the CI-bounded DiD ("no detectable harm; harm
  worse than ~15pp ruled out"), and the decomposition of the −9.0pp all-40 figure
  into noise + fitted tasks. None of these needs a TOST pass.

## 6. Decision rule

- **P3 fails** (the new placebo gap diverges from the baseline −5.3pp beyond CI
  overlap): the no-op control itself moved, so the harness is not stable enough to
  attribute anything. Stop and investigate before interpreting — nothing else is
  readable.
- **P3 holds and P4's DiD CI includes 0**: this is the headline. Write the
  placebo-controlled paper — compression is task-success-neutral against a
  byte-identical control (no detectable harm, CI-bounded), the ~5pp noise floor is
  the reusable instrument, and the −9.0pp all-40 "penalty" is decomposed into
  noise + fitted diagnostic tasks. The claim is the DiD CI, not a TOST certificate
  (§5); a follow-up at the full ~115 tasks is the path to a formal equivalence
  bound if a reviewer demands one.
- **P4's DiD CI excludes 0 on the harmful side**: the fixes did not remove a real
  harm. Report it honestly as a measured penalty with the placebo floor as
  context; the equivalence claim is retracted, the noise-floor method still stands.
- **P5 (diagnostic collapse) persists**: the natural follow-up is a budget sweep
  (384/512/768) on the diagnostic tasks, reported as exploratory motivation — not
  a headline claim, since those tasks were used to find the bugs.

No outcome here is a failure — every branch produces a paper, and the noise-floor
method stands regardless of how the equivalence claim lands. That is the point of
writing this before the run.

## Amendments

_(none)_
