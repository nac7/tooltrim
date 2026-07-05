# Submitting to the TOFB Leaderboard

The **Tool-Output Faithfulness Benchmark (TOFB)** is open for submissions. If you
have a tool-output compression method (a RAG variant, a learned compressor, a
prompt-compression model, a new content-type router), you can add it to the
leaderboard by opening a pull request.

## Tracks

| track | question | judge |
|:--|:--|:--|
| `faithfulness` | Does the needed fact survive compression? | offline BM25 **or** real-LLM |
| `downstream_extractability` | Does the compressed output still parse in code? | deterministic (`json.loads`/CSV) |
| `agent_tasks` | Can an agent parse it and compute the right answer? | deterministic (parse + arithmetic) |

The `agent_tasks` and `downstream_extractability` tracks use **no LLM judge** —
success is decided by real code — so they are cheap to reproduce and impossible
to game with an answer an LLM merely reads around.

## How to submit

1. Reproduce your numbers with a script in the repo (or your own, linked in the
   entry's `source` field). The headline harnesses are `run_baselines.py`,
   `run_frontier.py`, and `run_agent_tasks.py`.
2. Add one JSON object per (method, track, model, budget) to the `entries` array
   in [`results.json`](results.json). Required fields: `method`, `track`,
   `metric`, `value` (a rate in `[0,1]`), `n`, `source`. A significance claim
   (`vs_rag_topk.significant: true`) **must** include `mcnemar_p < 0.05`.
3. Run the validator locally — it must pass, and it also regenerates the table:

   ```bash
   python leaderboard/validate.py --render
   ```

4. Open a PR. CI runs the same validator; a green check is required to merge.

## Ground rules

- **Same cases, same budgets.** Compare on the released TOFB cases at the
  standard budgets (128 / 256 / 800 tokens) so entries are comparable.
- **No LLM judge on the deterministic tracks.** `agent_tasks` and
  `downstream_extractability` are code-scored by design.
- **Claims carry evidence.** Any significance or "beats X" claim must be backed
  by the paired McNemar `p`-value in the entry.

## Citation

If you use TOFB, please cite:

> Lele, N. *Faithful, Query-Aware Compression of LLM Agent Tool Outputs.*
> https://github.com/nac7/tooltrim
