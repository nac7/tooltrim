# Comparative faithfulness benchmark — tooltrim vs the baselines

`benchmarks/COMPARISON.md` and the faithfulness harness measure tooltrim against
*full context*. That answers "does compression keep the model correct?" but not
the question a reviewer actually asks: **"does it beat the obvious
alternatives?"** This benchmark scores tooltrim against those alternatives on the
same 62 cases, at the same budgets, with the same judge and a paired
significance test.

## Methods

| method | what it does | needs |
|:--|:--|:--|
| `full` | no compression (accuracy ceiling, token upper bound) | — |
| `truncate-head` | keep the first *budget* tokens (the naive default) | — |
| `truncate-tail` | keep the last *budget* tokens | — |
| `rag-topk` | chunk → BM25-rank against the query → keep best chunks in original order (content-type-agnostic RAG selection) | — |
| `rag-embed` | same, but semantic | `tooltrim[embeddings]` |
| `llmlingua-2` | Microsoft LLMLingua-2 token-level compression | `pip install llmlingua` |
| `tooltrim` | per-content-type router + query-aware relevance | — |

## Run it

```bash
# offline, no keys, no model downloads (uses the deterministic retrieval judge)
python run_baselines.py --out benchmarks/runs

# add optional baselines
python run_baselines.py --methods full,truncate-head,rag-topk,rag-embed,tooltrim
python run_baselines.py --with-llmlingua        # needs: pip install llmlingua

# score with a real LLM judge and cache answers so reruns don't re-spend
python run_baselines.py --model groq --cache .cache/groq.json --out benchmarks/runs
```

## Headline result (offline judge, 62 cases)

Full-context accuracy is **90.3%** (56/62). Retention = method accuracy ÷
full-context accuracy.

| budget | truncate-head | truncate-tail | rag-topk | **tooltrim** |
|---:|---:|---:|---:|---:|
| 128 | 1.8% | 1.8% | 100.0% | **100.0%** |
| 256 | 3.6% | 3.6% | 100.0% | **100.0%** |
| 400 | 7.1% | 8.9% | 100.0% | **100.0%** |
| 800 | 12.5% | 14.3% | 100.0% | **100.0%** |

Query-aware compression (tooltrim, rag-topk) retains **100%** of full-context
accuracy while cutting **94–99%** of tokens; naive truncation throws away the
needed fact and collapses to single-digit retention. The gap over truncation is
significant at every budget (McNemar, p < 0.001).

## Honest reading of this table

- The **offline judge is itself a lexical retriever**, so a lexical *selection*
  baseline (`rag-topk`) ties tooltrim here — as it should. This run proves the
  thesis *against truncation* and validates the harness; it does **not**
  differentiate tooltrim from RAG selection.
- tooltrim's edge — **content-type structure** (keeping a JSON schema, a table
  header, a log's error lines rather than an arbitrary word window) and robust
  handling of HTML/tabular/log outputs — shows up with a **real LLM judge** and
  on **structured** content, not against a BM25 proxy. Re-run with
  `--model claude|openai|groq` on API budget to surface it; that is exactly the
  comparison the paper's baseline/Pareto section is built from.

## Real-LLM pilot — Groq `llama-3.3-70b-versatile` (n=8)

First run against a live hosted model (a small subset; the full-context
reference pass is the token hog and free-tier caps limit n). Retention vs
full-context accuracy (8/8):

| budget | truncate-head | truncate-tail | rag-topk | tooltrim |
|---:|---:|---:|---:|---:|
| 256 | 0% | 0% | 100% | 87.5% |
| 800 | 0% | 12.5% | 100% | 87.5% |

Reading this honestly:

- The **robust** finding holds on a real model: query-aware compression beats
  naive truncation decisively, and it is **significant even at n=8** (McNemar
  p = 0.023–0.041). Truncation to a fixed budget throws the needle away.
- On this tiny sample, **rag-topk edged tooltrim by a single case** (8/8 vs 7/8;
  the difference is *not* significant, p = 1.0). n=8 is a smoke test, not a
  verdict — tooltrim-vs-RAG needs the full set at real-LLM budget to separate
  them, and that one lost case is worth inspecting (`benchmarks/runs/comparison_llama-3.3-70b-versatile.md`).

So: the harness runs end-to-end on a hosted LLM and cleanly proves query-aware ≫
truncation; distinguishing tooltrim from RAG selection is the open question the
full-budget paper run must answer.

## Outputs

`run_baselines.py --out DIR` writes `comparison_<model>.md` (tables +
significance) and `comparison_<model>.csv` (one row per method×budget, for
plotting the accuracy/token Pareto).
