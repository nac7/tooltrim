# Frontier-model matrix: accuracy/token Pareto

Cross-model comparison at a **256-token** budget. `full` is uncompressed tool output; `tooltrim` is query-aware compression to the budget. Retention = tooltrim accuracy / full-context accuracy.

| model | full acc | full tokens | tooltrim acc | tooltrim tokens | retention | tokens saved |
|---|---:|---:|---:|---:|---:|---:|
| claude-haiku-4-5 | 97% | 6,635 | 100% | 177 | 103% | 97.3% |
| claude-sonnet-5 | 69% | 6,634 | 90% | 177 | 130% | 97.3% |

## tooltrim vs RAG top-k (paired McNemar)

Does content-type structure separate tooltrim from plain query-aware RAG selection under a real LLM judge?

| model | budget | Δ acc (tooltrim − rag-topk) | p-value | significant (p<0.05) |
|---|---:|---:|---:|---:|
| claude-haiku-4-5 | 128 | +1.6pp | 1.000 | no |
| claude-haiku-4-5 | 256 | +0.0pp | 1.000 | no |
| claude-haiku-4-5 | 800 | +0.0pp | 1.000 | no |
| claude-sonnet-5 | 128 | -1.6pp | 1.000 | no |
| claude-sonnet-5 | 256 | -3.2pp | 0.617 | no |
| claude-sonnet-5 | 800 | -1.6pp | 1.000 | no |
