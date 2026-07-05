# Frontier-model matrix: accuracy/token Pareto

Cross-model comparison at a **256-token** budget. `full` is uncompressed tool output; `tooltrim` is query-aware compression to the budget. Retention = tooltrim accuracy / full-context accuracy.

| model | full acc | full tokens | tooltrim acc | tooltrim tokens | retention | tokens saved | tooltrim downstream | rag-topk downstream |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| claude-haiku-4-5 | 97% | 6,813 | 100% | 139 | 103% | 98.0% | 57% | 39% |
| gpt-4o-mini | 98% | 6,813 | 97% | 139 | 98% | 98.0% | 57% | 39% |
| claude-sonnet-5 | 73% | 6,813 | 92% | 139 | 127% | 98.0% | 57% | 39% |

*downstream* = fraction of json/tabular cases whose gold fact is recoverable from a valid parse in code (the agent's next step). tooltrim keeps valid structure; rag-topk shreds single-line JSON into unparseable fragments.

## tooltrim vs RAG top-k (paired McNemar)

Does content-type structure separate tooltrim from plain query-aware RAG selection under a real LLM judge?

| model | budget | Δ acc (tooltrim − rag-topk) | p-value | significant (p<0.05) |
|---|---:|---:|---:|---:|
| claude-haiku-4-5 | 128 | +1.5pp | 1.000 | no |
| claude-haiku-4-5 | 256 | +3.0pp | 0.480 | no |
| claude-haiku-4-5 | 800 | +0.0pp | 1.000 | no |
| claude-sonnet-5 | 128 | +0.0pp | 0.480 | no |
| claude-sonnet-5 | 256 | +0.0pp | 0.617 | no |
| claude-sonnet-5 | 800 | -1.5pp | 1.000 | no |
| gpt-4o-mini | 128 | +1.5pp | 1.000 | no |
| gpt-4o-mini | 256 | +1.5pp | 1.000 | no |
| gpt-4o-mini | 800 | +0.0pp | 1.000 | no |
