## Comparative faithfulness — `claude-haiku-4-5`

Full-context accuracy: **64/66 (97.0%)** [90-99%], avg 6,813 tokens/case. CIs are 95% Wilson; each method is scored on the *same* cases so accuracies are directly comparable.

### Budget 128 tokens

| method | comp tokens | tokens saved | accuracy | 95% CI | retention | parseable | downstream |
|:--|---:|---:|---:|:--:|---:|---:|---:|
| `tooltrim` **★** | 85 | 98.8% | 64/66 (97%) | [90-99%] | 100.0% | 67% | 50% |
| `full` | 6,813 | 0.0% | 64/66 (97%) | [90-99%] | 100.0% | 85% | 100% |
| `rag-topk` | 57 | 99.2% | 63/66 (95%) | [87-98%] | 98.4% | 61% | 39% |
| `truncate-head` | 128 | 98.1% | 15/66 (23%) | [14-34%] | 23.4% | 55% | 0% |
| `truncate-tail` | 128 | 98.1% | 13/66 (20%) | [12-31%] | 20.3% | 27% | 0% |

### Budget 256 tokens

| method | comp tokens | tokens saved | accuracy | 95% CI | retention | parseable | downstream |
|:--|---:|---:|---:|:--:|---:|---:|---:|
| `tooltrim` **★** | 139 | 98.0% | 66/66 (100%) | [94-100%] | 103.1% | 67% | 57% |
| `rag-topk` | 103 | 98.5% | 64/66 (97%) | [90-99%] | 100.0% | 61% | 39% |
| `full` | 6,813 | 0.0% | 64/66 (97%) | [90-99%] | 100.0% | 85% | 100% |
| `truncate-tail` | 256 | 96.2% | 22/66 (33%) | [23-45%] | 34.4% | 27% | 0% |
| `truncate-head` | 256 | 96.2% | 17/66 (26%) | [17-37%] | 26.6% | 41% | 0% |

### Budget 800 tokens

| method | comp tokens | tokens saved | accuracy | 95% CI | retention | parseable | downstream |
|:--|---:|---:|---:|:--:|---:|---:|---:|
| `rag-topk` | 271 | 96.0% | 66/66 (100%) | [94-100%] | 103.1% | 61% | 39% |
| `tooltrim` **★** | 345 | 94.9% | 66/66 (100%) | [94-100%] | 103.1% | 67% | 57% |
| `full` | 6,813 | 0.0% | 64/66 (97%) | [90-99%] | 100.0% | 85% | 100% |
| `truncate-tail` | 800 | 88.3% | 26/66 (39%) | [29-51%] | 40.6% | 29% | 0% |
| `truncate-head` | 800 | 88.3% | 24/66 (36%) | [26-48%] | 37.5% | 35% | 4% |

*parseable* = fraction of compressed outputs that still parse as their content type (what the agent's next `json.loads`/CSV read does). *downstream* = fraction of json/tabular cases whose gold fact is recoverable from a valid parse in code.

### Significance — `tooltrim` vs each baseline (McNemar, paired)

| budget | baseline | Δ accuracy | wins/losses | p-value | significant (p<0.05) |
|---:|:--|---:|:--:|---:|:--:|
| 128 | `full` | +0.0% | 2/2 | 0.617 | no |
| 128 | `truncate-head` | +74.2% | 49/0 | 0.000 | yes |
| 128 | `truncate-tail` | +77.3% | 51/0 | 0.000 | yes |
| 128 | `rag-topk` | +1.5% | 1/0 | 1.000 | no |
| 256 | `full` | +3.0% | 2/0 | 0.480 | no |
| 256 | `truncate-head` | +74.2% | 49/0 | 0.000 | yes |
| 256 | `truncate-tail` | +66.7% | 44/0 | 0.000 | yes |
| 256 | `rag-topk` | +3.0% | 2/0 | 0.480 | no |
| 800 | `full` | +3.0% | 2/0 | 0.480 | no |
| 800 | `truncate-head` | +63.6% | 42/0 | 0.000 | yes |
| 800 | `truncate-tail` | +60.6% | 40/0 | 0.000 | yes |
| 800 | `rag-topk` | +0.0% | 0/0 | 1.000 | no |
