## Comparative faithfulness — `claude-sonnet-5`

Full-context accuracy: **48/66 (72.7%)** [61-82%], avg 6,813 tokens/case. CIs are 95% Wilson; each method is scored on the *same* cases so accuracies are directly comparable.

### Budget 128 tokens

| method | comp tokens | tokens saved | accuracy | 95% CI | retention | parseable | downstream |
|:--|---:|---:|---:|:--:|---:|---:|---:|
| `rag-topk` | 57 | 99.2% | 63/66 (95%) | [87-98%] | 131.2% | 61% | 39% |
| `tooltrim` **★** | 85 | 98.8% | 63/66 (95%) | [87-98%] | 131.2% | 67% | 50% |
| `full` | 6,813 | 0.0% | 48/66 (73%) | [61-82%] | 100.0% | 85% | 100% |
| `truncate-head` | 128 | 98.1% | 7/66 (11%) | [5-20%] | 14.6% | 55% | 0% |
| `truncate-tail` | 128 | 98.1% | 4/66 (6%) | [2-15%] | 8.3% | 27% | 0% |

### Budget 256 tokens

| method | comp tokens | tokens saved | accuracy | 95% CI | retention | parseable | downstream |
|:--|---:|---:|---:|:--:|---:|---:|---:|
| `rag-topk` | 103 | 98.5% | 61/66 (92%) | [83-97%] | 127.1% | 61% | 39% |
| `tooltrim` **★** | 139 | 98.0% | 61/66 (92%) | [83-97%] | 127.1% | 67% | 57% |
| `full` | 6,813 | 0.0% | 48/66 (73%) | [61-82%] | 100.0% | 85% | 100% |
| `truncate-head` | 256 | 96.2% | 2/66 (3%) | [1-10%] | 4.2% | 41% | 0% |
| `truncate-tail` | 256 | 96.2% | 2/66 (3%) | [1-10%] | 4.2% | 27% | 0% |

### Budget 800 tokens

| method | comp tokens | tokens saved | accuracy | 95% CI | retention | parseable | downstream |
|:--|---:|---:|---:|:--:|---:|---:|---:|
| `rag-topk` | 271 | 96.0% | 62/66 (94%) | [85-98%] | 129.2% | 61% | 39% |
| `tooltrim` **★** | 345 | 94.9% | 61/66 (92%) | [83-97%] | 127.1% | 67% | 57% |
| `full` | 6,813 | 0.0% | 48/66 (73%) | [61-82%] | 100.0% | 85% | 100% |
| `truncate-head` | 800 | 88.3% | 7/66 (11%) | [5-20%] | 14.6% | 35% | 4% |
| `truncate-tail` | 800 | 88.3% | 6/66 (9%) | [4-18%] | 12.5% | 29% | 0% |

*parseable* = fraction of compressed outputs that still parse as their content type (what the agent's next `json.loads`/CSV read does). *downstream* = fraction of json/tabular cases whose gold fact is recoverable from a valid parse in code.

### Significance — `tooltrim` vs each baseline (McNemar, paired)

| budget | baseline | Δ accuracy | wins/losses | p-value | significant (p<0.05) |
|---:|:--|---:|:--:|---:|:--:|
| 128 | `full` | +22.7% | 17/2 | 0.001 | yes |
| 128 | `truncate-head` | +84.8% | 56/0 | 0.000 | yes |
| 128 | `truncate-tail` | +89.4% | 59/0 | 0.000 | yes |
| 128 | `rag-topk` | +0.0% | 1/1 | 0.480 | no |
| 256 | `full` | +19.7% | 13/0 | 0.001 | yes |
| 256 | `truncate-head` | +89.4% | 59/0 | 0.000 | yes |
| 256 | `truncate-tail` | +89.4% | 59/0 | 0.000 | yes |
| 256 | `rag-topk` | +0.0% | 2/2 | 0.617 | no |
| 800 | `full` | +19.7% | 13/0 | 0.001 | yes |
| 800 | `truncate-head` | +81.8% | 54/0 | 0.000 | yes |
| 800 | `truncate-tail` | +83.3% | 55/0 | 0.000 | yes |
| 800 | `rag-topk` | -1.5% | 0/1 | 1.000 | no |
