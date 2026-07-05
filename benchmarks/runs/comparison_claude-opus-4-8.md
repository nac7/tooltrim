## Comparative faithfulness — `claude-opus-4-8`

Full-context accuracy: **48/66 (72.7%)** [61-82%], avg 6,813 tokens/case. CIs are 95% Wilson; each method is scored on the *same* cases so accuracies are directly comparable.

### Budget 128 tokens

| method | comp tokens | tokens saved | accuracy | 95% CI | retention | parseable | downstream |
|:--|---:|---:|---:|:--:|---:|---:|---:|
| `tooltrim` **★** | 85 | 98.8% | 64/66 (97%) | [90-99%] | 133.3% | 67% | 50% |
| `rag-topk` | 57 | 99.2% | 63/66 (95%) | [87-98%] | 131.2% | 61% | 39% |
| `full` | 6,813 | 0.0% | 48/66 (73%) | [61-82%] | 100.0% | 85% | 100% |
| `truncate-tail` | 128 | 98.1% | 12/66 (18%) | [11-29%] | 25.0% | 27% | 0% |
| `truncate-head` | 128 | 98.1% | 11/66 (17%) | [10-27%] | 22.9% | 55% | 0% |

### Budget 256 tokens

| method | comp tokens | tokens saved | accuracy | 95% CI | retention | parseable | downstream |
|:--|---:|---:|---:|:--:|---:|---:|---:|
| `tooltrim` **★** | 139 | 98.0% | 66/66 (100%) | [94-100%] | 137.5% | 67% | 57% |
| `rag-topk` | 103 | 98.5% | 63/66 (95%) | [87-98%] | 131.2% | 61% | 39% |
| `full` | 6,813 | 0.0% | 48/66 (73%) | [61-82%] | 100.0% | 85% | 100% |
| `truncate-head` | 256 | 96.2% | 13/66 (20%) | [12-31%] | 27.1% | 41% | 0% |
| `truncate-tail` | 256 | 96.2% | 6/66 (9%) | [4-18%] | 12.5% | 27% | 0% |

### Budget 800 tokens

| method | comp tokens | tokens saved | accuracy | 95% CI | retention | parseable | downstream |
|:--|---:|---:|---:|:--:|---:|---:|---:|
| `rag-topk` | 271 | 96.0% | 63/66 (95%) | [87-98%] | 131.2% | 61% | 39% |
| `tooltrim` **★** | 345 | 94.9% | 62/66 (94%) | [85-98%] | 129.2% | 67% | 57% |
| `full` | 6,813 | 0.0% | 48/66 (73%) | [61-82%] | 100.0% | 85% | 100% |
| `truncate-head` | 800 | 88.3% | 21/66 (32%) | [22-44%] | 43.8% | 35% | 4% |
| `truncate-tail` | 800 | 88.3% | 15/66 (23%) | [14-34%] | 31.2% | 29% | 0% |

*parseable* = fraction of compressed outputs that still parse as their content type (what the agent's next `json.loads`/CSV read does). *downstream* = fraction of json/tabular cases whose gold fact is recoverable from a valid parse in code.

### Significance — `tooltrim` vs each baseline (McNemar, paired)

| budget | baseline | Δ accuracy | wins/losses | p-value | significant (p<0.05) |
|---:|:--|---:|:--:|---:|:--:|
| 128 | `full` | +24.2% | 18/2 | 0.001 | yes |
| 128 | `truncate-head` | +80.3% | 53/0 | 0.000 | yes |
| 128 | `truncate-tail` | +78.8% | 52/0 | 0.000 | yes |
| 128 | `rag-topk` | +1.5% | 1/0 | 1.000 | no |
| 256 | `full` | +27.3% | 18/0 | 0.000 | yes |
| 256 | `truncate-head` | +80.3% | 53/0 | 0.000 | yes |
| 256 | `truncate-tail` | +90.9% | 60/0 | 0.000 | yes |
| 256 | `rag-topk` | +4.5% | 3/0 | 0.248 | no |
| 800 | `full` | +21.2% | 14/0 | 0.001 | yes |
| 800 | `truncate-head` | +62.1% | 41/0 | 0.000 | yes |
| 800 | `truncate-tail` | +71.2% | 47/0 | 0.000 | yes |
| 800 | `rag-topk` | -1.5% | 0/1 | 1.000 | no |
