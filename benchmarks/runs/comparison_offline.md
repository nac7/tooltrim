## Comparative faithfulness — `offline`

Full-context accuracy: **57/66 (86.4%)** [76-93%], avg 6,813 tokens/case. CIs are 95% Wilson; each method is scored on the *same* cases so accuracies are directly comparable.

### Budget 128 tokens

| method | comp tokens | tokens saved | accuracy | 95% CI | retention | parseable | downstream |
|:--|---:|---:|---:|:--:|---:|---:|---:|
| `tooltrim` **★** | 85 | 98.8% | 58/66 (88%) | [78-94%] | 101.8% | 67% | 50% |
| `rag-topk` | 57 | 99.2% | 57/66 (86%) | [76-93%] | 100.0% | 61% | 39% |
| `full` | 6,813 | 0.0% | 57/66 (86%) | [76-93%] | 100.0% | 85% | 100% |
| `truncate-head` | 128 | 98.1% | 1/66 (2%) | [0-8%] | 1.8% | 55% | 0% |
| `truncate-tail` | 128 | 98.1% | 1/66 (2%) | [0-8%] | 1.8% | 27% | 0% |

### Budget 256 tokens

| method | comp tokens | tokens saved | accuracy | 95% CI | retention | parseable | downstream |
|:--|---:|---:|---:|:--:|---:|---:|---:|
| `tooltrim` **★** | 139 | 98.0% | 60/66 (91%) | [82-96%] | 105.3% | 67% | 57% |
| `rag-topk` | 103 | 98.5% | 57/66 (86%) | [76-93%] | 100.0% | 61% | 39% |
| `full` | 6,813 | 0.0% | 57/66 (86%) | [76-93%] | 100.0% | 85% | 100% |
| `truncate-head` | 256 | 96.2% | 2/66 (3%) | [1-10%] | 3.5% | 41% | 0% |
| `truncate-tail` | 256 | 96.2% | 2/66 (3%) | [1-10%] | 3.5% | 27% | 0% |

### Budget 800 tokens

| method | comp tokens | tokens saved | accuracy | 95% CI | retention | parseable | downstream |
|:--|---:|---:|---:|:--:|---:|---:|---:|
| `tooltrim` **★** | 345 | 94.9% | 60/66 (91%) | [82-96%] | 105.3% | 67% | 57% |
| `rag-topk` | 271 | 96.0% | 57/66 (86%) | [76-93%] | 100.0% | 61% | 39% |
| `full` | 6,813 | 0.0% | 57/66 (86%) | [76-93%] | 100.0% | 85% | 100% |
| `truncate-tail` | 800 | 88.3% | 9/66 (14%) | [7-24%] | 15.8% | 29% | 0% |
| `truncate-head` | 800 | 88.3% | 7/66 (11%) | [5-20%] | 12.3% | 35% | 4% |

*parseable* = fraction of compressed outputs that still parse as their content type (what the agent's next `json.loads`/CSV read does). *downstream* = fraction of json/tabular cases whose gold fact is recoverable from a valid parse in code.

### Significance — `tooltrim` vs each baseline (McNemar, paired)

| budget | baseline | Δ accuracy | wins/losses | p-value | significant (p<0.05) |
|---:|:--|---:|:--:|---:|:--:|
| 128 | `full` | +1.5% | 1/0 | 1.000 | no |
| 128 | `truncate-head` | +86.4% | 57/0 | 0.000 | yes |
| 128 | `truncate-tail` | +86.4% | 57/0 | 0.000 | yes |
| 128 | `rag-topk` | +1.5% | 2/1 | 1.000 | no |
| 256 | `full` | +4.5% | 3/0 | 0.248 | no |
| 256 | `truncate-head` | +87.9% | 58/0 | 0.000 | yes |
| 256 | `truncate-tail` | +87.9% | 58/0 | 0.000 | yes |
| 256 | `rag-topk` | +4.5% | 3/0 | 0.248 | no |
| 800 | `full` | +4.5% | 3/0 | 0.248 | no |
| 800 | `truncate-head` | +80.3% | 53/0 | 0.000 | yes |
| 800 | `truncate-tail` | +77.3% | 51/0 | 0.000 | yes |
| 800 | `rag-topk` | +4.5% | 3/0 | 0.248 | no |
