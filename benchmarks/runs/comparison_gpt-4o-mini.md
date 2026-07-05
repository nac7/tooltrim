## Comparative faithfulness — `gpt-4o-mini`

Full-context accuracy: **65/66 (98.5%)** [92-100%], avg 6,813 tokens/case. CIs are 95% Wilson; each method is scored on the *same* cases so accuracies are directly comparable.

### Budget 128 tokens

| method | comp tokens | tokens saved | accuracy | 95% CI | retention | parseable | downstream |
|:--|---:|---:|---:|:--:|---:|---:|---:|
| `full` | 6,813 | 0.0% | 65/66 (98%) | [92-100%] | 100.0% | 85% | 100% |
| `tooltrim` **★** | 85 | 98.8% | 62/66 (94%) | [85-98%] | 95.4% | 67% | 50% |
| `rag-topk` | 57 | 99.2% | 61/66 (92%) | [83-97%] | 93.8% | 61% | 39% |
| `truncate-head` | 128 | 98.1% | 1/66 (2%) | [0-8%] | 1.5% | 55% | 0% |
| `truncate-tail` | 128 | 98.1% | 1/66 (2%) | [0-8%] | 1.5% | 27% | 0% |

### Budget 256 tokens

| method | comp tokens | tokens saved | accuracy | 95% CI | retention | parseable | downstream |
|:--|---:|---:|---:|:--:|---:|---:|---:|
| `full` | 6,813 | 0.0% | 65/66 (98%) | [92-100%] | 100.0% | 85% | 100% |
| `tooltrim` **★** | 139 | 98.0% | 64/66 (97%) | [90-99%] | 98.5% | 67% | 57% |
| `rag-topk` | 103 | 98.5% | 63/66 (95%) | [87-98%] | 96.9% | 61% | 39% |
| `truncate-head` | 256 | 96.2% | 2/66 (3%) | [1-10%] | 3.1% | 41% | 0% |
| `truncate-tail` | 256 | 96.2% | 2/66 (3%) | [1-10%] | 3.1% | 27% | 0% |

### Budget 800 tokens

| method | comp tokens | tokens saved | accuracy | 95% CI | retention | parseable | downstream |
|:--|---:|---:|---:|:--:|---:|---:|---:|
| `full` | 6,813 | 0.0% | 65/66 (98%) | [92-100%] | 100.0% | 85% | 100% |
| `rag-topk` | 271 | 96.0% | 64/66 (97%) | [90-99%] | 98.5% | 61% | 39% |
| `tooltrim` **★** | 345 | 94.9% | 64/66 (97%) | [90-99%] | 98.5% | 67% | 57% |
| `truncate-tail` | 800 | 88.3% | 9/66 (14%) | [7-24%] | 13.8% | 29% | 0% |
| `truncate-head` | 800 | 88.3% | 6/66 (9%) | [4-18%] | 9.2% | 35% | 4% |

*parseable* = fraction of compressed outputs that still parse as their content type (what the agent's next `json.loads`/CSV read does). *downstream* = fraction of json/tabular cases whose gold fact is recoverable from a valid parse in code.

### Significance — `tooltrim` vs each baseline (McNemar, paired)

| budget | baseline | Δ accuracy | wins/losses | p-value | significant (p<0.05) |
|---:|:--|---:|:--:|---:|:--:|
| 128 | `full` | -4.5% | 0/3 | 0.248 | no |
| 128 | `truncate-head` | +92.4% | 61/0 | 0.000 | yes |
| 128 | `truncate-tail` | +92.4% | 61/0 | 0.000 | yes |
| 128 | `rag-topk` | +1.5% | 2/1 | 1.000 | no |
| 256 | `full` | -1.5% | 0/1 | 1.000 | no |
| 256 | `truncate-head` | +93.9% | 62/0 | 0.000 | yes |
| 256 | `truncate-tail` | +93.9% | 62/0 | 0.000 | yes |
| 256 | `rag-topk` | +1.5% | 1/0 | 1.000 | no |
| 800 | `full` | -1.5% | 0/1 | 1.000 | no |
| 800 | `truncate-head` | +87.9% | 58/0 | 0.000 | yes |
| 800 | `truncate-tail` | +83.3% | 55/0 | 0.000 | yes |
| 800 | `rag-topk` | +0.0% | 0/0 | 1.000 | no |
