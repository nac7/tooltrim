## Comparative faithfulness — `claude-haiku-4-5`

Full-context accuracy: **60/62 (96.8%)** [89-99%], avg 6,635 tokens/case. CIs are 95% Wilson; each method is scored on the *same* cases so accuracies are directly comparable.

### Budget 128 tokens

| method | comp tokens | tokens saved | accuracy | 95% CI | retention |
|:--|---:|---:|---:|:--:|---:|
| `tooltrim` **★** | 94 | 98.6% | 62/62 (100%) | [94-100%] | 103.3% |
| `rag-topk` | 55 | 99.2% | 61/62 (98%) | [91-100%] | 101.7% |
| `full` | 6,635 | 0.0% | 60/62 (97%) | [89-99%] | 100.0% |
| `truncate-head` | 128 | 98.1% | 15/62 (24%) | [15-36%] | 25.0% |
| `truncate-tail` | 128 | 98.1% | 13/62 (21%) | [13-33%] | 21.7% |

### Budget 256 tokens

| method | comp tokens | tokens saved | accuracy | 95% CI | retention |
|:--|---:|---:|---:|:--:|---:|
| `rag-topk` | 100 | 98.5% | 62/62 (100%) | [94-100%] | 103.3% |
| `tooltrim` **★** | 177 | 97.3% | 62/62 (100%) | [94-100%] | 103.3% |
| `full` | 6,635 | 0.0% | 60/62 (97%) | [89-99%] | 100.0% |
| `truncate-tail` | 256 | 96.1% | 22/62 (35%) | [25-48%] | 36.7% |
| `truncate-head` | 256 | 96.1% | 17/62 (27%) | [18-40%] | 28.3% |

### Budget 800 tokens

| method | comp tokens | tokens saved | accuracy | 95% CI | retention |
|:--|---:|---:|---:|:--:|---:|
| `rag-topk` | 260 | 96.1% | 62/62 (100%) | [94-100%] | 103.3% |
| `tooltrim` **★** | 380 | 94.3% | 62/62 (100%) | [94-100%] | 103.3% |
| `full` | 6,635 | 0.0% | 60/62 (97%) | [89-99%] | 100.0% |
| `truncate-tail` | 800 | 87.9% | 25/62 (40%) | [29-53%] | 41.7% |
| `truncate-head` | 800 | 87.9% | 24/62 (39%) | [28-51%] | 40.0% |

### Significance — `tooltrim` vs each baseline (McNemar, paired)

| budget | baseline | Δ accuracy | wins/losses | p-value | significant (p<0.05) |
|---:|:--|---:|:--:|---:|:--:|
| 128 | `full` | +3.2% | 2/0 | 0.480 | no |
| 128 | `truncate-head` | +75.8% | 47/0 | 0.000 | yes |
| 128 | `truncate-tail` | +79.0% | 49/0 | 0.000 | yes |
| 128 | `rag-topk` | +1.6% | 1/0 | 1.000 | no |
| 256 | `full` | +3.2% | 2/0 | 0.480 | no |
| 256 | `truncate-head` | +72.6% | 45/0 | 0.000 | yes |
| 256 | `truncate-tail` | +64.5% | 40/0 | 0.000 | yes |
| 256 | `rag-topk` | +0.0% | 0/0 | 1.000 | no |
| 800 | `full` | +3.2% | 2/0 | 0.480 | no |
| 800 | `truncate-head` | +61.3% | 38/0 | 0.000 | yes |
| 800 | `truncate-tail` | +59.7% | 37/0 | 0.000 | yes |
| 800 | `rag-topk` | +0.0% | 0/0 | 1.000 | no |
