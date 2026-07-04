## Comparative faithfulness — `claude-sonnet-5`

Full-context accuracy: **43/62 (69.4%)** [57-79%], avg 6,634 tokens/case. CIs are 95% Wilson; each method is scored on the *same* cases so accuracies are directly comparable.

### Budget 128 tokens

| method | comp tokens | tokens saved | accuracy | 95% CI | retention |
|:--|---:|---:|---:|:--:|---:|
| `rag-topk` | 54 | 99.2% | 61/62 (98%) | [91-100%] | 141.9% |
| `tooltrim` **★** | 94 | 98.6% | 60/62 (97%) | [89-99%] | 139.5% |
| `full` | 6,634 | 0.0% | 43/62 (69%) | [57-79%] | 100.0% |
| `truncate-head` | 128 | 98.1% | 7/62 (11%) | [6-22%] | 16.3% |
| `truncate-tail` | 128 | 98.1% | 2/62 (3%) | [1-11%] | 4.7% |

### Budget 256 tokens

| method | comp tokens | tokens saved | accuracy | 95% CI | retention |
|:--|---:|---:|---:|:--:|---:|
| `rag-topk` | 100 | 98.5% | 58/62 (94%) | [85-97%] | 134.9% |
| `tooltrim` **★** | 177 | 97.3% | 56/62 (90%) | [80-95%] | 130.2% |
| `full` | 6,634 | 0.0% | 43/62 (69%) | [57-79%] | 100.0% |
| `truncate-tail` | 256 | 96.1% | 2/62 (3%) | [1-11%] | 4.7% |
| `truncate-head` | 256 | 96.1% | 1/62 (2%) | [0-9%] | 2.3% |

### Budget 800 tokens

| method | comp tokens | tokens saved | accuracy | 95% CI | retention |
|:--|---:|---:|---:|:--:|---:|
| `rag-topk` | 259 | 96.1% | 57/62 (92%) | [82-97%] | 132.6% |
| `tooltrim` **★** | 379 | 94.3% | 56/62 (90%) | [80-95%] | 130.2% |
| `full` | 6,634 | 0.0% | 43/62 (69%) | [57-79%] | 100.0% |
| `truncate-head` | 800 | 87.9% | 6/62 (10%) | [5-20%] | 14.0% |
| `truncate-tail` | 800 | 87.9% | 4/62 (6%) | [3-15%] | 9.3% |

### Significance — `tooltrim` vs each baseline (McNemar, paired)

| budget | baseline | Δ accuracy | wins/losses | p-value | significant (p<0.05) |
|---:|:--|---:|:--:|---:|:--:|
| 128 | `full` | +27.4% | 18/1 | 0.000 | yes |
| 128 | `truncate-head` | +85.5% | 53/0 | 0.000 | yes |
| 128 | `truncate-tail` | +93.5% | 58/0 | 0.000 | yes |
| 128 | `rag-topk` | -1.6% | 1/2 | 1.000 | no |
| 256 | `full` | +21.0% | 14/1 | 0.002 | yes |
| 256 | `truncate-head` | +88.7% | 55/0 | 0.000 | yes |
| 256 | `truncate-tail` | +87.1% | 54/0 | 0.000 | yes |
| 256 | `rag-topk` | -3.2% | 1/3 | 0.617 | no |
| 800 | `full` | +21.0% | 13/0 | 0.001 | yes |
| 800 | `truncate-head` | +80.6% | 50/0 | 0.000 | yes |
| 800 | `truncate-tail` | +83.9% | 52/0 | 0.000 | yes |
| 800 | `rag-topk` | -1.6% | 1/2 | 1.000 | no |
