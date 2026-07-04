## Comparative faithfulness — `offline`

Full-context accuracy: **56/62 (90.3%)** [80-95%], avg 6,635 tokens/case. CIs are 95% Wilson; each method is scored on the *same* cases so accuracies are directly comparable.

### Budget 128 tokens

| method | comp tokens | tokens saved | accuracy | 95% CI | retention |
|:--|---:|---:|---:|:--:|---:|
| `rag-topk` | 55 | 99.2% | 56/62 (90%) | [80-95%] | 100.0% |
| `tooltrim` **★** | 94 | 98.6% | 56/62 (90%) | [80-95%] | 100.0% |
| `full` | 6,635 | 0.0% | 56/62 (90%) | [80-95%] | 100.0% |
| `truncate-head` | 128 | 98.1% | 1/62 (2%) | [0-9%] | 1.8% |
| `truncate-tail` | 128 | 98.1% | 1/62 (2%) | [0-9%] | 1.8% |

### Budget 256 tokens

| method | comp tokens | tokens saved | accuracy | 95% CI | retention |
|:--|---:|---:|---:|:--:|---:|
| `rag-topk` | 100 | 98.5% | 56/62 (90%) | [80-95%] | 100.0% |
| `tooltrim` **★** | 177 | 97.3% | 56/62 (90%) | [80-95%] | 100.0% |
| `full` | 6,635 | 0.0% | 56/62 (90%) | [80-95%] | 100.0% |
| `truncate-head` | 256 | 96.1% | 2/62 (3%) | [1-11%] | 3.6% |
| `truncate-tail` | 256 | 96.1% | 2/62 (3%) | [1-11%] | 3.6% |

### Budget 400 tokens

| method | comp tokens | tokens saved | accuracy | 95% CI | retention |
|:--|---:|---:|---:|:--:|---:|
| `rag-topk` | 143 | 97.8% | 56/62 (90%) | [80-95%] | 100.0% |
| `tooltrim` **★** | 229 | 96.5% | 56/62 (90%) | [80-95%] | 100.0% |
| `full` | 6,635 | 0.0% | 56/62 (90%) | [80-95%] | 100.0% |
| `truncate-tail` | 400 | 94.0% | 5/62 (8%) | [3-18%] | 8.9% |
| `truncate-head` | 400 | 94.0% | 4/62 (6%) | [3-15%] | 7.1% |

### Budget 800 tokens

| method | comp tokens | tokens saved | accuracy | 95% CI | retention |
|:--|---:|---:|---:|:--:|---:|
| `rag-topk` | 260 | 96.1% | 56/62 (90%) | [80-95%] | 100.0% |
| `tooltrim` **★** | 380 | 94.3% | 56/62 (90%) | [80-95%] | 100.0% |
| `full` | 6,635 | 0.0% | 56/62 (90%) | [80-95%] | 100.0% |
| `truncate-tail` | 800 | 87.9% | 8/62 (13%) | [7-23%] | 14.3% |
| `truncate-head` | 800 | 87.9% | 7/62 (11%) | [6-22%] | 12.5% |

### Significance — `tooltrim` vs each baseline (McNemar, paired)

| budget | baseline | Δ accuracy | wins/losses | p-value | significant (p<0.05) |
|---:|:--|---:|:--:|---:|:--:|
| 128 | `full` | +0.0% | 0/0 | 1.000 | no |
| 128 | `truncate-head` | +88.7% | 55/0 | 0.000 | yes |
| 128 | `truncate-tail` | +88.7% | 55/0 | 0.000 | yes |
| 128 | `rag-topk` | +0.0% | 2/2 | 0.617 | no |
| 256 | `full` | +0.0% | 0/0 | 1.000 | no |
| 256 | `truncate-head` | +87.1% | 54/0 | 0.000 | yes |
| 256 | `truncate-tail` | +87.1% | 54/0 | 0.000 | yes |
| 256 | `rag-topk` | +0.0% | 0/0 | 1.000 | no |
| 400 | `full` | +0.0% | 0/0 | 1.000 | no |
| 400 | `truncate-head` | +83.9% | 52/0 | 0.000 | yes |
| 400 | `truncate-tail` | +82.3% | 51/0 | 0.000 | yes |
| 400 | `rag-topk` | +0.0% | 0/0 | 1.000 | no |
| 800 | `full` | +0.0% | 0/0 | 1.000 | no |
| 800 | `truncate-head` | +79.0% | 49/0 | 0.000 | yes |
| 800 | `truncate-tail` | +77.4% | 48/0 | 0.000 | yes |
| 800 | `rag-topk` | +0.0% | 0/0 | 1.000 | no |
