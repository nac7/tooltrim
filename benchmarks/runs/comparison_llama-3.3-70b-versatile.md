## Comparative faithfulness — `llama-3.3-70b-versatile`

Full-context accuracy: **8/8 (100.0%)** [68-100%], avg 4,068 tokens/case. CIs are 95% Wilson; each method is scored on the *same* cases so accuracies are directly comparable.

### Budget 256 tokens

| method | comp tokens | tokens saved | accuracy | 95% CI | retention |
|:--|---:|---:|---:|:--:|---:|
| `rag-topk` | 70 | 98.3% | 8/8 (100%) | [68-100%] | 100.0% |
| `tooltrim` **★** | 111 | 97.3% | 7/8 (88%) | [53-98%] | 87.5% |
| `truncate-head` | 256 | 93.7% | 0/8 (0%) | [0-32%] | 0.0% |
| `truncate-tail` | 256 | 93.7% | 0/8 (0%) | [0-32%] | 0.0% |

### Budget 800 tokens

| method | comp tokens | tokens saved | accuracy | 95% CI | retention |
|:--|---:|---:|---:|:--:|---:|
| `rag-topk` | 202 | 95.0% | 8/8 (100%) | [68-100%] | 100.0% |
| `tooltrim` **★** | 248 | 93.9% | 7/8 (88%) | [53-98%] | 87.5% |
| `truncate-tail` | 800 | 80.3% | 1/8 (12%) | [2-47%] | 12.5% |
| `truncate-head` | 800 | 80.3% | 0/8 (0%) | [0-32%] | 0.0% |

### Significance — `tooltrim` vs each baseline (McNemar, paired)

| budget | baseline | Δ accuracy | wins/losses | p-value | significant (p<0.05) |
|---:|:--|---:|:--:|---:|:--:|
| 256 | `truncate-head` | +87.5% | 7/0 | 0.023 | yes |
| 256 | `truncate-tail` | +87.5% | 7/0 | 0.023 | yes |
| 256 | `rag-topk` | -12.5% | 0/1 | 1.000 | no |
| 800 | `truncate-head` | +87.5% | 7/0 | 0.023 | yes |
| 800 | `truncate-tail` | +75.0% | 6/0 | 0.041 | yes |
| 800 | `rag-topk` | -12.5% | 0/1 | 1.000 | no |
