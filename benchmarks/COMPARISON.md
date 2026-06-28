# tooltrim faithfulness — model comparison

Dataset: 62 cases. Token savings (same for all models): @128: 98.6%, @256: 97.3%, @400: 96.5%. Accuracy shown as point estimate with 95% Wilson CI.

| model | full | @128 | @256 | @400 |
|---|---:|---:|---:|---:|
| `mistral:7b` | 13% [7-23%] | 84% [73-91%] | 81% [69-89%] | 82% [71-90%] |
| `llama3.1:8b` | 23% [14-34%] | 73% [60-82%] | 66% [54-77%] | 66% [54-77%] |

### Full-context accuracy by category

| model | distractor | multi | single |
|---|---:|---:|---:|
| `mistral:7b` | 0% | 0% | 16% |
| `llama3.1:8b` | 17% | 0% | 26% |

*Distractor cases need reasoning (current vs deprecated value); they separate strong models from retrieval.*
