# tooltrim faithfulness run — `llama3.1:8b`

- **Run (UTC):** 2026-06-28T15:30:48.155450+00:00
- **tooltrim:** v0.1.0  |  **commit:** `17f4a8695918a5211e4b24aad67008e6ea47b41c`
- **Python:** 3.13.5  |  **Platform:** Windows-11-10.0.26200-SP0
- **Token counts:** exact (tiktoken cl100k_base)
- **Dataset:** 62 curated cases (text:18, html:10, json:14, logs:10, tabular:10)
- **Note:** Local Ollama 8B; 62-case dataset (single+multi+distractor); Wilson 95% CIs.

Reproduce: `python run_faithfulness.py --model ollama --model-id llama3.1:8b --budgets 128,256,400`

### Faithfulness under compression — `llama3.1:8b`

Full-context accuracy: **14/62 (22.6%)** [14-34%], avg 6,635 tokens/case. CIs are 95% Wilson.

| budget | comp tokens | tokens saved | accuracy | 95% CI | retention |
|---:|---:|---:|---:|:--:|---:|
| 128 | 94 | 98.6% | 45/62 (73%) | [60-82%] | 321.4% |
| 256 | 177 | 97.3% | 41/62 (66%) | [54-77%] | 292.9% |
| 400 | 229 | 96.5% | 41/62 (66%) | [54-77%] | 292.9% |


### By category (full vs best budget)

| category | n | full acc | @128 | @256 | @400 |
|---|---:|---:|---:|---:|---:|
| distractor | 6 | 17% | 100% | 83% | 83% |
| multi | 6 | 0% | 67% | 67% | 83% |
| single | 50 | 26% | 70% | 64% | 62% |

*Distractor cases require reasoning (pick the current value, not the deprecated one); a pure retriever scores ~0 on them by design, so they separate strong models from weak ones.*
