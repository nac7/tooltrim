# tooltrim faithfulness run — `mistral:7b`

- **Run (UTC):** 2026-06-28T15:36:37.168596+00:00
- **tooltrim:** v0.1.0  |  **commit:** `17f4a8695918a5211e4b24aad67008e6ea47b41c`
- **Python:** 3.13.5  |  **Platform:** Windows-11-10.0.26200-SP0
- **Token counts:** exact (tiktoken cl100k_base)
- **Dataset:** 62 curated cases (text:18, html:10, json:14, logs:10, tabular:10)
- **Note:** Local Ollama 7B; 62-case dataset (single+multi+distractor); Wilson 95% CIs.

Reproduce: `python run_faithfulness.py --model ollama --model-id mistral:7b --budgets 128,256,400`

### Faithfulness under compression — `mistral:7b`

Full-context accuracy: **8/62 (12.9%)** [7-23%], avg 6,635 tokens/case. CIs are 95% Wilson.

| budget | comp tokens | tokens saved | accuracy | 95% CI | retention |
|---:|---:|---:|---:|:--:|---:|
| 128 | 94 | 98.6% | 52/62 (84%) | [73-91%] | 650.0% |
| 256 | 177 | 97.3% | 50/62 (81%) | [69-89%] | 625.0% |
| 400 | 229 | 96.5% | 51/62 (82%) | [71-90%] | 637.5% |


### By category (full vs best budget)

| category | n | full acc | @128 | @256 | @400 |
|---|---:|---:|---:|---:|---:|
| distractor | 6 | 0% | 83% | 67% | 67% |
| multi | 6 | 0% | 100% | 100% | 100% |
| single | 50 | 16% | 82% | 80% | 82% |

*Distractor cases require reasoning (pick the current value, not the deprecated one); a pure retriever scores ~0 on them by design, so they separate strong models from weak ones.*
