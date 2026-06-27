# tooltrim faithfulness run — `llama3.1:8b`

- **Run (UTC):** 2026-06-27T23:23:18.489432+00:00
- **tooltrim:** v0.1.0  |  **commit:** `2986765c947cc7cba3d4aa08201088619523621d`
- **Python:** 3.13.5  |  **Platform:** Windows-11-10.0.26200-SP0
- **Token counts:** exact (tiktoken cl100k_base)
- **Dataset:** 16 curated cases (text:3, html:3, json:3, logs:3, tabular:4)
- **Note:** Local Ollama, 8B; small-n (16) pilot. Compression more than doubles accuracy vs full context on this small model.

Reproduce: `python run_faithfulness.py --model ollama --model-id llama3.1:8b --budgets 128,256,400`

### Faithfulness under compression — `llama3.1:8b`

Full-context accuracy: **6/16 (37.5%)**, avg 6,671 tokens/case.

| budget | comp tokens | tokens saved | accuracy | retention |
|---:|---:|---:|---:|---:|
| 128 | 76 | 98.9% | 14/16 (88%) | 233.3% |
| 256 | 157 | 97.6% | 12/16 (75%) | 200.0% |
| 400 | 214 | 96.8% | 12/16 (75%) | 200.0% |
