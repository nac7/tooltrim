# tooltrim faithfulness run — `llama3.1:8b`

- **Run (UTC):** 2026-06-27T23:35:56.935462+00:00
- **tooltrim:** v0.1.0  |  **commit:** `a190bbe2c39240f763e18c1b476b36481ce389f9`
- **Python:** 3.13.5  |  **Platform:** Windows-11-10.0.26200-SP0
- **Token counts:** exact (tiktoken cl100k_base)
- **Dataset:** 50 curated cases (text:10, html:10, json:10, logs:10, tabular:10)
- **Note:** Local Ollama, 8B; 50-case dataset; Wilson 95% CIs. Compression vs full-context on a small model.

Reproduce: `python run_faithfulness.py --model ollama --model-id llama3.1:8b --budgets 128,256,400`

### Faithfulness under compression — `llama3.1:8b`

Full-context accuracy: **10/50 (20.0%)** [11-33%], avg 6,587 tokens/case. CIs are 95% Wilson.

| budget | comp tokens | tokens saved | accuracy | 95% CI | retention |
|---:|---:|---:|---:|:--:|---:|
| 128 | 76 | 98.8% | 37/50 (74%) | [60-84%] | 370.0% |
| 256 | 159 | 97.6% | 34/50 (68%) | [54-79%] | 340.0% |
| 400 | 217 | 96.7% | 35/50 (70%) | [56-81%] | 350.0% |
