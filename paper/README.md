# Paper: Faithful, Query-Aware Compression of LLM Agent Tool Outputs

arXiv-ready source for the tooltrim paper. Self-contained: compiles with a stock
`pdflatex` + `bibtex` (no exotic style files).

## Build

```bash
cd paper
pdflatex tooltrim
bibtex tooltrim
pdflatex tooltrim
pdflatex tooltrim
```

Output: `tooltrim.pdf`.

## Filling the frontier Pareto (Table 2)

Table 2 is populated from a real run:

```bash
export ANTHROPIC_API_KEY=...  OPENAI_API_KEY=...  GROQ_API_KEY=...
python ../run_frontier.py --limit 40 --budgets 128,256,800
# then paste the rows of ../benchmarks/FRONTIER.md into the tabular in tooltrim.tex
```

## Status

Draft. All results cite committed artifacts under `../benchmarks/`. Table 2 (the
frontier Pareto) is filled from a full $n{=}62$ run on Claude Haiku 4.5 and Claude
Sonnet 5 (`../benchmarks/FRONTIER.md`, `../benchmarks/runs/comparison_claude-*.md`):
compression significantly raises Sonnet 5 accuracy from 69\% to 90\% ($p{=}0.002$),
and \tooltrim{} ties RAG top-$k$ under the LLM judge ($p\geq0.6$). Extending the
matrix to OpenAI/Groq rows is a re-run of `run_frontier.py` with those keys set.
