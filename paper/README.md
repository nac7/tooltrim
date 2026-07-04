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

Draft. The offline-judge, small-model, and rate-limit-admission results are final
and cite committed artifacts under `../benchmarks/`. The frontier Pareto table is
wired to `run_frontier.py` and is filled once a budgeted matrix run completes.
