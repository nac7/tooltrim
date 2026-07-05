# LlamaIndex integration PR (draft)

**Title:** Add `tooltrim` tool-output compression integration

**Target:** `run-llama/llama_index` — a community integration package under
`llama-index-integrations/` (LlamaIndex ships integrations as small installable
packages, e.g. `llama-index-postprocessor-*` / tool wrappers).

## What it adds

A thin wrapper that compresses the output of any LlamaIndex `FunctionTool` /
`BaseTool` before it is returned to the agent, using the agent's current query as
the relevance signal. Built directly on the existing adapter
`tooltrim/integrations/llamaindex.py` (`compress_llamaindex_tool`,
`compress_llamaindex_tools`).

## Usage

```python
from llama_index.core.tools import FunctionTool
from tooltrim.integrations.llamaindex import compress_llamaindex_tool

raw = FunctionTool.from_defaults(fn=fetch_web_page)
tool = compress_llamaindex_tool(raw, budget=256)  # drop-in; same interface
```

## PR description body

> **Problem.** Tool outputs (web pages, JSON API responses, logs) are frequently
> the largest token sink in an agent loop and routinely blow the context budget.
> **This PR** adds a drop-in wrapper that compresses a tool's output with
> query-aware, content-type-aware extraction before it re-enters context, keeping
> the full output retrievable on demand. On the Tool-Output Faithfulness Benchmark
> it retains 100% of full-context answer accuracy while cutting 94–99% of tokens,
> and reaches 100% vs 20% end-to-end agent-task success against RAG top-k at a
> 256-token budget (p=0.013). MIT-licensed, zero required deps.

## Checklist before opening

- [ ] Package skeleton under `llama-index-integrations/tools/llama-index-tools-tooltrim/`
- [ ] `pyproject.toml` with `tooltrim` as dependency
- [ ] Unit test mirroring `tests/test_integrations_llamaindex.py`
- [ ] Example notebook / README section
- [ ] Sign the LlamaIndex CLA
