# LangChain integration PR (draft)

**Title:** Add `tooltrim` query-aware tool-output compression wrapper

**Target:** `langchain-ai/langchain` (`langchain-community`) or a standalone
`langchain-tooltrim` package following LangChain's current partner-package
guidance. Check the contributing guide for whether new integrations go to
`langchain-community` or a standalone package before opening.

## What it adds

A wrapper that compresses the output of any LangChain `BaseTool` before it is
returned to the agent. Built on the existing adapter
`tooltrim/integrations/langchain.py` (`compress_langchain_tool`,
`compress_langchain_tools`).

## Usage

```python
from tooltrim.integrations.langchain import compress_langchain_tool

tool = compress_langchain_tool(raw_tool, budget=256)  # same BaseTool interface
```

## PR description body

> Tool outputs are the dominant, least-managed token sink in agent loops. This
> integration wraps any `BaseTool` so its output is compressed with query-aware,
> content-type-aware extraction before re-entering context, with expand-on-demand
> retrieval of the full result. Benchmarked on TOFB: 100% answer-accuracy
> retention while cutting 94–99% of tokens; 100% vs 20% end-to-end agent-task
> success vs RAG top-k at 256 tokens (p=0.013). MIT.

## Checklist before opening

- [ ] Confirm target (langchain-community vs standalone) per current contrib guide
- [ ] Test mirroring `tests/test_integrations.py`
- [ ] `docs/` integration page
- [ ] Sign the LangChain CLA
