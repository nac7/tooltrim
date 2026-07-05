# Upstream adoption

Getting `tooltrim` into widely-used projects is the strongest independent-use
signal. The adapters already exist in [`tooltrim/integrations/`](../tooltrim/integrations);
this directory holds **submission-ready** PR drafts and registry entries so each
upstream contribution is copy-paste ready.

> These are drafts. Actually opening the PRs is a manual step (each target has its
> own CLA / contribution flow) — do not automate it.

## Targets, ordered by tractability × payoff

| target | vehicle | why it's a fit | draft |
|:--|:--|:--|:--|
| **MCP server registry** | listing entry | MCP-boundary compression is novel and standards-adjacent; a registry listing is a small, high-visibility PR | [mcp-registry-entry.md](mcp-registry-entry.md) |
| **LlamaIndex** | `postprocessor` integration package | tool-output compression maps cleanly onto a node postprocessor; LlamaIndex takes community integration packages | [llamaindex-pr.md](llamaindex-pr.md) |
| **LangChain** | community tool wrapper | `compress_langchain_tool` already wraps a `BaseTool`; fits `langchain-community` | [langchain-pr.md](langchain-pr.md) |

## Adapter inventory (what each PR builds on)

- `integrations/mcp.py` — `compress_tool_result`, `compressing_call_tool`
- `integrations/llamaindex.py` — `compress_llamaindex_tool`, `compress_llamaindex_tools`
- `integrations/langchain.py` — `compress_langchain_tool`, `compress_langchain_tools`
- `integrations/openai_agents.py` — OpenAI Agents SDK wrapper

## Tracking (becomes NIW exhibits)

Record over time — these are the citable adoption evidence:

- PyPI downloads (`pypistats recent tooltrim`)
- GitHub stars / forks / dependents
- Merged-upstream links (the PRs above, once accepted)
- Any third-party repo that imports `tooltrim`
