# MCP registry / awesome-list submission

`tooltrim` compresses tool results **at the MCP boundary** — a gateway that wraps
an upstream server's `call_tool` and shrinks large results before they re-enter
the model's context, with expand-on-demand over a content-addressed store. This
is a natural registry listing.

## Where to submit

- `modelcontextprotocol/servers` community list (README PR), and/or
- `punkpeye/awesome-mcp-servers` (and similar awesome-lists) — add one line.

## Entry (README table row / list item)

```markdown
- **[tooltrim](https://github.com/nac7/tooltrim)** — Query-aware compression
  gateway for MCP tool results. Wraps an upstream server's `call_tool` and
  compresses large outputs (HTML/JSON/logs/tabular/text) before they re-enter
  context, with expand-on-demand retrieval over a content-addressed store.
  Cuts 94–99% of tokens while keeping the needed fact. MIT.
```

## Minimal proof-of-use snippet (include in the PR if asked)

```python
from tooltrim.integrations.mcp import compressing_call_tool
from tooltrim import ToolCompressor

# Wrap an existing MCP client session so every tool result is compressed
# at the protocol boundary — no changes to the upstream server.
call = compressing_call_tool(session.call_tool, compressor=ToolCompressor())
result = await call("fetch", {"url": "https://example.com/big"})
```

## Talking points for the PR description

- Compresses at the **protocol boundary**, so it works with any MCP server
  without modifying it.
- **Fails open**: on any compression error the original result passes through.
- **Lossless-on-demand**: full output stays retrievable via an `expand` id.
- Benchmarked: see the TOFB leaderboard and paper in the repo.
