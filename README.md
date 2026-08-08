# tooltrim

**Drop-in compression for LLM agent tool outputs.** Shrink bloated tool results
— fetched web pages, paginated JSON, log dumps, CSV exports, long documents —
*before* they re-enter your agent's context window. Keep the facts the model
needs, drop the boilerplate, and keep the full output one `expand()` away.

```python
from tooltrim import compressed_tool

@compressed_tool(max_tokens=400)
def web_fetch(url: str) -> str:
    ...                      # returns a 3,000-token HTML page
# your agent now receives a compact, on-topic extract instead
```

- **Zero dependencies** in the core. Pure-stdlib, deterministic, reproducible.
- **Provider-agnostic.** Works with OpenAI, Anthropic, local models, LangChain,
  LlamaIndex, OpenAI-Agents, MCP, raw function-calling — anything. It compresses
  *strings*, not APIs.
- **Lossless by reference.** Compression is extractive, and the full output stays
  retrievable via a short `ref` — so it's compression *plus retrieval*, not
  blind truncation.
- **Content-aware.** Separate compressors for HTML, JSON, tabular data, logs,
  and free text. Optionally **query-aware** (BM25) to keep what the agent is
  actually looking for.
- **Faithfulness-tested.** A built-in harness measures whether the model still
  answers correctly on compressed output (with Wilson 95% CIs) — not just how
  many tokens you saved.
- **Deploy as a proxy.** An OpenAI-compatible compression proxy trims
  `role:"tool"` messages in flight, so any app/language adopts it with zero code
  changes — just a `base_url`.

---

## Why

In a real agent loop, the prompt isn't what blows up your context — **tool
outputs are.** A single `web_fetch` returns thousands of tokens of nav bars and
footers; a REST call returns a 300-item paginated array; a log tool dumps
10,000 lines of `INFO heartbeat`. And because the agent's transcript is replayed
on **every** turn, you pay for that bloat again and again — slower responses,
higher bills, and a model that loses the thread.

Routers, caches, and prompt compressors don't touch this. `tooltrim` targets the
tool output directly, at the exact point it enters context.

## Benchmark

Realistic tool outputs compressed to a **400-token** budget, exact `tiktoken`
(`cl100k_base`) counts. Each output contains one planted fact ("needle") that the
agent needs; `tooltrim` is given the task as its relevance query.
Reproduce with [`benchmark.py`](benchmark.py).

| Tool output           |  before |  after |  saved | needle kept |
|-----------------------|--------:|-------:|-------:|:-----------:|
| Web page (HTML)       |   2,816 |     13 |  99.5% |     yes     |
| REST response (JSON)  |  15,119 |    325 |  97.9% |     yes     |
| Server logs           |   7,606 |    390 |  94.9% |     yes     |
| CSV export            |   7,895 |    373 |  95.3% |     yes     |
| Long document (text)  |   6,139 |     10 |  99.8% |     yes     |
| **Total**             | **39,575** | **1,111** | **97.2%** | **5/5** |

**39,575 → 1,111 tokens — a 35.6× smaller context, with the relevant fact kept
in every case.** (HTML/text collapse to the matching passage when the query
pinpoints it; structured types keep a representative, schema-preserving sample.)

## Does compression lose information? (it can *help*)

Throwing away 99% of the tokens is only safe if the model still answers
correctly. We measure that directly: for **62 curated `(tool output, question,
gold answer)` cases** across all five content types — including **multi-fact**
cases (the answer needs several facts from different parts of the output) and
**distractor** cases (a deprecated value sits next to the current one) — a model
is asked the question twice: once on the **full** output, once on the
**tooltrim-compressed** output. Accuracy is reported with **Wilson 95%
confidence intervals**. Reproduce with [`run_faithfulness.py`](run_faithfulness.py)
— it runs **offline by default (no API key)** and has adapters for
Claude / OpenAI / Groq / Ollama.

On small local models, compression doesn't just preserve accuracy — it
**improves** it, because the model is no longer distracted by thousands of tokens
of noise. The effect reproduces across two independent model families:

| model | full | @128 (−98.6%) | @256 (−97.3%) | @400 (−96.5%) |
|---|---:|---:|---:|---:|
| `mistral:7b`  | 13% [7–23%]  | **84% [73–91%]** | 81% [69–89%] | 82% [71–90%] |
| `llama3.1:8b` | 23% [14–34%] | **73% [60–82%]** | 66% [54–77%] | 66% [54–77%] |

The compressed intervals don't overlap the full-context intervals — at n=62 this
is a **significant** improvement for both models, not noise. Full provenance,
per-case answers, and the cross-model table are saved as citable artifacts under
[`benchmarks/runs/`](benchmarks/runs/) and [`benchmarks/COMPARISON.md`](benchmarks/COMPARISON.md).

*Stated plainly:* these are small 7–8B models. A frontier long-context model
handles the full context far better, so its baseline is higher and the accuracy
*uplift* shrinks — but the token/cost savings remain. The uplift is largest for
smaller/cheaper models and longer contexts. The harness is wired so a frontier
run (`--model claude`) drops a new row into the same table when an API key is
available; n=62 is a pilot, which is why the CIs are reported.

### How does it compare to truncation and RAG?

Preserving accuracy vs *full context* only matters if it beats the obvious
alternatives. [`run_baselines.py`](run_baselines.py) scores tooltrim against
**naive truncation**, **query-aware RAG top-k**, **RAG-embed**, and
**LLMLingua-2** on the same cases and budgets, with a paired **McNemar**
significance test. Retention (accuracy ÷ full-context accuracy), offline judge:

| budget | truncate-head | truncate-tail | rag-topk | **tooltrim** |
|---:|---:|---:|---:|---:|
| 128 | 1.8% | 1.8% | 100% | **100%** |
| 256 | 3.6% | 3.6% | 100% | **100%** |
| 800 | 12.5% | 14.3% | 100% | **100%** |

Query-aware compression retains **100%** of accuracy while cutting 94–99% of
tokens; blind truncation drops the needed fact and collapses (p < 0.001 at every
budget). The offline judge is itself lexical, so RAG top-k ties tooltrim here —
tooltrim's content-type structure advantage surfaces with a real-LLM judge on
structured output. Details, caveats, and the full grid:
[`benchmarks/BASELINES.md`](benchmarks/BASELINES.md).

### End-to-end: does it preserve *task success*? (tau-bench, multi-step)

Single-turn faithfulness isn't the whole story — in a real agent loop a
compressor can drop a field the agent only needs three turns later.
[`run_taubench.py`](run_taubench.py) measures that directly: it wraps
[tau-bench](https://github.com/sierra-research/tau-bench)'s own environment so
**every tool observation is compressed before it re-enters the agent's context**,
while tau-bench's reward function, LLM user simulator, and agent stay untouched.
A compressor that shreds an output the agent needs later shows up as *lower task
success* — the outcome metric, not a proxy.

The harness reports task success with **Wilson 95% CIs** over all (task, trial)
observations and a paired **McNemar** test vs tooltrim, sweeps multiple token
budgets to trace the accuracy-vs-budget curve, and emits a **reproducibility
manifest** (pinned tau-bench commit, resolved model snapshot, seeds) alongside
raw per-task JSON so every number is re-derivable offline without re-spending on
the API. It runs against any litellm-supported model for both the agent and the
user simulator.

**Status: harness implemented and under pilot** on tau-bench `retail` with
gpt-4o-mini; results land in [`benchmarks/TAUBENCH.md`](benchmarks/TAUBENCH.md).
One design note surfaced by the pilot: retail's native tool outputs are modest
(~130–650 tokens), so runs use a tight budget (≈128 tokens) where compression
actually engages rather than passing through.

## Install

```bash
pip install tooltrim          # zero-dependency core (heuristic token counts)
pip install tooltrim[tokens]  # add tiktoken for exact token counts
```

Extras: `tooltrim[langchain]`, `tooltrim[redis]`, `tooltrim[s3]`.

## CLI

```bash
tooltrim demo                                   # 10-second self-contained savings tour
cat big.json | tooltrim compress -q "refund status" --stats   # pipe in, compressed out
tooltrim compress page.html -q "rate limits" -m 400
tooltrim proxy --upstream https://api.openai.com/v1           # run the proxy
```

## Usage (library)

### 1. Decorate a tool

```python
from tooltrim import compressed_tool

@compressed_tool(max_tokens=400)
def read_file(path: str) -> str:
    return open(path).read()
```

### 2. Make it query-aware

Pull the relevance query from the call arguments…

```python
@compressed_tool(max_tokens=400, query_from=lambda query, **_: query)
def web_search(query: str) -> str:
    ...
```

…or set the agent's current goal ambiently, so every tool call this turn keeps
what's relevant to it:

```python
from tooltrim import query_scope

with query_scope("find the customer's refund status"):
    result = run_agent_step()   # all @compressed_tool calls inside use this query
```

### 3. Imperative API + expand-on-demand

```python
from tooltrim import ToolCompressor

tc = ToolCompressor(max_tokens=400)
res = tc.compress(huge_json_response, query="refund status for customer C-1007")

res.text             # compact text to feed back to the model
res.saved_tokens     # e.g. 14794
res.saved_ratio      # e.g. 0.979
res.ref              # e.g. "a1b2c3d4"

full = tc.expand(res.ref)                    # get the original back
slice_ = tc.expand(res.ref, start=0, length=2000)
```

By default the compressed output ends with a small footer the model can act on:

```
…compressed extract…

[tooltrim: compressed 15119->325 tokens (saved 14794); full output ref=a1b2c3d4]
```

Expose an `expand(ref)` tool to your agent and it can pull the full output back
whenever the extract isn't enough — turning aggressive compression into a safe
default. tooltrim hands you both the tool schema and the handler:

```python
tools = my_tools + [tc.expand_tool_spec(style="openai")]   # or style="anthropic"

# when the model calls expand_tool_output(ref=..., start=..., length=...):
result_text = tc.handle_expand(ref, start=start, length=length)   # paged, safe
```

See [`examples/04_expand_tool.py`](examples/04_expand_tool.py) for a full wiring.
Extractive compressors also keep **neighbor context** (a line/sentence around each
match) so the model gets context, not just the bare matching line.

### 4. Optional: LLM distillation (any provider)

The deterministic compressors need no LLM. When you want summarization instead
of extraction, plug in *any* model with a one-line completion function — use a
small/cheap one; distilling 15k → 300 tokens once saves your expensive model
from re-reading the blob every turn.

```python
from tooltrim import LLMDistiller

def complete(prompt: str) -> str:
    # wrap OpenAI / Anthropic / local — your choice
    return my_client.responses(prompt)

distiller = LLMDistiller(complete, max_tokens=300)
summary = distiller.compress(huge_output, query="refund status")
```

### 5. Drop into LangChain — one line per tool

Already have LangChain tools? Wrap any of them and you get back a tool with the
**same name, description, and argument schema**, so the agent calls it unchanged —
but its (string) output is compressed before it lands in the scratchpad. The
relevance query comes from the tool's own arguments.

```bash
pip install tooltrim[langchain]
```

```python
from tooltrim.integrations import compress_langchain_tool, compress_langchain_tools

fetch = compress_langchain_tool(my_tool, max_tokens=400,
                                query_from=lambda query, **_: query)

# or wrap the whole toolset at once (sharing one compressor + expand store):
tools = compress_langchain_tools(my_tools, max_tokens=400)
```

See [`examples/03_langchain_tool.py`](examples/03_langchain_tool.py).

### 6. Or LlamaIndex — same one-liner

```bash
pip install tooltrim[llamaindex]
```

```python
from tooltrim.integrations import compress_llamaindex_tool, compress_llamaindex_tools

fetch = compress_llamaindex_tool(my_tool, max_tokens=400,
                                 query_from=lambda topic: topic)
tools = compress_llamaindex_tools(my_tools, max_tokens=400)
```

A LlamaIndex tool returns a `ToolOutput`; only its `content` (what the LLM reads)
is compressed — the structured `raw_output` is preserved. See
[`examples/05_llamaindex_tool.py`](examples/05_llamaindex_tool.py).

### 7. Or the OpenAI Agents SDK — same one-liner

```bash
pip install tooltrim[openai-agents]
```

```python
from tooltrim.integrations import compress_openai_agents_tool, compress_openai_agents_tools

fetch = compress_openai_agents_tool(my_tool, max_tokens=400,
                                    query_from=lambda url: url)
tools = compress_openai_agents_tools(my_tools, max_tokens=400)
```

Only the tool's `on_invoke_tool` is wrapped — name, JSON schema, and guardrails
are preserved. See [`examples/06_openai_agents_tool.py`](examples/06_openai_agents_tool.py).

### 8. Or at the MCP boundary — a compressing gateway

[MCP](https://modelcontextprotocol.io) tool results (`tools/call`) are exactly
the bloated outputs tooltrim targets. Run a gateway in front of any MCP server and
point your MCP client (Claude Desktop, an IDE, an agent) at it — every result is
compressed in flight, no code change:

```bash
pip install tooltrim[mcp]
tooltrim mcp -- npx -y @modelcontextprotocol/server-filesystem /path
```

Or wrap the result-handling in your own server:

```python
from tooltrim.integrations import compressing_call_tool, compress_tool_result

# wrap an upstream call_tool coroutine...
call = compressing_call_tool(session.call_tool, max_tokens=400)
# ...or compress a single CallToolResult (errors / non-text pass through)
result = compress_tool_result(result, compressor=tc, query=query)
```

See [`examples/08_mcp_gateway.py`](examples/08_mcp_gateway.py).

### 9. Or run it as a proxy — zero code changes

Point your client at the tooltrim proxy; every tool result is compressed (using
the latest user message as the relevance query) before being forwarded upstream.
Both wire formats are understood, routed by request path — you only change
`base_url`.

```bash
python run_proxy.py --upstream https://api.openai.com/v1     # OpenAI-compatible
python run_proxy.py --upstream https://api.anthropic.com/v1  # Claude
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8800/v1", api_key="<upstream key>")

from anthropic import Anthropic
client = Anthropic(base_url="http://127.0.0.1:8800")
```

`/v1/chat/completions` compresses OpenAI `role:"tool"` messages; `/v1/messages`
compresses Anthropic `tool_result` blocks. The proxy is stdlib-only and **fails
open**: if anything goes wrong it forwards the original request untouched, so it
never breaks a production call.

**Online, it also keeps you under provider rate limits.** Against a live hosted
model (Groq free tier, 6,000-tokens-per-request cap), **45% of raw tool outputs
are rejected (HTTP 413) but 100% of tooltrim-compressed calls fit** — a 14,415-token
result is compressed to 26 tokens in flight and the call succeeds. See
[`benchmarks/ONLINE_GROQ.md`](benchmarks/ONLINE_GROQ.md).

### 10. Scale out — shared expand-store + metrics

The default expand-store is in-process, fine for one worker. To run multiple
workers/replicas behind a load balancer, the store must be **shared** — otherwise
a `ref` minted by one worker can't be expanded by another. Swap in a backend
(all are content-addressed, so writes dedup automatically):

```python
from tooltrim import ToolCompressor, FileStore, RedisStore, S3Store

tc = ToolCompressor(store=FileStore("/mnt/shared/tooltrim"))         # zero-dep, shared volume
tc = ToolCompressor(store=RedisStore(url="redis://cache:6379/0",     # pip install tooltrim[redis]
                                     ttl_seconds=86_400))
tc = ToolCompressor(store=S3Store(bucket="my-bucket"))               # pip install tooltrim[s3]
```

The proxy exposes **Prometheus metrics** at `GET /metrics` (tokens in/out/saved,
messages compressed, fail-open count, upstream errors, latency) — scrape it to
quantify savings fleet-wide:

```
tooltrim_tokens_saved_total 14389
tooltrim_messages_compressed_total 1
tooltrim_fail_open_total 0
```

## How it works

1. **Pass-through** if the output already fits the budget (zero overhead).
2. **Detect** the content type (JSON / HTML / tabular / logs / text).
3. **Compress** with a type-specific strategy:
   - **JSON** — preserve structure; sample arrays (keeping the key schema), note
     `(+N more items)`, truncate long strings; tighten until it fits.
   - **HTML** — extract readable text (drop `script`/`style`/`nav`/`footer`),
     then fit the budget.
   - **Tabular** — keep the header + a sample of rows + `(+N more rows)`.
   - **Logs** — collapse repeated lines (`x42`), always keep errors/warnings,
     fill with head/tail context.
   - **Text** — query-aware extractive selection (BM25 or embeddings), `[…]` elisions.
4. **Stash** the full output under a content-addressed `ref` for `expand()`.

With a query, every compressor keeps the most *relevant* parts; without one, it
falls back to structure-preserving head/tail selection.

### Semantic relevance (optional)

Scoring defaults to lexical **BM25** (zero-dependency). For semantic matching —
so a query for `"car"` keeps a chunk about `"automobiles"` — pass an
`EmbeddingScorer`. It's provider-agnostic: give it any `embed(texts) -> vectors`
callable (OpenAI, Cohere, local), or let it load `sentence-transformers`
(`pip install tooltrim[embeddings]`). The scorer threads through **every** content
type:

```python
from tooltrim import ToolCompressor, EmbeddingScorer

tc = ToolCompressor(max_tokens=400,
                    scorer=EmbeddingScorer(embed=my_client.embed))
```

### Streaming (bounded memory)

Some outputs are too big to hold in memory — a multi-GB log, a subprocess's
stdout, an HTTP stream. `compress_stream` consumes an iterable incrementally with
**constant memory** (bounded head/tail/top-K/important-line buffers), then fits
the survivors to the budget:

```python
from tooltrim import compress_stream

text = compress_stream(open("huge.log"), max_tokens=400, query="disk error")
```

## How it's different

| Tool class           | What it optimizes            | tooltrim |
|----------------------|------------------------------|----------|
| Routers (RouteLLM…)  | *which model* gets the call  | orthogonal |
| Semantic caches      | repeated *identical* calls   | orthogonal |
| Prompt compressors (LLMLingua) | the *prompt/instructions* | different target |
| Memory frameworks (MemGPT…) | conversation history, as a framework you adopt | tooltrim is a drop-in on the *tool boundary* |

tooltrim targets the **tool-output boundary** — the largest and most-ignored
token sink in agentic apps — and works alongside all of the above.

## Status

v0.2 — deterministic zero-dependency core, 104-test suite, reproducible token +
**faithfulness** benchmarks (with Wilson CIs, cross-model), a **proxy** speaking
both **OpenAI and Anthropic** wire formats with Prometheus **/metrics**,
**LangChain**, **LlamaIndex**, and **OpenAI-Agents** adapters, an **MCP**
compressing gateway, pluggable **File/Redis/S3 expand-stores** for horizontal
scale, optional **embedding-based relevance**, **streaming** compression for
outputs too big to hold in memory, a `tooltrim` **CLI**, a **multi-step
task-success harness** that compresses tool outputs inside tau-bench's own agent
loop (in-loop, McNemar vs baselines, reproducibility manifest), and citable run
artifacts under [`benchmarks/`](benchmarks/). Published on
[PyPI](https://pypi.org/project/tooltrim/).

Roadmap: frontier-model faithfulness runs, the scaled tau-bench task-success
sweep (multi-budget, multi-trial) and its benchmark release, and native
streaming passthrough in the proxy.

Contributions and benchmark cases welcome. MIT licensed.
