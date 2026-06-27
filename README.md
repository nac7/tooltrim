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
  LlamaIndex, raw function-calling — anything. It compresses *strings*, not APIs.
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
correctly. We measure that directly: for **50 curated `(tool output, question,
gold answer)` cases** across all five content types, a model is asked the
question twice — once on the **full** output, once on the **tooltrim-compressed**
output — and accuracy is reported with **Wilson 95% confidence intervals**.
Reproduce with [`run_faithfulness.py`](run_faithfulness.py) — it runs **offline
by default (no API key)** and has adapters for Claude / OpenAI / Groq / Ollama.

On a small local model (`llama3.1:8b`), compression doesn't just preserve
accuracy — it **improves** it, because the model is no longer distracted by
thousands of tokens of noise:

| condition | tokens/case | accuracy | 95% CI |
|---|---:|---:|---:|
| full context | 6,587 | 10/50 (20%) | [11–33%] |
| **compressed @128** | **76 (−98.8%)** | **37/50 (74%)** | **[60–84%]** |
| compressed @256 | 159 (−97.6%) | 34/50 (68%) | [54–79%] |
| compressed @400 | 217 (−96.7%) | 35/50 (70%) | [56–81%] |

The intervals don't overlap — at n=50 this is a **significant** improvement, not
noise. Full provenance and per-case answers are saved as a citable artifact under
[`benchmarks/runs/`](benchmarks/runs/).

*Stated plainly:* this is one small 8B model. A frontier long-context model
handles the full context far better, so its baseline is higher and the accuracy
*uplift* shrinks — but the token/cost savings remain. The uplift is largest for
smaller/cheaper models and longer contexts; n=50 is a pilot, which is why the CIs
are reported.

## Install

```bash
pip install tooltrim          # zero-dependency core (heuristic token counts)
pip install tooltrim[tokens]  # add tiktoken for exact token counts
```

## Usage

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
default.

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

### 5. Or run it as a proxy — zero code changes

Point any OpenAI-compatible client at the tooltrim proxy; every `role:"tool"` /
`role:"function"` message is compressed (using the latest user message as the
relevance query) before being forwarded upstream. Works for any language or
framework — you only change `base_url`.

```bash
python run_proxy.py --upstream https://api.openai.com/v1   # any OpenAI-compatible endpoint
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8800/v1", api_key="<upstream key>")
# use the API normally — bloated tool results are trimmed in flight
```

The proxy is stdlib-only and **fails open**: if anything goes wrong it forwards
the original request untouched, so it never breaks a production call.

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
   - **Text** — query-aware extractive selection (BM25), with `[…]` elisions.
4. **Stash** the full output under a content-addressed `ref` for `expand()`.

With a query, every compressor keeps the most *relevant* parts; without one, it
falls back to structure-preserving head/tail selection.

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

v0.1 — deterministic zero-dependency core, 47-test suite, reproducible token +
**faithfulness** benchmarks (with Wilson CIs), an OpenAI-compatible **proxy**, and
citable run artifacts under [`benchmarks/`](benchmarks/).

Roadmap: frontier-model faithfulness runs, embedding-based relevance, streaming
compression, a pluggable Redis/S3 expand-store for horizontal scale, and native
LangChain / LlamaIndex / OpenAI-Agents wrappers.

Contributions and benchmark cases welcome. MIT licensed.
