# Changelog

All notable changes to tooltrim are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.2] — 2026-08-10

### Added
- **Standalone MCP server (`tooltrim serve`).** In addition to the compressing
  *gateway* (which fronts an upstream server), tooltrim can now run as a complete
  MCP server in its own right — no upstream required — exposing its capability as
  two stdio tools: `compress(text, query=None, max_tokens=None)` and
  `expand_tool_output(ref, start=0, length=None)`. New
  `integrations.mcp.run_stdio_server` / `build_tool_server` (the latter is
  test-friendly). Published to the MCP Registry as `io.github.nac7/tooltrim`, so
  an MCP client can launch it with `uvx --from tooltrim[mcp] tooltrim serve`.
  `mcp` stays an optional extra — the core library remains dependency-free.

## [0.3.1] — 2026-08-09

### Fixed
- **Reported version now matches the release.** `tooltrim.__version__` was still
  `0.2.1` in the 0.3.0 release, so `import tooltrim; tooltrim.__version__`
  disagreed with the installed distribution. The in-package version now tracks
  `pyproject.toml`.

## [0.3.0] — 2026-08-09

Research-grade evaluation release: the compressor gains substantial robustness
under tight budgets, and the repository now ships the full benchmark harness,
the paper source, and a pre-registered live-agent study.

### Added
- **Benchmark & evaluation harness.** Frontier accuracy/token Pareto driver
  (`run_frontier.py`), end-to-end agent-task suite (`run_agent_tasks.py`),
  multi-step tool-chain harness, component ablation (`run_ablation.py`), and a
  `tau-bench` compression adapter + runner (`run_taubench.py`) that compresses
  tool observations inside tau-bench's own loop, leaving its reward function and
  LLM user simulator untouched.
- **Tool-Output Faithfulness Benchmark (TOFB)** export for HuggingFace.
- **Downstream-extractability metric** — whether a compressed output still parses
  in code and the gold fact is recoverable from that parse.
- **Pre-registered multi-trial tau-bench study.** A committed pre-registration
  (`paper/PREREGISTRATION_multitrial.md`) plus the frozen analysis: the
  placebo-controlled difference-in-differences headline (`eval/did_analysis.py`),
  the noise-floor / paired-equivalence estimator (`eval/noise_floor.py`, with
  `--headline` to score on the non-diagnostic tasks), and reproducible strata
  derivation (`eval/freeze_strata.py`, exposing `compute_strata()` as the single
  source of truth shared with the analysis).
- **Paper source** (`paper/tooltrim.tex`) and reproducible-run notes.

### Fixed
- **JSON compressor no longer collapses under a tight budget.** For a deeply
  nested object (e.g. a retail order) whose smallest sampled form still exceeded
  the budget, the compressor cliffed to the text fallback, which splits on commas
  and could drop top-level scalar fields an agent depends on (e.g. `status`),
  emitting invalid JSON — as little as ~21 tokens. Two tighter sampling rungs
  now let a tight budget land on still-valid, structure-preserving JSON that
  retains those scalars. Purely additive: only affects cases that previously hit
  the broken fallback.
- **Id-keyed record maps and flat lookup maps are no longer gutted.** A dict of
  records (e.g. tau-bench retail `variants` keyed by item id) had no sampling
  lever, so a tight budget depth-elided every value wholesale, destroying every
  `item_id`/`available`/`options` field. Dicts now get the same relevance-based
  entry sampling arrays already had (keep *k* whole entries + a truncation
  marker), and large flat `name → id` lookup tables are sampled by relevance
  rather than positionally.
- **Elision markers are always emitted.** The incompleteness marker is now
  reserved before filling the budget, so a compressed catalogue can no longer
  look complete when records were dropped.

### Changed
- **Budget is a cap, not a target.** Array selection uses a relevance cliff (keep
  records scoring at least half the best match) instead of filling the record
  quota to the budget, so accuracy no longer decays as the budget grows.

## [0.2.1] — 2026-07-04

First PyPI release since 0.1.0, collecting a large amount of work that had
landed on `main` but was never published to the index.

### Added
- **Framework adapters**: LangChain (`[langchain]`), LlamaIndex
  (`[llamaindex]`), and OpenAI-Agents (`[openai-agents]`) — wrap a tool so its
  output is compressed transparently while name/schema/guardrails are preserved.
- **MCP compressing gateway** (`[mcp]`): `compress_tool_result`,
  `compressing_call_tool`, `run_stdio_gateway`, and `tooltrim mcp -- <cmd>` to
  front any Model Context Protocol server and compress tool results at the
  protocol boundary.
- **Pluggable expand-stores** (`store.py`): `BaseStore` with `FileStore`,
  `RedisStore` (`[redis]`), and `S3Store` (`[s3]`), all content-addressed.
- **Prometheus metrics** (`metrics.py`): the proxy serves `GET /metrics`;
  tracks tokens in/out, savings, and fail-open events.
- **CLI** (`tooltrim`): `compress`, `proxy`, `demo`, `mcp`, `version`.
- **Pluggable relevance scorer** (`relevance.py`) with an optional
  embedding-based `EmbeddingScorer` (`[embeddings]`) alongside the default BM25.
- **Streaming compression** (`streaming.py`): `StreamingCompressor` /
  `compress_stream` with bounded (constant) memory.
- **Comparative baseline harness** (`eval/baselines.py`): tooltrim vs
  truncate-head/tail, RAG top-k, RAG-embed, LLMLingua-2, and full context on the
  same cases and budgets, with a paired **McNemar** significance test and
  `run_baselines.py`. LLMLingua-2 defaults to CPU so it runs without a GPU.

### Changed
- Proxy routes by path to support both OpenAI Chat Completions and Anthropic
  Messages APIs; forwards client headers (fixes a Cloudflare 1010 rejection).
- `split_paragraphs` sub-splits oversize newline-free blobs so large single-line
  outputs are still compressed.

### Fixed
- Newline-free blobs previously passed through uncompressed.
- Baseline `available()` now reports honestly — a dependency that imports but
  fails to construct (CPU-only / offline) is reported unavailable rather than
  crashing a run.

## [0.1.0] — 2026-06-28

Initial release: per-content-type compressors (HTML/JSON/logs/tabular/text),
BM25 query-aware extraction, expand-on-demand content-addressed store,
`@compressed_tool` decorator, OpenAI-compatible compression proxy (fail-open),
and a faithfulness evaluation harness with Wilson 95% confidence intervals.

[0.3.2]: https://github.com/nac7/tooltrim/releases/tag/v0.3.2
[0.3.1]: https://github.com/nac7/tooltrim/releases/tag/v0.3.1
[0.3.0]: https://github.com/nac7/tooltrim/releases/tag/v0.3.0
[0.2.1]: https://github.com/nac7/tooltrim/releases/tag/v0.2.1
[0.1.0]: https://github.com/nac7/tooltrim/releases/tag/v0.1.0
