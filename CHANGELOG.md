# Changelog

All notable changes to tooltrim are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.2.1]: https://github.com/nac7/tooltrim/releases/tag/v0.2.1
[0.1.0]: https://github.com/nac7/tooltrim/releases/tag/v0.1.0
