# Dockerfile for the tooltrim MCP server (Glama / MCP registry listing).
#
# Builds a container that runs `tooltrim serve` — a stdio MCP server exposing
# the `compress` and `expand_tool_output` tools. An MCP client (e.g. Glama)
# connects over stdin/stdout and verifies the server starts and answers
# tools/list introspection.
#
#   docker build -t tooltrim-mcp .
#   docker run --rm -i tooltrim-mcp        # -i keeps stdin open for stdio JSON-RPC
FROM python:3.12-slim

# Unbuffered stdout so stdio MCP framing isn't delayed.
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app

# Install tooltrim with the MCP extra from source (stays in sync with the repo).
RUN pip install --no-cache-dir ".[mcp]"

# stdio transport: the MCP client speaks JSON-RPC over stdin/stdout.
ENTRYPOINT ["tooltrim", "serve"]
