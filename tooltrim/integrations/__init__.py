"""Framework integrations for tooltrim.

Each integration imports its framework lazily, so importing this package never
pulls in LangChain/LlamaIndex/etc. Install the matching extra to use one, e.g.
``pip install tooltrim[langchain]``.
"""

from __future__ import annotations

from .langchain import compress_langchain_tool, compress_langchain_tools
from .llamaindex import compress_llamaindex_tool, compress_llamaindex_tools
from .mcp import (
    compress_tool_result,
    compressing_call_tool,
    run_stdio_gateway,
)
from .openai_agents import (
    compress_openai_agents_tool,
    compress_openai_agents_tools,
)

__all__ = [
    "compress_langchain_tool",
    "compress_langchain_tools",
    "compress_llamaindex_tool",
    "compress_llamaindex_tools",
    "compress_openai_agents_tool",
    "compress_openai_agents_tools",
    "compress_tool_result",
    "compressing_call_tool",
    "run_stdio_gateway",
]
