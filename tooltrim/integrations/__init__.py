"""Framework integrations for tooltrim.

Each integration imports its framework lazily, so importing this package never
pulls in LangChain/LlamaIndex/etc. Install the matching extra to use one, e.g.
``pip install tooltrim[langchain]``.
"""

from __future__ import annotations

from .langchain import compress_langchain_tool, compress_langchain_tools

__all__ = ["compress_langchain_tool", "compress_langchain_tools"]
