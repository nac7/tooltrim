"""Component ablation for tooltrim's compressor.

The paper claims tooltrim's advantage is *structural*: a content-type router, a
JSON relevance cliff, and a neighbor-context window, layered on plain query-aware
BM25 extraction. This module isolates each component's marginal contribution by
building a ladder of compressors that turn the mechanisms on one at a time:

    bm25            generic BM25 extraction, no routing, no neighbors      (floor)
    +router         + content-type router (JSON/tabular/HTML/log aware)
    +cliff          + JSON relevance cliff (budget as cap, not fill-to-k)
    +neighbors      + neighbor-context window   == full tooltrim

Each rung is a drop-in ``Compressor`` (``name`` + ``compress(text, query,
budget)``) so it plugs straight into ``evaluate_methods`` and the agent-task
harness. The knobs are set via the context-scoped overrides in
``tooltrim._config`` — production behavior is untouched.
"""

from __future__ import annotations

from typing import List, Optional

from tooltrim import ToolCompressor
from tooltrim._config import using_config


class _AblationRung:
    """One rung of the ablation ladder, as a Compressor.

    Args:
        name: rung label used in reports.
        route: if False, force ``content_type="text"`` so the content-type router
            is bypassed and everything goes through generic extraction.
        relevance_floor: JSON array cliff fraction (0.0 = fill-to-k quota, the
            pre-cliff behavior; 0.5 = the shipped cliff).
        neighbor: neighbor-context window (0 = off; 1 = shipped).
    """

    def __init__(self, name: str, *, route: bool, relevance_floor: float,
                 neighbor: int):
        self.name = name
        self.route = route
        self.relevance_floor = relevance_floor
        self.neighbor = neighbor

    def compress(self, text: str, query: Optional[str], budget: int) -> str:
        tc = ToolCompressor(max_tokens=budget, add_footer=False, store=None)
        # content_type="text" pins the generic compressor, disabling the router.
        ctype = None if self.route else "text"
        with using_config(neighbor=self.neighbor,
                          relevance_floor=self.relevance_floor):
            return tc.compress(text, query=query, content_type=ctype).text

    def available(self) -> bool:
        return True


def ablation_ladder() -> List[_AblationRung]:
    """The cumulative ablation ladder, weakest to full tooltrim."""
    return [
        # No routing, no cliff (irrelevant without JSON routing), no neighbors.
        _AblationRung("bm25", route=False, relevance_floor=0.0, neighbor=0),
        # Turn on the content-type router; cliff still off (fill-to-k), no neighbors.
        _AblationRung("+router", route=True, relevance_floor=0.0, neighbor=0),
        # Turn on the JSON relevance cliff.
        _AblationRung("+cliff", route=True, relevance_floor=0.5, neighbor=0),
        # Turn on neighbor context == shipped tooltrim.
        _AblationRung("+neighbors", route=True, relevance_floor=0.5, neighbor=1),
    ]
