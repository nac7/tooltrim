"""Tests for the component ablation ladder and the config seam it rides on."""

import json

from eval.ablation import ablation_ladder
from eval.agent_tasks import default_agent_tasks, task_success_rate
from tooltrim._config import using_config
from tooltrim.compressors._budget import fit_chunks
from tooltrim.compressors.json_ import compress as json_compress
from tooltrim.relevance import using_scorer


def _fixed_scorer(chunks, query):
    """Deterministic scorer: 'strong'->1.0, 'weak'->0.3, else 0.0 — so the test
    isolates the relevance-cliff logic from BM25's length-normalization quirks."""
    out = []
    for c in chunks:
        out.append(1.0 if "strong" in c else 0.3 if "weak" in c else 0.0)
    return out


def test_ladder_is_cumulative_and_shaped():
    rungs = ablation_ladder()
    assert [r.name for r in rungs] == ["bm25", "+router", "+cliff", "+neighbors"]
    # Router turns on at +router and stays on; cliff at +cliff; neighbor at +neighbors.
    assert [r.route for r in rungs] == [False, True, True, True]
    assert [r.relevance_floor for r in rungs] == [0.0, 0.0, 0.5, 0.5]
    assert [r.neighbor for r in rungs] == [0, 0, 0, 1]


def test_router_is_the_load_bearing_component_on_agent_tasks():
    # The headline of the ablation: without the content-type router, generic BM25
    # extraction cannot keep JSON/tabular output parseable, so the agent's next
    # parse fails and task success is 0; adding the router recovers it.
    tasks = default_agent_tasks()
    rungs = {r.name: r for r in ablation_ladder()}
    bm25 = task_success_rate(
        (t, rungs["bm25"].compress(t.tool_output, t.query, 256)) for t in tasks)
    routed = task_success_rate(
        (t, rungs["+router"].compress(t.tool_output, t.query, 256)) for t in tasks)
    assert bm25 == 0.0
    assert routed > bm25
    # The full ladder (shipped tooltrim) is at least as good as +router.
    full = task_success_rate(
        (t, rungs["+neighbors"].compress(t.tool_output, t.query, 256)) for t in tasks)
    assert full >= routed


def test_neighbor_override_controls_context_window():
    # Middle chunk is the only match; neighbor=1 pulls in its two neighbors,
    # neighbor=0 keeps just the match.
    chunks = ["alpha filler one", "beta filler two", "the needle gamma value",
              "delta filler four", "epsilon filler five"]
    with using_config(neighbor=0):
        bare = fit_chunks(chunks, "needle gamma", 1000)
    with using_config(neighbor=1):
        widened = fit_chunks(chunks, "needle gamma", 1000)
    assert "beta filler two" not in bare        # neighbor excluded
    assert "beta filler two" in widened          # neighbor included
    # Default (no override) is the shipped neighbor=1 behavior.
    assert "beta filler two" in fit_chunks(chunks, "needle gamma", 1000)


def test_relevance_floor_override_controls_array_selection():
    # One strong match ("alpha" x3) and one weak-but-positive match ("alpha" x1,
    # ~<50% score). The cliff (floor=0.5) drops the weak record; fill-to-k keeps it.
    blob = json.dumps({"results": [
        {"id": 1, "note": "strong match here"},
        {"id": 2, "note": "unrelated bravo charlie delta echo foxtrot"},
        {"id": 3, "note": "weak single mention among other padding"},
    ]})
    with using_scorer(_fixed_scorer):
        with using_config(relevance_floor=0.5):
            cliffed = json_compress(blob, "match", 200)
        with using_config(relevance_floor=0.0):
            filled = json_compress(blob, "match", 200)
    assert '"id":1' in cliffed and '"id":1' in filled  # strong match kept by both
    assert '"id":3' not in cliffed                       # weak match falls off cliff
    assert '"id":3' in filled                            # fill-to-k keeps it


def test_rungs_expose_compressor_interface():
    for r in ablation_ladder():
        assert hasattr(r, "name") and callable(r.compress) and r.available() is True
