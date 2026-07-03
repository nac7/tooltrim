from tooltrim import (
    BM25Scorer,
    EmbeddingScorer,
    ToolCompressor,
    score_chunks,
    using_scorer,
)

# A tiny deterministic "embedding": map words onto 3 concept axes so synonyms
# share a vector. No model / torch needed.
_CONCEPTS = {
    "car": [1, 0, 0], "automobile": [1, 0, 0], "vehicle": [1, 0, 0],
    "sedan": [1, 0, 0], "drives": [1, 0, 0],
    "rain": [0, 1, 0], "weather": [0, 1, 0], "storm": [0, 1, 0],
    "pizza": [0, 0, 1], "food": [0, 0, 1], "recipe": [0, 0, 1],
}


def fake_embed(texts):
    out = []
    for t in texts:
        v = [0, 0, 0]
        for raw in t.lower().split():
            w = "".join(ch for ch in raw if ch.isalnum())
            if w in _CONCEPTS:
                for i in range(3):
                    v[i] += _CONCEPTS[w][i]
        out.append(v)
    return out


def test_embedding_scorer_matches_synonyms_bm25_misses():
    chunks = ["the automobile reached top speed", "a recipe for pizza",
              "heavy rain and storm tonight"]
    emb = EmbeddingScorer(embed=fake_embed)(chunks, "car")
    bm25 = BM25Scorer()(chunks, "car")

    # embeddings: the automobile chunk wins; unrelated chunks are 0
    assert emb[0] > 0 and emb[1] == 0 and emb[2] == 0
    # lexical BM25: "car" shares no token with any chunk -> all zero
    assert bm25 == [0.0, 0.0, 0.0]


def test_using_scorer_overrides_active_scorer():
    chunks = ["automobile", "pizza"]
    # default (BM25) sees no "car" overlap
    assert score_chunks(chunks, "car") == [0.0, 0.0]
    with using_scorer(EmbeddingScorer(embed=fake_embed)):
        s = score_chunks(chunks, "car")
    assert s[0] > 0 and s[1] == 0
    # resets afterwards
    assert score_chunks(chunks, "car") == [0.0, 0.0]


def test_compressor_with_embedding_scorer_keeps_semantic_needle():
    paras = [f"Paragraph {i} about pizza recipe and food." for i in range(80)]
    paras[40] = "The automobile reached 200 units on the open road."
    blob = "\n\n".join(paras)
    query = "car"  # lexical miss, semantic hit

    semantic = ToolCompressor(max_tokens=60, add_footer=False,
                              scorer=EmbeddingScorer(embed=fake_embed))
    out = semantic.compress(blob, query=query)
    assert "automobile" in out.text          # embeddings surfaced it

    # the default lexical compressor can't (no shared word with "car")
    lexical = ToolCompressor(max_tokens=60, add_footer=False)
    assert "automobile" not in lexical.compress(blob, query=query).text


def test_embedding_scorer_empty_query_is_all_zero():
    assert EmbeddingScorer(embed=fake_embed)(["automobile"], "") == [0.0]
