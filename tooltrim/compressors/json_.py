"""JSON compressor: structure-preserving pruning + sampling.

Strategy (deterministic):
  * Long strings are truncated with an elision marker.
  * Large arrays keep a sample of items plus a "(+N more items)" note; for arrays
    of objects the key schema is preserved via the sampled items.
  * Dicts used as id-keyed record maps are sampled entry-wise, like arrays.
  * With a query, array items are sampled by BM25 relevance instead of position.
  * The whole thing is rendered compactly and the sampling budget is tightened
    until it fits ``max_tokens``. Output is always valid JSON: the last resort is
    a scalar-field skeleton, never a comma-split of the rendered text.
"""

from __future__ import annotations

import json
from typing import Any, List

from .._config import relevance_floor_override
from ..relevance import score_chunks
from ..tokens import count_tokens
from ._budget import fit_chunks

# A kept record must score at least this fraction of the best-matching record's
# score. This turns array selection into a *relevance cliff* rather than a fill-to-k
# quota: single-needle queries keep just the needle (everything else falls off the
# cliff), while genuine multi-record matches (aggregation) keep the whole cluster
# because those records score similarly high. A fixed prior — deliberately not tuned
# per benchmark — that stops a larger budget from padding in keyword-stuffed noise.
_RELEVANCE_FLOOR = 0.5

# Arrays no larger than this are treated as *enumerable entities* (an order's line
# items, a user's addresses/orders) where every above-floor record can be load-
# bearing for a later step. We keep all of their relevance-positive records rather
# than sampling down to k — sampling is what collapses a multi-item order to a
# single item at a tight budget. Larger arrays (search-result-like) are still
# capped at k. The relevance floor removes distractors in both cases, so this
# doesn't reintroduce padding; it only stops entity lists from being decimated.
_KEEP_ALL_MAX = 10

# Whitespace-free strings this long or shorter are treated as identifiers/codes
# (item_ids, skus, urls, statuses) and never truncated: every character is load-
# bearing and they carry almost no tokens anyway. Truncation is reserved for
# free-text prose (has spaces), which is where the tokens actually live.
_ID_MAX_LEN = 128

# A dict of purely scalar values with at least this many entries is a lookup
# *table* (product name -> id, code -> label), not a record with named fields.
# Set well above the field count of a realistic record so that sampling never
# starts dropping named fields off an entity like an order.
_FLAT_MAP_MIN = 20


def _truncate_str(s: str, max_str_len: int) -> str:
    if len(s) <= max_str_len:
        return s
    if " " not in s and len(s) <= _ID_MAX_LEN:
        return s
    return s[:max_str_len] + "…(+%d chars)" % (len(s) - max_str_len)


def _select(rendered: List[str], k: int, query: str | None, keep_all: bool) -> List[int]:
    # Query-aware selection treats the budget as a *cap*, not a target: keep only
    # the relevance-positive records (top-k of them), never padding out to k with
    # zero-score noise. Padding is what let a larger budget reintroduce distractors
    # and cost accuracy — a bigger array kept *more* wrong records, not more signal.
    # (Plain top-k-by-position, below, is the no-query fallback.)
    #
    # ``keep_all`` is the small-entity protection (see _KEEP_ALL_MAX). The tight
    # rungs of the ladder in ``compress`` turn it off, because decimating to a few
    # *whole* records beats keeping every record with all of its fields elided.
    n = len(rendered)
    if query:
        scores = score_chunks(rendered, query)
        positive = [i for i in range(n) if scores[i] > 0]
        if positive:
            positive.sort(key=lambda i: scores[i], reverse=True)
            # An ablation can lower the cliff to 0 (fill-to-k quota, the pre-cliff
            # behavior) to isolate the relevance-cliff's contribution.
            floor_frac = relevance_floor_override()
            if floor_frac is None:
                floor_frac = _RELEVANCE_FLOOR
            floor = scores[positive[0]] * floor_frac
            kept = [i for i in positive if scores[i] >= floor]
            # Small entity arrays: keep every above-floor record (don't cap at k);
            # the string/depth rungs find the budget instead. Large arrays: sample.
            if not keep_all or n > _KEEP_ALL_MAX:
                kept = kept[:k]
            return sorted(kept)
    if keep_all and n <= max(k, _KEEP_ALL_MAX):
        return list(range(n))
    return list(range(min(k, n)))


def _sample_indices(items: List[Any], k: int, query: str | None,
                    keep_all: bool = True) -> List[int]:
    rendered = [json.dumps(it, ensure_ascii=False, default=str) for it in items]
    return _select(rendered, k, query, keep_all)


def _sample_entries(entries: List[Any], k: int, query: str | None,
                    keep_all: bool = True) -> List[int]:
    """Same selection, but over the *entries* of a record map (see _is_record_map).

    Each candidate is rendered as a one-entry object so the key — which for a
    record map is the identifier (an item_id, sku, variant id) — participates in
    relevance scoring instead of being invisible to it.
    """
    rendered = [json.dumps({k_: v}, ensure_ascii=False, default=str)
                for k_, v in entries]
    return _select(rendered, k, query, keep_all)


def _is_collection_map(obj: dict) -> bool:
    """True for a dict used as a *collection*, not as a single record.

    Two shapes qualify:

    * **Record map** — every value is a dict. tau-bench's ``get_product_details``
      returns ``{"variants": {"<item_id>": {...}}}``: semantically an array of
      records that happens to be keyed by id.
    * **Flat lookup map** — every value is a scalar and there are a lot of them
      (``list_all_product_types`` is 50 ``name -> product_id`` pairs).

    Either way the dict is a collection and must be *sampled* like an array. The
    only other lever a dict has is depth elision, which destroys every record's
    fields at once — or, for a flat map, no lever at all, so it overruns the
    budget and falls through to the last-resort skeleton.

    A normal record — ``{"order_id": ..., "status": ..., "items": [...]}`` — must
    keep all of its keys. It has few keys and mixed value types, so neither branch
    matches: the record-map branch requires *every* value to be a dict, and the
    flat-map branch requires many more entries than a record carries fields.
    """
    if len(obj) >= 3 and all(isinstance(v, dict) for v in obj.values()):
        return True
    return (len(obj) >= _FLAT_MAP_MIN
            and all(not isinstance(v, (dict, list)) for v in obj.values()))


def _prune(obj: Any, depth: int, *, max_list_items: int, max_str_len: int,
           max_depth: int, query: str | None, keep_all: bool = True) -> Any:
    if depth >= max_depth and isinstance(obj, (dict, list)):
        if isinstance(obj, dict):
            return {"…": "(%d keys elided)" % len(obj)}
        return ["(%d items elided)" % len(obj)]

    if isinstance(obj, str):
        return _truncate_str(obj, max_str_len)

    kw = dict(max_list_items=max_list_items, max_str_len=max_str_len,
              max_depth=max_depth, query=query, keep_all=keep_all)

    if isinstance(obj, dict):
        # An id-keyed record map is a collection: sample whole entries, exactly as
        # for an array. Without this a dict has no sampling lever at all and the
        # only way to shrink it is depth elision, which wipes every record's
        # identifiers and flags at once.
        if _is_collection_map(obj):
            entries = list(obj.items())
            idxs = _sample_entries(entries, max_list_items, query, keep_all)
            out_map = {entries[i][0]: _prune(entries[i][1], depth + 1, **kw)
                       for i in idxs}
            if len(entries) > len(idxs):
                out_map["…"] = "(+%d more entries)" % (len(entries) - len(idxs))
            return out_map
        return {k: _prune(v, depth + 1, **kw) for k, v in obj.items()}

    if isinstance(obj, list):
        idxs = _sample_indices(obj, max_list_items, query, keep_all)
        out: List[Any] = [_prune(obj[i], depth + 1, **kw) for i in idxs]
        if len(obj) > len(idxs):
            out.append("(+%d more items)" % (len(obj) - len(idxs)))
        return out

    return obj


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=None,
                      separators=(",", ":"), default=str)


def _scalar_skeleton(data: Any, max_tokens: int) -> str:
    """Last resort that is still *parseable*: as many scalar leaves as fit.

    Containers are dropped wholesale and counted in an "…" marker, so the result
    is always valid JSON and always tells the agent that something was removed.
    """
    if isinstance(data, dict):
        kept: dict = {}
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                continue
            trial = dict(kept)
            trial[k] = _truncate_str(v, 32) if isinstance(v, str) else v
            if count_tokens(_dumps(trial)) > max_tokens:
                break
            kept = trial
        # Scalars alone usually leave budget on the table. Spend it: add container
        # fields back in heavily pruned form, in declaration order, keeping each
        # only if it still fits. One sampled line item beats none.
        for k, v in data.items():
            if not isinstance(v, (dict, list)) or k in kept:
                continue
            trial = dict(kept)
            trial[k] = _prune(v, 1, max_list_items=1, max_str_len=16,
                              max_depth=3, query=None, keep_all=False)
            if count_tokens(_dumps(trial)) <= max_tokens:
                kept = trial
        # The elision marker is not optional. If it does not fit, give back fields
        # until it does: an agent that can see something was removed can re-query,
        # while a silently truncated result is indistinguishable from a complete
        # one — it reads as "the store does not sell water bottles".
        if len(kept) < len(data):
            while True:
                trial = dict(kept)
                trial["…"] = "(%d fields elided)" % (len(data) - len(kept))
                if count_tokens(_dumps(trial)) <= max_tokens or not kept:
                    kept = trial
                    break
                kept.pop(next(reversed(kept)))
        if kept and count_tokens(_dumps(kept)) <= max_tokens:
            return _dumps(kept)
    elif isinstance(data, list):
        kept_l: List[Any] = []
        for v in data:
            if isinstance(v, (dict, list)):
                continue
            trial_l = kept_l + [_truncate_str(v, 32) if isinstance(v, str) else v]
            if count_tokens(_dumps(trial_l)) > max_tokens:
                break
            kept_l = trial_l
        if len(kept_l) < len(data):
            while True:
                trial_l = kept_l + ["(%d items elided)" % (len(data) - len(kept_l))]
                if count_tokens(_dumps(trial_l)) <= max_tokens or not kept_l:
                    kept_l = trial_l
                    break
                kept_l.pop()
        if kept_l and count_tokens(_dumps(kept_l)) <= max_tokens:
            return _dumps(kept_l)
    # Degenerate budget: too small even to name what was dropped. A bare "…" is
    # still valid JSON and still says "this is not the whole answer", which is the
    # one thing the agent must not be misled about.
    return '"…"'


def compress(text: str, query: str | None, max_tokens: int) -> str:
    try:
        data = json.loads(text)
    except Exception:
        # Not actually parseable JSON: treat as text.
        return fit_chunks([p for p in text.split("\n") if p.strip()], query, max_tokens)

    # Progressively tighter sampling until it fits the budget. The last few rungs
    # (shallow max_depth, short strings) exist so a *tight* budget still lands on
    # structure-preserving, still-valid JSON — keeping top-level scalar fields
    # like "status" that a downstream agent needs — instead of cliffing to the
    # text fallback below, which shatters the object on commas and can drop those
    # very fields. See the retail multi-step benchmark: at budget 128 the object
    # previously collapsed to just the id fields.
    # Rung order matters: we shrink free-text strings (and sample large arrays)
    # hard while holding depth high, so record *structure and identifiers* survive
    # a tight budget. Then we *decimate* — keep fewer, but complete, records — and
    # only after that do we reduce depth. Depth elision collapses a record to
    # "(N keys elided)", destroying every id, flag and price it held, so a subset
    # of whole records is strictly more useful to an agent than the full set with
    # nothing in it. It stays the genuine last resort for a pathologically deep
    # single object. (The earlier ladder dropped depth while the small-entity
    # protection still forbade decimating, so tight budgets went straight to the
    # destructive lever — that is what wiped every item_id/available flag out of
    # a 5-variant product at budget 128.)
    for max_list_items, max_str_len, max_depth, keep_all in (
        (10, 200, 8, True), (10, 120, 8, True), (10, 60, 8, True),
        (10, 30, 8, True), (8, 16, 8, True),
        (10, 60, 8, False), (8, 60, 8, False), (6, 60, 8, False),
        (8, 16, 8, False), (6, 16, 8, False), (4, 16, 8, False),
        (3, 16, 8, False), (2, 16, 8, False), (1, 16, 8, False),
        (1, 16, 4, False), (1, 16, 3, False), (1, 16, 2, False),
    ):
        pruned = _prune(data, 0, max_list_items=max_list_items,
                        max_str_len=max_str_len, max_depth=max_depth,
                        query=query, keep_all=keep_all)
        rendered = json.dumps(pruned, ensure_ascii=False, indent=None,
                              separators=(",", ":"), default=str)
        if count_tokens(rendered) <= max_tokens:
            return rendered

    # Still too big (e.g. one enormous flat object). Degrade to a *valid* JSON
    # skeleton of the scalar fields. Never fall through to splitting the rendered
    # JSON on commas: that shatters the object into fragments that no longer parse,
    # which is worse than useless to a downstream agent — it drops the very
    # top-level fields ("status", ids) the ladder above exists to protect.
    return _scalar_skeleton(data, max_tokens)
