from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .common import item_key


BM25RecallItem = Tuple[int, Any, float]
EmbRecallItem = Tuple[int, Any, float]


def dedup_fuse(bm25_list: List[BM25RecallItem], emb_list: List[EmbRecallItem], *, top_k: int) -> List[Any]:
    top_k = max(1, int(top_k))
    seen = set()
    out: List[Any] = []

    for _idx, item, _score in sorted(bm25_list, key=lambda x: x[2], reverse=True):
        key = item_key(item)
        if key in seen:
            continue
        out.append(item)
        seen.add(key)
        if len(out) >= top_k:
            return out

    for _idx, item, _dist in sorted(emb_list, key=lambda x: x[2]):
        key = item_key(item)
        if key in seen:
            continue
        out.append(item)
        seen.add(key)
        if len(out) >= top_k:
            return out

    return out

