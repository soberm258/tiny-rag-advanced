from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .common import item_key


BM25RecallItem = Tuple[int, Any, float]
EmbRecallItem = Tuple[int, Any, float]


def rrf_fuse(
    bm25_list: List[BM25RecallItem],
    emb_list: List[EmbRecallItem],
    *,
    top_k: int,
    k: int = 60,
    bm25_weight: float = 1.0,
    emb_weight: float = 1.0,
) -> List[Any]:
    """
    Reciprocal Rank Fusion (RRF)：
    - BM25：按分数从高到低排序
    - 向量：按相似度从高到低排序（余弦相似度，越大越相似）
    """
    top_k = max(1, int(top_k))
    k = max(1, int(k))

    score_map: Dict[str, float] = {}
    item_map: Dict[str, Any] = {}

    bm25_sorted = sorted(bm25_list, key=lambda x: x[2], reverse=True)
    emb_sorted = sorted(emb_list, key=lambda x: x[2], reverse=True)
    
    for rank, (_idx, item, _score) in enumerate(bm25_sorted, start=1):
        key = item_key(item)
        item_map.setdefault(key, item)
        score_map[key] = score_map.get(key, 0.0) + float(bm25_weight) * (1.0 / (k + rank))

    for rank, (_idx, item, _sim) in enumerate(emb_sorted, start=1):
        key = item_key(item)
        item_map.setdefault(key, item)
        score_map[key] = score_map.get(key, 0.0) + float(emb_weight) * (1.0 / (k + rank))

    fused = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    return [item_map[key] for key, _ in fused[:top_k]]
