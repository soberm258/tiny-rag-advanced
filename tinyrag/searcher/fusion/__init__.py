from __future__ import annotations

from typing import Any, List

from ..types import BM25RecallItem, EmbRecallItem
from .dedup import dedup_fuse
from .rrf import rrf_fuse

__all__ = [
    "dedup_fuse",
    "fuse_candidates",
    "rrf_fuse",
]


def fuse_candidates(
    bm25_list: List[BM25RecallItem],
    emb_list: List[EmbRecallItem],
    *,
    recall_k: int,
    method: str = "rrf",
    rrf_k: int = 60,
    bm25_weight: float = 1.0,
    emb_weight: float = 1.0,
) -> List[Any]:
    """
    融合候选集合：把 BM25 与向量召回列表合并为一份候选列表（去重后截断到 recall_k）。

    - method="rrf"：Reciprocal Rank Fusion
    - method="dedup"：BM25 优先 + 追加向量 + 去重
    """
    recall_k = max(1, int(recall_k))
    method = (method or "rrf").lower().strip()
    if method == "rrf":
        return rrf_fuse(
            bm25_list,
            emb_list,
            top_k=recall_k,
            k=rrf_k,
            bm25_weight=bm25_weight,
            emb_weight=emb_weight,
        )
    elif method == "dedup":
        return dedup_fuse(bm25_list, emb_list, top_k=recall_k)
    else:
        raise ValueError(f"不支持的融合方法：{method}")