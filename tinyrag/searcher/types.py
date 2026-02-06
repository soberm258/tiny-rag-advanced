from __future__ import annotations

from typing import Any, List, Tuple

# 召回结果统一类型：
# - BM25：score 越大越相关
# - 向量：score 越大越相关（当前为余弦相似度）
BM25RecallItem = Tuple[int, Any, float]
EmbRecallItem = Tuple[int, Any, float]

BM25RecallList = List[BM25RecallItem]
EmbRecallList = List[EmbRecallItem]
