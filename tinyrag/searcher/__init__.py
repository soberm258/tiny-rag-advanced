"""
tinyrag.searcher：检索入口集合。

说明：
为了避免 `import tinyrag.searcher` 时强制加载 sentence-transformers/torch 等重依赖，
这里也采用延迟导入（PEP 562：__getattr__）。
"""

from __future__ import annotations

from typing import Any

__all__ = ["Searcher", "MultiDBSearcher"]


def __getattr__(name: str) -> Any:
    if name == "Searcher":
        from .searcher import Searcher

        return Searcher
    if name == "MultiDBSearcher":
        from .multi_db_searcher import MultiDBSearcher

        return MultiDBSearcher
    raise AttributeError(name)
