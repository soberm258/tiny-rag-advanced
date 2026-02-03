"""
RAG 相关的可复用组件

检索系统可复用这些模块来完成：
1) 提示词构造
2) 文档切分为 chunk
3) 检索结果的上下文与引用格式化
"""

from .prompts import build_hyde_prompt, build_rag_prompt
from .chunking import chunk_doc_item
from .observation import format_observation_for_llm

__all__ = [
    "build_hyde_prompt",
    "build_rag_prompt",
    "chunk_doc_item",
    "format_observation_for_llm",
]
