"""
tinyrag：最小 RAG 检索组件集合。

说明：
本仓库已移除自研 agent/LLM 编排层，仅保留解析、切分、建库、检索与评测相关能力。
为避免导入时强制依赖 faiss/torch 等重组件，这里采用延迟导入（PEP 562：__getattr__）。
"""

from __future__ import annotations

from typing import Any

__all__ = [
    # 句子切分
    "SentenceSplitter",
    # 摄取
    "load_docs_for_build",
    # 检索
    "Searcher",
    "MultiDBSearcher",
    "BM25Retriever",
    "EmbRetriever",
    "EmbIndex",
    "RerankerBGEM3",
    # BM25 算法
    "BM25Okapi",
    "BM25L",
    "BM25Plus",
    # Embeddings
    "BaseEmbedding",
    "HFSTEmbedding",
    "ImgEmbedding",
    "OpenAIEmbedding",
    "ZhipuEmbedding",
]


def __getattr__(name: str) -> Any:
    if name == "SentenceSplitter":
        from .sentence_splitter import SentenceSplitter

        return SentenceSplitter

    if name == "load_docs_for_build":
        from .ingest import load_docs_for_build

        return load_docs_for_build

    if name == "Searcher":
        from .searcher.searcher import Searcher

        return Searcher
    if name == "MultiDBSearcher":
        from .searcher.multi_db_searcher import MultiDBSearcher

        return MultiDBSearcher

    if name in ("BM25Retriever", "BM25Okapi", "BM25L", "BM25Plus"):
        from .searcher.bm25_recall.bm25_retriever import BM25Retriever
        from .searcher.bm25_recall.rank_bm25 import BM25Okapi, BM25L, BM25Plus

        return {
            "BM25Retriever": BM25Retriever,
            "BM25Okapi": BM25Okapi,
            "BM25L": BM25L,
            "BM25Plus": BM25Plus,
        }[name]

    if name in ("EmbRetriever", "EmbIndex"):
        from .searcher.emb_recall.emb_index import EmbIndex
        from .searcher.emb_recall.emb_retriever import EmbRetriever

        return {
            "EmbRetriever": EmbRetriever,
            "EmbIndex": EmbIndex,
        }[name]

    if name == "RerankerBGEM3":
        from .searcher.reranker.reanker_bge_m3 import RerankerBGEM3

        return RerankerBGEM3

    if name in ("BaseEmbedding", "HFSTEmbedding", "ImgEmbedding", "OpenAIEmbedding", "ZhipuEmbedding"):
        from .embedding.base_emb import BaseEmbedding
        from .embedding.hf_emb import HFSTEmbedding
        from .embedding.img_emb import ImgEmbedding
        from .embedding.openai_emb import OpenAIEmbedding
        from .embedding.zhipu_emb import ZhipuEmbedding

        return {
            "BaseEmbedding": BaseEmbedding,
            "HFSTEmbedding": HFSTEmbedding,
            "ImgEmbedding": ImgEmbedding,
            "OpenAIEmbedding": OpenAIEmbedding,
            "ZhipuEmbedding": ZhipuEmbedding,
        }[name]

    raise AttributeError(name)
