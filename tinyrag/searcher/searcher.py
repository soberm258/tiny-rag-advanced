from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple

from tinyrag.logging_utils import logger
from tqdm import tqdm

from tinyrag.embedding.hf_emb import HFSTEmbedding
from tinyrag.searcher.bm25_recall.bm25_retriever import BM25Retriever
from tinyrag.searcher.emb_recall.emb_retriever import EmbRetriever
from tinyrag.searcher.reranker.reanker_bge_m3 import RerankerBGEM3
from tinyrag.searcher.fusion import fuse_candidates
from tinyrag.searcher.types import BM25RecallItem, EmbRecallItem


def _to_text(item: Any) -> str:
    if isinstance(item, dict):
        # 索引增强：优先使用 index_text（包含法名/编章节条等定位信息）
        return str(item.get("index_text") or item.get("text") or "")
    return str(item or "")
 

class Searcher:
    def __init__(
        self,
        *,
        emb_model_id: str,
        ranker_model_id: str,
        device: str = "cpu",
        base_dir: str = "data/db",
    ) -> None:
        self.base_dir = base_dir
        self.device = device

        bm25_dir = os.path.join(self.base_dir, "bm_corpus")
        faiss_dir = os.path.join(self.base_dir, "faiss_idx")

        # 召回
        self.bm25_retriever = BM25Retriever(base_dir=bm25_dir)
        self.emb_model = HFSTEmbedding(path=emb_model_id, device=self.device)
        try:
            index_dim = self.emb_model.st_model.get_sentence_embedding_dimension()
        except Exception:
            index_dim = len(self.emb_model.get_embedding("test_dim"))
        self.emb_retriever = EmbRetriever(index_dim=index_dim, base_dir=faiss_dir)

        # 排序
        self.ranker = RerankerBGEM3(model_id_key=ranker_model_id, device=self.device)

        logger.info("Searcher init build success...")

    def build_db(self, docs: List[Any]) -> None:
        if not docs:
            raise ValueError("构建失败：docs 为空，无法构建 BM25/向量索引。")

        self.bm25_retriever.build(docs)
        logger.info("bm25 retriever build success...")

        device_lower = str(self.device).lower()
        default_bs = "96" if "cuda" in device_lower else "16"
        batch_size = int(os.getenv("TINYRAG_EMB_BATCH_SIZE", default_bs))
        batch_size = max(1, batch_size)

        for start in tqdm(range(0, len(docs), batch_size), desc="emb build ", ascii=True):
            batch_docs = docs[start : start + batch_size]
            batch_texts = [_to_text(x) for x in batch_docs]
            batch_embs = self.emb_model.get_embeddings(batch_texts, batch_size=batch_size)
            self.emb_retriever.batch_insert(batch_embs, batch_docs)

        logger.info("emb retriever build success...")

    def save_db(self) -> None:
        self.bm25_retriever.save_bm25_data()
        logger.info("bm25 retriever save success...")
        self.emb_retriever.save()
        logger.info("emb retriever save success...")

    def load_db(self) -> None:
        self.bm25_retriever.load_bm25_data()
        logger.info("bm25 retriever load success...")
        self.emb_retriever.load()
        logger.info("emb retriever load success...")

    def search_advanced(
        self,
        *,
        rerank_query: str,
        bm25_query: str,
        emb_query_text: str,
        top_n: int = 3,
        recall_k: Optional[int] = None,
        fusion_method: str = "rrf",
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        emb_weight: float = 1.0,
    ) -> List[Tuple[float, Any]]:
        top_n = max(1, int(top_n))
        recall_k = int(recall_k) if recall_k is not None else 2 * top_n
        recall_k = max(top_n, recall_k)

        bm25_list: List[BM25RecallItem] = self.bm25_retriever.search(bm25_query, recall_k)
        logger.info("bm25 recall text num: {}", len(bm25_list))

        query_emb = self.emb_model.get_embedding(emb_query_text)
        emb_list: List[EmbRecallItem] = self.emb_retriever.search(query_emb, recall_k)
        logger.info("emb recall text num: {}", len(emb_list))

        candidates = fuse_candidates(
            bm25_list,
            emb_list,
            recall_k=recall_k,
            method=fusion_method,
            rrf_k=rrf_k,
            bm25_weight=bm25_weight,
            emb_weight=emb_weight,
        )
        logger.info("fusion candidate text num: {}", len(candidates))
        return self.ranker.rank(rerank_query, candidates, top_n)
