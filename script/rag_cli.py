import sys

sys.path.append(".")

import argparse
import os
from typing import Any, Dict, List

from tinyrag.ingest import load_docs_for_build
from tinyrag.rag import chunk_doc_item, format_observation_for_llm
from tinyrag.searcher.searcher import Searcher
from tinyrag.sentence_splitter import SentenceSplitter
from tinyrag.utils import write_list_to_jsonl


_DB_ROOT_DIR = os.path.join("data", "db")
_EMB_MODEL_ID = os.path.join("models", "bge-base-zh-v1.5")
_RERANK_MODEL_ID = os.path.join("models", "bge-reranker-base")
_DEVICE = os.getenv("TINYRAG_DEVICE", "cpu")


def _ensure_db_dir(db_name: str) -> str:
    base_dir = os.path.join(_DB_ROOT_DIR, db_name)
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def build_db(*, db_name: str, input_path: str, min_chunk_len: int = 20, sentence_size: int = 512) -> None:
    base_dir = _ensure_db_dir(db_name)
    docs = load_docs_for_build(input_path, recursive=True, pdf_mode="auto", txt_mode="auto")
    print(f"读取文档数：{len(docs)}", flush=True)

    splitter = SentenceSplitter(use_model=False, sentence_size=sentence_size)

    chunks: List[Dict[str, Any]] = []
    for doc in docs:
        chunks.extend(chunk_doc_item(doc, splitter, min_chunk_len=min_chunk_len))

    if not chunks:
        raise ValueError("建库失败：切分后没有任何 chunk。请检查输入数据或调小 min_chunk_len。")

    write_list_to_jsonl(chunks, os.path.join(base_dir, "split_sentence.jsonl"))
    print(f"切分 chunk 数：{len(chunks)}", flush=True)

    searcher = Searcher(
        emb_model_id=_EMB_MODEL_ID,
        ranker_model_id=_RERANK_MODEL_ID,
        device=_DEVICE,
        base_dir=base_dir,
    )
    searcher.build_db(chunks)
    searcher.save_db()
    print(f"建库完成：{base_dir}", flush=True)


def search_db(*, db_name: str, query: str, topk: int,
              is_hyde:bool=False,bm25_weight: float=1.0, emb_weight: float=1.0, fusion_method: str="rrf"
              ) -> None:
    base_dir = os.path.join(_DB_ROOT_DIR, db_name)
    searcher = Searcher(
        emb_model_id=_EMB_MODEL_ID,
        ranker_model_id=_RERANK_MODEL_ID,
        device=_DEVICE,
        base_dir=base_dir,
    )
    searcher.load_db()
    if is_hyde:
        from transformers import pipeline
        from tinyrag.rag.prompts import build_hyde_prompt
        import torch
        pipe = pipeline(
            "text-generation",
            model="./models/Qwen2-1.5B-Instruct",
            torch_dtype=torch.bfloat16,  # 使用bfloat16减少内存
            device_map="auto",  # 自动分配设备
        )
        hyde_prompt = build_hyde_prompt()
        message = [{"role": "system", "content": hyde_prompt},{"role": "user", "content": f"用户问题：{query}"}]
        response = pipe(message, max_new_tokens=256, do_sample=False)
        hyde_raw = response[0]['generated_text'][-1]["content"]
        # 防止 HyDE 丢失原问题信息：把原 query 与扩展短语拼在一起做向量检索
        hyde_q = f"{query} {str(hyde_raw or '').strip()}".strip()
    else:
        hyde_q = query
    reranked = searcher.search_advanced(
        rerank_query=query,
        bm25_query=query,
        emb_query_text=hyde_q,
        top_n=topk,
        recall_k=max(1, topk * 4),
        fusion_method=fusion_method,
        rrf_k=60,
        bm25_weight=bm25_weight,
        emb_weight=emb_weight,
    )
    items: List[Dict[str, Any]] = []
    for i, (score, item) in enumerate(reranked, start=1):
        if isinstance(item, dict):
            items.append(
                {
                    "rank": i,
                    "score": float(score),
                    "id": str(item.get("id") or ""),
                    "text": str(item.get("text") or ""),
                    "meta": item.get("meta") or {},
                }
            )
        else:
            items.append({"rank": i, "score": float(score), "id": "", "text": str(item), "meta": {}})

    print(format_observation_for_llm({"items": items}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="最小 RAG CLI（建库/检索；不含 agent）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="建库：解析 -> 切分 -> BM25/向量索引落盘")
    p_build.add_argument("--db-name", type=str, required=True, help="数据库名（目录名），建议为 law 或 case")
    p_build.add_argument("--path", type=str, required=True, help="输入文件或目录路径")
    p_build.add_argument("--min-chunk-len", type=int, default=20)
    p_build.add_argument("--sentence-size", type=int, default=512)


    p_search = sub.add_parser("search", help="检索：返回带 source 的 Observation 文本")
    p_search.add_argument("--db-name", type=str, required=True, help="数据库名（law/case）")
    p_search.add_argument("--query", type=str, required=True)
    p_search.add_argument("--topk", type=int, default=5)
    p_search.add_argument("--is_hyde", action='store_true', default=False, help="是否使用 HyDE 生成假设查询")
    p_search.add_argument("--bm25-weight", type=float, default=1.0, help="BM25 权重")
    p_search.add_argument("--emb-weight", type=float, default=1.0, help="向量相似度权重")
    p_search.add_argument("--fusion-method", type=str, default="rrf", help="融合方法，默认 rrf")


    args = parser.parse_args()

    if args.cmd == "build":
        build_db(
            db_name=str(args.db_name).strip(),
            input_path=str(args.path),
            min_chunk_len=int(args.min_chunk_len),
            sentence_size=int(args.sentence_size),
        )
    elif args.cmd == "search":
        """
        此接口留给eval快速检验用
        """
        search_db(db_name=str(args.db_name).strip(), query=str(args.query).strip(), topk=int(args.topk)
                  ,is_hyde=bool(args.is_hyde), bm25_weight=float(args.bm25_weight), emb_weight=float(args.emb_weight),
                    fusion_method=str(args.fusion_method))


if __name__ == "__main__":
    main()
