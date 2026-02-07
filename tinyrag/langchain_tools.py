from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field


from langchain.tools import tool
import torch  



from tinyrag.rag.observation import format_observation_for_llm

from tinyrag.searcher.searcher import Searcher


_DB_ROOT_DIR = os.path.join("data", "db")
_EMB_MODEL_ID = os.path.join("models", "bge-base-zh-v1.5")
_RERANK_MODEL_ID = os.path.join("models", "bge-reranker-base")
_DEVICE = os.getenv("TINYRAG_DEVICE","cuda" if torch.cuda.is_available() else "cpu")
_RECALL_FACTOR = 4
_RRF_K = 60
_BM25_WEIGHT = 1.0
_EMB_WEIGHT = 1.0

@lru_cache(maxsize=8)
def _get_searcher(db_name: str) -> Searcher:
    base_dir = os.path.join(_DB_ROOT_DIR, str(db_name))
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"数据库目录不存在：{base_dir}")
    searcher = Searcher(
        emb_model_id=_EMB_MODEL_ID,
        ranker_model_id=_RERANK_MODEL_ID,
        device=_DEVICE,
        base_dir=base_dir,
    )
    searcher.load_db()
    return searcher


class RAGSearchInput(BaseModel):
    query: str = Field(description="用户问题/检索查询（必填）")
    topk: int = Field(default=5, ge=1, le=20, description="返回条数（1~20）")
    db_name: Literal["law", "case", "law_big","law_huge"] = Field(description="数据库名：law 或 case（必填）")
    is_hyde: bool = Field(default=False, description="是否启用 HyDE 查询扩展用于向量检索（默认 False）")

@tool("rag_search", args_schema=RAGSearchInput)
def rag_search(query: str, topk: int = 5, db_name: str = "law", is_hyde: bool = False) -> str:
    """
      在当前数据库中进行证据检索（默认策略：HyDE + RRF + rerank），返回带元数据的片段列表。" \
       "目前支持两个数据库：law（法律法规）和 case（司法案例）。" \
       "当你需要从法律法规或司法案例中寻找答案时使用。" \
       "用户询问法律问题时，必须查找law库，而case库可选择作为案例补充使用" \
       "注意，使用case库时,topk不宜过大，推荐为'topk: 3'，以免返回过多无关案例片段影响回答质量。
    """
    q = (query or "").strip()
    if not q:
        return format_observation_for_llm({"items": [], "error": "query 不能为空"})

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
        message = [{"role": "system", "content": hyde_prompt},{"role": "user", "content": f"用户问题：{q}"}]
        response = pipe(message, max_new_tokens=256, do_sample=False)
        hyde_raw = response[0]['generated_text'][-1]["content"]
        # 防止 HyDE 丢失原问题信息：把原 query 与扩展短语拼在一起做向量检索
        hyde_q = f"{q} {str(hyde_raw or '').strip()}".strip()
    else:
        hyde_q = q

    topk = int(topk) if topk is not None else 5
    topk = max(1, min(20, topk))

    name = str(db_name or "").strip()
    if name not in ("law", "case", "law_big","law_huge"):
        return format_observation_for_llm({"items": [], "error": "db_name 必须是 law 或 case"})

    searcher = _get_searcher(name)
    recall_k = max(1, topk * _RECALL_FACTOR)

    # import time
    # time_start = time.perf_counter()
    reranked = searcher.search_advanced(
        rerank_query=q,
        bm25_query=q,
        emb_query_text=hyde_q,
        top_n=topk,
        recall_k=recall_k,
        fusion_method="rrf",
        rrf_k=_RRF_K,
        bm25_weight=_BM25_WEIGHT,
        emb_weight=_EMB_WEIGHT,
        k_percent=0.8,
    )
    # time_end = time.perf_counter()
    # print("total search time:", (time_end - time_start))

    items: List[Dict[str, Any]] = []
    for i, (score, item) in enumerate(reranked, start=1):
        if isinstance(item, dict):
            text = str(item.get("text") or "")
            meta = item.get("meta") or {}
            chunk_id = item.get("id") or ""
        else:
            text = str(item or "")
            meta = {}
            chunk_id = ""
        items.append(
            {
                "rank": i,
                "score": float(score),
                "id": str(chunk_id),
                "text": text,
                "meta": meta if isinstance(meta, dict) else {},
            }
        )

    return format_observation_for_llm({"query": q, "topk": topk, "db_name": name, "items": items})

def __main__():
    # 示例调用
    load_dotenv()
    while True:
        user_query = input("请输入检索问题（输入exit退出）: ")
        if user_query.lower() == "exit":
            print("程序结束")
            break
        user_use_db_name = input("请输入数据库名称（law或case）: ")
        user_topk = input("请输入返回条数（1~20，默认5）: ")
        try:
            user_topk_int = int(user_topk)
        except ValueError:
            user_topk_int = 5
    
        tool_input = {
            "query": user_query,
            "topk": user_topk_int,
            "db_name": user_use_db_name,
            "is_hyde": False,
        }
        response = rag_search.invoke(tool_input)
        print("RAG Search Response:")
        print(response)
    
    # tool_input = {
    #     "query": "酒驾违法吗",
    #     "topk": 5,
    #     "db_name": "case",
    #     "is_hyde": False,
    # }
    # response = rag_search.invoke(tool_input)
    # print("RAG Search Response:")
    # print(response)

if __name__ == "__main__":
    __main__()
