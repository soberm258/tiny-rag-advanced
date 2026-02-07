from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from functools import lru_cache
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from eval.law_eval_schema import LawEvalSample



@dataclass(frozen=True)
class Experiment:
    name: str
    topk: int = 5
    recall_factor: int = 4
    fusion_method: str = "rrf"
    rrf_k: int = 60
    bm25_weight: float = 1.0
    emb_weight: float = 1.0
    is_hyde: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "topk": self.topk,
            "recall_factor": self.recall_factor,
            "fusion_method": self.fusion_method,
            "rrf_k": self.rrf_k,
            "bm25_weight": self.bm25_weight,
            "emb_weight": self.emb_weight,
            "is_hyde": self.is_hyde,
        }


def _read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            yield json.loads(line)


def _load_samples(path: str) -> List[LawEvalSample]:
    samples: List[LawEvalSample] = []
    for obj in _read_jsonl(path):
        samples.append(LawEvalSample.from_dict(obj))
    return samples


def _match_law_article(item: Any, gold_law: str, gold_article: str) -> bool:
    if not isinstance(item, dict):
        return False
    meta = item.get("meta") or {}
    if not isinstance(meta, dict):
        return False
    law = str(meta.get("law") or "").strip()
    article = str(meta.get("article") or "").strip()
    return (law == gold_law) and (article == gold_article)


def _hyde_query(text: str) -> str:
    """
    与 script/rag_cli.py 保持一致的 HyDE 生成逻辑。
    注意：该路径会触发本地生成模型推理，评测耗时会明显增加。
    """
    pipe = _get_hyde_pipe()
    from tinyrag.rag.prompts import build_hyde_prompt
    hyde_prompt = build_hyde_prompt()
    message = [{"role": "system", "content": hyde_prompt}, {"role": "user", "content": f"用户问题：{text}"}]
    response = pipe(message, max_new_tokens=256, do_sample=False)
    return response[0]["generated_text"][-1]["content"]


@lru_cache(maxsize=1)
def _get_hyde_pipe() -> Any:
    """
    HyDE 生成模型缓存。
    不缓存会导致每条样本都重复加载模型，评测耗时会非常夸张。
    """
    from transformers import pipeline
    import torch

    return pipeline(
        "text-generation",
        model="./models/Qwen2-1.5B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )


def _safe_mean(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None]  # type: ignore[comparison-overlap]
    return float(statistics.mean(xs)) if xs else 0.0


def eval_one_experiment(
    *,
    exp: Experiment,
    searcher: Any,
    samples: List[LawEvalSample],
) -> Dict[str, Any]:
    try:
        from tqdm import tqdm  # type: ignore
    except ModuleNotFoundError:
        tqdm = None  # type: ignore[assignment]

    topk = max(1, int(exp.topk))
    recall_k = max(1, topk * max(1, int(exp.recall_factor)))

    hit_list: List[float] = []
    mrr_list: List[float] = []
    hyde_costs: List[float] = []
    search_costs: List[float] = []
    total_costs: List[float] = []

    by_tag: Dict[str, Dict[str, List[float]]] = {}

    it = samples
    if tqdm is not None:
        it = tqdm(
            samples,
            total=len(samples),
            desc=f"eval[{exp.name}]",
            unit="q",
            ascii=True,
            dynamic_ncols=True,
        )

    for s in it:  # type: ignore[assignment]
        q = (s.query or "").strip()
        if not q:
            continue

        t0 = time.perf_counter()
        if exp.is_hyde:
            th0 = time.perf_counter()
            hyde_raw = _hyde_query(q)
            th1 = time.perf_counter()
            hyde_costs.append(th1 - th0)
            #避免 HyDE 丢失原问题信息
            hyde_q = f"{q} {str(hyde_raw or '').strip()}".strip()
            # print(f"\nHyDE 扩展后：{hyde_q}\n", flush=True)
        else:
            hyde_q = q
            hyde_costs.append(0.0)
        t1 = time.perf_counter()

        reranked = searcher.search_advanced(
            rerank_query=q,
            bm25_query=q,
            emb_query_text=hyde_q,
            top_n=topk,
            recall_k=recall_k,
            fusion_method=str(exp.fusion_method or "rrf"),
            rrf_k=int(exp.rrf_k),
            bm25_weight=float(exp.bm25_weight),
            emb_weight=float(exp.emb_weight),
            k_percent=None,
        )
        t2 = time.perf_counter()

        search_costs.append(t2 - t1)
        total_costs.append(t2 - t0)

        gold_law = s.gold.law
        gold_article = s.gold.article

        first_hit_rank: Optional[int] = None
        for rank, (_score, item) in enumerate(reranked, start=1):
            if _match_law_article(item, gold_law, gold_article):
                first_hit_rank = rank
                break

        hit = 1.0 if first_hit_rank is not None else 0.0
        mrr = 1.0 / float(first_hit_rank) if first_hit_rank else 0.0

        hit_list.append(hit)
        mrr_list.append(mrr)

        tag = str(s.tag)
        if tag not in by_tag:
            by_tag[tag] = {"hit": [], "mrr": []}
        by_tag[tag]["hit"].append(hit)
        by_tag[tag]["mrr"].append(mrr)

        if tqdm is not None:
            try:
                # 只展示一个粗略平均，避免频繁更新带来额外开销
                mean_total = _safe_mean(total_costs)
                mean_search = _safe_mean(search_costs)
                mean_hyde = _safe_mean(hyde_costs)
                it.set_postfix(  # type: ignore[union-attr]
                    mean_total=f"{mean_total:.3f}s",
                    mean_search=f"{mean_search:.3f}s",
                    mean_hyde=f"{mean_hyde:.3f}s",
                )
            except Exception:
                pass

    report: Dict[str, Any] = {
        "experiment": exp.to_dict(),
        "n_samples": len(samples),
        "metrics": {
            "hit@k": _safe_mean(hit_list),
            "mrr@k": _safe_mean(mrr_list),
        },
        "timing_sec": {
            "hyde_mean": _safe_mean(hyde_costs),
            "search_mean": _safe_mean(search_costs),
            "total_mean": _safe_mean(total_costs),
        },
        "by_tag": {},
    }

    for tag, m in by_tag.items():
        report["by_tag"][tag] = {
            "hit@k": _safe_mean(m["hit"]),
            "mrr@k": _safe_mean(m["mrr"]),
            "n": len(m["hit"]),
        }

    return report


def _load_experiments(path: str) -> List[Experiment]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "experiments" in obj:
        obj = obj["experiments"]
    if not isinstance(obj, list):
        raise ValueError("实验配置文件应为 JSON 数组，或包含 experiments 数组的对象")

    exps: List[Experiment] = []
    for it in obj:
        if not isinstance(it, dict):
            continue
        exps.append(
            Experiment(
                name=str(it.get("name") or "").strip() or "exp",
                topk=int(it.get("topk") or 5),
                recall_factor=int(it.get("recall_factor") or 4),
                fusion_method=str(it.get("fusion_method") or "rrf"),
                rrf_k=int(it.get("rrf_k") or 60),
                bm25_weight=float(it.get("bm25_weight") or 1.0),
                emb_weight=float(it.get("emb_weight") or 1.0),
                is_hyde=bool(it.get("is_hyde") or False),
            )
        )
    if not exps:
        raise ValueError("未读取到任何实验配置")
    return exps


def _default_experiments() -> List[Experiment]:
    return [
        Experiment(name="bm25_only", fusion_method="rrf", bm25_weight=1.0, emb_weight=0.0),
        Experiment(name="emb_only", fusion_method="rrf", bm25_weight=0.0, emb_weight=1.0),
        Experiment(name="hyde_emb_only", fusion_method="rrf", bm25_weight=0.0, emb_weight=1.0, is_hyde=True),
        Experiment(name="dedup_default", fusion_method="dedup", bm25_weight=1.0, emb_weight=1.0),
        Experiment(name="rrf_default", fusion_method="rrf", bm25_weight=1.0, emb_weight=1.0),
        Experiment(name="hyde_rrf_default", fusion_method="rrf", bm25_weight=1.0, emb_weight=1.0, is_hyde=True),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="law 检索评测（只评检索，不跑最终回答生成）")
    parser.add_argument(
        "--eval-jsonl",
        type=str,
        default=os.path.join("eval", "law_eval_set.jsonl"),
        help="评测集 JSONL 路径（由 build_law_eval_set.py 生成）",
    )
    parser.add_argument("--db-dir", type=str, default=os.path.join("data", "db", "law"), help="数据库目录")
    parser.add_argument("--emb-model", type=str, default=os.path.join("models", "bge-base-zh-v1.5"))
    parser.add_argument("--rerank-model", type=str, default=os.path.join("models", "bge-reranker-base"))
    parser.add_argument("--device", type=str, default=os.getenv("TINYRAG_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--exp-json", type=str, default="", help="实验配置 JSON（可选）")
    parser.add_argument("--out-json", type=str, default=os.path.join("eval", "law_eval_report.json"))
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        help="日志级别：DEBUG/INFO/WARNING/ERROR（默认 WARNING，避免评测时刷屏）",
    )

    args = parser.parse_args()

    samples = _load_samples(str(args.eval_jsonl))
    if not samples:
        raise ValueError(f"评测集为空：{args.eval_jsonl}")

    if args.exp_json:
        exps = _load_experiments(str(args.exp_json))
    else:
        exps = _default_experiments()

    # 评测阶段默认关闭 INFO 刷屏（例如 Searcher.search_advanced 的 fusion 候选数日志）
    try:
        from tinyrag.logging_utils import logger

        logger.remove()
        logger.add(sys.stderr, level=str(args.log_level).upper().strip() or "WARNING")
    except Exception:
        pass

    # 延迟导入，避免仅做数据处理时引入 sentence-transformers/torch 等重依赖
    from tinyrag.searcher.searcher import Searcher

    searcher = Searcher(
        emb_model_id=str(args.emb_model),
        ranker_model_id=str(args.rerank_model),
        device=str(args.device),
        base_dir=str(args.db_dir),
    )
    searcher.load_db()

    reports: List[Dict[str, Any]] = []
    for exp in exps:
        rep = eval_one_experiment(
            exp=exp,
            searcher=searcher,
            samples=samples,
        )
        reports.append(rep)
        print(json.dumps(rep["experiment"], ensure_ascii=False), flush=True)
        print(json.dumps(rep["metrics"], ensure_ascii=False), flush=True)
        print(json.dumps(rep["timing_sec"], ensure_ascii=False), flush=True)

    out = {
        "eval_jsonl": str(args.eval_jsonl),
        "db_dir": str(args.db_dir),
        "reports": reports,
    }
    os.makedirs(os.path.dirname(str(args.out_json)) or ".", exist_ok=True)
    with open(str(args.out_json), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"eval_report_written={args.out_json}", flush=True)


if __name__ == "__main__":
    main()
