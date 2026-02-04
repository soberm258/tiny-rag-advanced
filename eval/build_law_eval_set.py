from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

from eval.law_eval_schema import LawEvalSample, LawGold, validate_samples


_RE_ARTICLE_PREFIX = re.compile(r"^\s*第[一二三四五六七八九十百千零〇两0-9]+条\s*")
_RE_CJK = re.compile(r"[\u4e00-\u9fff]+")


def _read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            yield json.loads(line)


def _load_stopwords(path: str) -> set[str]:
    stop: set[str] = set()
    if not path or not os.path.isfile(path):
        return stop
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            w = (raw or "").strip()
            if w:
                stop.add(w)
    return stop


def _clean_text_for_query(text: str) -> str:
    t = (text or "").strip()
    t = _RE_ARTICLE_PREFIX.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _extract_keywords(text: str, *, stopwords: set[str], max_kw: int = 3) -> List[str]:
    """
    用最朴素的方式从法条正文抽关键词，避免引入复杂依赖，便于 debug。
    """
    t = _clean_text_for_query(text)
    if not t:
        return []

    # jieba 是项目依赖的一部分，但为了方便在最小环境中生成评测集，这里做一次可降级处理
    try:
        import jieba  # type: ignore

        tokens = list(jieba.cut_for_search(t))
    except ModuleNotFoundError:
        # 降级：按连续汉字片段切分，再做简单 n-gram，效果不如 jieba 但足够用于生成模板 query
        cjk_segs = _RE_CJK.findall(t)
        tokens = []
        for seg in cjk_segs:
            seg = seg.strip()
            if len(seg) <= 1:
                continue
            tokens.append(seg)
            # 追加 2-4 字 n-gram，避免整个长段落变成一个 token
            for n in (2, 3, 4):
                if len(seg) < n:
                    continue
                for i in range(0, min(len(seg) - n + 1, 20)):
                    tokens.append(seg[i : i + n])
    tokens = [x.strip() for x in tokens if x and x.strip()]
    tokens = [x for x in tokens if _RE_CJK.search(x)]
    tokens = [x for x in tokens if len(x) >= 2]
    tokens = [x for x in tokens if x not in stopwords]

    if not tokens:
        return []

    cnt = Counter(tokens)
    # 先按频率降序，再按首次出现位置升序，保证稳定可复现
    first_pos: Dict[str, int] = {}
    for idx, tok in enumerate(tokens):
        if tok not in first_pos:
            first_pos[tok] = idx

    sorted_toks = sorted(cnt.keys(), key=lambda k: (-cnt[k], first_pos.get(k, 10**9)))
    kws: List[str] = []
    for tok in sorted_toks:
        if tok not in kws:
            kws.append(tok)
        if len(kws) >= max_kw:
            break
    return kws


def _semantic_queries(text: str, *, kws: List[str]) -> List[str]:
    """
    生成不泄漏条号的语义型问题模板。
    这里故意写死模板，优先保证可控与可 debug。
    """
    base = "、".join(kws[:2]).strip("、") if kws else ""
    if not base:
        base = "相关事项"

    t = _clean_text_for_query(text)
    has_should = "应当" in t
    has_must_not = ("不得" in t) or ("禁止" in t)
    has_can = "可以" in t
    has_right = ("有权" in t) or ("权利" in t)
    has_resp = ("责任" in t) or ("承担" in t) or ("赔偿" in t) or ("免除" in t)

    q1 = f"关于{base}的法律规定是什么？"
    if has_resp:
        q1 = f"在{base}方面需要承担哪些责任，或者在什么条件下可以免除责任？"
    elif has_must_not:
        q1 = f"在{base}方面哪些行为是被禁止的，违反会有什么后果？"
    elif has_should:
        q1 = f"在{base}方面应当如何处理或者遵循什么原则？"
    elif has_can:
        q1 = f"在{base}情形下当事人可以怎么做，有哪些选择？"
    elif has_right:
        q1 = f"关于{base}涉及哪些权利，权利主体是谁？"

    q2 = f"请解释{base}的定义、适用范围以及常见边界情形。"
    q3 = f"当出现{base}相关争议时，通常依据哪些规则来判断和处理？"
    return [q1, q2, q3]


def _locate_queries(law: str, article: str) -> List[str]:
    law = (law or "").strip()
    article = (article or "").strip()
    if not (law and article):
        return []
    return [
        f"{law}{article}规定了什么？",
        f"{law}中{article}的核心内容是什么？",
    ]


def _group_by_article(items: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    以 (law, article) 聚合，取每条法条的一个代表 chunk。
    """
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for it in items:
        meta = it.get("meta") or {}
        if not isinstance(meta, dict):
            continue
        law = str(meta.get("law") or "").strip()
        article = str(meta.get("article") or "").strip()
        if not (law and article):
            continue
        key = (law, article)
        if key not in out:
            out[key] = it
    return out


def build_eval_set(
    *,
    split_jsonl: str,
    out_jsonl: str,
    max_articles: int,
    seed: int,
    semantic_per_article: int,
    include_locate: bool,
    stopwords_path: Optional[str],
) -> None:
    stopwords = _load_stopwords(stopwords_path or "")
    # 基础停用词兜底（避免 stopwords 文件缺失导致问题过于“虚”）
    stopwords |= {
        "以及",
        "或者",
        "以及其他",
        "根据",
        "规定",
        "法律",
        "本法",
        "本条",
        "当事人",
        "应当",
        "可以",
        "不得",
        "但是",
        "除外",
        "依照",
        "有关",
        "情形",
        "行为",
        "责任",
        "权利",
    }

    grouped = _group_by_article(_read_jsonl(split_jsonl))
    keys = list(grouped.keys())
    if not keys:
        raise ValueError(f"未从 {split_jsonl} 读取到任何可用法条 chunk（缺少 meta.law/meta.article？）")

    rnd = random.Random(seed)
    rnd.shuffle(keys)
    keys = keys[: max(1, min(len(keys), int(max_articles)))]

    samples: List[LawEvalSample] = []
    for idx, (law, article) in enumerate(keys, start=1):
        it = grouped[(law, article)]
        text = str(it.get("text") or "").strip()
        meta = it.get("meta") or {}
        gold = LawGold.from_meta(meta if isinstance(meta, dict) else {})

        kws = _extract_keywords(text, stopwords=stopwords, max_kw=3)
        sem_qs = _semantic_queries(text, kws=kws)[: max(1, int(semantic_per_article))]
        for j, q in enumerate(sem_qs, start=1):
            sample_id = f"law_sem_{idx:05d}_{j}"
            samples.append(
                LawEvalSample(
                    sample_id=sample_id,
                    db_name="law",
                    tag="semantic",
                    query=q,
                    gold=gold,
                    note=f"{law}{article}",
                )
            )

        if include_locate:
            for j, q in enumerate(_locate_queries(law, article)[:1], start=1):
                sample_id = f"law_loc_{idx:05d}_{j}"
                samples.append(
                    LawEvalSample(
                        sample_id=sample_id,
                        db_name="law",
                        tag="locate",
                        query=q,
                        gold=gold,
                        note=f"{law}{article}",
                    )
                )

    errs = validate_samples(samples)
    if errs:
        raise ValueError("评测集生成失败：" + "；".join(errs[:10]))

    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 law 检索评测集（JSONL）")
    parser.add_argument(
        "--split-jsonl",
        type=str,
        default=os.path.join("data", "db", "law", "split_sentence.jsonl"),
        help="建库产物 split_sentence.jsonl 路径",
    )
    parser.add_argument(
        "--out-jsonl",
        type=str,
        default=os.path.join("eval", "law_eval_set.jsonl"),
        help="输出评测集 JSONL 路径",
    )
    parser.add_argument("--max-articles", type=int, default=200, help="抽样的法条数量上限")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--semantic-per-article", type=int, default=2, help="每条法条生成的语义型问题数")
    parser.add_argument("--include-locate", action="store_true", default=False, help="是否额外生成定位型问题")
    parser.add_argument(
        "--stopwords",
        type=str,
        default=os.path.join("tinyrag", "searcher", "bm25_recall", "stopwords_hit.txt"),
        help="停用词文件路径（可选）",
    )

    args = parser.parse_args()
    build_eval_set(
        split_jsonl=str(args.split_jsonl),
        out_jsonl=str(args.out_jsonl),
        max_articles=int(args.max_articles),
        seed=int(args.seed),
        semantic_per_article=int(args.semantic_per_article),
        include_locate=bool(args.include_locate),
        stopwords_path=str(args.stopwords),
    )

    # 为避免不同 Windows 控制台编码导致中文乱码，这里只输出 ASCII 信息
    print(f"eval_set_written={args.out_jsonl}", flush=True)


if __name__ == "__main__":
    main()
