from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional


EvalTag = Literal["semantic", "locate"]


@dataclass(frozen=True)
class LawGold:
    source_path: str
    law: str
    article: str

    @staticmethod
    def from_meta(meta: Dict[str, Any]) -> "LawGold":
        return LawGold(
            source_path=str(meta.get("source_path") or "").strip(),
            law=str(meta.get("law") or "").strip(),
            article=str(meta.get("article") or "").strip(),
        )

    def is_valid(self) -> bool:
        return bool(self.law and self.article)


@dataclass(frozen=True)
class LawEvalSample:
    """
    law 检索评测样本。

    设计目标：
    1) 每条样本都有可自动判定的 gold（不依赖 LLM 打分）。
    2) semantic 样本尽量不泄漏条号，不复制原文长片段，主要评估“语义召回能力”。
    3) locate 样本允许条号定位，主要评估“系统健康度/字面检索上限”。
    """

    sample_id: str
    db_name: Literal["law"]
    tag: EvalTag
    query: str
    gold: LawGold
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "db_name": self.db_name,
            "tag": self.tag,
            "query": self.query,
            "gold": {
                "source_path": self.gold.source_path,
                "law": self.gold.law,
                "article": self.gold.article,
            },
            "note": self.note or "",
        }

    @staticmethod
    def from_dict(obj: Dict[str, Any]) -> "LawEvalSample":
        gold = obj.get("gold") or {}
        return LawEvalSample(
            sample_id=str(obj.get("sample_id") or "").strip(),
            db_name="law",
            tag=str(obj.get("tag") or "semantic").strip(),  # type: ignore[assignment]
            query=str(obj.get("query") or "").strip(),
            gold=LawGold(
                source_path=str(gold.get("source_path") or "").strip(),
                law=str(gold.get("law") or "").strip(),
                article=str(gold.get("article") or "").strip(),
            ),
            note=str(obj.get("note") or "").strip() or None,
        )


def validate_samples(samples: List[LawEvalSample]) -> List[str]:
    errs: List[str] = []
    seen_ids: set[str] = set()
    for s in samples:
        if not s.sample_id:
            errs.append("存在 sample_id 为空的样本")
            continue
        if s.sample_id in seen_ids:
            errs.append(f"sample_id 重复：{s.sample_id}")
        seen_ids.add(s.sample_id)
        if not s.query:
            errs.append(f"{s.sample_id}: query 为空")
        if s.db_name != "law":
            errs.append(f"{s.sample_id}: db_name 非 law")
        if s.tag not in ("semantic", "locate"):
            errs.append(f"{s.sample_id}: tag 非法：{s.tag}")
        if not s.gold.is_valid():
            errs.append(f"{s.sample_id}: gold 无效（law/article 为空）")
    return errs

