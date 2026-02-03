import re
from typing import List


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def split_sentences(text: str, *, sentence_size: int = 256, prefer_zh: bool = True) -> List[str]:
    """
    统一分句入口：
    - 中文（含CJK字符）默认走 SentenceSplitter 的规则切分
    - 非中文优先使用 nltk.sent_tokenize；若缺少资源/不可用则回退到 SentenceSplitter
    """
    text = (text or "").strip()
    if not text:
        return []

    if prefer_zh and _contains_cjk(text):
        from .sentence_splitter import SentenceSplitter

        splitter = SentenceSplitter(use_model=False, sentence_size=sentence_size)
        return [s for s in splitter.split_text(text) if s]

    try:
        from nltk.tokenize import sent_tokenize

        return [s for s in sent_tokenize(text) if s]
    except Exception:
        from .sentence_splitter import SentenceSplitter

        splitter = SentenceSplitter(use_model=False, sentence_size=sentence_size)
        return [s for s in splitter.split_text(text) if s]
