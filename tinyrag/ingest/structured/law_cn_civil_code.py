from __future__ import annotations

import re
from typing import Any, Dict, List


_CN_NUM = r"一二三四五六七八九十百千零〇两0-9"
_RE_BOOK = re.compile(rf"^\s*第([{_CN_NUM}]+)编\s*(.+)?\s*$")
_RE_CHAPTER = re.compile(rf"^\s*第([{_CN_NUM}]+)章\s*(.+)?\s*$")
_RE_SECTION = re.compile(rf"^\s*第([{_CN_NUM}]+)节\s*(.+)?\s*$")
_RE_ARTICLE = re.compile(rf"^\s*第([{_CN_NUM}]+)条\s*(.*)\s*$")
_RE_LAW_TITLE_LINE = re.compile(r"^\s*(中华人民共和国.{0,40}?(?:法典|宪法|法|条例|规定|解释))\s*$")


def _compact_cjk_spaces(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    prev = None
    cur = text
    while prev != cur:
        prev = cur
        cur = re.sub(r"([\u4e00-\u9fff])\s+([\u4e00-\u9fff])", r"\1\2", cur)
    cur = re.sub(r"\s+", " ", cur).strip()
    return cur


def _compact_title(text: str) -> str:
    return _compact_cjk_spaces(text or "")


def infer_law_title(*, text: str, source_path: str = "") -> str:
    """
    推断法律名称：
    1) 优先从正文开头若干行中抽取形如“中华人民共和国XX法/法典/宪法...”的标题行
    2) 回退到文件名（不含后缀）
    3) 再回退到“法律文本”
    """
    text = text or ""
    head_lines = text.splitlines()[:60]
    for line in head_lines:
        m = _RE_LAW_TITLE_LINE.match((line or "").strip())
        if m:
            return _compact_cjk_spaces(m.group(1))

    if source_path:
        try:
            import os

            base = os.path.basename(source_path)
            name, _ = os.path.splitext(base)
            name = _compact_cjk_spaces(name)
            if name:
                return name
        except Exception:
            pass
    return "法律文本"


def detect_cn_law_like(text: str) -> bool:
    text = text or ""
    # 防止误判：至少需要出现较多条文
    article_hits = len(re.findall(rf"第[{_CN_NUM}]+条", text))
    if article_hits < 50:
        return False
    # 法律文本通常包含“中华人民共和国XX法/宪法”等标题；也允许只出现“宪法”关键字
    if ("中华人民共和国" not in text) and ("宪法" not in text) and ("刑法" not in text) and ("民法典" not in text) and ("法" not in text):
        return False
    return True


def parse_cn_law_text(text: str, *, source_path: str) -> List[Dict[str, Any]]:
    lines = (text or "").splitlines()

    law_title = infer_law_title(text=text, source_path=source_path)
    cur_book = ""
    cur_chapter = ""
    cur_section = ""

    docs: List[Dict[str, Any]] = []
    article_num = 0
    cur_article = ""
    buf: List[str] = []

    def flush() -> None:
        nonlocal article_num, cur_article, buf
        if not cur_article:
            return
        body = "\n".join([x for x in buf if x]).strip()
        if not body:
            cur_article = ""
            buf = []
            return
        meta = {
            "source_path": source_path,
            "type": "txt",
            "law": law_title,
            "book": cur_book,
            "chapter": cur_chapter,
            "section": cur_section,
            "article": cur_article,
            "record_index": article_num,
        }
        docs.append({"text": body, "meta": meta})
        cur_article = ""
        buf = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        # 目录/排版常有全角空格与多空格，统一压缩
        line = _compact_cjk_spaces(line)
        if not line:
            continue

        m = _RE_BOOK.match(line)
        if m:
            title = _compact_title(m.group(2) or "")
            cur_book = f"第{m.group(1)}编 {title}".strip()
            cur_chapter = ""
            cur_section = ""
            continue
        m = _RE_CHAPTER.match(line)
        if m:
            title = _compact_title(m.group(2) or "")
            cur_chapter = f"第{m.group(1)}章 {title}".strip()
            cur_section = ""
            continue
        m = _RE_SECTION.match(line)
        if m:
            title = _compact_title(m.group(2) or "")
            cur_section = f"第{m.group(1)}节 {title}".strip()
            continue

        m = _RE_ARTICLE.match(line)
        if m:
            flush()
            article_num += 1
            cur_article = f"第{m.group(1)}条"
            rest = (m.group(2) or "").strip()
            buf = [f"{cur_article} {rest}".strip()] if rest else [cur_article]
            continue

        if cur_article:
            buf.append(line)

    flush()
    return docs


# 兼容旧接口命名
def detect_civil_code_like(text: str) -> bool:
    return detect_cn_law_like(text)


def parse_civil_code_text(text: str, *, source_path: str) -> List[Dict[str, Any]]:
    return parse_cn_law_text(text, source_path=source_path)
