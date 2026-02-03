from __future__ import annotations

import re
from typing import Any, Dict, List, Union

from ..utils import make_chunk_id, make_doc_id


_RE_LAW_ENUM = re.compile(r"^\s*[（(][一二三四五六七八九十百千0-9]+[)）]\s*")
_CASE_DEFAULT_EMBED_SECTIONS = ("基本案情", "裁判理由")


def _is_law_doc(meta: Dict[str, Any]) -> bool:
    if not isinstance(meta, dict):
        return False
    return bool(meta.get("law") or meta.get("article") or meta.get("book") or meta.get("chapter"))


def _is_case_pdf_doc(meta: Dict[str, Any]) -> bool:
    if not isinstance(meta, dict):
        return False
    if meta.get("pdf_mode") == "case":
        return True
    if meta.get("case_paragraphs"):
        return True
    return False


def _law_index_prefix(meta: Dict[str, Any]) -> str:
    law = str(meta.get("law") or "").strip()
    book = str(meta.get("book") or "").strip()
    chapter = str(meta.get("chapter") or "").strip()
    section = str(meta.get("section") or "").strip() or "未分节"
    article = str(meta.get("article") or "").strip()

    # 常见写法：用户会输入“刑法/宪法/民法典”而不带“中华人民共和国”
    alias = ""
    if law.startswith("中华人民共和国") and len(law) > len("中华人民共和国"):
        alias = law[len("中华人民共和国") :].strip()

    parts = []
    if law:
        parts.append(f"《{law}》")
    if alias and alias != law:
        parts.append(f"（简称：{alias}）")
    for p in (book, chapter, section, article):
        if p:
            parts.append(p)
    return " ".join(parts).strip()


def _merge_law_sentences(
    sents: List[str],
    *,
    max_chars: int,
    min_chars: int = 120,
) -> List[str]:
    """
    针对法条文本做“条文内合并”：
    - 类似“（一）/（二）…”的枚举项不应单独成块，优先合并到同一 chunk
    - 以“：”结尾的引导句与后续枚举项优先同块
    - chunk 尽量达到 min_chars，但不超过 max_chars
    """
    max_chars = max(1, int(max_chars))
    min_chars = max(1, int(min_chars))

    out: List[str] = []
    buf: List[str] = []

    def buf_len() -> int:
        return sum(len(x) for x in buf) + max(0, len(buf) - 1)

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        text = "\n".join([x.strip() for x in buf if x and x.strip()]).strip()
        if text:
            out.append(text)
        buf = []

    for sent in [s.strip() for s in (sents or []) if s and s.strip()]:
        # 新块起始条件：当前 buffer 已经够长且再加会太长
        cur_len = buf_len()
        if buf and cur_len >= min_chars and (cur_len + 1 + len(sent)) > max_chars:
            flush()

        # 默认追加到当前 buffer
        buf.append(sent)

        # 如果遇到“引导句：”，先别急着 flush，等待至少一个枚举项进来
        if sent.endswith("：") or sent.endswith(":"):
            continue

        # 当 buffer 达到一定长度时，可以考虑 flush，但枚举项不要被拆散得过碎
        if buf_len() >= max_chars:
            flush()

    flush()
    return out


def _chunk_case_pdf_doc(
    *,
    doc_id: str,
    meta: Dict[str, Any],
    sent_split_model: Any,
    min_chunk_len: int,
) -> List[Dict[str, Any]]:
    base_meta = dict(meta or {})
    paragraphs = base_meta.pop("case_paragraphs", None) or []
    case_title = str(base_meta.get("case_title") or "").strip()

    embed_sections = base_meta.get("case_embed_sections")
    if isinstance(embed_sections, (list, tuple, set)):
        embed_sections_set = {str(x).strip() for x in embed_sections if str(x).strip()}
    else:
        embed_sections_set = set(_CASE_DEFAULT_EMBED_SECTIONS)

    # 生成“文本单元”：默认以段落为单位；若段落过长，则使用 splitter 切句后再打包
    max_chars = int(base_meta.get("case_chunk_max_chars") or 1200)
    max_chars = max(200, max_chars)
    overlap_units = int(base_meta.get("case_chunk_overlap_units") or 1)
    overlap_units = max(0, overlap_units)

    units: List[Dict[str, Any]] = []
    for p in paragraphs:
        if not isinstance(p, dict):
            continue
        section = str(p.get("section") or "").strip()
        if section and section not in embed_sections_set:
            continue
        text = str(p.get("text") or "").strip()
        if not text:
            continue
        para_index = int(p.get("para_index") or 0)
        page = int(p.get("page") or 0)

        if len(text) > max_chars:
            # 过长段落：先切句，再重新打包（避免 embedding 被截断）
            sents = [s.strip() for s in (sent_split_model.split_text(text) or []) if s and s.strip()]
            buf: List[str] = []
            buf_len = 0
            for s in sents:
                s_len = len(s) + (1 if buf else 0)
                if buf and (buf_len + s_len) > max_chars:
                    units.append({"section": section, "para_index": para_index, "page": page, "text": "\n".join(buf)})
                    buf = []
                    buf_len = 0
                buf.append(s)
                buf_len += s_len
            if buf:
                units.append({"section": section, "para_index": para_index, "page": page, "text": "\n".join(buf)})
        else:
            units.append({"section": section, "para_index": para_index, "page": page, "text": text})

    if not units:
        return []

    out: List[Dict[str, Any]] = []
    chunk_index = 0
    start = 0
    while start < len(units):
        buf_units: List[Dict[str, Any]] = []
        buf_len = 0
        end = start
        while end < len(units):
            t = str(units[end].get("text") or "")
            add_len = len(t) + (1 if buf_units else 0)
            if buf_units and (buf_len + add_len) > max_chars:
                break
            buf_units.append(units[end])
            buf_len += add_len
            end += 1

        if not buf_units:
            # 极端情况：单个 unit 超过 max_chars，但上面应该已经切过；这里兜底避免死循环
            buf_units = [units[start]]
            end = start + 1

        chunk_text = "\n".join([str(u.get("text") or "").strip() for u in buf_units if str(u.get("text") or "").strip()]).strip()
        if chunk_text and len(chunk_text) >= int(min_chunk_len):
            pages = [int(u.get("page") or 0) for u in buf_units if int(u.get("page") or 0) > 0]
            para_ids = [int(u.get("para_index") or 0) for u in buf_units if int(u.get("para_index") or 0) > 0]
            sections = [str(u.get("section") or "").strip() for u in buf_units if str(u.get("section") or "").strip()]
            # 去重但保序
            uniq_sections: List[str] = []
            seen = set()
            for s in sections:
                if s not in seen:
                    uniq_sections.append(s)
                    seen.add(s)

            out_meta = dict(base_meta)
            out_meta["chunk_index"] = chunk_index
            if pages:
                out_meta["page_start"] = min(pages)
                out_meta["page_end"] = max(pages)
            if para_ids:
                out_meta["case_para_start"] = min(para_ids)
                out_meta["case_para_end"] = max(para_ids)
            if uniq_sections:
                out_meta["case_sections"] = uniq_sections

            chunk_id = make_chunk_id(doc_id=doc_id, chunk_index=chunk_index)
            chunk: Dict[str, Any] = {"id": chunk_id, "text": chunk_text, "meta": out_meta}

            if case_title:
                # 索引增强：标题 + 分节名（若有）+ 正文
                sec = uniq_sections[0] if len(uniq_sections) == 1 else "案例"
                prefix = f"{case_title} [{sec}]".strip()
                chunk["index_text"] = (prefix + "\n" + chunk_text).strip()

            out.append(chunk)
            chunk_index += 1

        # 下一轮：按 overlap 回退
        if end <= start:
            start += 1
        else:
            start = max(end - overlap_units, start + 1) if overlap_units > 0 else end

    return out


def chunk_doc_item(
    doc_item: Union[str, Dict[str, Any]],
    sent_split_model: Any,
    *,
    min_chunk_len: int,
) -> List[Dict[str, Any]]:
    if isinstance(doc_item, dict):
        text = (doc_item.get("text") or "").strip()
        meta = doc_item.get("meta") or {}
        doc_id = str(doc_item.get("id") or meta.get("doc_id") or "")
        source_path = meta.get("source_path", "")
        page = int(meta.get("page") or 0)
        if not doc_id:
            doc_id = make_doc_id(
                source_path=source_path,
                page=page,
                record_index=int(meta.get("record_index") or 0),
            )
    else:
        text = (doc_item or "").strip()
        meta = {"source_path": ""}
        doc_id = make_doc_id(source_path="", page=0, record_index=0)

    if not text:
        return []

    # 案例 PDF：按“段落打包”切块，sentence splitter 仅用于过长段落的兜底切句
    if _is_case_pdf_doc(meta):
        return _chunk_case_pdf_doc(doc_id=doc_id, meta=meta, sent_split_model=sent_split_model, min_chunk_len=min_chunk_len)

    sent_res = sent_split_model.split_text(text)
    sent_res = [s for s in sent_res if s and s.strip()]

    # 法律条文：先做合并再按 min_chunk_len 过滤，避免“（一）（二）”这类短句被单独成块
    if _is_law_doc(meta):
        sent_res = _merge_law_sentences(sent_res, max_chars=getattr(sent_split_model, "sentence_size", 512))

    sent_res = [s for s in sent_res if s and len(s) >= int(min_chunk_len)]

    out: List[Dict[str, Any]] = []
    law_prefix = _law_index_prefix(meta) if _is_law_doc(meta) else ""
    for idx, sent in enumerate(sent_res):
        out_meta = dict(meta)
        out_meta["chunk_index"] = idx
        chunk_id = make_chunk_id(doc_id=doc_id, chunk_index=idx)
        chunk: Dict[str, Any] = {"id": chunk_id, "text": sent, "meta": out_meta}
        if law_prefix:
            chunk["index_text"] = (law_prefix + "\n" + sent).strip()
        out.append(chunk)
    return out
