from __future__ import annotations

from typing import Any, Dict, List

"""
将检索结果格式化为适合回传给大模型的 Observation 文本。
"""
def _format_law_location(meta: Dict[str, Any]) -> str:
    law = str(meta.get("law") or "").strip()
    book = str(meta.get("book") or "").strip() or "未知编"
    chapter = str(meta.get("chapter") or "").strip() or "未知章"
    section = str(meta.get("section") or "").strip() or "未分节"
    article = str(meta.get("article") or "").strip() or "未知条"
    parts = [p for p in [law, book, chapter, section, article] if p]
    return " | ".join(parts)


def _is_case_chunk(meta: Dict[str, Any]) -> bool:
    return bool(meta.get("pdf_mode") == "case" or meta.get("case_title") or meta.get("case_para_start") or meta.get("case_para_end"))


def _format_case_source(meta: Dict[str, Any]) -> str:
    src = str(meta.get("source_path") or "").strip()
    title = str(meta.get("case_title") or "").strip()
    ps = meta.get("page_start")
    pe = meta.get("page_end")
    parts = [p for p in [src, title] if p]
    if ps and pe:
        parts.append(f"第{ps}~{pe}页")
    sections = meta.get("case_sections") or []
    if isinstance(sections, list) and sections:
        uniq: List[str] = []
        seen = set()
        for s in [str(x).strip() for x in sections if str(x).strip()]:
            if s not in seen:
                uniq.append(s)
                seen.add(s)
        if uniq:
            parts.append("章节=" + ",".join(uniq))
    return " | ".join(parts).strip() or (src or "未知来源")


def _expand_case_blocks(meta: Dict[str, Any], *, max_chars: int = 6000) -> str:
    """
    case 输出风格：把同一案例的关键章节整块返回。
    - 优先返回：基本案情 / 裁判理由 / 裁判要旨
    - 加缓存：避免同一个 PDF 被重复解析
    """
    source_path = str(meta.get("source_path") or "").strip()
    if not source_path:
        return ""

    cache: Dict[str, Dict[str, Any]] = getattr(_expand_case_blocks, "_cache", {})
    if not isinstance(cache, dict):
        cache = {}

    if source_path not in cache:
        try:
            from pathlib import Path

            from tinyrag.ingest.structured.case_pdf import read_case_pdf_sections

            cache[source_path] = read_case_pdf_sections(Path(source_path))
        except Exception:
            cache[source_path] = {}
        setattr(_expand_case_blocks, "_cache", cache)

    data = cache.get(source_path) or {}
    title = str(data.get("case_title") or meta.get("case_title") or "").strip()
    secs = data.get("sections") or {}
    if not isinstance(secs, dict):
        secs = {}

    def sec_block(name: str) -> str:
        body = str(secs.get(name) or "").strip()
        if not body:
            return ""
        return f"【{name}】\n{body}".strip()

    blocks = [b for b in [sec_block("基本案情"), sec_block("裁判理由"), sec_block("裁判要旨")] if b]
    if not blocks:
        return ""

    text = ((title + "\n") if title else "") + "\n\n".join(blocks)
    text = text.strip()
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…（已截断）"
    return text


def _format_source(meta: Dict[str, Any]) -> str:
    source_path = str(meta.get("source_path") or "").strip()
    page = meta.get("page", None)
    if meta.get("law") or meta.get("article") or meta.get("book") or meta.get("chapter"):
        loc = _format_law_location(meta)
        return f"{source_path} | {loc}" if source_path else loc
    if _is_case_chunk(meta):
        return _format_case_source(meta)
    if source_path:
        if page:
            return f"{source_path} 第{page}页"
        return source_path
    return "未知来源"


def format_observation_for_llm(result: Dict[str, Any], *, max_chars_per_item: int = 500) -> str:
    """
    将检索结果格式化为适合回传给大模型的 Observation 文本。

    约定输入结构（与 rag_search 工具一致）：
      {
        "items": [{"rank": 1, "text": "...", "meta": {...}}, ...],
        "error": "..."（可选）
      }
    """
    items = result.get("items") or []
    err = result.get("error")
    if not isinstance(items, list):
        items = []

    lines: List[str] = []
    if err:
        lines.append(f"error={err}")

    # law：每个 item 一条证据 + 一条 source
    # case：按“案例维度”去重，并尽量输出关键章节整块
    display_rank = 0
    seen_case_sources: set[str] = set()
    for it in items:
        if not isinstance(it, dict):
            continue
        meta = it.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}
        text = str(it.get("text") or "").strip()
        if not text:
            continue

        if _is_case_chunk(meta):
            source_path = str(meta.get("source_path") or "").strip()
            if source_path and source_path in seen_case_sources:
                continue
            if source_path:
                seen_case_sources.add(source_path)

            expanded = _expand_case_blocks(meta)
            display_rank += 1
            if expanded:
                lines.append(f"[{display_rank}] {expanded}")
            else:
                t = text.replace("\r\n", "\n").replace("\r", "\n")
                t = " ".join([x.strip() for x in t.splitlines() if x.strip()]).strip()
                if max_chars_per_item and len(t) > max_chars_per_item:
                    t = t[:max_chars_per_item].rstrip() + "..."
                lines.append(f"[{display_rank}] {t}")
            lines.append(f"source={_format_source(meta)}")
            continue

        display_rank += 1
        t = text.replace("\r\n", "\n").replace("\r", "\n")
        t = " ".join([x.strip() for x in t.splitlines() if x.strip()]).strip()
        if max_chars_per_item and len(t) > max_chars_per_item:
            t = t[:max_chars_per_item].rstrip() + "..."
        lines.append(f"[{display_rank}] {t}")
        lines.append(f"source={_format_source(meta)}")

    if not items and not err:
        lines.append("（无结果）")

    return "\n".join(lines).strip()
