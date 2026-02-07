from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from ..readers.pdf_reader import read_pdf_pages


_RE_PAGE_NUM = re.compile(r"第\s*\d+\s*页")
_RE_CASE_HEADING = re.compile(r"(关键词|基本案情|裁判理由|裁判要旨|关联索引|[一二三四]审：)")
_RE_CASE_PAGE_MARK = re.compile(r"^<<<PAGE:(\d+)>>>$")


def _clean_extracted_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = _RE_PAGE_NUM.sub("", text)
    text = text.replace("人民法院案例库", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_case_to_paragraphs(full_text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    将案例库 PDF 的全文（包含页标记）解析为：
    - case_title：标题/案号行（取首段直到第一个分节标题之前）
    - paragraphs：按分节组织的段落列表，每条包含 section/page/text/para_index
    """
    # 说明：
    # - 上游会插入形如 <<<PAGE:3>>> 的页标记，这里用它给段落打 page。
    # - headings 是案例库 PDF 常见分节标题；这里只做轻量规则切分，最终切块在 rag/chunking.py 完成。
    full_text = _RE_CASE_HEADING.sub(lambda m: "\n" + m.group(1) + "\n", full_text or "")
    lines = [ln.strip() for ln in (full_text or "").splitlines()]
    lines = [ln for ln in lines if ln]

    headings = {"关键词", "基本案情", "裁判理由", "裁判要旨", "关联索引", "一审：", "二审：", "三审：", "四审："}

    title_lines: List[str] = []
    for ln in lines:
        if ln in headings:
            break
        if not _RE_CASE_PAGE_MARK.match(ln):
            title_lines.append(ln)
    case_title = (title_lines[1] if title_lines else "").strip()

    paragraphs: List[Dict[str, Any]] = []
    current_section = ""
    current_page = 0
    para_index = 0
    for ln in lines:
        m = _RE_CASE_PAGE_MARK.match(ln)
        if m:
            # 更新“当前页码”，后续段落沿用该页码直到下一次页标记出现
            current_page = int(m.group(1))
            continue
        if ln in headings:
            # 更新“当前分节”，例如 基本案情/裁判理由 等；用于后续筛选嵌入章节
            current_section = ln.rstrip("：")
            continue
        para_index += 1
        paragraphs.append(
            {
                "para_index": para_index,
                "section": current_section,
                "page": current_page,
                "text": ln,
            }
        )
    return case_title, paragraphs


def detect_case_pdf_like(path: Path) -> bool:
    """
    轻量判断：抽取首页文本，看是否包含案例库常见结构分节标题。
    """
    try:
        pages = read_pdf_pages(path)
    except Exception:
        return False
    head = (pages[0] if pages else "")[:4000]
    return ("基本案情" in head) and ("裁判理由" in head or "裁判要旨" in head)


def read_case_pdf_paragraphs(
    path: Path,
    *,
    watermark_font_contains: str = "SimHei",
    watermark_min_size: float = 24.0,
    watermark_chars: Optional[Tuple[str, ...]] = ("人", "民", "法", "院", "案", "例", "库"),
) -> Dict[str, Any]:
    """
    读取“人民法院案例库”类 PDF，并做水印过滤与分节解析。

    返回：
      {"case_title": str, "paragraphs": List[{para_index, section, page, text}]}

    说明：
    - 优先使用 pdfplumber 获取 char 粒度信息进行水印过滤
    - 若缺少 pdfplumber，则退化为 PyMuPDF 抽文本并做分节切分（无法过滤水印）
    """
    try:
        import pdfplumber  # type: ignore
    except ModuleNotFoundError:
        pages = read_pdf_pages(path)
        marked: List[str] = []
        for i, t in enumerate(pages, start=1):
            marked.append(f"<<<PAGE:{i}>>>")
            marked.append(_clean_extracted_text(t))
        full_text = "\n".join([x for x in marked if x and x.strip()])
        case_title, paragraphs = _split_case_to_paragraphs(full_text)
        return {"case_title": case_title, "paragraphs": paragraphs, "degraded": True}

    def keep_obj(obj: Dict[str, Any]) -> bool:
        if obj.get("object_type") != "char":
            return True
        text = str(obj.get("text") or "")
        font = str(obj.get("fontname") or "")
        size = float(obj.get("size") or 0.0)
        if watermark_chars and text in set(watermark_chars) and size >= float(watermark_min_size) and watermark_font_contains in font:
            return False
        return True

    marked2: List[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            page = page.filter(keep_obj)
            page_text = _clean_extracted_text(page.extract_text() or "")
            if not page_text:
                continue
            marked2.append(f"<<<PAGE:{i}>>>")
            marked2.append(page_text)

    full_text = "\n".join(marked2)
    case_title, paragraphs = _split_case_to_paragraphs(full_text)
    return {"case_title": case_title, "paragraphs": paragraphs, "degraded": False}


def read_case_pdf_sections(
    path: Path,
    *,
    sections: Tuple[str, ...] = ("基本案情", "裁判理由", "裁判要旨"),
) -> Dict[str, Any]:
    """
    读取案例 PDF，并按章节聚合为整块文本，便于在“返回给大模型”阶段提供完整上下文。

    返回：
      {
        "case_title": str,
        "sections": {section_name: section_text},
        "degraded": bool,
      }
    """
    parsed = read_case_pdf_paragraphs(path)
    case_title = str(parsed.get("case_title") or "").strip()
    paras = parsed.get("paragraphs") or []

    want = {str(s).strip() for s in (sections or ()) if str(s).strip()}
    sec_texts: Dict[str, List[str]] = {s: [] for s in want}
    for p in paras:
        if not isinstance(p, dict):
            continue
        sec = str(p.get("section") or "").strip()
        if sec not in want:
            continue
        t = str(p.get("text") or "").strip()
        if t:
            sec_texts[sec].append(t)

    merged: Dict[str, str] = {}
    for sec in sections:
        sec = str(sec).strip()
        if not sec:
            continue
        merged[sec] = "\n".join(sec_texts.get(sec) or []).strip()

    return {"case_title": case_title, "sections": merged, "degraded": bool(parsed.get("degraded"))}


__all__ = [
    "detect_case_pdf_like",
    "read_case_pdf_paragraphs",
    "read_case_pdf_sections",
]
