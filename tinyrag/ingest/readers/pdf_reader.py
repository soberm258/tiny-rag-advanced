from __future__ import annotations

from pathlib import Path
import re
from typing import List


def read_pdf_pages(path: Path) -> List[str]:
    import fitz  # PyMuPDF

    pdf_doc: fitz.Document = fitz.open(str(path))
    pages: List[str] = []
    for page in pdf_doc:
        # 读取该页“可复制文本层”（不是 OCR）。
        # 注意：如果 PDF 本身是扫描件/图片，get_text("text") 可能返回空或非常少的内容。
        text = page.get_text("text") or ""
        text = text.replace("\r\n", "\n")
        pages.append(text)
    return pages
