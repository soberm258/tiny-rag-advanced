from __future__ import annotations

from pathlib import Path
import re
from typing import List


def read_pdf_pages(path: Path) -> List[str]:
    import fitz  # PyMuPDF

    pdf_doc: fitz.Document = fitz.open(str(path))
    pages: List[str] = []
    for page in pdf_doc:
        text = page.get_text("text") or ""
        text = text.replace("\r\n", "\n")
        pages.append(text)
    return pages
