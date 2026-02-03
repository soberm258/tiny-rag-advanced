from __future__ import annotations

from pathlib import Path


def read_docx_to_text(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    lines = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(lines)

