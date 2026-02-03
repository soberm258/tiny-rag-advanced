from __future__ import annotations

from pathlib import Path

from .common import read_text_file


def read_md_file_to_text(path: Path) -> str:
    raw = read_text_file(path)
    try:
        import markdown
        from bs4 import BeautifulSoup

        html_content = markdown.markdown(raw)
        soup = BeautifulSoup(html_content, "html.parser")
        return soup.get_text()
    except Exception:
        return raw

