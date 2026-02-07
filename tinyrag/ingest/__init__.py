from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from ..logging_utils import logger

from ..utils import make_doc_id
from .readers.common import read_text_file
from .readers.docx_reader import read_docx_to_text
from .readers.json_reader import extract_texts_from_json_obj, read_json_file, read_jsonl_file
from .readers.md_reader import read_md_file_to_text
from .readers.pdf_reader import read_pdf_pages
from .readers.pptx_reader import read_pptx_to_text
from .structured.case_pdf import detect_case_pdf_like, read_case_pdf_paragraphs
from .structured.law_cn_civil_code import detect_cn_law_like, parse_cn_law_text


def load_docs_for_build(
    input_path: Union[str, Path],
    *,
    json_text_key: str = "completion",
    recursive: bool = True,
    suffix_allowlist: Optional[Iterable[str]] = None,
    txt_mode: str = "auto",
    pdf_mode: str = "auto",
) -> List[Dict[str, Any]]:
    """
    将输入（文件或目录）读取为 build 可用的文档列表。
    返回值为 List[doc]，每个元素是：
      {"text": str, "meta": {...}}
    其中 meta 至少包含 source_path，PDF 还会包含 page；法律条文会包含 book/chapter/section/article 等定位字段。
    """
    input_path = Path(str(input_path))
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在：{input_path}")

    allow = None
    if suffix_allowlist is not None:
        allow = {s.lower().lstrip(".") for s in suffix_allowlist}

    def allow_suffix(path: Path) -> bool:
        if allow is None:
            return True
        return path.suffix.lower().lstrip(".") in allow

    files: List[Path] = []
    if input_path.is_dir():
        it = input_path.rglob("*") if recursive else input_path.glob("*")
        for p in it:
            if p.is_file() and allow_suffix(p):
                files.append(p)
    else:
        files = [input_path]

    docs: List[Dict[str, Any]] = []
    for file_path in files:
        suffix = file_path.suffix.lower().lstrip(".")
        if not suffix:
            continue

        try:
            if suffix == "pdf":
                mode = (pdf_mode or "auto").lower().strip()
                if mode not in ("auto", "pages", "case"):
                    mode = "auto"

                is_case = False
                if mode == "case":
                    is_case = True
                elif mode == "auto":
                    # 约定：路径包含 case/案例 的优先按案例 PDF 解析；否则尝试轻量探测
                    parts = {p.lower() for p in file_path.parts}
                    is_case = ("case" in parts) or any("案例" in p for p in file_path.parts) or detect_case_pdf_like(file_path)

                if is_case:
                    parsed = read_case_pdf_paragraphs(file_path)
                    meta = {
                        "source_path": str(file_path),
                        "type": "pdf",
                        "pdf_mode": "case",
                        "case_title": str(parsed.get("case_title") or "").strip(),
                        "case_degraded": bool(parsed.get("degraded")),
                        # 默认只嵌入关键章节：可在调用方覆盖
                        "case_embed_sections": ["基本案情", "裁判理由"],
                        # 结构化段落列表：每条包含 section/page/para_index/text
                        # 后续切块不会直接用当前 doc 的 text，而是从这里按章节挑段落再打包成 chunk
                        "case_paragraphs": list(parsed.get("paragraphs") or []),
                    }
                    doc_id = make_doc_id(source_path=str(file_path), page=0, record_index=0)
                    meta["doc_id"] = doc_id
                    # 这里的 doc 是“容器/占位”：对案例库 PDF，真正入库的是 chunking 阶段从 case_paragraphs 聚合出来的 chunk。
                    # 因此 text 只用于保证非空与便于调试（标题比 doc_id 更可读）。
                    docs.append({"text": meta["case_title"] or doc_id, "meta": meta})
                else:
                    pages = read_pdf_pages(file_path)
                    for idx, t in enumerate(pages, start=1):
                        meta = {"source_path": str(file_path), "page": idx, "type": "pdf"}
                        doc_id = make_doc_id(source_path=str(file_path), page=idx, record_index=0)
                        meta["doc_id"] = doc_id
                        # 普通 PDF 模式：每页作为一个 doc，后续会再切句/切块。
                        docs.append({"text": t, "meta": meta})

            elif suffix == "txt":
                text = read_text_file(file_path)
                if not text.strip():
                    continue

                mode = (txt_mode or "auto").lower().strip()
                if mode not in ("auto", "plain", "law"):
                    mode = "auto"

                if mode in ("law", "auto") and detect_cn_law_like(text):
                    parsed = parse_cn_law_text(text, source_path=str(file_path))
                    for d in parsed:
                        meta = d.get("meta") or {}
                        if isinstance(meta, dict):
                            meta["doc_id"] = make_doc_id(
                                source_path=str(file_path),
                                page=0,
                                record_index=int(meta.get("record_index") or 0),
                            )
                            d["meta"] = meta
                    docs.extend(parsed)
                else:
                    meta = {"source_path": str(file_path), "type": "txt"}
                    doc_id = make_doc_id(source_path=str(file_path), page=0, record_index=0)
                    meta["doc_id"] = doc_id
                    docs.append({"text": text, "meta": meta})

            elif suffix == "md":
                text = read_md_file_to_text(file_path)
                if text.strip():
                    meta = {"source_path": str(file_path), "type": "md"}
                    doc_id = make_doc_id(source_path=str(file_path), page=0, record_index=0)
                    meta["doc_id"] = doc_id
                    docs.append({"text": text, "meta": meta})

            elif suffix == "docx":
                text = read_docx_to_text(file_path)
                if text.strip():
                    meta = {"source_path": str(file_path), "type": "docx"}
                    doc_id = make_doc_id(source_path=str(file_path), page=0, record_index=0)
                    meta["doc_id"] = doc_id
                    docs.append({"text": text, "meta": meta})

            elif suffix == "pptx":
                text = read_pptx_to_text(file_path)
                if text.strip():
                    meta = {"source_path": str(file_path), "type": "pptx"}
                    doc_id = make_doc_id(source_path=str(file_path), page=0, record_index=0)
                    meta["doc_id"] = doc_id
                    docs.append({"text": text, "meta": meta})

            elif suffix in ("json", "jsonl"):
                if suffix == "json":
                    obj = read_json_file(file_path)
                else:
                    obj = read_jsonl_file(file_path)
                texts = extract_texts_from_json_obj(obj, text_key=json_text_key)
                for idx, t in enumerate([x for x in texts if x and str(x).strip()]):
                    meta = {
                        "source_path": str(file_path),
                        "type": suffix,
                        "record_index": idx,
                        "text_key": json_text_key,
                    }
                    doc_id = make_doc_id(source_path=str(file_path), page=0, record_index=idx)
                    meta["doc_id"] = doc_id
                    docs.append({"text": str(t), "meta": meta})

            else:
                logger.warning("不支持的文件类型，已跳过：{}", str(file_path))

        except Exception as e:
            logger.error("读取失败：{}，错误：{}", str(file_path), str(e))
            raise

    return docs


__all__ = [
    "load_docs_for_build",
]
