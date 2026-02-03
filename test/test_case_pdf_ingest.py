import sys

sys.path.append(".")


def test_case_pdf_ingest_and_chunking_should_work() -> None:
    try:
        import fitz  # noqa: F401
    except ModuleNotFoundError:
        # 环境缺少 PyMuPDF 时跳过（该项目在 conda tr 环境中应可用）
        return

    from tinyrag.ingest import load_docs_for_build
    from tinyrag.rag.chunking import chunk_doc_item
    from tinyrag.sentence_splitter import SentenceSplitter

    pdf_path = "data/raw_data/case/杨某海交通肇事案.pdf"
    docs = load_docs_for_build(pdf_path, recursive=False, pdf_mode="case")
    assert docs, "应当至少解析出一个案例文档"
    doc = docs[0]
    meta = doc.get("meta") or {}
    assert meta.get("pdf_mode") == "case"
    assert meta.get("case_paragraphs"), "应当包含按分节解析的段落列表"

    splitter = SentenceSplitter(use_model=False, sentence_size=2048)
    chunks = chunk_doc_item(doc, splitter, min_chunk_len=20)
    assert chunks, "应当生成至少一个 chunk"

    # 至少应包含“基本案情/裁判理由”中的一种
    all_sections = []
    for c in chunks:
        m = c.get("meta") or {}
        all_sections.extend(m.get("case_sections") or [])
    assert any(s in ("基本案情", "裁判理由") for s in all_sections)

    # 清洗后应出现该关键短语（用于粗略验证水印过滤有效）
    joined = "\n".join([c.get("text") or "" for c in chunks])
    assert "公安机关处理" in joined


if __name__ == "__main__":
    test_case_pdf_ingest_and_chunking_should_work()
    print("ok")
