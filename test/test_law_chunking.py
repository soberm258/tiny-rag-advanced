import sys

sys.path.append(".")

from tinyrag.rag.chunking import chunk_doc_item
from tinyrag.sentence_splitter import SentenceSplitter


def test_law_article_enumeration_should_merge() -> None:
    doc = {
        "id": "doc-1",
        "text": "第一百七十九条 承担民事责任的方式主要有：\n（一）停止侵害；\n（二）排除妨碍；\n（三）消除危险；",
        "meta": {
            "source_path": "minfadian.txt",
            "type": "txt",
            "law": "中华人民共和国民法典",
            "book": "第一编 总则",
            "chapter": "第八章 民事责任",
            "section": "",
            "article": "第一百七十九条",
            "doc_id": "doc-1",
        },
    }
    splitter = SentenceSplitter(use_model=False, sentence_size=256)
    chunks = chunk_doc_item(doc, splitter, min_chunk_len=1)

    assert chunks, "应当生成至少一个 chunk"
    # 期望枚举项不要每行一个 chunk（至少前两行应当在同一个 chunk 里）
    first_text = chunks[0]["text"]
    first_index = chunks[0].get("index_text", "")
    assert "第一百七十九条" in first_text
    assert "（一）停止侵害" in first_text
    assert "中华人民共和国民法典" in first_index
    assert "第一编 总则" in first_index
    assert "第八章 民事责任" in first_index
    assert "第一百七十九条" in first_index


if __name__ == "__main__":
    test_law_article_enumeration_should_merge()
    print("ok")
