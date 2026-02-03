import sys

sys.path.append(".")

from tinyrag.rag.observation import format_observation_for_llm


def test_format_observation_includes_law_location() -> None:
    result = {
        "items": [
            {
                "rank": 1,
                "text": "第一百七十九条 承担民事责任的方式主要有：",
                "meta": {
                    "source_path": "E:\\\\code\\\\tiny-agents\\\\data\\\\raw_data\\\\minfadian.txt",
                    "law": "中华人民共和国民法典",
                    "book": "第一编 总则",
                    "chapter": "第八章 民事责任",
                    "section": "",
                    "article": "第一百七十九条",
                },
            }
        ]
    }
    out = format_observation_for_llm(result)
    assert "source=" in out
    assert "第一编 总则" in out
    assert "第八章 民事责任" in out
    assert "未分节" in out
    assert "第一百七十九条" in out


if __name__ == "__main__":
    test_format_observation_includes_law_location()
    print("ok")
