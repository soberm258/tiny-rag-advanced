import sys

sys.path.append(".")

from tinyrag.ingest.structured.law_cn_civil_code import parse_civil_code_text


def test_parse_articles_with_zero() -> None:
    text = """中华人民共和国民法典
第五编　婚姻家庭
第五章　收养
第一节　收养关系的成立
第一千条 行为人因侵害人格权承担消除影响、恢复名誉、赔礼道歉等民事责任的，应当与行为的具体方式和造成的影响范围相当。
第一千零一条 对自然人因婚姻家庭关系等产生的身份权利的保护，适用本法第一编、第五编和其他法律的相关规定。
第一千零二条 自然人享有生命权。
"""
    docs = parse_civil_code_text(text, source_path="minfadian.txt")
    assert len(docs) == 3
    assert docs[0]["meta"]["article"] == "第一千条"
    assert docs[1]["meta"]["article"] == "第一千零一条"
    assert docs[2]["meta"]["article"] == "第一千零二条"
    assert docs[0]["meta"]["book"].startswith("第五编")
    assert docs[0]["meta"]["chapter"].startswith("第五章")
    assert docs[0]["meta"]["section"].startswith("第一节")


def test_infer_law_title_should_match_other_laws() -> None:
    text = "中华人民共和国刑法\n第一条 为了惩罚犯罪，保护人民，维护国家安全和社会秩序，制定本法。\n"
    docs = parse_civil_code_text(text, source_path="xingfa.txt")
    assert docs
    assert docs[0]["meta"]["law"] == "中华人民共和国刑法"


if __name__ == "__main__":
    test_parse_articles_with_zero()
    test_infer_law_title_should_match_other_laws()
    print("ok")
