import pdfplumber
import pandas as pd
import re
_remove_extra_char = re.compile(r'\s+(人|民|法|院|案|例|库)',re.DOTALL)
_remove_page_num = re.compile(r'第\s*\d+\s*页',re.DOTALL)
_remove_blank = re.compile(r'\n*',re.DOTALL)
_add_split = re.compile(r'(关键词|基本案情|裁判理由|裁判要旨|关联索引|[一二三四]审：)',re.DOTALL)

def keep_obj(obj):
    if obj.get("object_type") != "char":
        return True
    text = obj.get("text", "")
    font = obj.get("fontname", "")
    size = float(obj.get("size") or 0)
    if text in {"人","民","法","院","案", "例", "库"} and size >= 24 and "SimHei" in font:
        return False
    return True

with pdfplumber.open("./data/raw_data/case/杨某海交通肇事案.pdf") as pdf:
    pdf_page = pdf.pages
    text = ""
    for page in pdf_page:
        page = page.filter(keep_obj)
        page_text = _remove_extra_char.sub('', page.extract_text())
        page_text = _remove_page_num.sub('', page_text)
        page_text = _remove_blank.sub('', page_text)
        page_text = _add_split.sub(lambda x: '\n'+x.group(0)+'\n', page_text)
        text += page_text 
    print(text)