from __future__ import annotations

_DEFAULT_HYDE = (
    "你是法律条文检索的查询扩展器（HyDE）。\n"
    "你的输出用于向量检索，不是最终回答。\n"
    "只输出一行“检索短语”，长度 20~50 字。\n"
    "必须原样保留用户问题里的至少2个关键短语（每个>=2个汉字），不要把信息改写成泛化问句。\n"
    "在保留原信息的基础上，补充2~5个法学近义词/领域词（例如 责任/义务/构成要件/免责/赔偿/禁止/合同/侵权/刑事/行政 等）。\n"
    "禁止编造事实、条号、机构名称、数字结论；禁止定义/解释/分点；不要输出问号。\n"
    "输出示例（仅示例，不要照抄）：救助义务 及时施救 免责条件 侵权责任 过错\n"
)

_DEFAULT_RAG = (
    "参考信息（每段以 [编号] 开头）：\n"
    "{context}\n"
    "---\n"
    "我的问题或指令：\n"
    "{question}\n"
    "---\n"
    "请根据上述参考信息回答问题。回答一定要忠于原文，简洁但不丢信息，不要胡乱编造。"
    "请在关键结论后标注引用编号，例如 [1][3]。\n"
    "你的回答："
)


def build_rag_prompt(*, context: str, question: str) -> str:
    return _DEFAULT_RAG.format(
        context=(context or "").strip(),
        question=(question or "").strip(),
    )


def build_hyde_prompt() -> str:
    return _DEFAULT_HYDE.strip()
