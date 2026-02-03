from __future__ import annotations

_DEFAULT_HYDE = (
    "你是一名检索增强系统的查询改写器。\n"
    "请根据用户问题，写一段“可能出现在知识库/百科/说明文中的答案段落”，用于向量检索召回相关资料。\n"
    "要求：只输出正文，不要标题，不要编号，不要引用，不要出现“根据/可能/我认为”等措辞；"
    "尽量包含关键实体、别名、时间、地点、定义、要点等信息；长度控制在 200~400 字。\n"
    "用户问题：{question}\n"
    "正文："
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


def build_hyde_prompt(question: str) -> str:
    return _DEFAULT_HYDE.format(question=(question or "").strip())
