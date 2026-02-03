
import re
from typing import List

class SentenceSplitter:
    """ 句子分割模型
        use_model: 指定是否用语义切分文档, 采取的文档语义分割模型为 nlp_bert_document-segmentation_chinese-base， 论文见https://arxiv.org/abs/2107.09278
    """
    def __init__(self, 
                 use_model: bool = False, 
                 sentence_size = 256,
                 model_path: str = "damo/nlp_bert_document-segmentation_chinese-base", 
                 device="cpu"
        ):
        self.sentence_size = sentence_size
        self.use_model = use_model
        if self.use_model:
            try:
                from modelscope.pipelines import pipeline
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "已启用句子切分模型(use_model=True)，但当前环境缺少 modelscope 或其依赖。"
                    "请安装 modelscope 及其依赖（常见缺失为 addict）。"
                ) from e
            # assert model_path == "" "模型路径为空"
            self.sent_split_pp = pipeline(
                task="document-segmentation",
                model=model_path,
                device=device
            )

    def split_text(self, sentence: str) -> List[str]:
        if self.use_model:
            # TODO: modelscope install unable to find candidates for en-core-web-sm
            result = self.sent_split_pp(documents=sentence)
            sent_list = [i for i in result["text"].split("\n\t") if i]
        else:
            # 单字符断句符
            # 注意：英文句点 . 在数字小数（如 2.4）中不应断句，因此这里对 "." 做特殊处理
            text = re.sub(r'([;；!?。！？\?])([^”’])', r"\1\n\2", sentence)
            text = re.sub(r'(\.)(?!\d)([^”’])', r"\1\n\2", text)
            text = re.sub(r'(\.{6})([^"’”」』])', r"\1\n\2", text)  # 英文省略号
            text = re.sub(r'(\…{2})([^"’”」』])', r"\1\n\2", text)  # 中文省略号
            text = re.sub(r'([;；!?。！？\?]["’”」』]{0,2})([^;；!?，。！？\?])', r'\1\n\2', text)
            text = re.sub(r'(\.(?!\d)["’”」』]{0,2})([^;；!?，。！？\?])', r'\1\n\2', text)
            # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
            text = text.rstrip()  # 段尾如果有多余的\n就去掉它
            # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
            sent_list = [i for i in text.split("\n") if i]
        for ele in sent_list:
            if len(ele) > self.sentence_size:
                # 过长时优先按逗号/顿号拆分，避免把数字小数中的 "." 拆碎
                ele1 = re.sub(r'([,，]["’”」』]{0,2})([^,，])', r'\1\n\2', ele)
                ele1_ls = ele1.split("\n")
                for ele_ele1 in ele1_ls:
                    if len(ele_ele1) > self.sentence_size:
                        ele_ele2 = re.sub(r'([\n]{1,}| {2,}["’”」』]{0,2})([^\s])', r'\1\n\2', ele_ele1)
                        ele2_ls = ele_ele2.split("\n")
                        for ele_ele2 in ele2_ls:
                            if len(ele_ele2) > self.sentence_size:
                                ele_ele3 = re.sub('( ["’”」』]{0,2})([^ ])', r'\1\n\2', ele_ele2)
                                ele2_id = ele2_ls.index(ele_ele2)
                                ele2_ls = ele2_ls[:ele2_id] + [i for i in ele_ele3.split("\n") if i] + ele2_ls[
                                                                                                       ele2_id + 1:]
                        ele_id = ele1_ls.index(ele_ele1)
                        ele1_ls = ele1_ls[:ele_id] + [i for i in ele2_ls if i] + ele1_ls[ele_id + 1:]

                id = sent_list.index(ele)
                sent_list = sent_list[:id] + [i for i in ele1_ls if i] + sent_list[id + 1:]

        return sent_list
    # def split_text(self, sentence: str) -> List[str]:

    #     if self.use_model:
    #         # TODO: modelscope install unable to find candidates for en-core-web-sm
    #         result = self.sent_split_pp(documents=sentence)
    #         sent_list = [i for i in result["text"].split("\n\t") if i]
    #     else:
    #         sent_sep_pattern = re.compile('([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))')  # del ：；
    #         sent_list = []
    #         for ele in sent_sep_pattern.split(sentence):
    #             if sent_sep_pattern.match(ele) and sent_list:
    #                 sent_list[-1] += ele
    #             elif ele:
    #                 sent_list.append(ele)

    #     # 合并长度不足的句子
    #     merged_sent_list = []
    #     curr_sentence = ''
    #     for sent in sent_list:
    #         if len(curr_sentence) + len(sent) < self.min_sent_len:
    #             curr_sentence += sent
    #         else:
    #             if curr_sentence:
    #                 merged_sent_list.append(curr_sentence)
    #             curr_sentence = sent
    #     if curr_sentence:  # 添加最后一个句子
    #         merged_sent_list.append(curr_sentence)

    #     return merged_sent_list
