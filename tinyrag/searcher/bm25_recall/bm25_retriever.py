import os
import pickle
import jieba
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, Union

from tinyrag.searcher.bm25_recall.rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, txt_list: List[Any]=[], base_dir="data/db/bm_corpus") -> None:
        self.data_list = txt_list
        self.stopwords = set()

        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir, exist_ok=True)

        with open("tinyrag/searcher/bm25_recall/stopwords_hit.txt", 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    self.stopwords.add(word)
                    
        if len(self.data_list) != 0:
            self.build(self.data_list)
            # 初始化 BM25Okapi 实例
            # self.bm25 = BM25Okapi(self.tokenized_corpus)
            print("初始化数据库！")
        else:
            print("未初始化数据库，请加载数据库！ ")
        
        # 触发 jieba 冷启动初始化，避免第一次 search 才付出代价
        list(jieba.cut_for_search("热启动"))

        
    def build(self, txt_list: List[Any]):
        self.data_list = txt_list
        self.tokenized_corpus = []
        for doc in tqdm(self.data_list, desc="bm25 build "):
            self.tokenized_corpus.append(self.tokenize(doc))
        # 初始化 BM25Okapi 实例
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def tokenize(self, item: Any) -> List[str]:
        """ 使用jieba进行中文分词。
        """
        #注：原项目 Tiny-rag 中对BM-25的分词仅仅使用了 jieba 分词，应该需要考虑停用词等优化手段。
        #后续可以考虑增加更多分词选项。
        #已优化

        if isinstance(item, dict):
            # 索引增强：优先使用 index_text（包含法名/编章节条等定位信息）
            text = item.get("index_text") or item.get("text") or ""
        else:
            text = str(item or "")


        result = list(jieba.cut_for_search(text))
        #停用词过滤
        result = [word for word in result if word not in self.stopwords and len(word.strip()) > 0]
        return result

    def save_bm25_data(self, db_name=""):
        """ 对数据进行分词并保存到文件中。
        """
        db_name = db_name if db_name != "" else "bm25_data"
        db_file_path = os.path.join(self.base_dir, db_name + ".pkl")
        # 保存分词结果
        data_to_save = {
            "data_list": self.data_list,
            "tokenized_corpus": self.tokenized_corpus
        }
        
        with open(db_file_path, 'wb') as f:
            pickle.dump(data_to_save, f)

    def load_bm25_data(self, db_name=""):
        """ 从文件中读取分词后的语料库，并重新初始化 BM25Okapi 实例。
        """
        db_name = db_name if db_name != "" else "bm25_data"
        db_file_path = os.path.join(self.base_dir, db_name + ".pkl")
        
        with open(db_file_path, 'rb') as f:
            data = pickle.load(f)
        
        self.data_list = data["data_list"]
        self.tokenized_corpus = data["tokenized_corpus"]
        
        # 重新初始化 BM25Okapi 实例
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def search(self, query: str, top_n=5, k_percent: Optional[float] = None) -> List[Tuple[int, Any, float]]:
        """ 使用BM25算法检索最相似的文本。
        """
        if self.tokenized_corpus is None:
            raise ValueError("Tokenized corpus is not loaded or generated.")

        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # 获取分数最高的前 N 个文本的索引
        top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        
        threshold: Optional[float] = None

        #BM25的召回结果，长尾分布导致截掉小于k_percent*max_score的结果特别狠
        #需要考虑更合适的方法，这里暂时将k_percent/2作为阈值
        #待优化
        if k_percent is not None and top_n_indices:
            kp = float(k_percent)
            if kp < 0.0:
                kp = 0.0
            if kp > 1.0:
                kp = 1.0
            max_score = max(float(scores[i]) for i in top_n_indices)
            threshold = max_score * kp/2

        result: List[Tuple[int, Any, float]] = []
        for i in top_n_indices:
            s = float(scores[i])
            if threshold is not None and s < threshold:
                continue
            result.append((i, self.data_list[i], s))

        return result

