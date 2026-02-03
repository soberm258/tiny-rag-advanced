import os
import json
import time
from typing import Any, List

# from emb_index import EmbIndex
from tinyrag.searcher.emb_recall.emb_index import EmbIndex
from tqdm import tqdm

class EmbRetriever:
    def __init__(self, index_dim: int, base_dir="data/db/faiss_idx") -> None:
        self.index_dim = index_dim
        self.invert_index = EmbIndex(index_dim)
        self.forward_index = []
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir, exist_ok=True)

    def insert(self, emb: list, doc: Any):
        # print("Inserting document into index...")
        self.invert_index.insert(emb)
        self.forward_index.append(doc)
        # print("Document inserted")

    def batch_insert(self, embs: list, docs: List[Any]):
        if not docs:
            return
        if embs is None or len(embs) != len(docs):
            raise ValueError("batch_insert: embs 与 docs 长度不一致")
        self.invert_index.batch_insert(embs)
        self.forward_index.extend(docs)

    def save(self, index_name=""):
        self.index_name = index_name if index_name != "" else "index_" + str(self.index_dim)
        self.index_folder_path = os.path.join(self.base_dir, self.index_name)
        if not os.path.exists(self.index_folder_path):
            os.makedirs(self.index_folder_path, exist_ok=True)

        with open(self.index_folder_path + "/forward_index.txt", "w", encoding="utf8") as f:
            for data in self.forward_index:
                f.write("{}\n".format(json.dumps(data, ensure_ascii=False)))

        self.invert_index.save(self.index_folder_path + "/invert_index.faiss")
    
    def load(self, index_name=""):
        self.index_name = index_name if index_name != "" else "index_" + str(self.index_dim)
        self.index_folder_path = os.path.join(self.base_dir, self.index_name)

        self.invert_index = EmbIndex(self.index_dim)
        inv_path = self.index_folder_path + "/invert_index.faiss"
        fwd_path = self.index_folder_path + "/forward_index.txt"
        if not os.path.isfile(inv_path) or not os.path.isfile(fwd_path):
            raise FileNotFoundError(
                f"向量索引文件不存在：{self.index_folder_path}（需要包含 invert_index.faiss 与 forward_index.txt）。"
                "请先完成建库并保存索引。"
            )
        print(f"正在加载向量倒排索引：{inv_path} ...", flush=True)
        t0 = time.time()
        self.invert_index.load(inv_path)
        print(f"向量倒排索引加载完成，耗时 {time.time()-t0:.1f}s", flush=True)

        self.forward_index = []
        total_bytes = os.path.getsize(fwd_path)
        print(f"正在加载向量前排索引：{fwd_path} ...", flush=True)
        t1 = time.time()
        with open(fwd_path, "rb") as f:
            pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, desc="load forward_index", ascii=True)
            for raw in f:
                pbar.update(len(raw))
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                self.forward_index.append(json.loads(line))
            pbar.close()
        print(f"向量前排索引加载完成，耗时 {time.time()-t1:.1f}s", flush=True)

    def search(self, embs: list, top_n=5):
        search_res = self.invert_index.search(embs, top_n)
        recall_list = []
        indices = search_res[1][0].tolist()
        distances = search_res[0][0].tolist()
        for idx, doc_idx in enumerate(indices):
            if doc_idx is None or doc_idx < 0:
                continue
            if doc_idx >= len(self.forward_index):
                continue
            recall_list.append((doc_idx, self.forward_index[doc_idx], distances[idx]))
        return recall_list
