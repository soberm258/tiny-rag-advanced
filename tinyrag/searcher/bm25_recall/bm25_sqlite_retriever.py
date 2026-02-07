from __future__ import annotations

from array import array
import json
import math
import os
import sqlite3
import struct
from typing import Any, Dict, List, Optional, Tuple

import jieba
from tqdm import tqdm


class BM25SQLiteRetriever:
    """
    使用 SQLite 持久化倒排索引的 BM25（Okapi）召回器。

    设计目标：
    - 离线建库、只读检索
    - 查询时只遍历 query term 的 postings，避免全库 get_scores 扫描
    - 与现有 BM25Retriever 的 tokenize 逻辑保持一致（jieba.cut_for_search + case/pdf 停用词策略）
    """

    def __init__(
        self,
        txt_list: List[Any] = [],
        *,
        base_dir: str = "data/db/bm_corpus",
        sqlite_name: str = "bm25.sqlite",
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

        self.sqlite_path = os.path.join(self.base_dir, sqlite_name)
        self.k1 = float(k1)
        self.b = float(b)
        self.epsilon = float(epsilon)

        self.stopwords: set[str] = set()
        stopwords_path = os.path.join("tinyrag", "searcher", "bm25_recall", "stopwords_hit.txt")
        if os.path.isfile(stopwords_path):
            with open(stopwords_path, "r", encoding="utf-8") as f:
                for line in f:
                    w = line.strip()
                    if w:
                        self.stopwords.add(w)

        self._conn: Optional[sqlite3.Connection] = None
        self._meta: Dict[str, Any] = {}

        if txt_list:
            self.build(txt_list)
            self.save_bm25_data()
            print("初始化数据库！")
        else:
            print("未初始化数据库，请加载数据库！ ")
        # 触发 jieba 冷启动初始化，避免第一次 search 才付出代价
        list(jieba.cut_for_search("热启动"))

    def _connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        conn = sqlite3.connect(self.sqlite_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        self._conn = conn
        return conn

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta_stats (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS doc_stats (
              doc_id INTEGER PRIMARY KEY,
              dl INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS postings_blob (
              term TEXT PRIMARY KEY,
              df INTEGER NOT NULL,
              blob BLOB NOT NULL
            );
            """
        )

    def tokenize(self, item: Any) -> List[str]:
        if isinstance(item, dict):
            text = item.get("index_text") or item.get("text") or ""
            meta = item.get("meta") or {}
        else:
            text = str(item or "")

        tokens = list(jieba.cut_for_search(str(text)))

        tokens = [t for t in tokens if t not in self.stopwords and t.strip()]
        return tokens

    def build(self, txt_list: List[Any]) -> None:
        self.data_list = txt_list

        conn = self._connect()
        self._ensure_schema(conn)

        # 清空旧数据（离线重建模式）
        conn.execute("DELETE FROM meta_stats;")
        conn.execute("DELETE FROM doc_stats;")
        conn.execute("DELETE FROM postings_blob;")
        conn.commit()

        N = int(len(txt_list))
        total_dl = 0
        df_map: Dict[str, int] = {}
        postings_map: Dict[str, List[Tuple[int, int]]] = {}

        doc_stats_rows: List[Tuple[int, int]] = []

        for doc_id, doc in tqdm(list(enumerate(txt_list)), desc="bm25(sqlite) build ", ascii=True):
            #doc_id =  ids,doc = chunk
            tokens = self.tokenize(doc)
            #tokenize 返回index_text/text 分词列表
            dl = int(len(tokens))
            total_dl += dl

            doc_stats_rows.append((doc_id, dl))

            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            for term, freq in tf.items():
                df_map[term] = df_map.get(term, 0) + 1
                postings_map.setdefault(term, []).append((doc_id, int(freq)))

        avgdl = (float(total_dl) / float(N)) if N > 0 else 0.0

        # 计算 IDF 下限（与 rank_bm25.BM25Okapi 的 epsilon 规则一致）
        idf_sum = 0.0
        negative_terms: List[str] = []
        for term, df in df_map.items():
            idf = math.log(N - df + 0.5) - math.log(df + 0.5)
            idf_sum += float(idf)
            if idf < 0:
                negative_terms.append(term)
        average_idf = (idf_sum / float(len(df_map))) if df_map else 0.0
        eps = float(self.epsilon) * float(average_idf)
        _ = negative_terms  # 仅用于构造 eps_floor，不在库中存每个 term 的 idf

        postings_blob_rows: List[Tuple[str, int, bytes]] = []
        for term, plist in postings_map.items():
            # term 一行 blob：n + doc_ids(uint32*n) + tfs(uint32*n)
            plist.sort(key=lambda x: x[0])
            n = int(len(plist))
            doc_ids = array("I", (int(doc_id) for doc_id, _ in plist))
            tfs = array("I", (int(tf) for _, tf in plist))
            blob = struct.pack("<I", n) + doc_ids.tobytes() + tfs.tobytes()
            postings_blob_rows.append((term, int(df_map.get(term, n)), blob))

        conn.execute("BEGIN;")
        conn.executemany("INSERT INTO doc_stats(doc_id, dl) VALUES (?, ?);", doc_stats_rows)
        conn.executemany("INSERT INTO postings_blob(term, df, blob) VALUES (?, ?, ?);", postings_blob_rows)

        meta = {
            "version": 2,
            "N": N,
            "avgdl": avgdl,
            "k1": self.k1,
            "b": self.b,
            "epsilon": self.epsilon,
            "average_idf": average_idf,
            "eps_floor": eps,
        }
        meta_rows = [(k, json.dumps(v, ensure_ascii=False)) for k, v in meta.items()]
        conn.executemany("INSERT INTO meta_stats(key, value) VALUES (?, ?);", meta_rows)
        conn.commit()

        self._meta = meta

    def save_bm25_data(self, db_name: str = "") -> None:
        # SQLite 已在 build 时写入；这里保留同名方法，便于 Searcher.save_db 复用
        _ = db_name
        conn = self._connect()
        conn.commit()

    def _load_split_sentence_docs(self) -> List[Any]:
        """
        从 data/db/<db_name>/split_sentence.jsonl 加载 doc 列表，用于 doc_id -> doc 的映射。
        注意：doc_id 与 split_sentence.jsonl 的行号（从 0 开始）必须一致。
        """
        db_root = os.path.dirname(self.base_dir.rstrip("/\\"))
        path = os.path.join(db_root, "split_sentence.jsonl")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"split_sentence.jsonl 不存在：{path}")

        docs: List[Any] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = (line or "").strip()
                if not line:
                    continue
                docs.append(json.loads(line))
        return docs

    def load_bm25_data(self, db_name: str = "") -> None:
        _ = db_name
        if not os.path.isfile(self.sqlite_path):
            raise FileNotFoundError(f"BM25 SQLite 索引不存在：{self.sqlite_path}（请先建库）")

        conn = self._connect()
        self._ensure_schema(conn)

        meta: Dict[str, Any] = {}
        for k, v in conn.execute("SELECT key, value FROM meta_stats;").fetchall():
            try:
                meta[k] = json.loads(v)
            except Exception:
                meta[k] = v
        self._meta = meta

        N = int(self._meta.get("N") or 0)
        self.doc_lens: List[int] = [0] * max(0, N)
        for doc_id, dl in conn.execute("SELECT doc_id, dl FROM doc_stats ORDER BY doc_id;"):
            if 0 <= int(doc_id) < len(self.doc_lens):
                self.doc_lens[int(doc_id)] = int(dl)
        self.docs = self._load_split_sentence_docs()

    def search(self, query: str, top_n: int = 5, k_percent: Optional[float] = None) -> List[Tuple[int, Any, float]]:
        conn = self._connect()
        if not self._meta:
            raise ValueError("BM25SQLiteRetriever 尚未加载索引：请先调用 load_bm25_data()。")

        top_n = max(1, int(top_n))
        N = int(self._meta.get("N") or 0)
        avgdl = float(self._meta.get("avgdl") or 0.0)
        k1 = float(self._meta.get("k1") or self.k1)
        b = float(self._meta.get("b") or self.b)
        eps_floor = float(self._meta.get("eps_floor") or 0.0)

        if N <= 0 or avgdl <= 0.0:
            return []

        q_terms = self.tokenize(query)
        if not q_terms:
            return []

        q_tf: Dict[str, int] = {}
        for t in q_terms:
            q_tf[t] = q_tf.get(t, 0) + 1
        terms = list(q_tf.keys())

        score_map: Dict[int, float] = {}
        placeholders = ",".join(["?"] * len(terms))
        cur = conn.execute(
            f"""
            SELECT term, df, blob
            FROM postings_blob
            WHERE term IN ({placeholders});
            """,
            tuple(terms),
        )

        for term, df, blob in cur:
            term = str(term)
            df = int(df)
            if df <= 0:
                continue

            idf_raw = math.log(N - df + 0.5) - math.log(df + 0.5)
            idf = float(eps_floor if idf_raw < 0 else idf_raw)

            q_mult = int(q_tf.get(term, 1))
            if not blob:
                continue

            mv = memoryview(blob)
            n = struct.unpack_from("<I", mv, 0)[0]
            if n <= 0:
                continue
            off = 4
            doc_bytes = mv[off : off + 4 * n]
            off += 4 * n
            tf_bytes = mv[off : off + 4 * n]

            doc_ids = array("I")
            tfs = array("I")
            doc_ids.frombytes(doc_bytes)
            tfs.frombytes(tf_bytes)

            for doc_id, tf in zip(doc_ids, tfs):
                doc_id = int(doc_id)
                if doc_id < 0 or doc_id >= len(self.doc_lens):
                    continue
                dl = int(self.doc_lens[doc_id])
                tf = int(tf)
                denom = tf + k1 * (1.0 - b + b * float(dl) / avgdl)
                if denom <= 0:
                    continue
                inc = idf * float(q_mult) * (tf * (k1 + 1.0) / denom)
                score_map[doc_id] = score_map.get(doc_id, 0.0) + float(inc)

        if not score_map:
            return []

        # 取 top_n doc_id
        scored = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:top_n]

        threshold: Optional[float] = None
        if k_percent is not None and scored:
            kp = float(k_percent)
            if kp < 0.0:
                kp = 0.0
            if kp > 1.0:
                kp = 1.0
        max_score = float(scored[0][1])
        threshold = max_score * kp

        kept = [(doc_id, s) for doc_id, s in scored if threshold is None or float(s) >= threshold]

        result: List[Tuple[int, Any, float]] = []
        for doc_id, s in kept:
            if not hasattr(self, "docs"):
                continue
            if int(doc_id) < 0 or int(doc_id) >= len(self.docs):
                continue
            result.append((int(doc_id), self.docs[int(doc_id)], float(s)))
        return result
