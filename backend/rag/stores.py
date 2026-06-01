import json
from pathlib import Path

import faiss
import numpy as np

from backend.config import FAISS_CHUNKS_FILE, FAISS_INDEX_FILE
from backend.rag.embeddings import get_embeddings
from backend.rag.splitters import chunks_to_texts


class FaissVectorStore:
    """RAG 示例项目使用的 FAISS 向量库。

    默认索引类型：IndexFlatIP。
    文档向量和查询向量都会先进行 L2 归一化，
    因此内积相似度可以近似当作余弦相似度使用。
    """

    def __init__(self):
        self.index = None
        self.chunks = []

    def build(self, chunks):
        if not chunks:
            raise ValueError("chunks 为空，无法构建 FAISS 索引。")

        texts = chunks_to_texts(chunks)
        embeddings = get_embeddings(texts)

        if len(embeddings) != len(chunks):
            raise ValueError("Embedding 数量和 chunk 数量不一致，无法建立正确的索引映射。")

        vectors = np.array(embeddings, dtype="float32")

        if vectors.ndim != 2:
            raise ValueError("Embedding 结果不是二维数组，无法构建 FAISS 索引。")

        faiss.normalize_L2(vectors)

        dimension = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(vectors)

        self.chunks = list(chunks)

    def search(self, query, top_k=5):
        if self.index is None:
            raise ValueError("FAISS 索引还没有构建，请先调用 build() 或 load()。")

        if not query or not query.strip():
            return []

        if top_k <= 0:
            return []

        query_embeddings = get_embeddings([query])
        if not query_embeddings:
            return []

        query_embedding = query_embeddings[0]
        query_vector = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(query_vector)

        actual_top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, actual_top_k)


        results = []
        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue

            chunk = dict(self.chunks[index])
            chunk["score"] = float(score)
            chunk["source"] = "faiss"
            results.append(chunk)

        return results

    def save(
        self,
        index_path=FAISS_INDEX_FILE,
        chunks_path=FAISS_CHUNKS_FILE,
    ):
        if self.index is None:
            raise ValueError("FAISS 索引还没有构建，无法保存。")

        index_path = Path(index_path)
        chunks_path = Path(chunks_path)

        index_path.parent.mkdir(parents=True, exist_ok=True)
        chunks_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))

        with chunks_path.open("w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    def load(
        self,
        index_path=FAISS_INDEX_FILE,
        chunks_path=FAISS_CHUNKS_FILE,
    ):
        index_path = Path(index_path)
        chunks_path = Path(chunks_path)

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS 索引文件不存在：{index_path}")

        if not chunks_path.exists():
            raise FileNotFoundError(f"chunks 文件不存在：{chunks_path}")

        self.index = faiss.read_index(str(index_path))

        with chunks_path.open("r", encoding="utf-8") as f:
            self.chunks = json.load(f)
