import re
import jieba
from rank_bm25 import BM25Okapi


def tokenize(text):
    if not text:
        return []

    text = text.lower()
    english_tokens = re.findall(r"[a-zA-Z0-9_.%-]+", text)
    chinese_text = re.sub(r"[a-zA-Z0-9_.%-]+", " ", text)

    chinese_tokens = [
        token.strip()
        for token in jieba.lcut(chinese_text)
        if token.strip()
    ]

    return english_tokens + chinese_tokens


class BM25Retriever:
    def __init__(self):
        self.chunks = []
        self.bm25 = None

    def build(self, chunks):
        self.chunks = list(chunks)
        tokenized_chunks = [tokenize(chunk.get("text", "")) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

    def search(self, query, top_k=5):
        if self.bm25 is None:
            raise ValueError("BM25索引还没有构建。")

        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        top_k = min(top_k, len(scores))
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        results = []
        for index in ranked_indices:
            chunk = dict(self.chunks[index])
            chunk["score"] = float(scores[index])
            chunk["source"] = "bm25"
            results.append(chunk)

        return results



def reciprocal_rank_fusion(result_groups, final_top_k=10, rrf_k=60):
    """用 RRF 融合多组检索结果列表。"""
    merged = {}

    for results in result_groups:
        for rank, item in enumerate(results, start=1):
            chunk_id = item["chunk_id"]

            if chunk_id not in merged:
                merged[chunk_id] = dict(item)
                merged[chunk_id]["sources"] = set()
                merged[chunk_id]["scores"] = {}
                merged[chunk_id]["rrf_score"] = 0.0

            source = item.get("source", "unknown")
            score = item.get("score", 0.0)

            merged[chunk_id]["sources"].add(source)
            merged[chunk_id]["scores"][source] = score
            merged[chunk_id]["rrf_score"] += 1.0 / (rrf_k + rank)

    fused = []
    for item in merged.values():
        item["sources"] = sorted(item["sources"])
        fused.append(item)

    fused.sort(key=lambda item: item["rrf_score"], reverse=True)
    return fused[:final_top_k]


def hybrid_retrieve(
    query,
    vector_store,
    bm25_retriever=None,
    vector_top_k=10,
    bm25_top_k=10,
    final_top_k=10,
):
    vector_results = vector_store.search(query, top_k=vector_top_k)

    result_groups = [vector_results]

    if bm25_retriever is not None:
        bm25_results = bm25_retriever.search(query, top_k=bm25_top_k)
        result_groups.append(bm25_results)

    return reciprocal_rank_fusion(result_groups, final_top_k=final_top_k)
