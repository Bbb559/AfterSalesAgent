import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from backend.config import DASHSCOPE_BASE_URL, DEFAULT_CHAT_MODEL
from backend.rag.prompts import build_query_rewrite_prompt, build_rerank_prompt
from backend.rag.retrievers import reciprocal_rank_fusion


load_dotenv()

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client

    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("缺少 API_KEY，请在项目根目录创建 .env，或设置 API_KEY 环境变量。")

    if _client is None:
        _client = OpenAI(
            api_key=api_key,
            base_url=DASHSCOPE_BASE_URL,
        )

    return _client


def get_completion(prompt, model=DEFAULT_CHAT_MODEL):
    response = _get_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


def rewrite_query(question):
    """返回原始问题和改写后的检索查询。"""
    prompt = build_query_rewrite_prompt(question)
    content = get_completion(prompt)

    try:
        rewritten = _extract_json(content)
        if isinstance(rewritten, list):
            queries = [question]
            queries.extend([item.strip() for item in rewritten if isinstance(item, str) and item.strip()])
            return _dedupe_strings(queries)
    except Exception:
        pass

    return [question]


def rerank_with_llm(query, chunks, top_n=5):
    """用 LLM 对检索到的 chunks 重新打分排序。

    如果 LLM 返回 JSON 解析失败，就回退到原始检索顺序。
    """
    if not chunks:
        return []

    prompt = build_rerank_prompt(query, chunks)
    content = get_completion(prompt)

    try:
        rankings = _extract_json(content)
        if not isinstance(rankings, list):
            raise ValueError("LLM rerank 返回结果不是 JSON 数组。")

        score_map = {}

        for item in rankings:
            chunk_id = item.get("chunk_id")
            if not chunk_id:
                continue

            score = _safe_float(item.get("relevance_score", 0.0))
            score = max(0.0, min(1.0, score))

            score_map[chunk_id] = {
                "rerank_score": score,
                "rerank_reason": item.get("reason", ""),
            }

        reranked = []
        for chunk in chunks:
            new_chunk = dict(chunk)
            rank_info = score_map.get(chunk["chunk_id"], {})
            new_chunk["rerank_score"] = rank_info.get("rerank_score", 0.0)
            new_chunk["rerank_reason"] = rank_info.get("rerank_reason", "")
            reranked.append(new_chunk)

        reranked.sort(key=lambda item: item.get("rerank_score", 0.0), reverse=True)
        return reranked[:top_n]

    except Exception:
        fallback = []
        for chunk in chunks[:top_n]:
            new_chunk = dict(chunk)
            new_chunk["rerank_score"] = None
            new_chunk["rerank_reason"] = "LLM 重排序结果解析失败，已回退到原始检索顺序。"
            fallback.append(new_chunk)
        return fallback


def retrieve_with_query_rewrite(
    question,
    vector_store,
    bm25_retriever,
    hybrid_retrieve_func,
    vector_top_k=10,
    bm25_top_k=10,
    final_top_k=10,
):
    if not question or not question.strip():
        return []
    
    queries = rewrite_query(question)
    result_groups = []

    for query in queries:
        results = hybrid_retrieve_func(
            query=query,
            vector_store=vector_store,
            bm25_retriever=bm25_retriever,
            vector_top_k=vector_top_k,
            bm25_top_k=bm25_top_k,
            final_top_k=final_top_k,
        )
        result_groups.append(results)

    return reciprocal_rank_fusion(result_groups, final_top_k=final_top_k)


def _extract_json(content):
    content = content.strip()

    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?", "", content, flags=re.I)
        content = re.sub(r"^```", "", content)
        content = re.sub(r"```$", "", content).strip()

    match = re.search(r"(\[.*\]|\{.*\})", content, re.S)
    if match:
        content = match.group(1)

    return json.loads(content)


def _dedupe_strings(items):
    seen = set()
    result = []

    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result

def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
