from __future__ import annotations

from typing import Any

from backend.rag.rag_service import RAGService


class RetrievalAgent:
    """围绕现有 RAG 检索流程的适配器。"""

    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service

    def retrieve(
        self,
        query: str,
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        retrieval_options = dict(options or {})
        retrieval_options.update(kwargs)
        return self.rag_service.retrieve(query, retrieval_options)
