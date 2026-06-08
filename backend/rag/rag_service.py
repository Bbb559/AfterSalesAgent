from __future__ import annotations

from typing import Any, Callable

from backend.rag.kb_manager import KnowledgeBaseManager


class RAGService:
    """围绕现有 RAG 检索功能的小型适配器。"""

    def __init__(
        self,
        retrieval_func: Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]] | None = None,
        kb_manager: KnowledgeBaseManager | None = None,
    ) -> None:
        self.retrieval_func = retrieval_func
        self.kb_manager = kb_manager or KnowledgeBaseManager()

    def retrieve(self, question: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        options = options or {}
        try:
            if self.retrieval_func is not None:
                results, trace = self.retrieval_func(question=question, **options)
            else:
                results, trace = self.kb_manager.retrieve(question, options)
        except Exception as exc:
            return {
                "query": question,
                "results": [],
                "trace": {"mode": options.get("retrieval_mode", "unknown"), "error": str(exc)},
                "sources": [],
                "error": str(exc),
            }

        return {
            "query": question,
            "results": results or [],
            "trace": trace or {},
            "sources": self._format_sources(results or []),
            "error": "",
        }

    def answer(self, question: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.retrieve(question, options)

    def build_knowledge_base(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.kb_manager.build_knowledge_base(*args, **kwargs)

    def load_knowledge_base(self, database_id: str) -> dict[str, Any]:
        return self.kb_manager.load_knowledge_base(database_id)

    def delete_knowledge_base(self, database_id: str) -> dict[str, Any]:
        return self.kb_manager.delete_knowledge_base(database_id)

    def list_knowledge_bases(self) -> list[dict[str, Any]]:
        return self.kb_manager.list_knowledge_bases()

    def status(self) -> dict[str, Any]:
        return self.kb_manager.status()

    def _format_sources(self, results: list[dict[str, Any]]) -> list[str]:
        sources = []
        for item in results:
            file_name = item.get("file_name", "unknown")
            page = item.get("page", "unknown")
            source = f"{file_name} 第{page}页"
            if source not in sources:
                sources.append(source)
        return sources

