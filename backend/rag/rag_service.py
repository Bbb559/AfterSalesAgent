from __future__ import annotations

from typing import Any, Callable

from backend.rag.kb_manager import KnowledgeBaseManager


class RAGService:
    """围绕现有 RAG 检索功能的小型适配器."""

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

        prioritized = self._prioritize_by_after_sales_metadata(question, results or []) # 优先排序，提升售后相关文档的排名
        return {
            "query": question,
            "results": prioritized,
            "trace": trace or {},
            "sources": self._format_sources(prioritized),
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

    def _prioritize_by_after_sales_metadata( 
        self,
        question: str,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not results:
            return []

        preferred_doc_type = ""
        if any(word in question for word in ["保修", "质保", "免费", "收费", "过保"]):
            preferred_doc_type = "warranty_policy"
        elif any(word in question for word in ["故障码", "E0", "F0", "报错", "显示"]):
            preferred_doc_type = "fault_code_table"
        elif any(word in question for word in ["维修", "上门", "排查", "处理"]):
            preferred_doc_type = "repair_guide"

        if not preferred_doc_type:
            return results

        return sorted(
            results,
            key=lambda item: 0 if item.get("doc_type") == preferred_doc_type else 1,
        )
