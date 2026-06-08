from __future__ import annotations

from typing import Any

from backend.memory import MemoryManager
from backend.tools.base import BaseTool


def default_memory_context(session_id: str = "", policy: str = "未配置 memory_manager，本轮不读取长期记忆。") -> dict[str, Any]:
    return {
        "session": {"session_id": session_id},
        "last_case": {},
        "missing_info": [],
        "recent_safety": {},
        "recent_ticket": {},
        "session_summary": {"session_id": session_id, "message_count": 0},
        "session_search": {
            "available": False,
            "query": "",
            "matches": [],
            "summary": "",
            "summary_method": "sqlite_fts5_extractive",
            "error": "",
        },
        "last_customer_reply": "",
        "customer": {},
        "charger": {},
        "site": {},
        "ticket": {},
        "matched_ids": {"session_id": session_id},
        "isolation": {
            "scope": "session/customer/charger/site/ticket/repo",
            "session_id": session_id,
            "session_isolated": True,
            "long_term_store": "local_json",
            "session_search_store": "sqlite_fts5",
            "repo_knowledge_separated": True,
            "used_as_diagnostic_evidence": False,
            "policy": policy,
        },
    }


class MemoryContextReadTool(BaseTool):
    name = "memory_context_read"
    description = "读取当前会话和长期记忆摘要，只作为上下文补全，不作为诊断证据。"

    def run(self, **kwargs: Any) -> dict[str, Any]:
        manager = kwargs.get("memory_manager")
        session_id = str(kwargs.get("session_id") or "")
        case = kwargs.get("case") if isinstance(kwargs.get("case"), dict) else {}
        if manager is None:
            return default_memory_context(session_id=session_id)
        if not isinstance(manager, MemoryManager):
            raise TypeError("memory_manager must be a MemoryManager")
        context = manager.recall_context(case, session_id=session_id)
        context.setdefault("isolation", {})
        context["isolation"]["used_as_diagnostic_evidence"] = False
        return context


class MemoryWorkflowWriteTool(BaseTool):
    name = "memory_workflow_write"
    description = "沉淀 workflow 结构化结果到 Session/Customer/Charger/Site/Ticket 记忆。"

    def run(self, **kwargs: Any) -> dict[str, Any]:
        manager = kwargs.get("memory_manager")
        session_id = str(kwargs.get("session_id") or "")
        user_input = str(kwargs.get("user_input") or "")
        result = kwargs.get("result") if isinstance(kwargs.get("result"), dict) else {}
        if manager is None:
            return {
                "memory_ids": {"session_id": session_id},
                "isolation": default_memory_context(session_id=session_id)["isolation"],
                "skipped": True,
                "reason": "未配置 memory_manager，本轮不写入记忆。",
            }
        if not isinstance(manager, MemoryManager):
            raise TypeError("memory_manager must be a MemoryManager")
        memory_ids = manager.remember_workflow_result(user_input, result, session_id=session_id)
        return {
            "memory_ids": memory_ids,
            "isolation": {
                "scope": "session/customer/charger/site/ticket/repo",
                "session_id": memory_ids.get("session_id", session_id),
                "repo_knowledge_separated": True,
                "used_as_diagnostic_evidence": False,
                "policy": "记忆写入只沉淀摘要和工单快照，不作为本轮诊断依据。",
            },
        }
