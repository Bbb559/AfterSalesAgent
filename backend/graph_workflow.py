from __future__ import annotations

import time
from typing import Any, Callable, TypedDict

from backend.agents.action_agent import ActionAgent
from backend.agents.audit_agent import AuditAgent
from backend.agents.case_extract_agent import CaseExtractAgent
from backend.agents.diagnosis_agent import DiagnosisAgent
from backend.agents.intent_agent import IntentAgent
from backend.agents.retrieval_agent import RetrievalAgent
from backend.rag.rag_service import RAGService
from backend.schemas import WorkflowResult
from backend.tools.local_runner import run_after_sales_tools_sync

try:
    from langgraph.graph import END, StateGraph
except ModuleNotFoundError:  # pragma: no cover - 当前环境未安装 LangGraph 时走顺序兜底
    END = "__end__"
    StateGraph = None


class AfterSalesGraphState(TypedDict, total=False):
    user_input: str
    retrieval_options: dict[str, Any]
    intent: dict[str, Any]
    case: dict[str, Any]
    retrieval: dict[str, Any]
    diagnosis: dict[str, Any]
    warranty: dict[str, Any]
    escalation: dict[str, Any]
    ticket: dict[str, Any]
    action: dict[str, Any]
    audit: dict[str, Any]
    tool_history: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    result: dict[str, Any]


class AfterSalesGraphWorkflow:
    """使用 LangGraph 编排售后 Agent 流程，依赖缺失时自动退回顺序执行。"""

    def __init__(
        self,
        retrieval_func: Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]] | None = None,
        rag_service: RAGService | None = None,
        tool_runner: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.intent_agent = IntentAgent()
        self.case_agent = CaseExtractAgent()
        self.rag_service = rag_service or RAGService(retrieval_func=retrieval_func)
        self.retrieval_agent = RetrievalAgent(self.rag_service)
        self.diagnosis_agent = DiagnosisAgent()
        self.action_agent = ActionAgent()
        self.audit_agent = AuditAgent()
        self.tool_runner = tool_runner or run_after_sales_tools_sync
        self._compiled_graph = self._build_graph()

    def run(self, user_input: str, retrieval_options: dict[str, Any] | None = None) -> dict[str, Any]:
        state: AfterSalesGraphState = {
            "user_input": user_input or "",
            "retrieval_options": retrieval_options or {},
            "tool_history": [],
            "trace": [],
        }

        try:
            if self._compiled_graph is not None:
                final_state = self._compiled_graph.invoke(state)
            else:
                final_state = self._run_sequential(state)
            return final_state.get("result") or self._build_result(final_state)
        except Exception as exc:  # pragma: no cover - 工作流最外层防御边界
            self._add_trace(state, "error", "流程异常", "failed", {}, {"error": str(exc)})
            result = WorkflowResult().to_dict()
            result["audit"] = {
                "passed": False,
                "warnings": [f"售后流程运行异常：{exc}"],
                "final_note": "建议人工确认后再回复客户。",
                "risk_level": "unknown",
            }
            result["trace"] = state.get("trace", [])
            result["tool_history"] = state.get("tool_history", [])
            return result

    def _build_graph(self) -> Any:
        if StateGraph is None:
            return None

        graph = StateGraph(AfterSalesGraphState)
        graph.add_node("intent", self._intent_node)
        graph.add_node("case_extract", self._case_extract_node)
        graph.add_node("retrieval", self._retrieval_node)
        graph.add_node("diagnosis", self._diagnosis_node)
        graph.add_node("tools", self._tools_node)
        graph.add_node("action", self._action_node)
        graph.add_node("audit", self._audit_node)
        graph.add_node("final", self._final_node)

        graph.set_entry_point("intent")
        graph.add_edge("intent", "case_extract")
        graph.add_edge("case_extract", "retrieval")
        graph.add_edge("retrieval", "diagnosis")
        graph.add_edge("diagnosis", "tools")
        graph.add_edge("tools", "action")
        graph.add_edge("action", "audit")
        graph.add_edge("audit", "final")
        graph.add_edge("final", END)
        return graph.compile()

    def _run_sequential(self, state: AfterSalesGraphState) -> AfterSalesGraphState:
        for node in [
            self._intent_node,
            self._case_extract_node,
            self._retrieval_node,
            self._diagnosis_node,
            self._tools_node,
            self._action_node,
            self._audit_node,
            self._final_node,
        ]:
            state = node(state)
        return state

    def _intent_node(self, state: AfterSalesGraphState) -> AfterSalesGraphState:
        start = time.time()
        intent = self.intent_agent.determine_intent(state["user_input"])
        state["intent"] = intent
        self._add_trace(state, "intent", "意图识别", "completed", {"user_input": state["user_input"]}, intent, start)
        return state

    def _case_extract_node(self, state: AfterSalesGraphState) -> AfterSalesGraphState:
        start = time.time()
        case = self.case_agent.extract(state["user_input"])
        state["case"] = case
        self._add_trace(state, "case_extract", "售后信息提取", "completed", {"user_input": state["user_input"]}, case, start)
        return state

    def _retrieval_node(self, state: AfterSalesGraphState) -> AfterSalesGraphState:
        start = time.time()
        query = self._build_retrieval_query(state["user_input"], state.get("case", {}), state.get("intent", {}))
        retrieval = self.retrieval_agent.retrieve(query, options=state.get("retrieval_options", {}))
        state["retrieval"] = retrieval
        status = "warning" if retrieval.get("error") or not retrieval.get("results") else "completed"
        self._add_trace(
            state,
            "retrieval",
            "知识库检索",
            status,
            {"query": query},
            {"result_count": len(retrieval.get("results", [])), "error": retrieval.get("error", "")},
            start,
        )
        return state

    def _diagnosis_node(self, state: AfterSalesGraphState) -> AfterSalesGraphState:
        start = time.time()
        diagnosis = self.diagnosis_agent.diagnose(state.get("case", {}), state.get("retrieval", {}))
        state["diagnosis"] = diagnosis
        self._add_trace(state, "diagnosis", "故障诊断", "completed", {"case": state.get("case", {})}, diagnosis, start)
        return state

    def _tools_node(self, state: AfterSalesGraphState) -> AfterSalesGraphState:
        start = time.time()
        tool_bundle = self.tool_runner(state.get("case", {}), state.get("diagnosis", {}))
        state["warranty"] = tool_bundle.get("warranty", {})
        state["escalation"] = tool_bundle.get("escalation", {})
        state["ticket"] = tool_bundle.get("ticket", {})
        state["tool_history"] = tool_bundle.get("tool_history", [])
        errors = tool_bundle.get("errors", [])
        self._add_trace(
            state,
            "tools",
            "本地工具调用",
            "warning" if errors else "completed",
            {},
            {"tool_count": len(state["tool_history"]), "errors": errors},
            start,
        )
        return state

    def _action_node(self, state: AfterSalesGraphState) -> AfterSalesGraphState:
        start = time.time()
        action = self.action_agent.generate(
            case=state.get("case", {}),
            diagnosis=state.get("diagnosis", {}),
            warranty=state.get("warranty", {}),
            escalation=state.get("escalation", {}),
            ticket=state.get("ticket", {}),
        )
        state["action"] = action
        self._add_trace(
            state,
            "action",
            "客户回复与工单输出",
            "completed",
            {},
            {"has_customer_reply": bool(action.get("customer_reply"))},
            start,
        )
        return state

    def _audit_node(self, state: AfterSalesGraphState) -> AfterSalesGraphState:
        start = time.time()
        audit = self.audit_agent.audit(
            state.get("case", {}),
            state.get("diagnosis", {}),
            state.get("retrieval", {}),
            state.get("action", {}),
        )
        tool_errors = [
            item.get("error", "")
            for item in state.get("tool_history", [])
            if item.get("status") != "success" and item.get("error")
        ]
        if tool_errors:
            audit["passed"] = False
            audit.setdefault("warnings", []).extend(tool_errors)
            audit["final_note"] = "建议先检查本地工具调用结果，再人工确认后回复客户。"
        state["audit"] = audit
        self._add_trace(state, "audit", "风险审核", "completed", {}, audit, start)
        return state

    def _final_node(self, state: AfterSalesGraphState) -> AfterSalesGraphState:
        state["result"] = self._build_result(state)
        self._add_trace(state, "final", "结果汇总", "completed", {}, {"keys": list(state["result"].keys())})
        return state

    def _build_result(self, state: AfterSalesGraphState) -> dict[str, Any]:
        result = WorkflowResult().to_dict()
        for key in ["intent", "case", "retrieval", "diagnosis", "warranty", "escalation", "action", "audit"]:
            result[key] = state.get(key, {})
        result["tool_history"] = state.get("tool_history", [])
        result["trace"] = state.get("trace", [])
        return result

    def _build_retrieval_query(self, user_input: str, case: dict[str, Any], intent: dict[str, Any]) -> str:
        parts = [user_input]
        for key in ["product_model", "fault_code"]:
            if case.get(key):
                parts.append(str(case[key]))
        if case.get("symptoms"):
            parts.append(" ".join(case["symptoms"]))
        if intent.get("name") == "warranty_consultation":
            parts.append("质保 保修 免费 收费")
        if intent.get("name") == "ticket_creation":
            parts.append("售后 SOP 工单 上门")
        return " ".join(part for part in parts if part)

    def _add_trace(
        self,
        state: AfterSalesGraphState,
        node: str,
        title: str,
        status: str,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
        start_time: float | None = None,
    ) -> None:
        item = {
            "node": node,
            "title": title,
            "status": status,
            "input": input_data,
            "output": output_data,
            "timestamp": round(time.time(), 3),
        }
        if start_time is not None:
            item["duration"] = round(time.time() - start_time, 3)
        state.setdefault("trace", []).append(item)
