from __future__ import annotations

import json
import time
from typing import Any, Callable, TypedDict

from backend import config
from backend.agents.action_agent import ChargerActionAgent
from backend.agents.audit_agent import ChargerAuditAgent
from backend.agents.case_extract_agent import ChargerCaseExtractAgent
from backend.agents.diagnosis_agent import ChargerDiagnosisAgent
from backend.agents.intent_agent import ChargerTriageAgent
from backend.agents.llm_utils import invoke_json
from backend.agents.retrieval_agent import RetrievalAgent
from backend.llm.factory import get_chat_model
from backend.memory import MemoryManager, get_memory_manager
from backend.prompts.memory_query import (
    FTS5_FIELD_EXTRACTION_PROMPT,
    MEMORY_ANSWER_GENERATION_PROMPT,
    MEMORY_QUERY_PARSE_PROMPT,
    _MEMORY_ANSWER_FIELD_LABEL,
)
from backend.rag.rag_service import RAGService
from backend.rules import case_rules, dispatch_rules, input_rules, output_rules, safety_rules
from backend.schemas import (
    ChargerActionResult,
    ChargerAuditResult,
    ChargerCase,
    ChargerDiagnosisResult,
    ChargerWorkflowResult,
    DispatchDraft,
    MemoryFieldResolution,
    MemoryQueryResult,
    SafetyResult,
    TriageResult,
    WarrantyResult,
    normalize_power_kw,
)
from backend.tools.memory import MemoryContextReadTool, MemoryWorkflowWriteTool, default_memory_context
from backend.tools.warranty import WarrantyTool

try:
    from langgraph.graph import END, StateGraph
except ModuleNotFoundError:  # pragma: no cover - 未安装 LangGraph 时走顺序兜底
    END = "__end__"
    StateGraph = None


_AUTO_LLM = object()


class ChargerDiagnosisGraphState(TypedDict, total=False):
    user_input: str
    retrieval_options: dict[str, Any]
    llm_available: bool
    triage: dict[str, Any]
    case: dict[str, Any]
    safety: dict[str, Any]
    retrieval: dict[str, Any]
    diagnosis: dict[str, Any]
    warranty: dict[str, Any]
    dispatch: dict[str, Any]
    action: dict[str, Any]
    audit: dict[str, Any]
    tool_history: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    progress_callback: Callable[[dict[str, Any]], None]
    session_id: str
    memory_manager: MemoryManager | None
    input_safety: dict[str, Any]
    memory_context: dict[str, Any]
    case_memory_merge: dict[str, Any]
    governance: dict[str, Any]
    result: dict[str, Any]


def _format_field_value(value: Any) -> str:
    """将字段值格式化为可读字符串（deterministic）。"""
    if isinstance(value, list):
        return "、".join(str(v).strip() for v in value if str(v).strip())
    if isinstance(value, bool):
        return "是" if value else "否"
    return str(value).strip()


class ChargerDiagnosisWorkflow:
    """使用 LangGraph 编排家用充电桩售后安全诊断流程。"""

    def __init__(
        self,
        retrieval_func: Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]] | None = None,
        rag_service: RAGService | None = None,
        llm: Any = _AUTO_LLM,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        self.llm = self._safe_get_llm() if llm is _AUTO_LLM else llm
        self.triage_agent = ChargerTriageAgent(llm=self.llm)
        self.case_agent = ChargerCaseExtractAgent(llm=self.llm)
        self.rag_service = rag_service or RAGService(retrieval_func=retrieval_func)
        self.retrieval_agent = RetrievalAgent(self.rag_service)
        self.diagnosis_agent = ChargerDiagnosisAgent(llm=self.llm)
        self.action_agent = ChargerActionAgent(llm=self.llm)
        self.audit_agent = ChargerAuditAgent(llm=self.llm)
        self.memory_read_tool = MemoryContextReadTool()
        self.memory_write_tool = MemoryWorkflowWriteTool()
        self.warranty_tool = WarrantyTool()
        self.memory_manager = memory_manager or get_memory_manager()
        self._compiled_graph = self._build_graph()

    def run(
        self,
        user_input: str,
        retrieval_options: dict[str, Any] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        session_id: str | None = None,
        memory_manager: MemoryManager | None = None,
    ) -> dict[str, Any]:
        active_memory = memory_manager or self.memory_manager
        active_session_id = session_id or ""
        if active_memory is not None:
            active_session_id = active_memory.get_or_create_session(session_id).session_id

        # 第一层：显式记忆标记（"刚才/之前/还记得"等关键词）→ 直接 memory_answer
        should_try_memory = self._is_memory_recall_query(user_input)
        # 第二层：确定性上下文追问 gate（不调 LLM）
        if not should_try_memory and active_memory is not None:
            should_try_memory = self._maybe_contextual_memory_query(
                user_input or "", active_memory, active_session_id
            )

        if should_try_memory:
            result = self._run_memory_answer(
                user_input=user_input or "",
                session_id=active_session_id,
                memory_manager=active_memory,
                progress_callback=progress_callback,
            )
            if result is not None:
                return result
            # result is None → LLM 判断非 memory_query → 回退主诊断链路

        state: ChargerDiagnosisGraphState = {
            "user_input": user_input or "",
            "retrieval_options": retrieval_options or {},
            "llm_available": self.llm is not None,
            "tool_history": [],
            "trace": [],
            "session_id": active_session_id,
            "memory_manager": active_memory,
        }
        if progress_callback is not None:
            state["progress_callback"] = progress_callback
        if self.llm is None:
            self._add_trace(state, "llm", "LLM 状态", "warning", {}, {"status": "llm_unavailable"})

        try:
            if self._compiled_graph is not None:
                final_state = self._compiled_graph.invoke(state)
            else:
                final_state = self._run_sequential(state)
            result = final_state.get("result") or self._build_result(final_state)
            self._remember_workflow_result(active_memory, active_session_id, user_input or "", result)
            return result
        except Exception as exc:  # pragma: no cover - 工作流最外层防御
            self._add_trace(state, "error", "流程异常", "failed", {}, {"error": str(exc)})
            result = ChargerWorkflowResult().to_dict()
            result["audit"] = {
                "passed": False,
                "warnings": [f"充电桩安全诊断流程运行异常：{exc}"],
                "final_note": "建议人工确认后再回复客户。",
                "risk_level": "unknown",
            }
            result["trace"] = state.get("trace", [])
            result["tool_history"] = state.get("tool_history", [])
            return result

    def _run_memory_answer(
        self,
        user_input: str,
        session_id: str,
        memory_manager: MemoryManager | None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        state: ChargerDiagnosisGraphState = {
            "user_input": user_input,
            "session_id": session_id,
            "trace": [],
            "tool_history": [],
            "memory_manager": memory_manager,
        }
        if progress_callback is not None:
            state["progress_callback"] = progress_callback

        self._input_guard_node(state)
        self._memory_context_node(state)
        self._memory_answer_node(state)

        # LLM 判断非 memory_query → 返回 None 通知 run() 回退主诊断链路
        if state.get("memory_answer_rejected"):
            return None  # type: ignore[return-value]

        self._final_node(state)
        return state.get("result") or self._build_result(state)

    def _memory_answer_node(self, state: ChargerDiagnosisGraphState) -> ChargerDiagnosisGraphState:
        start = time.time()
        user_input = state.get("user_input", "")
        memory_context = state.get("memory_context", {})

        # -----------------------------------------------------------------
        # memory_answer v2：parse → resolve → reply 三阶段
        # -----------------------------------------------------------------
        if config.MEMORY_ANSWER_V2:
            # 阶段 1: LLM 解析查询意图
            parsed = self._parse_memory_query(user_input, state)

            # LLM 明确判断非记忆查询 → 标记 rejected 回退主诊断链路
            if not parsed.is_memory_query and not parsed.fallback_reason:
                self._add_trace(
                    state, "memory_answer", "非记忆查询",
                    "warning",
                    {"user_input": user_input},
                    {"note": "LLM 判断 is_memory_query=false，回退到主诊断链路。",
                     "parse_result": parsed.to_dict()},
                )
                state["memory_answer_rejected"] = True
                return state

            # 阶段 2: field resolver v1 从结构化来源取值
            resolution = self._resolve_memory_fields(parsed, memory_context, user_input, state)

            self._emit_progress(
                state,
                "memory_answer",
                "会话记忆回答",
                "running",
                {"session_id": state.get("session_id", ""), "version": "v2",
                 "is_memory_query": parsed.is_memory_query,
                 "target_fields": parsed.target_fields,
                 "fallback_reason": parsed.fallback_reason,
                 "resolved_count": len(resolution.resolved_values),
                 "missing_count": len(resolution.missing_fields),
                 "confidence": resolution.confidence},
                {},
            )

            # 阶段 3: 回复生成（Answer LLM，不可用时回退 _build_memory_reply_v2）
            reply = self._build_memory_answer_llm(parsed, resolution, user_input, state)

            state["triage"] = TriageResult(
                intent="memory_answer",
                confidence="high",
                reason="命中会话记忆查询请求（memory_answer v2），直接读取当前 SessionMemory，不进入 RAG 诊断链路。",
            ).to_dict()
            state["case"] = ChargerCase(
                issue_type="memory_answer",
                issue_description=user_input,
                customer_requests=["会话记忆查询"],
                missing_info=[],
                raw_text=user_input,
            ).to_dict()
            state["retrieval"] = {
                "query": user_input,
                "results": [],
                "sources": [],
                "trace": {
                    "mode": "memory_answer",
                    "version": "v2",
                    "session_id": state.get("session_id", ""),
                    "parse_result": parsed.to_dict(),
                    "resolution": resolution.to_dict(),
                },
            }
            state["safety"] = SafetyResult(
                risk_level="p3_low",
                reason="会话记忆查询请求，不作为充电桩安全诊断。",
            ).to_dict()
            state["diagnosis"] = ChargerDiagnosisResult(
                summary="会话记忆查询请求，不做售后诊断。",
                evidence_status="insufficient",
                priority="p3_low",
                suggested_next_step="如需继续处理充电桩问题，请补充当前故障现象或故障码。",
            ).to_dict()
            state["warranty"] = WarrantyResult().to_dict()
            state["dispatch"] = DispatchDraft(
                customer_problem=user_input,
                suggested_dispatch="会话记忆查询请求，不创建派工。",
                priority="p3_low",
            ).to_dict()
            state["action"] = ChargerActionResult(
                customer_reply=reply,
                internal_advice="会话记忆查询请求（memory_answer v2）。",
            ).to_dict()
            state["audit"] = ChargerAuditResult(
                passed=True,
                final_note="会话记忆查询请求，无需售后诊断审核。",
                risk_level="p3_low",
            ).to_dict()
            state["governance"] = input_rules.build_governance_summary(
                input_safety=state.get("input_safety", {}),
                memory_context=memory_context,
                audit=state.get("audit", {}),
            )
            self._add_trace(
                state,
                "memory_answer",
                "会话记忆回答（v2）",
                "completed",
                {"session_id": state.get("session_id", ""), "version": "v2",
                 "target_fields": parsed.target_fields,
                 "resolved_count": len(resolution.resolved_values),
                 "missing_count": len(resolution.missing_fields),
                 "confidence": resolution.confidence,
                 "fallback_reason": parsed.fallback_reason},
                {"message": reply,
                 "parse_result": parsed.to_dict(),
                 "resolution": resolution.to_dict()},
                start,
            )
            return state

        # -----------------------------------------------------------------
        # 旧链路（MEMORY_ANSWER_V2=False 时保留）
        # -----------------------------------------------------------------
        answer_type = self._memory_query_type(user_input)  # deprecated
        self._emit_progress(
            state,
            "memory_answer",
            "会话记忆回答",
            "running",
            {"session_id": state.get("session_id", ""), "answer_type": answer_type},
            {},
        )
        reply = self._build_memory_reply(answer_type, user_input, memory_context)  # deprecated
        state["triage"] = TriageResult(
            intent="memory_answer",
            confidence="high",
            reason="命中会话记忆查询请求，直接读取当前 SessionMemory，不进入 RAG 诊断链路。",
        ).to_dict()
        state["case"] = ChargerCase(
            issue_type="memory_answer",
            issue_description=user_input,
            customer_requests=["会话记忆查询"],
            missing_info=[],
            raw_text=user_input,
        ).to_dict()
        state["retrieval"] = {
            "query": user_input,
            "results": [],
            "sources": [],
            "trace": {"mode": "memory_answer", "session_id": state.get("session_id", ""), "answer_type": answer_type},
        }
        state["safety"] = SafetyResult(
            risk_level="p3_low",
            reason="会话记忆查询请求，不作为充电桩安全诊断。",
        ).to_dict()
        state["diagnosis"] = ChargerDiagnosisResult(
            summary="会话记忆查询请求，不做售后诊断。",
            evidence_status="insufficient",
            priority="p3_low",
            suggested_next_step="如需继续处理充电桩问题，请补充当前故障现象或故障码。",
        ).to_dict()
        state["warranty"] = WarrantyResult().to_dict()
        state["dispatch"] = DispatchDraft(
            customer_problem=user_input,
            suggested_dispatch="会话记忆查询请求，不创建派工。",
            priority="p3_low",
        ).to_dict()
        state["action"] = ChargerActionResult(customer_reply=reply, internal_advice="会话记忆查询请求。").to_dict()
        state["audit"] = ChargerAuditResult(
            passed=True,
            final_note="会话记忆查询请求，无需售后诊断审核。",
            risk_level="p3_low",
        ).to_dict()
        state["governance"] = input_rules.build_governance_summary(
            input_safety=state.get("input_safety", {}),
            memory_context=memory_context,
            audit=state.get("audit", {}),
        )
        self._add_trace(
            state,
            "memory_answer",
            "会话记忆回答",
            "completed",
            {"session_id": state.get("session_id", ""), "answer_type": answer_type},
            {"answer_type": answer_type, "message": reply},
            start,
        )
        return state

    def _safe_get_llm(self) -> Any | None:
        try:
            return get_chat_model()
        except Exception:
            return None

    def _build_graph(self) -> Any:
        if StateGraph is None:
            return None

        graph = StateGraph(ChargerDiagnosisGraphState)
        graph.add_node("input_guard", self._input_guard_node)
        graph.add_node("triage", self._triage_node)
        graph.add_node("case_extract", self._case_extract_node)
        graph.add_node("memory_context", self._memory_context_node)
        graph.add_node("case_memory_merge", self._case_memory_merge_node)
        graph.add_node("safety_guard", self._safety_guard_node)
        graph.add_node("retrieval", self._retrieval_node)
        graph.add_node("diagnosis", self._diagnosis_node)
        graph.add_node("warranty_dispatch", self._warranty_dispatch_node)
        graph.add_node("action", self._action_node)
        graph.add_node("audit", self._audit_node)
        graph.add_node("final", self._final_node)

        graph.set_entry_point("input_guard")
        graph.add_edge("input_guard", "triage")
        graph.add_edge("triage", "case_extract")
        graph.add_edge("case_extract", "memory_context")
        graph.add_edge("memory_context", "case_memory_merge")
        graph.add_edge("case_memory_merge", "safety_guard")
        graph.add_edge("safety_guard", "retrieval")
        graph.add_edge("retrieval", "diagnosis")
        graph.add_edge("diagnosis", "warranty_dispatch")
        graph.add_edge("warranty_dispatch", "action")
        graph.add_edge("action", "audit")
        graph.add_edge("audit", "final")
        graph.add_edge("final", END)
        return graph.compile()

    def _run_sequential(self, state: ChargerDiagnosisGraphState) -> ChargerDiagnosisGraphState:
        for node in [
            self._input_guard_node,
            self._triage_node,
            self._case_extract_node,
            self._memory_context_node,
            self._case_memory_merge_node,
            self._safety_guard_node,
            self._retrieval_node,
            self._diagnosis_node,
            self._warranty_dispatch_node,
            self._action_node,
            self._audit_node,
            self._final_node,
        ]:
            state = node(state)
        return state

    def _input_guard_node(self, state: ChargerDiagnosisGraphState) -> ChargerDiagnosisGraphState:
        start = time.time()
        self._emit_progress(state, "input_guard", "输入安全扫描", "running", {"user_input": state["user_input"]}, {})
        input_safety = input_rules.scan_input_safety(state.get("user_input", ""))
        state["input_safety"] = input_safety
        status = "warning" if input_safety.get("warnings") else "completed"
        self._add_trace(
            state,
            "input_guard",
            "输入安全扫描",
            status,
            {"user_input": state.get("user_input", "")},
            {
                "status": input_safety.get("status"),
                "prompt_injection_detected": input_safety.get("prompt_injection_detected"),
                "warning_count": len(input_safety.get("warnings", [])),
            },
            start,
        )
        return state

    def _triage_node(self, state: ChargerDiagnosisGraphState) -> ChargerDiagnosisGraphState:
        start = time.time()
        self._emit_progress(state, "triage", "安全分诊", "running", {"user_input": state["user_input"]}, {})
        triage = self.triage_agent.triage(state["user_input"])
        state["triage"] = triage
        self._add_trace(state, "triage", "安全分诊", "completed", {"user_input": state["user_input"]}, triage, start)
        return state

    def _case_extract_node(self, state: ChargerDiagnosisGraphState) -> ChargerDiagnosisGraphState:
        start = time.time()
        self._emit_progress(state, "case_extract", "充电桩信息提取", "running", {"user_input": state["user_input"]}, {})
        extracted_case = self.case_agent.extract(state["user_input"])
        case = case_rules.normalize_charger_case(extracted_case, state["user_input"])
        state["case"] = case
        self._add_trace(state, "case_extract", "充电桩信息提取", "completed", {"user_input": state["user_input"]}, case, start)
        return state

    def _memory_context_node(self, state: ChargerDiagnosisGraphState) -> ChargerDiagnosisGraphState:
        start = time.time()
        memory_manager = state.get("memory_manager")
        session_id = state.get("session_id", "")
        self._emit_progress(
            state,
            "memory_context",
            "会话记忆读取",
            "running",
            {"session_id": session_id, "case_keys": list(state.get("case", {}).keys())},
            {},
        )
        read_result = self.memory_read_tool.execute(
            memory_manager=memory_manager,
            case=state.get("case", {}),
            session_id=session_id,
        )
        state.setdefault("tool_history", []).append(self._tool_history_item(
            read_result,
            {"session_id": session_id, "case": state.get("case", {})},
        ))
        memory_context = read_result.data if read_result.success else default_memory_context(
            session_id=session_id,
            policy=f"会话记忆读取失败：{read_result.error}",
        )
        state["memory_context"] = memory_context
        self._add_trace(
            state,
            "memory_context",
            "会话记忆读取",
            "completed" if read_result.success else "warning",
            {"session_id": session_id},
            {
                "matched_ids": memory_context.get("matched_ids", {}),
                "message_count": memory_context.get("session", {}).get("message_count", 0),
                "last_model": memory_context.get("last_case", {}).get("charger_model", ""),
                "missing_count": len(memory_context.get("missing_info", []) or []),
                "used_as_diagnostic_evidence": False,
                "tool_name": read_result.tool_name,
                "error": read_result.error,
            },
            start,
        )
        return state

    def _case_memory_merge_node(self, state: ChargerDiagnosisGraphState) -> ChargerDiagnosisGraphState:
        start = time.time()
        self._emit_progress(
            state,
            "case_memory_merge",
            "多轮 Case 合并",
            "running",
            {"case_keys": list(state.get("case", {}).keys()), "session_id": state.get("session_id", "")},
            {},
        )
        merged_case = case_rules.merge_case_with_memory(state.get("case", {}), state.get("memory_context", {}))
        merge_meta = merged_case.get("_memory_merge", {}) if isinstance(merged_case, dict) else {}
        state["case"] = merged_case
        state["case_memory_merge"] = merge_meta
        if isinstance(state.get("memory_context"), dict):
            state["memory_context"]["case_merge"] = merge_meta
        self._add_trace(
            state,
            "case_memory_merge",
            "多轮 Case 合并",
            "completed",
            {"session_id": state.get("session_id", "")},
            {
                "applied": bool(merge_meta.get("applied")),
                "filled_fields": merge_meta.get("filled_fields", []),
                "merged_list_fields": merge_meta.get("merged_list_fields", []),
                "used_as_diagnostic_evidence": False,
            },
            start,
        )
        return state

    def _safety_guard_node(self, state: ChargerDiagnosisGraphState) -> ChargerDiagnosisGraphState:
        start = time.time()
        self._emit_progress(state, "safety_guard", "安全护栏", "running", {"case": state.get("case", {})}, {})
        safety = safety_rules.evaluate_charger_safety(state.get("case", {}), raw_text=state.get("user_input", ""))
        state["safety"] = safety
        status = "warning" if safety.get("risk_level") in {"p0_emergency", "p1_high"} else "completed"
        self._add_trace(state, "safety_guard", "安全护栏", status, {"case": state.get("case", {})}, safety, start)
        return state

    def _retrieval_node(self, state: ChargerDiagnosisGraphState) -> ChargerDiagnosisGraphState:
        start = time.time()
        query = self._build_retrieval_query(state["user_input"], state.get("case", {}), state.get("triage", {}))
        self._emit_progress(state, "retrieval", "知识库检索", "running", {"query": query}, {})
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

    def _diagnosis_node(self, state: ChargerDiagnosisGraphState) -> ChargerDiagnosisGraphState:
        start = time.time()
        self._emit_progress(state, "diagnosis", "安全诊断", "running", {"case": state.get("case", {})}, {})
        tools = {"safety": state.get("safety", {})}
        diagnosis = self.diagnosis_agent.diagnose(state.get("case", {}), state.get("retrieval", {}), tools=tools)
        diagnosis = output_rules.enforce_diagnosis_grounding(diagnosis, state.get("case", {}), state.get("retrieval", {}))
        diagnosis = safety_rules.enforce_diagnosis(diagnosis, state.get("safety", {}))
        state["diagnosis"] = diagnosis
        self._add_trace(state, "diagnosis", "安全诊断", "completed", {"case": state.get("case", {})}, diagnosis, start)
        return state

    def _warranty_dispatch_node(self, state: ChargerDiagnosisGraphState) -> ChargerDiagnosisGraphState:
        start = time.time()
        self._emit_progress(state, "warranty_dispatch", "保修核验与派工草稿", "running", {}, {})
        case = state.get("case", {})
        warranty_result = self.warranty_tool.execute(
            purchase_or_install_time=case.get("purchase_or_install_time"),
            raw_text=case.get("raw_text", ""),
            retrieval=state.get("retrieval", {}),
        )
        dispatch = dispatch_rules.build_dispatch(
            case=case,
            diagnosis=state.get("diagnosis", {}),
            warranty=warranty_result.data,
            safety=state.get("safety", {}),
        )
        state["warranty"] = warranty_result.data
        state["dispatch"] = dispatch
        state.setdefault("tool_history", []).append(self._tool_history_item(warranty_result, {"case": case}))
        errors = [warranty_result.error] if not warranty_result.success and warranty_result.error else []
        self._add_trace(
            state,
            "warranty_dispatch",
            "保修核验与派工草稿",
            "warning" if errors else "completed",
            {},
            {"tool_count": 1, "errors": errors, "has_dispatch": bool(dispatch.get("title"))},
            start,
        )
        return state

    def _action_node(self, state: ChargerDiagnosisGraphState) -> ChargerDiagnosisGraphState:
        start = time.time()
        self._emit_progress(state, "action", "客户回复与派工输出", "running", {}, {})
        action = self.action_agent.generate(
            case=state.get("case", {}),
            diagnosis=state.get("diagnosis", {}),
            warranty=state.get("warranty", {}),
            safety=state.get("safety", {}),
            dispatch=state.get("dispatch", {}),
            retrieval=state.get("retrieval", {}),
            triage=state.get("triage", {}),
        )
        action = output_rules.enforce_reply(
            action,
            case=state.get("case", {}),
            safety=state.get("safety", {}),
            warranty=state.get("warranty", {}),
            retrieval=state.get("retrieval", {}),
            dispatch=state.get("dispatch", {}),
        )
        state["action"] = action
        self._add_trace(
            state,
            "action",
            "客户回复与派工输出",
            "completed",
            {},
            {"has_customer_reply": bool(action.get("customer_reply"))},
            start,
        )
        return state

    def _audit_node(self, state: ChargerDiagnosisGraphState) -> ChargerDiagnosisGraphState:
        start = time.time()
        self._emit_progress(state, "audit", "安全审核", "running", {}, {})
        audit = self.audit_agent.audit(
            state.get("case", {}),
            state.get("diagnosis", {}),
            state.get("retrieval", {}),
            state.get("action", {}),
            safety=state.get("safety", {}),
            warranty=state.get("warranty", {}),
        )
        audit = output_rules.merge_with_local_audit(
            audit,
            case=state.get("case", {}),
            diagnosis=state.get("diagnosis", {}),
            action=state.get("action", {}),
            safety=state.get("safety", {}),
            warranty=state.get("warranty", {}),
            retrieval=state.get("retrieval", {}),
            input_safety=state.get("input_safety", {}),
            memory_context=state.get("memory_context", {}),
        )
        if self.llm is None:
            audit["passed"] = False
            audit.setdefault("warnings", []).append("未启用 LLM，当前回复为规则和知识库兜底输出。")
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
        self._add_trace(state, "audit", "安全审核", "completed", {}, audit, start)
        return state

    def _final_node(self, state: ChargerDiagnosisGraphState) -> ChargerDiagnosisGraphState:
        self._emit_progress(state, "final", "结果汇总", "running", {}, {})
        state["governance"] = input_rules.build_governance_summary(
            input_safety=state.get("input_safety", {}),
            memory_context=state.get("memory_context", {}),
            audit=state.get("audit", {}),
        )
        state["result"] = self._build_result(state)
        self._add_trace(state, "final", "结果汇总", "completed", {}, {"keys": list(state["result"].keys())})
        return state

    def _build_result(self, state: ChargerDiagnosisGraphState) -> dict[str, Any]:
        result = ChargerWorkflowResult().to_dict()
        for key in [
            "input_safety",
            "triage",
            "case",
            "memory_context",
            "retrieval",
            "safety",
            "diagnosis",
            "warranty",
            "dispatch",
            "action",
            "audit",
            "governance",
        ]:
            result[key] = state.get(key, {})
        result["tool_history"] = state.get("tool_history", [])
        result["trace"] = state.get("trace", [])
        return result

    def _remember_workflow_result(
        self,
        memory_manager: MemoryManager | None,
        session_id: str,
        user_input: str,
        result: dict[str, Any],
    ) -> None:
        if memory_manager is None:
            return
        started = time.time()
        write_result = self.memory_write_tool.execute(
            memory_manager=memory_manager,
            session_id=session_id,
            user_input=user_input,
            result=result,
        )
        result.setdefault("tool_history", []).append(self._tool_history_item(
            write_result,
            {"session_id": session_id},
        ))
        result.setdefault("trace", []).append({
            "node": "memory",
            "title": "会话记忆沉淀",
            "status": "completed" if write_result.success else "warning",
            "input": {"session_id": session_id},
            "output": write_result.data if write_result.success else {"error": write_result.error},
            "timestamp": round(time.time(), 3),
            "duration": round(time.time() - started, 3),
        })

    def _is_memory_recall_query(self, text: str) -> bool:
        """[deprecated] 硬编码关键词粗筛（含 _memory_query_type 调用），仅用于 MEMORY_ANSWER_V2=False 的旧链路。

        memory_answer v2 上线后此函数保持不变，作为 _parse_memory_query() 之前的
        轻量本地粗判断——避免普通诊断请求每轮都调用 LLM。
        """
        # deterministic: 本地轻量粗筛，不依赖 LLM，关键词列表为确定性规则
        compact = str(text or "").strip()
        if not compact:
            return False
        if self._memory_query_type(compact) != "unknown":
            return True
        recent_markers = ["刚才", "刚刚", "前面", "之前", "前一个", "刚说", "刚提"]
        recall_markers = ["还记得", "记得", "复述", "说的是", "提到的是", "问的是", "我说的"]
        if any(marker in compact for marker in recent_markers) and any(marker in compact for marker in recall_markers):
            return True
        return any(pattern in compact for pattern in [
            "上一条问",
            "上一条用户问题",
            "上一个问题",
            "上条问",
            "刚才问",
            "刚刚问",
            "刚才提到",
            "刚刚提到",
            "刚才那个",
            "刚刚那个",
            "前面问",
            "上次问",
            "我问的什么",
        ])

    def _memory_query_type(self, text: str) -> str:
        """[deprecated] 硬编码字段级 if/else 匹配，仅用于 MEMORY_ANSWER_V2=False 的旧链路。

        memory_answer v2 中已替换为 _parse_memory_query() + MemoryQueryResult。
        旧链路移除时本函数应一并删除。
        """
        # deterministic: 字段级关键词匹配为确定性规则，不走 LLM
        compact = str(text or "").strip()
        if not compact:
            return "unknown"
        memory_markers = ["刚才", "刚刚", "前面", "之前", "上次", "那个", "还记得", "记得", "你记住", "当前记住"]
        has_memory_marker = any(marker in compact for marker in memory_markers)
        if has_memory_marker and any(token in compact for token in ["型号", "款型", "哪一款", "哪个型号"]):
            return "model"
        if has_memory_marker and any(token in compact for token in ["城市", "哪里", "哪儿", "在哪", "哪个地方", "什么地方"]):
            return "city"
        if any(token in compact for token in ["缺哪些", "缺少哪些", "还缺", "缺什么", "补充哪些", "缺失信息"]):
            return "missing_info"
        if has_memory_marker and any(token in compact for token in ["风险等级", "风险级别", "刚才风险", "安全等级"]):
            return "risk_level"
        if has_memory_marker and "工单" in compact and any(token in compact for token in ["优先级", "等级", "priority"]):
            return "ticket_priority"
        if any(token in compact for token in ["记住了哪些信息", "记了哪些信息", "记住哪些", "你记住了什么", "当前记住"]):
            return "remembered_summary"
        if any(token in compact for token in ["我刚才说了什么", "刚才说了什么", "刚刚说了什么", "上一条问", "上一个问题", "我问的什么"]):
            return "last_user_message"
        recent_markers = ["刚才", "刚刚", "前面", "之前", "上次"]
        recall_markers = ["还记得", "记得", "复述", "说的是", "问的是", "提到"]
        if any(marker in compact for marker in recent_markers) and any(marker in compact for marker in recall_markers):
            return "last_user_message"
        return "unknown"

    @staticmethod
    def _extract_recent_entities(session: Any) -> list[str]:
        """从 SessionMemory 提取上一轮实体值（确定性，不调 LLM）。"""
        entities: list[str] = []
        if session is None:
            return entities
        last_case = (getattr(session, "context", None) or {}).get("last_case") or {}
        for key in ("charger_model", "brand", "serial_number", "city"):
            val = str(last_case.get(key, "")).strip()
            if len(val) >= 2:
                entities.append(val)
        for code in last_case.get("fault_codes") or []:
            code = str(code).strip()
            if len(code) >= 2:
                entities.append(code)
        return entities

    def _maybe_contextual_memory_query(
        self, user_input: str, memory_manager: Any, session_id: str
    ) -> bool:
        """确定性二次判断：session 有上下文 & 输入短 & 可能为上下文追问。

        不调 LLM。仅判断"是否值得交给 _parse_memory_query 做 LLM 精细解析"。
        """
        compact = str(user_input or "").strip()
        if not compact:
            return False

        # 条件 1：session 有上下文
        session = getattr(memory_manager, "sessions", {}).get(session_id)
        if session is None:
            return False
        messages = getattr(session, "messages", []) or []
        if len(messages) == 0:
            return False
        last_case = (getattr(session, "context", None) or {}).get("last_case") or {}
        if not last_case:
            return False

        # 条件 2：用户输入较短（正常诊断输入通常 > 50 字符）
        if len(compact) >= 50:
            return False

        # 条件 3：包含上一轮实体 或 上下文追问标记
        entities = self._extract_recent_entities(session)
        lower_input = compact.lower()
        for entity in entities:
            if entity.lower() in lower_input:
                return True

        # 上下文追问标记（不含"漏保/跳闸/故障码"，它们需要实体命中才能进入）
        context_markers = [
            "这个", "那个", "它", "该",
            "多少功率", "功率是多少",
            "什么型号", "哪个型号",
            "还缺", "缺什么",
            "优先级", "风险等级", "工单",
        ]
        if any(marker in compact for marker in context_markers):
            return True

        return False

    # ------------------------------------------------------------------
    # memory_answer v2：LLM 驱动的记忆查询解析
    # ------------------------------------------------------------------

    @staticmethod
    def _build_session_context_for_parse(state: ChargerDiagnosisGraphState) -> str:
        """从 memory_context 构建轻量 session context 文本，供 parse LLM 使用。"""
        mem = state.get("memory_context") or {}
        lines: list[str] = []

        session = mem.get("session") or {}
        recent_msgs = session.get("recent_user_messages") or []
        if recent_msgs:
            lines.append(f"- 最近用户消息：{json.dumps(recent_msgs[-2:], ensure_ascii=False)}")

        last_case = mem.get("last_case") or {}
        for key, label in [
            ("brand", "品牌"), ("charger_model", "型号"), ("rated_power_kw", "额定功率"),
            ("city", "城市"), ("fault_codes", "故障码"),
        ]:
            val = last_case.get(key)
            if val is not None and val != "" and val != []:
                lines.append(f"- 当前案例{label}：{json.dumps(val, ensure_ascii=False)}")

        missing = mem.get("missing_info") or []
        if missing:
            lines.append(f"- 缺失信息：{json.dumps(missing, ensure_ascii=False)}")

        safety = mem.get("recent_safety") or {}
        risk = safety.get("risk_level", "")
        if risk:
            lines.append(f"- 安全风险等级：{risk}")

        ticket = mem.get("recent_ticket") or {}
        for key, label in [("ticket_id", "工单 ID"), ("priority", "工单优先级")]:
            val = ticket.get(key, "")
            if val:
                lines.append(f"- {label}：{val}")

        return "\n".join(lines) if lines else "（无当前会话上下文）"

    def _parse_memory_query(self, user_input: str, state: ChargerDiagnosisGraphState) -> MemoryQueryResult:
        """LLM 驱动的记忆查询解析（memory_answer v2）。

        调用 LLM 解析用户输入，判断是否为记忆查询，提取 target_fields、
        query_scope、entities、answer_style。输出经过 clean_fields()、
        normalize_scope()、normalize_answer_style() 清洗。

        LLM 不可用 / 解析异常时返回 is_memory_query=True + fallback_reason，
        而非 is_memory_query=False，以保持语义一致（resolver 回退到全字段扫描）。
        调用方通过检查 is_memory_query=False 且 fallback_reason 为空来判断
        LLM 主动判定非记忆查询，此时应回退主诊断链路。
        非法字段、非法 scope/style 会写入 state trace 的 warning 节点。
        """
        result = MemoryQueryResult(is_memory_query=True, fallback_reason="")

        if self.llm is None:
            result.fallback_reason = "llm_unavailable"
            self._add_trace(state, "memory_parse", "LLM 不可用", "warning",
                            {}, {"note": "memory_answer v2 parse 跳过，is_memory_query=True，resolver 将回退到全字段扫描。",
                                 "fallback_reason": result.fallback_reason})
            return result

        try:
            session_ctx = self._build_session_context_for_parse(state)
            parsed = invoke_json(self.llm, MEMORY_QUERY_PARSE_PROMPT, {
                "user_input": user_input,
                "session_context": session_ctx,
            })
            if not parsed:
                result.fallback_reason = "empty_response"
                self._add_trace(state, "memory_parse", "LLM 返回空", "warning",
                                {}, {"note": "invoke_json 返回空字典，is_memory_query=True，resolver 将回退到全字段扫描。",
                                     "fallback_reason": result.fallback_reason})
                return result

            result.is_memory_query = bool(parsed.get("is_memory_query", False))
            result.target_fields = parsed.get("target_fields", [])
            result.query_scope = str(parsed.get("query_scope", "recent"))
            result.entities = parsed.get("entities", [])
            result.answer_style = str(parsed.get("answer_style", "precise"))

            # --- 校验 + 清洗 ---
            illegal_fields = result.validate_fields()
            if illegal_fields:
                self._add_trace(
                    state, "memory_parse", "非法 target_fields 丢弃",
                    "warning",
                    {"illegal_fields": illegal_fields,
                     "raw_target_fields": parsed.get("target_fields", [])},
                    {"note": "LLM 返回了不在 MEMORY_QUERY_TARGET_FIELDS 中的字段，已丢弃。"},
                )
                result.clean_fields()

            if not result.validate_query_scope():
                old_scope = result.query_scope
                result.normalize_scope()
                self._add_trace(
                    state, "memory_parse", "非法 query_scope 回退",
                    "warning",
                    {"illegal_scope": old_scope},
                    {"normalized_scope": result.query_scope,
                     "note": f"query_scope 值 '{old_scope}' 不在允许列表中，已回退为 'recent'。"},
                )

            if not result.validate_answer_style():
                old_style = result.answer_style
                result.normalize_answer_style()
                self._add_trace(
                    state, "memory_parse", "非法 answer_style 回退",
                    "warning",
                    {"illegal_answer_style": old_style},
                    {"normalized_answer_style": result.answer_style,
                     "note": f"answer_style 值 '{old_style}' 不在允许列表中，已回退为 'precise'。"},
                )

            return result

        except Exception as exc:
            result.fallback_reason = "parse_failed"
            self._add_trace(state, "memory_parse", "LLM 解析异常", "failed",
                            {}, {"error": str(exc),
                                 "fallback_reason": result.fallback_reason,
                                 "note": "memory_answer v2 parse 异常，is_memory_query=True，resolver 将回退到全字段扫描。"})
            return result

    # ------------------------------------------------------------------
    # memory_answer v2：field resolver v1（结构化来源 + FTS5 fallback）
    # ------------------------------------------------------------------

    def _resolve_memory_fields(
        self,
        parsed: MemoryQueryResult,
        memory_context: dict[str, Any],
        user_input: str = "",
        state: ChargerDiagnosisGraphState | None = None,
    ) -> MemoryFieldResolution:
        """从 memory_context 中解析 target_fields 的值。

        Pass 1（结构化来源 → confidence=high）：
          - ChargerCase 字段 → memory_context["last_case"]
          - SafetyResult 字段 → memory_context["recent_safety"]
          - ChargerDiagnosisResult 字段 → memory_context["session"]["last_diagnosis"]
          - DispatchDraft 字段 → memory_context["recent_ticket"]
          - SessionMemory 字段 → memory_context["last_customer_reply"] /
            memory_context["session"]["recent_user_messages"]
          - missing_info → memory_context["missing_info"]

        Pass 2（FTS5 fallback → confidence=medium）：
          仅在 Pass 1 有缺失字段时触发。从 session_search 中获取候选片段，
          调用 LLM 从片段中提取字段值。FTS5 不覆盖 Pass 1 已命中的 high 置信度字段。
          提取失败的字段留在 missing_fields 中，不猜测。
        """
        state = state or {}
        resolution = MemoryFieldResolution()

        session = memory_context.get("session", {}) if isinstance(memory_context, dict) else {}
        last_case = memory_context.get("last_case", {}) if isinstance(memory_context, dict) else {}
        if not last_case and isinstance(session, dict):
            last_case = session.get("last_case") or session.get("recent_case") or {}
        recent_safety = memory_context.get("recent_safety", {}) if isinstance(memory_context, dict) else {}
        if not recent_safety and isinstance(session, dict):
            recent_safety = session.get("last_safety") or {}
        last_diagnosis = session.get("last_diagnosis", {}) if isinstance(session, dict) else {}
        recent_ticket = memory_context.get("recent_ticket", {}) if isinstance(memory_context, dict) else {}
        last_customer_reply = str(memory_context.get("last_customer_reply", "") or "").strip()
        recent_user_messages = session.get("recent_user_messages", []) if isinstance(session, dict) else []
        missing_info = memory_context.get("missing_info", []) if isinstance(memory_context, dict) else []

        # 来源路由表：field → (reader_fn, source_label)
        def _from_last_case(field: str):
            val = last_case.get(field)
            if val is None or val == "" or val == []:
                return None
            return val

        def _from_recent_safety(field: str):
            val = recent_safety.get(field)
            if val is None or val == "" or val == []:
                return None
            return val

        def _from_last_diagnosis(field: str):
            if field == "diagnosis_summary":
                val = last_diagnosis.get("summary")
            elif field == "suggested_next_step":
                val = last_diagnosis.get("suggested_next_step")
            else:
                val = last_diagnosis.get(field)
            if val is None or val == "" or val == []:
                return None
            return val

        def _from_recent_ticket(field: str):
            if field == "ticket_id":
                val = recent_ticket.get("ticket_id")
            elif field == "ticket_title":
                val = recent_ticket.get("title")
            elif field == "ticket_priority":
                val = recent_ticket.get("priority")
            else:
                val = recent_ticket.get(field)
            if val is None or val == "" or val == []:
                return None
            return val

        def _from_session(field: str):
            if field == "last_customer_reply":
                return last_customer_reply or None
            if field == "last_user_message":
                for msg in reversed(recent_user_messages or []):
                    content = str(msg or "").strip()
                    if content:
                        return content
                return None
            if field == "customer_request":
                reqs = last_case.get("customer_requests", [])
                if isinstance(reqs, list) and reqs:
                    return str(reqs[0])
                return None
            return None

        def _from_missing_info(field: str):
            if field == "missing_info":
                if missing_info:
                    return missing_info
                val = last_case.get("missing_info")
                if val:
                    return val
                return None
            return None

        # 字段 → (reader, source_label)
        _ROUTE_MAP: dict[str, tuple] = {}
        for f in ("brand", "charger_model", "charger_series", "rated_power_kw",
                  "charger_type", "connector_type", "serial_number",
                  "city", "contact_address", "installation_type",
                  "purchase_or_install_time",
                  "fault_codes", "observed_symptoms", "safety_signals",
                  "environment_factors", "trip_status", "indicator_status"):
            _ROUTE_MAP[f] = (_from_last_case, f"last_case.{f}")
        for f in ("risk_level", "need_onsite", "need_electrician"):
            _ROUTE_MAP[f] = (_from_recent_safety, f"recent_safety.{f}")
        for f in ("diagnosis_summary", "suggested_next_step"):
            _ROUTE_MAP[f] = (_from_last_diagnosis, f"session.last_diagnosis.{f}")
        for f in ("ticket_id", "ticket_title", "ticket_priority"):
            _ROUTE_MAP[f] = (_from_recent_ticket, f"recent_ticket.{f}")
        for f in ("last_customer_reply", "last_user_message", "customer_request"):
            _ROUTE_MAP[f] = (_from_session, f"session.{f}")
        _ROUTE_MAP["missing_info"] = (_from_missing_info, "missing_info")

        fields_to_resolve = list(parsed.target_fields) if parsed.target_fields else list(_ROUTE_MAP.keys())

        # ================================================================
        # Pass 1：结构化来源（confidence=high）
        # ================================================================
        for field in fields_to_resolve:
            if field not in _ROUTE_MAP:
                resolution.missing_fields.append(field)
                continue
            reader, source_label = _ROUTE_MAP[field]
            value = reader(field)
            if value is not None:
                resolution.resolved_values[field] = value
                resolution.resolver_sources[field] = source_label
            else:
                resolution.missing_fields.append(field)

        # ================================================================
        # Pass 2：FTS5 fallback（仅对缺失字段，confidence=medium）
        # ================================================================
        if resolution.missing_fields and user_input:
            fts5_result = self._fts5_extract_fields(
                missing_fields=list(resolution.missing_fields),
                user_input=user_input,
                entities=parsed.entities,
                memory_context=memory_context,
                state=state,
            )
            if fts5_result:
                extracted = fts5_result.get("extracted_values", {})
                extracted_sources = fts5_result.get("extracted_sources", {})
                for field in list(resolution.missing_fields):
                    if field in extracted:
                        value = extracted[field]
                        # 防御：resolver 层也过滤空值（None / 空字符串 / 空列表 / 空 dict）
                        if value is None:
                            continue
                        if isinstance(value, str) and value.strip() == "":
                            continue
                        if isinstance(value, (list, dict)) and len(value) == 0:
                            continue
                        resolution.resolved_values[field] = value
                        # 使用 LLM 返回的 source_index，回退到 0
                        idx = extracted_sources.get(field, 0) if isinstance(extracted_sources, dict) else 0
                        resolution.resolver_sources[field] = f"fts5.message[{idx}]"
                        resolution.missing_fields.remove(field)

        # ------------------------------------------------------------------
        # 确定性归一化：确保 rated_power_kw 全链路统一为 "XkW" 格式
        # ------------------------------------------------------------------
        if "rated_power_kw" in resolution.resolved_values:
            raw = resolution.resolved_values["rated_power_kw"]
            normalized = normalize_power_kw(raw)
            if normalized:
                resolution.resolved_values["rated_power_kw"] = normalized

        # ------------------------------------------------------------------
        # 置信度计算（source-aware）
        #
        # 规则：
        #  - 只要 resolved_values 中有任何字段来源是 fts5，整体最高为 medium。
        #    因为 FTS5 + LLM 抽取本质上是推断，不是确定性读取。
        #  - 全部来自结构化来源 → high。
        #  - 全部缺失 → low。
        #  - 部分命中无 fts5 → medium。
        # ------------------------------------------------------------------
        total = len(fields_to_resolve)
        resolved = len(resolution.resolved_values)
        if total == 0:
            resolution.confidence = "low"
        elif resolved == total:
            # 检查是否任何字段来自 fts5
            has_fts5_source = any(
                str(resolution.resolver_sources.get(f, "")).startswith("fts5")
                for f in resolution.resolved_values
            )
            resolution.confidence = "medium" if has_fts5_source else "high"
        elif resolved > 0:
            resolution.confidence = "medium"
        else:
            resolution.confidence = "low"

        return resolution

    # ------------------------------------------------------------------
    # memory_answer v2：FTS5 候选片段 → LLM 字段值抽取
    # ------------------------------------------------------------------

    def _fts5_extract_fields(
        self,
        missing_fields: list[str],
        user_input: str,
        entities: list[str],
        memory_context: dict[str, Any],
        state: ChargerDiagnosisGraphState | None = None,
    ) -> dict[str, Any] | None:
        """从 FTS5 搜索结果中抽取缺失字段的值。

        返回 {"extracted_values": {field: value}, "extracted_sources": {field: source_index}, "missing_fields": [...]}，
        或 None（FTS5 不可用 / 无匹配 / LLM 抽取失败）。
        所有 debug 信息通过 _add_trace 写入 state。
        """
        if not missing_fields:
            return None

        state = state or {}
        session_id = str(state.get("session_id", "") or "")
        memory_manager = state.get("memory_manager")

        # 构建 FTS5 查询
        query_parts = list(entities or [])
        if user_input:
            query_parts.append(user_input)
        fts5_query = " ".join(query_parts).strip()
        if not fts5_query:
            fts5_query = " ".join(missing_fields)

        # ================================================================
        # 优先使用 memory_manager.session_search 重新搜索（基于当前 query）
        # 回退：memory_context 中 recall_context 时的已有搜索结果
        # ================================================================
        search_result: dict[str, Any] | None = None
        fts5_available = False
        matches: list[dict[str, Any]] = []

        if memory_manager is not None and hasattr(memory_manager, "session_search"):
            search_index = memory_manager.session_search
            search_result = search_index.search(fts5_query, session_id=session_id)
            fts5_available = bool(search_result.get("available", False))
            matches = search_result.get("matches", []) if fts5_available else []

        # 回退：使用 memory_context 中已有的 session_search
        if not matches:
            session_search = memory_context.get("session_search", {}) if isinstance(memory_context, dict) else {}
            if search_result is None:
                search_result = session_search
            fts5_available = bool(session_search.get("available", False)) if isinstance(session_search, dict) else False
            matches = session_search.get("matches", []) if isinstance(session_search, dict) else []

        self._add_trace(state, "memory_fts5", "FTS5 fallback 开始",
                        "running",
                        {"fts5_query": fts5_query,
                         "missing_fields": missing_fields,
                         "fts5_available": fts5_available,
                         "fts5_search_source": "re_search" if memory_manager is not None and hasattr(memory_manager, "session_search") else "pre_existing"},
                        {})

        if not fts5_available or not matches:
            self._add_trace(state, "memory_fts5", "FTS5 不可用或无匹配",
                            "warning",
                            {"fts5_query": fts5_query,
                             "fts5_matches_count": len(matches),
                             "fts5_available": fts5_available},
                            {"note": "FTS5 未命中，字段保持 missing，不猜测。"})
            return None

        # 构建候选证据（最多 5 条消息，带显式编号供 LLM 返回 source_index）
        candidate_messages: list[str] = []
        for i, match in enumerate(matches[:5]):
            role = str(match.get("role", "")).strip()
            content = str(match.get("content", "")).strip()
            if content:
                candidate_messages.append(f"[{i}][{role}] {content}")
        candidate_evidence = "\n".join(candidate_messages)

        if not candidate_evidence:
            self._add_trace(state, "memory_fts5", "FTS5 候选证据为空",
                            "warning",
                            {"fts5_matches_count": len(matches)},
                            {})
            return None

        self._add_trace(state, "memory_fts5", "FTS5 候选证据已准备",
                        "running",
                        {"fts5_query": fts5_query,
                         "fts5_matches_count": len(matches),
                         "fts5_candidate_evidence": candidate_evidence[:500]},
                        {})

        # 调用 LLM 从候选片段中抽取字段值
        if self.llm is None:
            self._add_trace(state, "memory_fts5", "LLM 不可用，跳过 FTS5 抽取",
                            "warning", {},
                            {"note": "LLM 未配置，FTS5 候选片段无法抽取。"})
            return None

        try:
            extracted_raw = invoke_json(
                self.llm,
                FTS5_FIELD_EXTRACTION_PROMPT,
                {
                    "target_fields": json.dumps(missing_fields, ensure_ascii=False),
                    "candidate_evidence": candidate_evidence,
                },
            )
        except Exception:
            extracted_raw = {}

        if not extracted_raw:
            self._add_trace(state, "memory_fts5", "LLM 抽取返回空",
                            "warning", {},
                            {"note": "FTS5 候选片段存在但 LLM 抽取失败，字段保持 missing。"})
            return None

        extracted_values = extracted_raw.get("extracted_values", {})
        fts5_missing = extracted_raw.get("missing_fields", [])
        extracted_sources = extracted_raw.get("extracted_sources", {})

        if not isinstance(extracted_values, dict):
            extracted_values = {}
        if not isinstance(extracted_sources, dict):
            extracted_sources = {}

        def _is_empty(val: Any) -> bool:
            """判定值是否为空：None、空字符串、空列表、空 dict 都视为空。"""
            if val is None:
                return True
            if isinstance(val, str) and val.strip() == "":
                return True
            if isinstance(val, (list, dict)) and len(val) == 0:
                return True
            return False

        # 只保留在 missing_fields 中的字段，防止 LLM 幻觉额外字段 + 过滤空值
        filtered_extracted: dict[str, Any] = {}
        filtered_sources: dict[str, int] = {}
        for k, v in extracted_values.items():
            if k not in missing_fields:
                continue
            if _is_empty(v):
                continue
            # 确定性归一化：rated_power_kw 统一为 "XkW"
            if k == "rated_power_kw":
                v = normalize_power_kw(v)
                if not v:
                    continue
            filtered_extracted[k] = v
            idx = extracted_sources.get(k)
            if isinstance(idx, int):
                filtered_sources[k] = idx

        # 未出现在 extracted 中且未出现在 fts5_missing 中的字段 → 也视为 FTS5 无法解决
        remaining_missing = [
            f for f in missing_fields
            if f not in filtered_extracted
        ]

        self._add_trace(state, "memory_fts5", "FTS5 字段抽取完成",
                        "completed",
                        {"fts5_query": fts5_query,
                         "fts5_matches_count": len(matches),
                         "fts5_candidate_evidence": candidate_evidence[:500]},
                        {"fts5_extracted_values": filtered_extracted,
                         "fts5_extracted_sources": filtered_sources,
                         "fts5_missing_fields": remaining_missing,
                         "note": f"FTS5 从 {len(matches)} 条匹配中抽取到 {len(filtered_extracted)} 个字段，{len(remaining_missing)} 个仍缺失。"})

        return {
            "extracted_values": filtered_extracted,
            "extracted_sources": filtered_sources,
            "missing_fields": remaining_missing,
        }

    # ------------------------------------------------------------------
    # memory_answer v2：临时 fallback 回复模板（不含 FTS5 / answer LLM）
    # 最终方案中将被 answer LLM 替换。
    # ------------------------------------------------------------------

    @staticmethod
    def _build_memory_reply_v2(parsed: MemoryQueryResult, resolution: MemoryFieldResolution) -> str:
        """基于 field resolver 输出构建自然语言回复。

        此方法为临时 fallback，不做 FTS5、不调 answer LLM。
        最终方案中将被 LLM 驱动的答案生成替换。
        """
        if parsed.fallback_reason:
            # parse 失败时的回退回复
            return (
                "我理解你在询问之前记录的信息，但查询解析遇到了问题。"
                "请尝试更具体地描述你想查询的信息，比如「刚才记录的型号是什么？」"
            )

        if not parsed.is_memory_query:
            return "请告诉我你需要查询什么信息？"

        if not parsed.target_fields and not resolution.resolved_values:
            return "请告诉我你具体想查询哪方面的信息？"

        # 字段 → 中文标签
        _FIELD_LABEL: dict[str, str] = {
            "brand": "品牌", "charger_model": "型号", "charger_series": "系列",
            "rated_power_kw": "额定功率", "charger_type": "充电桩类型",
            "connector_type": "连接器类型", "serial_number": "序列号",
            "city": "城市", "contact_address": "联系地址",
            "installation_type": "安装类型", "purchase_or_install_time": "购买/安装时间",
            "fault_codes": "故障码", "observed_symptoms": "观察到的问题",
            "safety_signals": "安全信号", "environment_factors": "环境因素",
            "trip_status": "跳闸状态", "indicator_status": "指示灯状态",
            "risk_level": "风险等级", "need_onsite": "是否需要现场",
            "need_electrician": "是否需要电工",
            "diagnosis_summary": "诊断小结", "suggested_next_step": "建议下一步",
            "ticket_id": "工单ID", "ticket_title": "工单标题",
            "ticket_priority": "工单优先级", "missing_info": "缺失信息",
            "last_customer_reply": "上一次回复", "last_user_message": "上一次用户问题",
            "customer_request": "客户诉求",
        }

        def _format_value(value: Any) -> str:
            if isinstance(value, list):
                return "、".join(str(v).strip() for v in value if str(v).strip())
            return str(value).strip()

        found: list[str] = []
        not_found: list[str] = []

        for field in parsed.target_fields if parsed.target_fields else list(resolution.resolved_values.keys()):
            label = _FIELD_LABEL.get(field, field)
            if field in resolution.resolved_values:
                value_str = _format_value(resolution.resolved_values[field])
                if parsed.answer_style == "summary":
                    found.append(f"{label}：{value_str}")
                else:
                    found.append(f"已记录的{label}是：{value_str}")
            else:
                not_found.append(label)

        if parsed.answer_style == "summary":
            result_parts: list[str] = []
            if found:
                result_parts.append("当前会话已记录：\n" + "\n".join(f"  - {f}" for f in found))
            else:
                result_parts.append("当前会话中暂未记录相关信息。")
            if not_found:
                result_parts.append("\n以下信息未找到，需要你补充：" + "、".join(not_found))
            # 附加 resolver 元信息（debug 用）
            result_parts.append(f"\n[confidence: {resolution.confidence}]")
            return "\n".join(result_parts)

        # precise 模式
        if not found:
            return (
                f"当前会话记忆中没有找到{'、'.join(not_found)}，需要你补充。"
                f"\n[confidence: {resolution.confidence}]"
            )

        reply = "；".join(found)
        if not_found:
            reply += f"\n\n另外，以下信息未找到：{'、'.join(not_found)}。"
        reply += f"\n[confidence: {resolution.confidence}]"
        return reply

    # ------------------------------------------------------------------
    # memory_answer v2：Answer LLM — 基于 resolver 输出生成自然语言回答
    # ------------------------------------------------------------------

    def _build_memory_answer_llm(
        self,
        parsed: MemoryQueryResult,
        resolution: MemoryFieldResolution,
        user_input: str,
        state: ChargerDiagnosisGraphState,
    ) -> str:
        """使用 LLM 生成自然语言回答。

        LLM 不可用或异常时回退到 _build_memory_reply_v2()。
        debug 信息（answer_llm_used / answer_prompt_input_summary /
        answer_fallback_reason / final_memory_answer）写入 state trace。
        """
        answer_llm_used = False
        fallback_reason = ""
        prompt_summary = ""

        # 格式化 resolved / missing 为 prompt 可读文本
        resolved_text = self._format_fields_for_answer(
            resolution.resolved_values, resolution.resolver_sources, resolution.confidence
        )
        missing_text = self._format_missing_for_answer(resolution.missing_fields)

        prompt_vars = {
            "user_input": user_input,
            "answer_style": parsed.answer_style,
            "confidence": resolution.confidence,
            "resolved_text": resolved_text,
            "missing_text": missing_text,
        }
        prompt_summary = (
            f"user_input={user_input[:80]}, "
            f"answer_style={parsed.answer_style}, "
            f"confidence={resolution.confidence}, "
            f"resolved_count={len(resolution.resolved_values)}, "
            f"missing_count={len(resolution.missing_fields)}"
        )

        if self.llm is not None:
            try:
                chain = MEMORY_ANSWER_GENERATION_PROMPT | self.llm
                response = chain.invoke(prompt_vars)
                answer = str(response.content).strip() if hasattr(response, "content") else str(response).strip()
                if answer:
                    answer_llm_used = True
                else:
                    fallback_reason = "llm_empty_response"
            except Exception as exc:
                fallback_reason = f"llm_error: {exc}"
        else:
            fallback_reason = "llm_unavailable"

        if answer_llm_used:
            # LLM 可能仍然在输出中包含技术标记，做一次轻量清洗
            answer = self._clean_answer_output(answer)
            # 防编造校验
            validation_warnings = self._validate_answer_fields(
                answer, resolution.resolved_values, resolution.missing_fields
            )
            if validation_warnings:
                self._add_trace(
                    state, "memory_answer_llm", "Answer 编造校验",
                    "warning", {},
                    {"validation_warnings": validation_warnings,
                     "note": "LLM 生成回答疑似对缺失字段做了肯定陈述，回退到 fallback 模板。"},
                )
                # 编造检测触发 → 回退到确定性模板，确保不向客户输出编造内容
                answer_llm_used = False
                fallback_reason = "answer_validation_failed"
                final_answer = self._build_memory_reply_v2(parsed, resolution)
            else:
                final_answer = answer
        else:
            final_answer = self._build_memory_reply_v2(parsed, resolution)

        # ---- debug trace ----
        self._add_trace(
            state, "memory_answer_llm", "Answer LLM 生成",
            "completed" if answer_llm_used else "warning",
            {
                "answer_llm_used": answer_llm_used,
                "answer_prompt_input_summary": prompt_summary,
                "answer_fallback_reason": fallback_reason,
            },
            {"final_memory_answer": final_answer},
        )

        return final_answer

    @staticmethod
    def _format_fields_for_answer(
        resolved_values: dict[str, Any],
        resolver_sources: dict[str, str],
        confidence: str,
    ) -> str:
        """将 resolved_values 格式化为 prompt 可读文本。

        每行格式：字段中文名：值（来源：确定性记录 / 从历史消息推断）
        """
        if not resolved_values:
            return "（无）"
        lines: list[str] = []
        for field, value in resolved_values.items():
            label = _MEMORY_ANSWER_FIELD_LABEL.get(field, field)
            formatted = _format_field_value(value)
            source = resolver_sources.get(field, "")
            if source.startswith("fts5"):
                source_note = "（来源：从历史消息推断）"
            else:
                source_note = "（来源：确定性记录）"
            lines.append(f"  {label}：{formatted} {source_note}")
        return "\n".join(lines)

    @staticmethod
    def _format_missing_for_answer(missing_fields: list[str]) -> str:
        """将 missing_fields 格式化为 prompt 可读文本。"""
        if not missing_fields:
            return "（全部找到）"
        labels = [_MEMORY_ANSWER_FIELD_LABEL.get(f, f) for f in missing_fields]
        return "、".join(labels)

    @staticmethod
    def _clean_answer_output(text: str) -> str:
        """轻量清洗 LLM 输出中可能泄露的技术标记。"""
        # deterministic: 正则替换，不依赖 LLM
        import re
        # 移除 [confidence: ...] 标记
        text = re.sub(r"\[confidence:\s*\w+\]", "", text)
        # 移除 （来源：...） / (来源：...) 标注
        text = re.sub(r"[（(]来源[：:][^）)]*[）)]", "", text)
        # 移除 leading/trailing 空白
        text = text.strip()
        return text

    @staticmethod
    def _validate_answer_fields(
        answer: str,
        resolved_values: dict[str, Any],
        missing_fields: list[str],
    ) -> list[str]:
        """轻量检查 LLM 回答是否对缺失字段做了肯定陈述（防编造）。

        返回警告列表，空列表表示无问题。
        这是确定性字符串匹配，不依赖 LLM。
        """
        # deterministic: 字符串匹配检查，不依赖 LLM
        import re

        warnings: list[str] = []
        for field in missing_fields:
            label = _MEMORY_ANSWER_FIELD_LABEL.get(field, field)
            # 检查是否出现了 "标签是/为/：XXX" 的肯定陈述模式
            if re.search(rf"{re.escape(label)}[是为：:]\s*\S", answer):
                warnings.append(f"回答可能对缺失字段 '{label}' 做了肯定陈述（疑似编造）")
        return warnings

    # ------------------------------------------------------------------
    # [deprecated] 旧链路回复函数
    # ------------------------------------------------------------------

    def _build_memory_reply(self, answer_type: str, user_input: str, memory_context: dict[str, Any]) -> str:
        """[deprecated] 硬编码 answer_type → 回复映射，仅用于 MEMORY_ANSWER_V2=False 的旧链路。

        memory_answer v2 中已替换为 _build_memory_reply_v2() + MemoryQueryResult.target_fields 路由。
        旧链路移除时本函数应一并删除。
        """
        session = memory_context.get("session", {}) if isinstance(memory_context, dict) else {}
        last_case = memory_context.get("last_case", {}) if isinstance(memory_context, dict) else {}
        if not last_case and isinstance(session, dict):
            last_case = session.get("last_case") or session.get("recent_case") or {}
        recent_safety = memory_context.get("recent_safety", {}) if isinstance(memory_context, dict) else {}
        if not recent_safety and isinstance(session, dict):
            recent_safety = session.get("last_safety") or {}
        recent_ticket = memory_context.get("recent_ticket", {}) if isinstance(memory_context, dict) else {}
        missing_info = memory_context.get("missing_info", []) if isinstance(memory_context, dict) else []
        if not missing_info and isinstance(last_case, dict):
            missing_info = last_case.get("missing_info", [])
        last_dispatch = session.get("last_dispatch", {}) if isinstance(session, dict) else {}

        if answer_type == "model":
            model = str(last_case.get("charger_model", "") or "").strip()
            return f"记得，刚才记录的充电桩型号是：{model}。" if model else self._missing_memory_reply("型号")

        if answer_type == "city":
            city = str(last_case.get("city", "") or "").strip()
            return f"刚才记录的城市是：{city}。" if city else "当前会话没有记录该信息：城市。"

        if answer_type == "missing_info":
            items = self._string_list(missing_info)
            if not items:
                return "当前会话里暂未记录仍需补充的信息。"
            return "当前还缺少这些信息：" + "、".join(items) + "。"

        if answer_type == "risk_level":
            risk_level = str(recent_safety.get("risk_level", "") or "").strip()
            return f"刚才记录的风险等级是：{risk_level}。" if risk_level else self._missing_memory_reply("风险等级")

        if answer_type == "ticket_priority":
            priority = str(recent_ticket.get("priority") or last_dispatch.get("priority") or "").strip()
            return f"刚才工单草稿的优先级是：{priority}。" if priority else self._missing_memory_reply("工单优先级")

        if answer_type == "remembered_summary":
            summary = self._remembered_summary(last_case, recent_safety, missing_info)
            return summary if summary else "当前会话未记录该信息。"

        previous_question = self._latest_non_memory_user_message(session.get("recent_user_messages", []), user_input)
        if previous_question:
            return f"你刚才说的是：{previous_question}"
        return self._missing_memory_reply("上一条用户问题")

    def _remembered_summary(self, last_case: dict[str, Any], recent_safety: dict[str, Any], missing_info: Any) -> str:
        """[deprecated] _build_memory_reply() 的辅助函数，旧链路移除时本函数一并删除。"""
        parts = []
        field_labels = [
            ("charger_model", "型号"),
            ("city", "城市"),
            ("issue_description", "问题描述"),
            ("fault_codes", "故障码"),
            ("observed_symptoms", "观察现象"),
            ("safety_signals", "安全信号"),
        ]
        for key, label in field_labels:
            value = last_case.get(key)
            if isinstance(value, list):
                value = "、".join(self._string_list(value))
            if value:
                parts.append(f"{label}：{value}")
        risk_level = recent_safety.get("risk_level")
        if risk_level:
            parts.append(f"风险等级：{risk_level}")
        missing = self._string_list(missing_info)
        if missing:
            parts.append(f"缺失信息：{'、'.join(missing)}")
        return "当前会话已记录：" + "；".join(parts) + "。" if parts else ""

    def _latest_non_memory_user_message(self, messages: Any, current_input: str) -> str:
        """[deprecated] _build_memory_reply() 的辅助函数，旧链路移除时本函数一并删除。"""
        if not isinstance(messages, list):
            return ""
        current = str(current_input or "").strip()
        for item in reversed(messages):
            content = str(item or "").strip()
            if content and content != current and self._memory_query_type(content) == "unknown":
                return content
        return ""

    def _missing_memory_reply(self, field_name: str) -> str:
        return f"当前会话未记录该信息：{field_name}。"

    def _string_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, tuple):
            return [str(item).strip() for item in value if str(item).strip()]
        if value:
            return [str(value).strip()]
        return []

    def _build_retrieval_query(self, user_input: str, case: dict[str, Any], triage: dict[str, Any]) -> str:
        parts = [user_input, triage.get("intent", "")]
        for key in [
            "brand", # 充电桩品牌
            "charger_model", # 充电桩型号
            "issue_description", # 问题描述
            "installation_type", # 安装类型
            "vehicle_brand_model", # 车辆品牌型号
            "purchase_or_install_time", # 购买或安装时间
        ]:
            if case.get(key):
                parts.append(str(case[key]))
        for key in [
            "fault_codes", # 故障码
            "observed_symptoms", # 观察症状
            "safety_signals", # 安全信号
            "environment_factors", # 环境因素
            "customer_requests", # 客户请求
        ]:
            if case.get(key):
                parts.append(" ".join(str(item) for item in case[key]))
        return " ".join(part for part in parts if part)

    def _tool_history_item(self, result: Any, input_data: dict[str, Any]) -> dict[str, Any]:
        return {
            "call_type": "local_python",
            "tool_name": result.tool_name,
            "input": input_data,
            "output": result.data,
            "status": "success" if result.success else "failed",
            "error": result.error,
            "latency_ms": int(result.execution_time * 1000),
        }

    def _add_trace(
        self,
        state: ChargerDiagnosisGraphState,
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
        self._emit_progress(state, node, title, status, input_data, output_data, item.get("duration"))

    def _emit_progress(
        self,
        state: ChargerDiagnosisGraphState,
        node: str,
        title: str,
        status: str,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
        duration: float | None = None,
    ) -> None:
        callback = state.get("progress_callback")
        if not callable(callback):
            return
        event = {
            "node": node,
            "title": title,
            "status": status,
            "input": input_data,
            "output": output_data,
            "timestamp": round(time.time(), 3),
        }
        if duration is not None:
            event["duration"] = duration
        try:
            callback(event)
        except Exception:
            pass
