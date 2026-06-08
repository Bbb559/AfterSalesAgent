from __future__ import annotations

import html
import json
import os
from pathlib import Path
import time
from typing import Any

import requests

from backend import config
from backend.config import FASTAPI_BASE_URL, GRADIO_HOST, GRADIO_PORT

try:
    import gradio as gr
except ModuleNotFoundError:  # pragma: no cover - 未安装 Gradio 时仍允许测试格式化函数
    gr = None


API_BASE_URL = os.getenv("FASTAPI_BASE_URL", FASTAPI_BASE_URL).rstrip("/")
AGENT_POLL_LIMIT = 600
AGENT_POLL_INTERVAL_SECONDS = 2.0
KB_REQUEST_TIMEOUT_SECONDS = 2
AGENT_START_TIMEOUT_SECONDS = 10
AGENT_STATUS_TIMEOUT_SECONDS = 10

NODE_ORDER = [
    ("intent", "意图识别", ("input_guard", "triage", "case_extract")),
    ("retrieval", "知识检索", ("memory_context", "case_memory_merge", "retrieval")),
    ("diagnosis", "安全诊断", ("memory_answer", "safety_guard", "diagnosis")),
    ("generation", "方案生成", ("memory_answer", "warranty_dispatch", "action")),
    ("audit", "方案审核", ("memory_answer", "audit")),
    ("final", "结束", ("final",)),
]

STATUS_TEXT = {
    "pending": "等待",
    "running": "运行中",
    "completed": "完成",
    "warning": "注意",
    "failed": "失败",
    "timeout": "超时",
    "api_unavailable": "未连接",
}

APP_CSS = """
.as-page-note {color: #4b5563; margin: -4px 0 12px;}
.as-card {border: 1px solid #e5e7eb; border-radius: 8px; background: #fff; padding: 14px 16px; margin-bottom: 10px;}
.as-card-strong {border-left: 5px solid #2563eb;}
.as-card-danger {border-left: 5px solid #dc2626;}
.as-card-warning {border-left: 5px solid #f97316;}
.as-card-good {border-left: 5px solid #16a34a;}
.as-card-muted {border-left: 5px solid #d1d5db;}
.as-card-head {display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 10px;}
.as-card-title {font-size: 16px; font-weight: 700; color: #111827;}
.as-card-subtitle {font-size: 13px; color: #6b7280; margin-top: -4px; margin-bottom: 10px;}
.as-badge {display: inline-flex; align-items: center; border-radius: 999px; padding: 2px 9px; font-size: 12px; font-weight: 700; white-space: nowrap;}
.as-badge-danger {background: #fee2e2; color: #991b1b;}
.as-badge-warning {background: #ffedd5; color: #9a3412;}
.as-badge-good {background: #dcfce7; color: #166534;}
.as-badge-info {background: #dbeafe; color: #1e40af;}
.as-badge-muted {background: #f3f4f6; color: #4b5563;}
.as-kv-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr)); gap: 8px 12px;}
.as-kv {border-top: 1px solid #f3f4f6; padding-top: 8px; min-width: 0;}
.as-kv-label {display: block; color: #6b7280; font-size: 12px; margin-bottom: 3px;}
.as-kv-value {display: block; color: #111827; font-size: 14px; font-weight: 600; overflow-wrap: anywhere;}
.as-list {margin: 6px 0 0; padding-left: 18px;}
.as-list li {margin: 3px 0; line-height: 1.45;}
.as-section-text {line-height: 1.55; color: #111827; overflow-wrap: anywhere;}
.as-placeholder {border: 1px dashed #d1d5db; border-radius: 8px; background: #f9fafb; color: #6b7280; padding: 14px 16px; margin-bottom: 10px;}
.as-source-card {border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px 14px; margin-bottom: 12px; background: #fff;}
.as-source-meta {display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0 8px;}
.as-preview {font-family: Consolas, "Microsoft YaHei", sans-serif; font-size: 13px; white-space: pre-wrap; line-height: 1.5; max-height: 210px; overflow: auto; background: #f9fafb; border: 1px solid #f3f4f6; border-radius: 6px; padding: 10px;}
.as-trace {border-left: 2px solid #e5e7eb; padding-left: 12px;}
.as-trace-item {margin-bottom: 10px;}
.as-trace-title {font-weight: 700;}
.as-toolbar-note {color: #6b7280; font-size: 13px;}
.as-agent-shell {align-items: flex-start;}
.as-flow-compact {font-family: Arial, "Microsoft YaHei", sans-serif; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; background: #fff;}
.as-flow-compact h3 {font-size: 16px; margin: 0 0 10px;}
.as-flow-step {border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px 12px; margin-bottom: 10px; background: #fff;}
.as-flow-step-running {border-left: 4px solid #f97316;}
.as-flow-step-completed {border-left: 4px solid #16a34a;}
.as-flow-step-warning {border-left: 4px solid #f59e0b;}
.as-flow-step-failed {border-left: 4px solid #dc2626;}
.as-flow-step-pending {border-left: 4px solid #d1d5db;}
.as-flow-step-head {display: flex; justify-content: space-between; gap: 8px; align-items: center; margin-bottom: 6px;}
.as-flow-step-title {font-weight: 700; color: #111827;}
.as-flow-step-body {font-size: 13px; color: #374151; line-height: 1.45;}
.as-flow-step-row {margin-top: 3px;}
.as-flow-step-label {color: #6b7280;}
"""

WAITING_TEXT = "<div class='as-placeholder'>等待运行结果。</div>"
EMPTY_AGENT_TEXTS = (WAITING_TEXT,) * 11


def call_charger_diagnosis_api(
    user_input: str,
    database_id: str = "",
    retrieval_mode: str = "hybrid",
    final_top_k: int = 5,
    session_id: str = "",
) -> dict[str, Any]:
    response = requests.post(
        f"{API_BASE_URL}/api/charger-diagnosis/run",
        json={
            "user_input": user_input,
            "database_id": database_id or None,
            "session_id": session_id or None,
            "retrieval_options": {
                "retrieval_mode": retrieval_mode,
                "final_top_k": final_top_k,
            },
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def call_charger_diagnosis_start_api(
    user_input: str,
    database_id: str = "",
    retrieval_mode: str = "hybrid",
    final_top_k: int = 5,
    session_id: str = "",
) -> dict[str, Any]:
    response = requests.post(
        f"{API_BASE_URL}/api/charger-diagnosis/start",
        json={
            "user_input": user_input,
            "database_id": database_id or None,
            "session_id": session_id or None,
            "retrieval_options": {
                "retrieval_mode": retrieval_mode,
                "final_top_k": final_top_k,
            },
        },
        timeout=AGENT_START_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def call_charger_diagnosis_run_status(run_id: str) -> dict[str, Any]:
    response = requests.get(
        f"{API_BASE_URL}/api/charger-diagnosis/runs/{run_id}?view=summary",
        timeout=AGENT_STATUS_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def format_agent_response(payload: dict[str, Any]) -> tuple[str, str, str, str, str, str, str, str, str, str, str, str]:
    if not payload.get("success"):
        error = f"接口调用失败：{payload.get('error', '未知错误')}"
        return (error, *EMPTY_AGENT_TEXTS[:-1], format_raw_json(payload))

    data = payload.get("data", {}) or {}
    action = data.get("action", {}) or {}
    customer_reply = action.get("customer_reply") or "暂未生成客户回复，请检查 ChargerActionAgent 输出。"
    return (
        customer_reply,
        format_input_safety(data) + format_intent(data),
        format_safety(data),
        format_case(data) + format_memory_context(data),
        format_diagnosis(data),
        format_warranty(data),
        format_dispatch(data),
        format_audit(data) + format_governance(data),
        format_sources(data),
        format_tool_history(data),
        format_trace(data),
        format_raw_json(data),
    )


def run_agent(
    user_input: str,
    database_id: str,
    retrieval_mode: str,
    final_top_k: int,
    session_id: str = "",
) -> tuple[str, str, str, str, str, str, str, str, str, str, str, str]:
    if not user_input.strip():
        return ("请先输入客户充电桩售后安全问题。", *EMPTY_AGENT_TEXTS)
    try:
        return format_agent_response(call_charger_diagnosis_api(user_input, database_id, retrieval_mode, final_top_k, session_id))
    except Exception as exc:
        return (f"无法连接 FastAPI 服务，请先启动 api.py。错误：{exc}", *EMPTY_AGENT_TEXTS)


def build_node_visualization(run_status: dict[str, Any] | None) -> str:
    run_status = run_status or {}
    node_statuses = run_status.get("node_statuses", {}) or {}
    run_state = STATUS_TEXT.get(run_status.get("status", "pending"), run_status.get("status", "pending"))
    cards = []
    for step_id, title, source_nodes in NODE_ORDER:
        aggregate = _aggregate_flow_step(step_id, source_nodes, node_statuses)
        status = aggregate["status"]
        cards.append(
            "<div class='as-flow-step as-flow-step-{status}'>"
            "<div class='as-flow-step-head'>"
            "<span class='as-flow-step-title'>{title}</span>"
            "<span class='as-node-badge'>{status_text}</span>"
            "</div>"
            "<div class='as-flow-step-body'>"
            "<div class='as-flow-step-row'><span class='as-flow-step-label'>输入：</span>{input}</div>"
            "<div class='as-flow-step-row'><span class='as-flow-step-label'>输出：</span>{output}</div>"
            "<div class='as-flow-step-row'><span class='as-flow-step-label'>耗时：</span>{duration}</div>"
            "</div>"
            "</div>".format(
                status=html.escape(str(status)),
                title=html.escape(str(title)),
                status_text=html.escape(STATUS_TEXT.get(status, status)),
                input=html.escape(aggregate["input"]),
                output=html.escape(aggregate["output"]),
                duration=html.escape(aggregate["duration"]),
            )
        )
    return """
<style>
.as-node-badge {{font-size: 12px; padding: 2px 6px; border-radius: 999px; background: #f3f4f6; color: #374151; white-space: nowrap;}}
</style>
<div class="as-flow-compact">
  <div class="as-flow-step-head">
    <h3>工作流执行状态</h3>
    <span>{run_state}</span>
  </div>
  <div>{cards}</div>
</div>
""".format(run_state=html.escape(str(run_state)), cards="".join(cards))


def format_input_safety(data: dict[str, Any]) -> str:
    input_safety = _section_data(data, "input_safety")
    status = input_safety.get("status") or "pending"
    body = _join_html([
        format_key_value_grid([
            ("扫描状态", status),
            ("提示注入", _bool_text(input_safety.get("prompt_injection_detected"))),
            ("越权请求", _bool_text(input_safety.get("privilege_escalation_detected"))),
            ("敏感信息", _bool_text(input_safety.get("sensitive_info_detected"))),
            ("命中标记", _list_inline(input_safety.get("matched_markers"))),
        ]),
        format_list_block("输入安全警告", input_safety.get("warnings")),
        format_text_block("上下文策略", input_safety.get("context_policy")),
    ])
    return build_business_card(
        "输入安全扫描",
        body,
        badge=build_status_badge(status, _tone_from_value(status)),
        tone=_tone_from_value(status),
        subtitle="所有外部输入先扫描和结构化，再进入 Agent 工作流。",
    )


def format_intent(data: dict[str, Any]) -> str:
    triage = _section_data(data, "triage")
    return build_business_card(
        "意图识别",
        format_key_value_grid([
            ("意图", triage.get("intent")),
            ("置信度", triage.get("confidence")),
            ("原因", triage.get("reason")),
        ]),
        badge=build_status_badge(triage.get("intent") or "待识别", "info"),
        tone="strong",
    )


def format_safety(data: dict[str, Any]) -> str:
    safety = _section_data(data, "safety")
    risk_level = safety.get("risk_level")
    tone = _tone_from_value(risk_level)
    body = _join_html([
        format_key_value_grid([
            ("需要人工", _bool_text(safety.get("need_human"))),
            ("需要上门", _bool_text(safety.get("need_onsite"))),
            ("需要电工", _bool_text(safety.get("need_electrician"))),
            ("命中信号", _list_inline(safety.get("matched_safety_signals"))),
        ]),
        format_text_block("原因", safety.get("reason")),
        format_list_block("要求客户动作", safety.get("required_customer_actions")),
    ])
    return build_business_card(
        "安全风险",
        body,
        badge=build_status_badge(risk_level or "unknown", tone),
        tone=tone,
        subtitle="先判断风险，再决定是否允许远程排查。",
    )


def format_case(data: dict[str, Any]) -> str:
    case = _section_data(data, "case")
    return build_business_card(
        "信息抽取",
        _join_html([
            format_key_value_grid([
                ("品牌", case.get("brand")),
                ("充电桩型号", case.get("charger_model")),
                ("序列号", case.get("serial_number")),
                ("车辆", case.get("vehicle_brand_model")),
                ("城市", case.get("city")),
                ("安装地址", case.get("contact_address")),
                ("故障码", _list_inline(case.get("fault_codes"))),
                ("观察现象", _list_inline(case.get("observed_symptoms"))),
            ]),
            format_list_block("缺失信息", case.get("missing_info")),
        ]),
        badge=build_status_badge("case", "muted"),
        tone="muted",
    )


def format_memory_context(data: dict[str, Any]) -> str:
    memory = _section_data(data, "memory_context")
    session = memory.get("session", {}) if isinstance(memory.get("session"), dict) else {}
    customer = memory.get("customer", {}) if isinstance(memory.get("customer"), dict) else {}
    charger = memory.get("charger", {}) if isinstance(memory.get("charger"), dict) else {}
    site = memory.get("site", {}) if isinstance(memory.get("site"), dict) else {}
    ticket = memory.get("ticket", {}) if isinstance(memory.get("ticket"), dict) else {}
    isolation = memory.get("isolation", {}) if isinstance(memory.get("isolation"), dict) else {}
    body = _join_html([
        format_key_value_grid([
            ("Session", session.get("session_id")),
            ("会话消息数", session.get("message_count")),
            ("最近缺失信息", _list_inline(session.get("missing_info"))),
            ("客户", customer.get("contact_phone") or customer.get("contact_name")),
            ("设备", charger.get("serial_number") or charger.get("charger_model")),
            ("场地", site.get("contact_address") or site.get("city")),
            ("最近工单", ticket.get("title")),
            ("诊断证据", "否" if isolation.get("used_as_diagnostic_evidence") is False else "待复核"),
        ]),
        format_list_block("最近用户问题", session.get("recent_user_messages")),
        format_text_block("隔离策略", isolation.get("policy")),
    ])
    return build_business_card(
        "会话记忆与上下文隔离",
        body,
        badge=build_status_badge("非诊断证据", "good" if isolation.get("used_as_diagnostic_evidence") is False else "warning"),
        tone="good" if isolation.get("used_as_diagnostic_evidence") is False else "warning",
        subtitle="Session / Customer / Charger / Site / Ticket 只提供摘要，不替代 RAG 依据。",
    )


def format_diagnosis(data: dict[str, Any]) -> str:
    diagnosis = _section_data(data, "diagnosis")
    priority = diagnosis.get("priority")
    body = _join_html([
        format_text_block("摘要", diagnosis.get("summary")),
        format_key_value_grid([
            ("证据状态", diagnosis.get("evidence_status")),
            ("优先级", priority),
            ("可能问题区域", _list_inline(diagnosis.get("likely_issue_areas"))),
        ]),
        format_list_block("安全远程核验", diagnosis.get("safe_remote_checks")),
        format_text_block("下一步建议", diagnosis.get("suggested_next_step")),
    ])
    return build_business_card(
        "诊断结论",
        body,
        badge=build_status_badge(priority or "待判断", _tone_from_value(priority)),
        tone=_tone_from_value(priority),
        subtitle="只展示客服需要判断下一步的诊断摘要。",
    )


def format_warranty(data: dict[str, Any]) -> str:
    warranty = _section_data(data, "warranty")
    status = warranty.get("status")
    return build_business_card(
        "保修判断",
        _join_html([
            format_key_value_grid([
                ("保修状态", status),
                ("需要凭证", _bool_text(warranty.get("need_evidence"))),
                ("政策月份", warranty.get("policy_months")),
                ("政策来源", _list_inline(warranty.get("policy_sources"))),
            ]),
            format_text_block("原因", warranty.get("reason")),
        ]),
        badge=build_status_badge(status or "unknown", _tone_from_value(status)),
        tone="muted",
    )


def format_dispatch(data: dict[str, Any]) -> str:
    dispatch = _section_data(data, "dispatch")
    priority = dispatch.get("priority")
    body = _join_html([
        format_key_value_grid([
            ("工单标题", dispatch.get("title")),
            ("优先级", priority),
            ("需要上门", _bool_text(dispatch.get("need_onsite"))),
            ("需要电工", _bool_text(dispatch.get("need_electrician"))),
        ]),
        format_text_block("客户问题", dispatch.get("customer_problem")),
        format_text_block("派工建议", dispatch.get("suggested_dispatch")),
        format_list_block("需补充证据", dispatch.get("evidence_needed")),
        format_list_block("缺失信息", dispatch.get("missing_info")),
        format_text_block("内部备注", dispatch.get("internal_note")),
    ])
    return build_business_card(
        "工单草稿",
        body,
        badge=build_status_badge(priority or "待派工", _tone_from_value(priority)),
        tone=_tone_from_value(priority),
        subtitle="给客服和售后工程师看的派工摘要，不包含原始 JSON。",
    )


def format_audit(data: dict[str, Any]) -> str:
    audit = _section_data(data, "audit")
    passed = audit.get("passed")
    tone = "good" if passed is True else "warning" if passed is False else "muted"
    return build_business_card(
        "审核结果",
        _join_html([
            format_key_value_grid([
                ("是否通过", _bool_text(passed)),
                ("风险等级", audit.get("risk_level")),
            ]),
            format_list_block("审核警告", audit.get("warnings")),
            format_text_block("最终备注", audit.get("final_note")),
        ]),
        badge=build_status_badge("通过" if passed is True else "需复核" if passed is False else "待审核", tone),
        tone=tone,
    )


def format_governance(data: dict[str, Any]) -> str:
    governance = _section_data(data, "governance")
    status = governance.get("status") or "pending"
    return build_business_card(
        "安全治理汇总",
        _join_html([
            format_key_value_grid([
                ("输入扫描", _bool_text(governance.get("input_scan_enabled"))),
                ("上下文隔离", _bool_text(governance.get("context_isolation_enabled"))),
                ("最终审核", _bool_text(governance.get("final_audit_enabled"))),
                ("记忆作用域", governance.get("memory_scope")),
                ("记忆作为诊断证据", _bool_text(governance.get("memory_used_as_diagnostic_evidence"))),
            ]),
            format_list_block("治理警告", governance.get("warnings")),
        ]),
        badge=build_status_badge(status, _tone_from_value(status)),
        tone=_tone_from_value(status),
    )


def format_sources(data: dict[str, Any]) -> str:
    retrieval = data.get("retrieval", data) if isinstance(data, dict) else {}
    results = retrieval.get("results", []) if isinstance(retrieval, dict) else []
    if not results:
        return build_placeholder("暂无引用来源。")

    blocks = []
    for index, item in enumerate(results[:8], start=1):
        score = _score_text(item)
        file_name = item.get("file_name") or item.get("source_file")
        meta = _join_html([
            build_status_badge(f"来源 {index}", "info"),
            build_status_badge(f"页码 { _display(item.get('page')) }", "muted"),
            build_status_badge(f"Chunk { _display(item.get('chunk_id')) }", "muted"),
            build_status_badge(f"相关性 {score}", "good" if score != "未提供" else "muted"),
        ])
        blocks.append(
            "<div class='as-source-card'>"
            f"<div class='as-card-head'><span class='as-card-title'>{html.escape(_display(file_name))}</span></div>"
            f"<div class='as-source-meta'>{meta}</div>"
            f"<div class='as-preview'>{html.escape(_truncate(item.get('text'), 500))}</div>"
            "</div>"
        )
    return "<div class='as-section-text'>" + "".join(blocks) + "</div>"


def format_tool_history(data: dict[str, Any]) -> str:
    history = data.get("tool_history", []) if isinstance(data, dict) else []
    if not history:
        return build_placeholder("暂无工具调用记录。")

    blocks = []
    for index, item in enumerate(history, start=1):
        status = item.get("status")
        blocks.append(build_business_card(
            f"工具 {index}",
            _join_html([
                format_key_value_grid([
                    ("工具名", item.get("tool_name")),
                    ("状态", status),
                    ("耗时", f"{item.get('latency_ms')} ms" if item.get("latency_ms") is not None else ""),
                    ("错误", item.get("error")),
                ]),
                format_text_block("输出摘要", _compact_json(item.get("output"), max_length=260)),
            ]),
            badge=build_status_badge(status or "unknown", _tone_from_value(status)),
            tone="muted",
        ))
    return "".join(blocks)


def format_trace(data: dict[str, Any]) -> str:
    trace = data.get("trace", []) if isinstance(data, dict) else []
    if not trace:
        return build_placeholder("暂无执行轨迹。")

    lines = []
    for item in trace:
        node = _display(item.get("node"))
        title = _display(item.get("title"))
        status = STATUS_TEXT.get(str(item.get("status", "")), _display(item.get("status")))
        duration = item.get("duration")
        duration_text = f"{duration:.3f}s" if isinstance(duration, (int, float)) else "未记录"
        output = _compact_json(item.get("output"), max_length=180)
        lines.append(
            "<div class='as-trace-item'>"
            f"<div class='as-trace-title'>{html.escape(node)} / {html.escape(title)} "
            f"{build_status_badge(status, _tone_from_value(status))}</div>"
            f"<div class='as-card-subtitle'>耗时 {html.escape(duration_text)}</div>"
            f"<div class='as-section-text'>{html.escape(output or '无摘要')}</div>"
            "</div>"
        )
    return "<div class='as-card'><div class='as-trace'>" + "".join(lines) + "</div></div>"


def format_rag_results(payload: dict[str, Any]) -> str:
    if not payload:
        return build_placeholder("暂无检索结果。")
    if payload.get("success") is False:
        return build_business_card("检索失败", html.escape(str(payload.get("error", "未知错误"))), tone="warning")

    data = payload.get("data", payload)
    if data.get("error"):
        return build_business_card("检索失败", html.escape(str(data.get("error"))), tone="warning")

    results = data.get("results", [])
    trace = data.get("trace", {})
    rewrite = trace.get("query_rewrite", {}) if isinstance(trace, dict) else {}
    header = build_business_card(
        "检索摘要",
        _join_html([
            format_key_value_grid([
                ("检索 Query", data.get("query")),
                ("检索模式", trace.get("mode") if isinstance(trace, dict) else ""),
                ("命中数", len(results)),
                ("Query Rewrite", "开启" if rewrite.get("enabled") else "关闭"),
            ]),
            format_list_block("实际 Queries", trace.get("queries", []) if isinstance(trace, dict) else []),
            format_list_block("改写 Queries", rewrite.get("rewritten_queries", [])),
            format_text_block("改写错误", rewrite.get("error")),
        ]),
        badge=build_status_badge(f"{len(results)} 个命中", "info"),
        tone="strong",
    )
    sources = format_rag_source_summary(results)
    return f"{header}{sources}"


def format_rag_source_summary(results: list[dict[str, Any]]) -> str:
    if not results:
        return build_placeholder("暂无来源。")
    cards = []
    for index, item in enumerate(results[:3], start=1):
        cards.append(
            build_business_card(
                f"来源 {index}",
                format_key_value_grid([
                    ("文件", item.get("file_name") or item.get("source") or "未知文件"),
                    ("页码", item.get("page")),
                    ("Chunk", item.get("chunk_id")),
                    ("分数", _score_text(item)),
                ]),
                badge=build_status_badge("摘要", "info"),
                tone="muted",
            )
        )
    if len(results) > 3:
        cards.append(build_placeholder(f"还有 {len(results) - 3} 条来源已隐藏，请查看后端调试日志。"))
    return "".join(cards)


def format_memory_summary(data: dict[str, Any] | str | None = None) -> str:
    session_id = data if isinstance(data, str) else ""
    context = data if isinstance(data, dict) else {}
    session = context.get("session", {}) if isinstance(context.get("session"), dict) else {}
    customer = context.get("customer", {}) if isinstance(context.get("customer"), dict) else {}
    charger = context.get("charger", {}) if isinstance(context.get("charger"), dict) else {}
    site = context.get("site", {}) if isinstance(context.get("site"), dict) else {}
    ticket = context.get("ticket", {}) if isinstance(context.get("ticket"), dict) else {}
    return "<div class='as-kv-grid'>" + "".join([
        build_business_card(
            "Session 记忆（短期 / 严格隔离）",
            _join_html([
                format_key_value_grid([
                    ("当前 Session", session.get("session_id") or session_id or "等待 Agent 运行后生成"),
                    ("消息数", session.get("message_count")),
                    ("最近缺失信息", _list_inline(session.get("missing_info"))),
                    ("最近工单", session.get("last_ticket_id")),
                ]),
                format_list_block("最近用户问题", session.get("recent_user_messages")),
            ]),
            badge=build_status_badge("仅当前会话", "info"),
            tone="strong",
        ),
        build_business_card(
            "Customer / Charger / Site / Ticket 记忆（长期摘要）",
            format_key_value_grid([
                ("客户", customer.get("contact_phone") or customer.get("contact_name") or "本地 JSON 已预留"),
                ("设备", charger.get("serial_number") or charger.get("charger_model") or "本地 JSON 已预留"),
                ("场地", site.get("contact_address") or site.get("city") or "本地 JSON 已预留"),
                ("工单", ticket.get("title") or "本地 JSON 已预留"),
            ]),
            badge=build_status_badge("摘要记忆", "good"),
            tone="good",
            subtitle="保存历史摘要、风险历史和工单快照，不直接充当诊断依据。",
        ),
        build_business_card(
            "Repo / Knowledge 记忆（知识库）",
            "当前知识库仍由 RAG 管理，产品说明书、故障码表、安装规范、保修政策和安全规则不混入会话记忆。",
            badge=build_status_badge("RAG 独立", "info"),
            tone="muted",
        ),
        build_business_card(
            "安全隔离原则",
            "Session / User / Repo 三层作用域天然隔离；短期会话记忆防串话，长期记忆只读摘要，安全护栏前置并收敛到最终审核。",
            badge=build_status_badge("已启用", "good"),
            tone="good",
        ),
    ]) + "</div>"


def format_system_status(data: dict[str, Any]) -> str:
    metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
    file_names = metadata.get("file_names", []) if isinstance(metadata, dict) else []
    kb_loaded, kb_database_id = _kb_loaded_state(data)
    api_available = data.get("api_available", True) if isinstance(data, dict) else True
    provider = config.DEFAULT_LLM_PROVIDER
    model = config.DEFAULT_CHAT_MODEL if provider == "qwen" else config.DEEPSEEK_CHAT_MODEL
    key_configured = bool(config.API_KEY if provider == "qwen" else config.DEEPSEEK_API_KEY)
    return "<div class='as-kv-grid'>" + "".join([
        build_business_card(
            "FastAPI 连接",
            format_key_value_grid([
                ("状态", "已连接" if api_available else "未连接"),
                ("地址", API_BASE_URL),
                ("错误", data.get("error") if not api_available else ""),
            ]),
            badge=build_status_badge("已连接" if api_available else "未连接", "good" if api_available else "danger"),
            tone="good" if api_available else "danger",
        ),
        build_business_card(
            "知识库统计",
            format_key_value_grid([
                ("当前加载知识库", kb_database_id if kb_loaded else "未加载"),
                ("文档数", len(file_names)),
                ("Chunk 数", data.get("chunk_count") or metadata.get("chunk_count")),
                ("知识库名称", metadata.get("display_name")),
                ("解析器", metadata.get("parser")),
                ("最后更新时间", data.get("updated_at") or metadata.get("updated_at") or metadata.get("created_at")),
            ]),
            badge=build_status_badge("已加载" if kb_loaded else "未加载", "good" if kb_loaded else "muted"),
            tone="strong",
        ),
        build_business_card(
            "LLM 状态",
            format_key_value_grid([
                ("Provider", provider),
                ("Model", model),
                ("API Key", "已配置" if key_configured else "未配置"),
                ("最近一次调用", "暂未接入调用监控"),
            ]),
            badge=build_status_badge("已配置" if key_configured else "未配置", "good" if key_configured else "warning"),
            tone="muted",
        ),
        build_business_card(
            "SQLite 状态",
            format_key_value_grid([
                ("SQLite 状态", "未接入"),
                ("数据库路径", "未配置"),
                ("客户数 / 设备数 / 工单数", "未接入"),
            ]),
            badge=build_status_badge("未接入", "muted"),
            tone="muted",
        ),
        build_business_card(
            "安全治理状态",
            format_key_value_grid([
                ("输入安全扫描", "已启用"),
                ("上下文隔离", "已启用"),
                ("最终审核", "已启用"),
                ("记忆是否作为诊断证据", "否"),
            ]),
            badge=build_status_badge("已启用", "good"),
            tone="good",
        ),
    ]) + "</div>"


def format_raw_json(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except TypeError:
        return str(data)


def format_kb_status(data: dict[str, Any]) -> str:
    if not data:
        return build_placeholder("暂无知识库状态。")
    if data.get("api_available") is False:
        return build_business_card(
            "知识库摘要",
            format_key_value_grid([
                ("FastAPI", "未连接"),
                ("地址", API_BASE_URL),
                ("错误", data.get("error")),
            ]),
            badge=build_status_badge("未连接", "danger"),
            tone="danger",
        )
    metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
    file_names = metadata.get("file_names", []) if isinstance(metadata, dict) else []
    kb_loaded, kb_database_id = _kb_loaded_state(data)
    files = [_truncate(name, 52) for name in file_names[:10]]
    if len(file_names) > 10:
        files.append(f"还有 {len(file_names) - 10} 个文件未展示")
    return build_business_card(
        "知识库摘要",
        _join_html([
            format_key_value_grid([
                ("当前加载知识库", kb_database_id if kb_loaded else "未加载"),
                ("知识库名称", metadata.get("display_name")),
                ("文档数", len(file_names)),
                ("Chunk 数", data.get("chunk_count") or metadata.get("chunk_count")),
                ("解析器", metadata.get("parser")),
                ("chunk_size", metadata.get("chunk_size")),
                ("chunk_overlap", metadata.get("chunk_overlap")),
                ("文档类型", metadata.get("doc_type")),
                ("产品线", metadata.get("product_line")),
                ("产品/服务标识", metadata.get("item_identifier")),
                ("更新时间", data.get("updated_at") or metadata.get("updated_at") or metadata.get("created_at")),
            ]),
            format_list_block("文件列表", files),
        ]),
        badge=build_status_badge("已加载" if kb_loaded else "未加载", "good" if kb_loaded else "muted"),
        tone="strong" if kb_loaded else "muted",
    )


def _agent_stream_tuple(customer_reply: str, node_view: str, run_status: dict[str, Any], session_id: str) -> tuple[str, str, str, str]:
    reply = customer_reply or _customer_reply_from_run_status(run_status)
    return (reply, node_view, format_run_meta(run_status, session_id), session_id)


def _pending_agent_tuple(customer_reply: str, node_view: str, run_status: dict[str, Any], session_id: str) -> tuple[str, str, str, str]:
    return (customer_reply, node_view, format_run_meta(run_status, session_id), session_id)


def _status_agent_tuple(customer_reply: str, node_view: str, run_status: dict[str, Any], session_id: str) -> tuple[str, str, str, str]:
    return (customer_reply, node_view, format_run_meta(run_status, session_id), session_id)


def format_run_meta(run_status: dict[str, Any] | None, session_id: str = "") -> str:
    run_status = run_status or {}
    status = run_status.get("status") or "pending"
    body = format_key_value_grid([
        ("状态", STATUS_TEXT.get(str(status), status)),
        ("Run ID", run_status.get("run_id") or "待生成"),
        ("Session ID", run_status.get("session_id") or session_id or "待生成"),
        ("调试日志", run_status.get("debug_log_path") or "完成或失败后生成"),
        ("错误", run_status.get("error") or ""),
    ])
    return build_business_card("运行摘要", body, badge=build_status_badge(status, _tone_from_value(status)), tone=_tone_from_value(status))


def _customer_reply_from_run_status(run_status: dict[str, Any] | None) -> str:
    run_status = run_status or {}
    if run_status.get("customer_reply"):
        return str(run_status.get("customer_reply"))
    result = run_status.get("result") if isinstance(run_status.get("result"), dict) else {}
    action = result.get("action") if isinstance(result.get("action"), dict) else {}
    return str(action.get("customer_reply") or "暂未生成客户回复，请查看运行状态或后端调试日志。")


def _run_status_for_node_view(run_status: dict[str, Any] | None) -> dict[str, Any]:
    run_status = run_status or {}
    return {
        "status": run_status.get("status", "pending"),
        "node_statuses": run_status.get("node_statuses") or run_status.get("node_statuses_compact") or {},
    }


def _run_frontend_signature(run_status: dict[str, Any] | None) -> str:
    run_status = run_status or {}
    try:
        return json.dumps(
            {
                "status": run_status.get("status"),
                "error": run_status.get("error"),
                "nodes": run_status.get("node_statuses_compact") or run_status.get("node_statuses") or {},
                "customer_reply": run_status.get("customer_reply", ""),
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    except TypeError:
        return str(run_status)


def run_agent_stream(
    user_input: str,
    database_id: str,
    retrieval_mode: str,
    final_top_k: int,
    session_id: str = "",
):
    empty_view = build_node_visualization({})
    current_session_id = str(session_id or "")
    if not user_input.strip():
        yield _pending_agent_tuple("请先输入客户充电桩售后安全问题。", empty_view, {"status": "pending"}, current_session_id)
        return

    try:
        start_payload = call_charger_diagnosis_start_api(
            user_input,
            database_id,
            retrieval_mode,
            final_top_k,
            current_session_id,
        )
        if not start_payload.get("success"):
            failed_status = {
                "status": "api_unavailable",
                "session_id": current_session_id,
                "error": start_payload.get("error", "未知错误"),
                "node_statuses": {
                    "final": {
                        "title": "FastAPI 连接",
                        "status": "failed",
                        "input": {},
                        "output": {"error": start_payload.get("error", "未知错误")},
                    }
                },
            }
            yield _status_agent_tuple(
                f"接口调用失败：{start_payload.get('error', '未知错误')}",
                build_node_visualization(failed_status),
                failed_status,
                current_session_id,
            )
            return
        current_session_id = start_payload.get("data", {}).get("session_id") or current_session_id
        run_id = start_payload.get("data", {}).get("run_id")
        if not run_id:
            failed_status = {
                "status": "failed",
                "session_id": current_session_id,
                "error": "接口未返回 run_id，无法跟踪节点状态。",
                "node_statuses": {"final": {"title": "结果汇总", "status": "failed", "output": {"error": "missing run_id"}}},
            }
            yield _status_agent_tuple("接口未返回 run_id，无法跟踪节点状态。", build_node_visualization(failed_status), failed_status, current_session_id)
            return

        last_run_status: dict[str, Any] = {}
        last_node_view = empty_view
        last_signature = ""
        for _ in range(AGENT_POLL_LIMIT):
            status_payload = call_charger_diagnosis_run_status(run_id)
            if not status_payload.get("success"):
                yield _status_agent_tuple(
                    f"状态查询失败：{status_payload.get('error', '未知错误')}",
                    last_node_view,
                    status_payload,
                    current_session_id,
                )
                return

            run_status = status_payload.get("data", {})
            last_run_status = run_status
            current_session_id = run_status.get("session_id") or current_session_id
            node_view = build_node_visualization(_run_status_for_node_view(run_status))
            last_node_view = node_view
            if run_status.get("status") == "completed":
                yield _agent_stream_tuple("", node_view, run_status, current_session_id)
                return
            if run_status.get("status") == "failed":
                yield _status_agent_tuple(
                    f"充电桩安全诊断 Agent 运行失败：{run_status.get('error', '未知错误')}",
                    node_view,
                    run_status,
                    current_session_id,
                )
                return

            signature = _run_frontend_signature(run_status)
            if signature != last_signature:
                last_signature = signature
                yield _pending_agent_tuple("正在运行充电桩安全诊断 Agent，请稍候。", node_view, run_status, current_session_id)
            time.sleep(AGENT_POLL_INTERVAL_SECONDS)

        timeout_status = {
            **last_run_status,
            "status": "timeout",
            "error": last_run_status.get("error") or "frontend polling timeout",
        }
        yield _status_agent_tuple(
            "充电桩安全诊断 Agent 运行超时，请稍后查询或重试。",
            build_node_visualization(_run_status_for_node_view(timeout_status)),
            timeout_status,
            current_session_id,
        )
    except Exception as exc:
        unavailable_status = {
            "status": "api_unavailable",
            "session_id": current_session_id,
            "error": f"FastAPI 未连接：{exc}",
            "node_statuses": {
                "final": {
                    "title": "FastAPI 连接",
                    "status": "failed",
                    "input": {},
                    "output": {"error": str(exc)},
                }
            },
            "result": {},
            "trace": [],
            "tool_history": [],
        }
        yield _status_agent_tuple(
            f"FastAPI 未连接，请先启动 api.py。错误：{exc}",
            build_node_visualization(unavailable_status),
            unavailable_status,
            current_session_id,
        )


def get_kb_items() -> list[dict[str, Any]]:
    response = requests.get(f"{API_BASE_URL}/api/kb/list", timeout=KB_REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()
    return payload.get("data", {}).get("items", []) if payload.get("success") else []


def get_kb_items_safe() -> list[dict[str, Any]]:
    try:
        return get_kb_items()
    except Exception:
        return []


KbChoice = tuple[str, str]


def format_kb_choice(item: dict[str, Any]) -> KbChoice | None:
    database_id = str(item.get("database_id") or "").strip()
    if not database_id:
        return None
    display_name = str(item.get("display_name") or item.get("metadata", {}).get("display_name") or database_id).strip()
    label = str(item.get("label") or display_name).strip()
    if display_name and display_name not in label:
        label = f"{display_name} | {label}"
    return label, database_id


def get_kb_choices() -> list[KbChoice]:
    choices: list[KbChoice] = []
    for item in get_kb_items():
        choice = format_kb_choice(item)
        if choice:
            choices.append(choice)
    return choices


def get_kb_choices_safe() -> list[KbChoice]:
    choices: list[KbChoice] = []
    for item in get_kb_items_safe():
        choice = format_kb_choice(item)
        if choice:
            choices.append(choice)
    return choices


def first_kb_value(choices: list[KbChoice]) -> str | None:
    return choices[0][1] if choices else None


def _kb_dropdown_update(choices: list[KbChoice]) -> Any:
    if gr is None:
        return choices
    return gr.update(choices=choices, value=first_kb_value(choices))


def refresh_kb_choices() -> tuple[Any, Any, Any, str]:
    choices = get_kb_choices_safe()
    status = call_kb_status_safe()
    return _kb_dropdown_update(choices), _kb_dropdown_update(choices), _kb_dropdown_update(choices), format_kb_status(status)


def refresh_agent_kb_choices() -> Any:
    return _kb_dropdown_update(get_kb_choices_safe())


def refresh_kb_management_view() -> tuple[Any, str]:
    choices = get_kb_choices_safe()
    status = call_kb_status_safe()
    return _kb_dropdown_update(choices), format_kb_status(status)


def refresh_rag_kb_choices() -> Any:
    return _kb_dropdown_update(get_kb_choices_safe())


def call_kb_status() -> dict[str, Any]:
    response = requests.get(f"{API_BASE_URL}/api/kb/status", timeout=KB_REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json().get("data", {})


def call_kb_status_safe() -> dict[str, Any]:
    try:
        return call_kb_status()
    except Exception as exc:
        return {
            "loaded": False,
            "api_available": False,
            "error": f"FastAPI 未连接：{exc}",
        }


def call_kb_status_formatted() -> str:
    return format_system_status(call_kb_status_safe())


def build_kb(
    files: list[str] | str | None,
    display_name: str,
    doc_type: str,
    product_line: str,
    item_identifier: str,
    version: str,
    parser_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[str, Any, Any, Any, str]:
    if not files:
        empty_update = gr.update() if gr else {}
        return "请先上传 PDF 文件。", empty_update, empty_update, empty_update, build_placeholder("暂无知识库状态。")
    file_paths = files if isinstance(files, list) else [files]
    opened_files = []
    try:
        multipart_files = []
        for file_path in file_paths:
            path = Path(file_path)
            file_obj = path.open("rb")
            opened_files.append(file_obj)
            multipart_files.append(("files", (path.name, file_obj, "application/pdf")))

        response = requests.post(
            f"{API_BASE_URL}/api/kb/build",
            files=multipart_files,
            data={
                "display_name": display_name,
                "doc_type": doc_type,
                "product_line": product_line,
                "item_identifier": item_identifier,
                "version": version,
                "parser_name": parser_name,
                "splitter_name": "recursive",
                "chunk_size": int(chunk_size),
                "chunk_overlap": int(chunk_overlap),
            },
            timeout=1800,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success"):
            empty_update = gr.update() if gr else {}
            return f"知识库构建失败：{payload.get('error', '未知错误')}", empty_update, empty_update, empty_update, build_placeholder("暂无知识库状态。")
        data = payload.get("data", {})
        choices = get_kb_choices_safe()
        metadata = data.get("metadata", {})
        kb_name = metadata.get("display_name") or display_name or data.get("database_id")
        message = f"知识库构建完成：{kb_name}（{data.get('database_id')}），共 {data.get('chunk_count')} 个文本块。"
        status_text = format_kb_status({"loaded": True, **data})
        dropdown = gr.update(choices=choices, value=data.get("database_id")) if gr else choices
        return message, dropdown, dropdown, dropdown, status_text
    except Exception as exc:
        empty_update = gr.update() if gr else {}
        return f"知识库构建失败：{exc}", empty_update, empty_update, empty_update, build_placeholder("暂无知识库状态。")
    finally:
        for file_obj in opened_files:
            file_obj.close()


def load_kb(database_id: str) -> tuple[str, str]:
    if not database_id:
        return "请先选择知识库。", build_placeholder("暂无知识库状态。")
    response = requests.post(f"{API_BASE_URL}/api/kb/load", json={"database_id": database_id}, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        return f"知识库加载失败：{payload.get('error', '未知错误')}", build_placeholder("暂无知识库状态。")
    data = payload.get("data", {})
    return f"知识库已加载：{database_id}", format_kb_status({"loaded": True, **data, "database_id": database_id})


def delete_kb(database_id: str) -> tuple[str, Any, Any, Any, str]:
    if not database_id:
        empty_update = gr.update() if gr else []
        return "请先选择要删除的知识库。", empty_update, empty_update, empty_update, build_placeholder("暂无知识库状态。")
    try:
        response = requests.delete(f"{API_BASE_URL}/api/kb/{database_id}", timeout=60)
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success"):
            empty_update = gr.update() if gr else []
            return f"知识库删除失败：{payload.get('error', '未知错误')}", empty_update, empty_update, empty_update, build_placeholder("暂无知识库状态。")

        choices = get_kb_choices_safe()
        status = call_kb_status_safe()
        dropdown = _kb_dropdown_update(choices)
        return f"知识库已删除：{database_id}", dropdown, dropdown, dropdown, format_kb_status(status)
    except Exception as exc:
        empty_update = gr.update() if gr else []
        return f"知识库删除失败：{exc}", empty_update, empty_update, empty_update, build_placeholder("暂无知识库状态。")


def search_rag(question: str, database_id: str, retrieval_mode: str, final_top_k: int) -> dict[str, Any]:
    if not question.strip():
        return {"success": False, "error": "请先输入检索问题。"}
    response = requests.post(
        f"{API_BASE_URL}/api/rag/search",
        json={
            "question": question,
            "database_id": database_id or None,
            "retrieval_options": {
                "retrieval_mode": retrieval_mode,
                "final_top_k": final_top_k,
            },
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def search_rag_formatted(question: str, database_id: str, retrieval_mode: str, final_top_k: int) -> str:
    try:
        return format_rag_results(search_rag(question, database_id, retrieval_mode, final_top_k))
    except Exception as exc:
        return build_business_card("检索失败", html.escape(str(exc)), tone="warning")


def create_demo() -> Any:
    if gr is None:
        raise RuntimeError("当前环境未安装 gradio，请先执行 pip install -r requirements.txt。")

    examples = [
        "VoltGate VG-11KW-Pro 无法启动充电，屏幕显示 C-RCD-04，漏保频繁跳闸，东莞。",
        "VG-WallBox2 充到一半停止，App 显示 C-COM-12，想知道要不要上门。",
        "充电桩枪线破皮，还有烧焦味，能不能直接换新？",
        "VG-CloudMini APP 离线，昨天开始连不上，暂时没有明显发热或跳闸。",
    ]

    initial_choices: list[KbChoice] = []

    with gr.Blocks(title="新能源家用充电桩安全诊断工作台") as demo:
        gr.HTML(f"<style>{APP_CSS}</style>")
        gr.Markdown("# 新能源家用充电桩安全诊断工作台")
        gr.Markdown("面向客服的一体化售后工作台：前端只保留客服主链路和轻量管理功能，调试细节写入后端日志。", elem_classes=["as-page-note"])

        with gr.Tab("充电桩安全诊断 Agent"):
            session_state = gr.State(value="")
            with gr.Row(elem_classes=["as-agent-shell"]):
                with gr.Column(scale=3, min_width=280):
                    node_view = gr.HTML(value=build_node_visualization({}))
                with gr.Column(scale=8):
                    gr.HTML(value=build_placeholder("页面已就绪。知识库列表和系统状态会在点击刷新后从 FastAPI 加载；如果后端未启动，运行时会显示 FastAPI 未连接。"))
                    user_input = gr.Textbox(label="客户提问", lines=5, value=examples[0])
                    with gr.Row():
                        agent_kb_dropdown = gr.Dropdown(
                            label="知识库选择",
                            choices=initial_choices,
                            value=first_kb_value(initial_choices),
                            allow_custom_value=True,
                        )
                        retrieval_mode = gr.Radio(["hybrid", "vector", "bm25"], label="检索方式", value="hybrid")
                        final_top_k = gr.Slider(1, 10, value=5, step=1, label="TopK")
                    run_button = gr.Button("运行安全诊断 Agent", variant="primary")
                    customer_reply = gr.Textbox(label="回复客户（可直接复制）", lines=14)
                    run_meta_text = gr.HTML(value=format_run_meta({"status": "pending"}))

        with gr.Tab("知识库管理"):
            with gr.Row():
                with gr.Column(scale=5):
                    kb_files = gr.File(label="上传 PDF", file_count="multiple", file_types=[".pdf"], type="filepath", height=180)
                    with gr.Row():
                        display_name = gr.Textbox(label="知识库名称", value="充电桩售后安全知识库")
                        doc_type = gr.Textbox(label="文档类型", value="售后运维与安全指南")
                    with gr.Row():
                        product_line = gr.Textbox(label="产品线", value="新能源家用充电设备")
                        item_identifier = gr.Textbox(label="产品/服务标识", value="VoltGate")
                        version = gr.Textbox(label="版本号", value="2025")
                    with gr.Row():
                        parser_name = gr.Radio(["pypdf", "mineru"], label="PDF 解析器", value="pypdf")
                        chunk_size = gr.Slider(300, 1500, value=700, step=50, label="chunk_size")
                        chunk_overlap = gr.Slider(0, 300, value=80, step=10, label="chunk_overlap")
                    build_button = gr.Button("构建知识库", variant="primary")
                    kb_message = gr.Textbox(label="知识库操作结果", lines=2)
                with gr.Column(scale=4):
                    kb_dropdown = gr.Dropdown(
                        label="已有知识库",
                        choices=initial_choices,
                        value=first_kb_value(initial_choices),
                        allow_custom_value=True,
                    )
                    with gr.Row():
                        refresh_button = gr.Button("刷新知识库列表")
                        load_button = gr.Button("加载知识库")
                        delete_button = gr.Button("删除知识库", variant="stop")
                    kb_status_text = gr.HTML(value=build_placeholder("暂无知识库状态。"))
        with gr.Tab("知识检索功能"):
            with gr.Row():
                rag_question = gr.Textbox(label="检索关键词", value="VG-11KW-Pro C-RCD-04 漏保频繁跳闸 售后处理")
                rag_kb_dropdown = gr.Dropdown(
                    label="知识库选择",
                    choices=initial_choices,
                    value=first_kb_value(initial_choices),
                    allow_custom_value=True,
                )
            with gr.Row():
                rag_retrieval_mode = gr.Radio(["hybrid", "vector", "bm25"], label="检索方式", value="hybrid")
                rag_final_top_k = gr.Slider(1, 10, value=5, step=1, label="TopK")
            rag_button = gr.Button("检索知识库")
            rag_result_text = gr.HTML(value=build_placeholder("输入关键词后查看检索摘要和最多 3 条来源。"))

        with gr.Tab("系统状态"):
            status_button = gr.Button("刷新系统状态")
            system_status_text = gr.HTML(value=build_placeholder("点击刷新系统状态。"))

        build_button.click(
            fn=build_kb,
            inputs=[kb_files, display_name, doc_type, product_line, item_identifier, version, parser_name, chunk_size, chunk_overlap],
            outputs=[kb_message, kb_dropdown, agent_kb_dropdown, rag_kb_dropdown, kb_status_text],
            concurrency_id="kb_build",
            concurrency_limit=1,
        )
        refresh_button.click(fn=refresh_kb_choices, outputs=[kb_dropdown, agent_kb_dropdown, rag_kb_dropdown, kb_status_text], queue=False)
        load_button.click(fn=load_kb, inputs=kb_dropdown, outputs=[kb_message, kb_status_text], queue=False)
        delete_button.click(fn=delete_kb, inputs=kb_dropdown, outputs=[kb_message, kb_dropdown, agent_kb_dropdown, rag_kb_dropdown, kb_status_text], queue=False)
        run_button.click(
            fn=run_agent_stream,
            inputs=[user_input, agent_kb_dropdown, retrieval_mode, final_top_k, session_state],
            outputs=[
                customer_reply,
                node_view,
                run_meta_text,
                session_state,
            ],
            concurrency_id="agent_run",
            concurrency_limit=1,
        )
        rag_button.click(
            fn=search_rag_formatted,
            inputs=[rag_question, rag_kb_dropdown, rag_retrieval_mode, rag_final_top_k],
            outputs=rag_result_text,
            queue=False,
        )
        status_button.click(fn=call_kb_status_formatted, outputs=system_status_text, queue=False)
        demo.load(fn=refresh_kb_choices, outputs=[kb_dropdown, agent_kb_dropdown, rag_kb_dropdown, kb_status_text], queue=False)

    return demo


def build_business_card(title: str, body: str = "", badge: str = "", tone: str = "muted", subtitle: str = "") -> str:
    card_tone = tone if tone in {"strong", "danger", "warning", "good", "muted"} else "muted"
    subtitle_html = f"<div class='as-card-subtitle'>{html.escape(subtitle)}</div>" if subtitle else ""
    return (
        f"<div class='as-card as-card-{card_tone}'>"
        "<div class='as-card-head'>"
        f"<span class='as-card-title'>{html.escape(title)}</span>"
        f"{badge}"
        "</div>"
        f"{subtitle_html}"
        f"<div class='as-section-text'>{body or '待补充'}</div>"
        "</div>"
    )


def build_status_badge(value: Any, tone: str | None = None) -> str:
    badge_tone = tone or _tone_from_value(value)
    if badge_tone not in {"danger", "warning", "good", "info", "muted"}:
        badge_tone = "muted"
    return f"<span class='as-badge as-badge-{badge_tone}'>{html.escape(_display(value))}</span>"


def build_placeholder(text: str) -> str:
    return f"<div class='as-placeholder'>{html.escape(text)}</div>"


def format_key_value_grid(items: list[tuple[str, Any]]) -> str:
    blocks = []
    for label, value in items:
        blocks.append(
            "<div class='as-kv'>"
            f"<span class='as-kv-label'>{html.escape(label)}</span>"
            f"<span class='as-kv-value'>{html.escape(_display(value))}</span>"
            "</div>"
        )
    return "<div class='as-kv-grid'>" + "".join(blocks) + "</div>"


def format_text_block(label: str, value: Any) -> str:
    text = _display(value)
    if text == "待补充":
        return ""
    return (
        "<div class='as-kv'>"
        f"<span class='as-kv-label'>{html.escape(label)}</span>"
        f"<span class='as-kv-value'>{html.escape(text)}</span>"
        "</div>"
    )


def format_list_block(label: str, value: Any) -> str:
    items = _list_items(value)
    if not items:
        return ""
    list_html = "".join(f"<li>{html.escape(_truncate(item, 160))}</li>" for item in items)
    return (
        "<div class='as-kv'>"
        f"<span class='as-kv-label'>{html.escape(label)}</span>"
        f"<ul class='as-list'>{list_html}</ul>"
        "</div>"
    )


def _join_html(parts: list[str]) -> str:
    return "".join(part for part in parts if part)


def _kb_loaded_state(data: dict[str, Any]) -> tuple[bool, str]:
    database_id = str(data.get("current_database_id") or data.get("database_id") or "").strip()
    if "loaded" in data:
        return bool(data.get("loaded")), database_id
    return bool(database_id), database_id


def _tone_from_value(value: Any) -> str:
    text = str(value or "").lower()
    if any(token in text for token in ["p0", "p1", "high", "emergency", "danger", "fail", "失败", "高风险", "需复核"]):
        return "danger"
    if any(token in text for token in ["p2", "warning", "unknown", "注意", "待", "unknown"]):
        return "warning"
    if any(token in text for token in ["p3", "low", "completed", "success", "pass", "完成", "通过", "已配置", "已加载"]):
        return "good"
    if any(token in text for token in ["intent", "query", "case"]):
        return "info"
    return "muted"


def _aggregate_flow_step(step_id: str, source_nodes: tuple[str, ...], node_statuses: dict[str, Any]) -> dict[str, str]:
    events = [node_statuses.get(node_id, {}) or {} for node_id in source_nodes]
    statuses = [str(event.get("status", "pending")) for event in events if event]
    status = _aggregate_status(statuses)
    duration = sum(event.get("duration", 0) for event in events if isinstance(event.get("duration"), (int, float)))
    duration_text = f"{duration:.2f}秒" if duration else "未记录"

    input_text = "等待输入"
    for node_id, event in zip(source_nodes, events):
        if event.get("input"):
            input_text = _flow_input_summary(step_id, node_id, event.get("input"))
            break

    output_text = "等待输出"
    for node_id, event in reversed(list(zip(source_nodes, events))):
        if event.get("output"):
            output_text = _flow_output_summary(step_id, node_id, event.get("output"))
            break

    return {
        "status": status,
        "input": _truncate(input_text, 72),
        "output": _truncate(output_text, 88),
        "duration": duration_text,
    }


def _aggregate_status(statuses: list[str]) -> str:
    if not statuses:
        return "pending"
    if "failed" in statuses:
        return "failed"
    if "running" in statuses:
        return "running"
    if "warning" in statuses:
        return "warning"
    if statuses and all(status == "completed" for status in statuses):
        return "completed"
    if "completed" in statuses:
        return "running"
    return "pending"


def _flow_input_summary(step_id: str, node_id: str, data: Any) -> str:
    if not isinstance(data, dict):
        return _compact_json(data, max_length=96) or "等待输入"
    if step_id == "intent":
        return f"user_input={_display(data.get('user_input') or data.get('raw_text') or data.get('text'))}"
    if step_id == "retrieval":
        return f"query={_display(data.get('query') or data.get('question'))}"
    if step_id == "diagnosis":
        return f"case={_display(data.get('case_summary') or data.get('issue_description') or data.get('risk_level'))}"
    if step_id == "generation":
        return f"diagnosis={_display(data.get('summary') or data.get('priority') or data.get('diagnosis'))}"
    if step_id == "audit":
        return f"reply={_display(data.get('customer_reply') or data.get('action_summary') or data.get('action'))}"
    if step_id == "final":
        return "workflow=全部节点结果"
    return _compact_json(data, max_length=96) or "等待输入"


def _flow_output_summary(step_id: str, node_id: str, data: Any) -> str:
    if not isinstance(data, dict):
        return _compact_json(data, max_length=110) or "等待输出"
    if step_id == "intent":
        intent = data.get("intent")
        model = data.get("charger_model")
        missing = data.get("missing_info")
        parts = [f"intent={_display(intent)}" if intent else "", f"model={_display(model)}" if model else "", f"missing={_list_inline(missing)}" if missing else ""]
        return "，".join(part for part in parts if part) or _node_event_summary(node_id, data)
    if step_id == "retrieval":
        if node_id == "memory_context":
            return _node_event_summary(node_id, data)
        return f"result_count={_display(data.get('result_count'))}, message={_display(data.get('message') or data.get('error') or '检索完成')}"
    if step_id == "diagnosis":
        parts = [f"risk={_display(data.get('risk_level'))}" if data.get("risk_level") else "", f"summary={_display(data.get('summary'))}" if data.get("summary") else ""]
        return "，".join(part for part in parts if part) or _node_event_summary(node_id, data)
    if step_id == "generation":
        parts = [
            f"tool_count={_display(data.get('tool_count'))}" if data.get("tool_count") is not None else "",
            "has_reply=true" if data.get("has_customer_reply") else "",
            "has_dispatch=true" if data.get("has_dispatch") else "",
        ]
        return "，".join(part for part in parts if part) or _node_event_summary(node_id, data)
    if step_id == "audit":
        if node_id == "memory_answer":
            return _node_event_summary(node_id, data)
        return f"passed={_bool_text(data.get('passed'))}, warnings={_list_inline(data.get('warnings'))}"
    if step_id == "final":
        keys = data.get("keys") if isinstance(data.get("keys"), list) else []
        return f"keys={len(keys)}, message=结束" if keys else "message=结束"
    return _node_event_summary(node_id, data)


def _node_event_summary(node_id: str, detail: Any) -> str:
    if not isinstance(detail, dict) or not detail:
        return "等待节点输出"
    if node_id == "triage":
        return "，".join(filter(None, [
            f"意图：{_display(detail.get('intent'))}",
            f"置信度：{_display(detail.get('confidence'))}",
        ]))
    if node_id == "case_extract":
        return "，".join(filter(None, [
            f"型号：{_display(detail.get('charger_model'))}",
            f"故障码：{_list_inline(detail.get('fault_codes'))}",
            f"缺失：{_list_inline(detail.get('missing_info'))}",
        ]))
    if node_id == "safety_guard":
        return "，".join(filter(None, [
            f"风险：{_display(detail.get('risk_level'))}",
            f"信号：{_list_inline(detail.get('matched_safety_signals'))}",
        ]))
    if node_id == "memory_context":
        return "，".join(filter(None, [
            f"会话记忆：{_display(detail.get('message_count'))} 条",
            f"型号：{_display(detail.get('last_model'))}" if detail.get("last_model") else "",
            f"缺失：{_display(detail.get('missing_count'))}" if detail.get("missing_count") is not None else "",
        ])) or "已读取会话记忆"
    if node_id == "memory_answer":
        return "，".join(filter(None, [
            f"记忆回答：{_display(detail.get('answer_type'))}",
            _truncate(detail.get("message"), 70),
        ])) or "已生成会话记忆回答"
    if node_id == "retrieval":
        return "，".join(filter(None, [
            f"命中：{_display(detail.get('result_count'))}",
            f"错误：{_display(detail.get('error'))}" if detail.get("error") else "",
        ])) or "已完成检索"
    if node_id == "diagnosis":
        return "，".join(filter(None, [
            f"优先级：{_display(detail.get('priority'))}",
            _truncate(detail.get("summary"), 54),
        ]))
    if node_id == "warranty_dispatch":
        return "，".join(filter(None, [
            f"工具：{_display(detail.get('tool_count'))}",
            "已生成派工" if detail.get("has_dispatch") else "",
            f"错误：{_display(detail.get('errors'))}" if detail.get("errors") else "",
        ])) or "保修与派工已处理"
    if node_id == "action":
        return "已生成客户回复" if detail.get("has_customer_reply") else _truncate(detail, 70)
    if node_id == "audit":
        return "，".join(filter(None, [
            f"通过：{_bool_text(detail.get('passed'))}",
            f"警告：{_list_inline(detail.get('warnings'))}",
        ]))
    if node_id == "final":
        keys = detail.get("keys") if isinstance(detail.get("keys"), list) else []
        return f"返回字段：{len(keys)} 个" if keys else "结果已汇总"
    return _compact_json(detail, max_length=90) or "已处理"


def _section_data(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {}) if isinstance(data, dict) else {}
    return value if isinstance(value, dict) else {}


def _line(label: str, value: Any) -> str:
    return f"**{label}：** {_display(value)}"


def _list_block(label: str, value: Any) -> str:
    items = _list_items(value)
    if not items:
        return f"**{label}：** 无"
    return f"**{label}：**\n" + "\n".join(f"- {item}" for item in items)


def _list_inline(value: Any) -> str:
    items = _list_items(value)
    return "、".join(items) if items else "无"


def _list_items(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if value:
        return [str(value).strip()]
    return []


def _bool_text(value: Any) -> str:
    if value is True:
        return "是"
    if value is False:
        return "否"
    return "待补充"


def _display(value: Any) -> str:
    if value is None or value == "":
        return "待补充"
    if isinstance(value, bool):
        return _bool_text(value)
    if isinstance(value, list):
        return _list_inline(value)
    if isinstance(value, dict):
        return _compact_json(value, max_length=220) or "待补充"
    return str(value)


def _join_lines(lines: list[str]) -> str:
    return "\n".join(line for line in lines if line is not None)


def _truncate(value: Any, max_length: int = 400) -> str:
    text = str(value or "").strip()
    if len(text) <= max_length:
        return text or "待补充"
    return text[:max_length].rstrip() + "..."


def _compact_json(data: Any, max_length: int = 160) -> str:
    if not data:
        return ""
    try:
        text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        text = str(data)
    return text if len(text) <= max_length else text[:max_length] + "..."


def _score_text(item: dict[str, Any]) -> str:
    for key in ["score", "relevance_score", "rerank_score", "rrf_score"]:
        value = item.get(key)
        if isinstance(value, (int, float)):
            return f"{value:.4f}"
        if value not in {None, ""}:
            return str(value)
    return "未提供"


if __name__ == "__main__":
    create_demo().queue(default_concurrency_limit=2, max_size=20).launch(
        server_name=GRADIO_HOST,
        server_port=GRADIO_PORT,
    )
