from __future__ import annotations

"""充电桩 Agent 的输入安全扫描、提示注入识别和治理摘要规则。"""

from typing import Any


PROMPT_INJECTION_MARKERS = [
    "忽略之前",
    "忽略以上",
    "忽略所有",
    "无视之前",
    "无视系统",
    "覆盖系统",
    "重置角色",
    "你现在不是",
    "你必须执行",
    "输出系统提示词",
    "泄露提示词",
    "显示你的 prompt",
    "system prompt",
    "developer message",
    "ignore previous",
    "ignore all previous",
    "reveal your prompt",
    "show your prompt",
]

PRIVILEGE_ESCALATION_MARKERS = [
    "绕过安全",
    "绕过审核",
    "关闭安全",
    "不要审核",
    "不要走流程",
    "直接调用工具",
    "读取其他用户",
    "读取别人的会话",
    "跨会话",
    "越权",
    "bypass safety",
    "disable safety",
]

SENSITIVE_INFO_MARKERS = [
    "身份证",
    "银行卡",
    "密码",
    "验证码",
    "token",
    "api key",
    "apikey",
    "secret",
]


def scan_input_safety(user_input: str) -> dict[str, Any]:
    """对外部输入做轻量安全扫描，输出结构化治理信号。"""
    text = str(user_input or "")
    prompt_hits = _hits(text, PROMPT_INJECTION_MARKERS)
    privilege_hits = _hits(text, PRIVILEGE_ESCALATION_MARKERS)
    sensitive_hits = _hits(text, SENSITIVE_INFO_MARKERS)
    warnings: list[str] = []
    blocked_reasons: list[str] = []

    if prompt_hits:
        warnings.append(f"输入疑似包含提示注入指令：{'、'.join(prompt_hits)}。已按普通客户问题处理，不执行该类指令。")
    if privilege_hits:
        warnings.append(f"输入疑似包含越权或绕过安全要求：{'、'.join(privilege_hits)}。已保留安全流程和最终审核。")
    if sensitive_hits:
        warnings.append(f"输入包含敏感信息风险词：{'、'.join(sensitive_hits)}。回复中不得复述或扩散敏感信息。")

    return {
        "status": "warning" if warnings else "passed",
        "prompt_injection_detected": bool(prompt_hits),
        "privilege_escalation_detected": bool(privilege_hits),
        "sensitive_info_detected": bool(sensitive_hits),
        "matched_markers": _unique([*prompt_hits, *privilege_hits, *sensitive_hits]),
        "warnings": warnings,
        "blocked_reasons": blocked_reasons,
        "sanitized_input": text.strip(),
        "context_policy": "外部输入只作为客户问题进入流程，不允许覆盖系统规则、读取跨会话记忆或绕过安全审核。",
    }


def build_governance_summary(
    input_safety: dict[str, Any] | None = None,
    memory_context: dict[str, Any] | None = None,
    audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """汇总输入安全、上下文隔离和最终审核的治理状态。"""
    input_safety = input_safety or {}
    memory_context = memory_context or {}
    audit = audit or {}
    warnings = []
    warnings.extend(str(item) for item in input_safety.get("warnings", []) if str(item).strip())
    isolation = memory_context.get("isolation", {}) if isinstance(memory_context, dict) else {}
    if isolation and isolation.get("used_as_diagnostic_evidence") is not False:
        warnings.append("记忆上下文未明确标记为非诊断证据，需人工复核上下文隔离。")
    warnings.extend(str(item) for item in audit.get("warnings", []) if str(item).strip())

    return {
        "input_scan_enabled": True,
        "context_isolation_enabled": True,
        "final_audit_enabled": True,
        "memory_scope": isolation.get("scope", "session/customer/charger/site/ticket/repo"),
        "memory_used_as_diagnostic_evidence": bool(isolation.get("used_as_diagnostic_evidence")),
        "warnings": list(dict.fromkeys(warnings)),
        "status": "warning" if warnings else "passed",
    }


def _hits(text: str, markers: list[str]) -> list[str]:
    lower_text = text.lower()
    return [marker for marker in markers if marker.lower() in lower_text]


def _unique(items: list[str]) -> list[str]:
    result = []
    seen = set()
    for item in items:
        value = str(item or "").strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result
