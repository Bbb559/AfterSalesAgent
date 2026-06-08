from __future__ import annotations

"""充电桩客户输出拦截、保修承诺过滤和本地审核规则。"""

from typing import Any

from backend.schemas import ChargerActionResult, ChargerAuditResult


FREE_PROMISES = [
    "肯定免费维修",
    "一定免费维修",
    "保证免费维修",
    "绝对免费维修",
    "肯定免费",
    "一定免费",
    "保证免费",
    "绝对免费",
    "一定换新",
    "保证换新",
    "马上免费换新",
    "无需费用",
]

DANGEROUS_ACTIONS = [
    "开盖检修",
    "拆开外壳",
    "自行拆盖",
    "带电测量",
    "测量电压",
    "绕过漏保",
    "绕过接地",
    "继续充电观察",
    "自行更换空开",
    "自行更换漏保",
    "拆改配电箱",
    "触摸发热",
]


def evidence_text_from_retrieval(retrieval: dict[str, Any] | None) -> str:
    """把检索结果压成用于依据检查的文本。"""
    if not isinstance(retrieval, dict):
        return ""
    return "\n".join(str(item.get("text", "")) for item in retrieval.get("results", []) if isinstance(item, dict))


def enforce_reply(
    action: dict[str, Any],
    case: dict[str, Any],
    safety: dict[str, Any],
    warranty: dict[str, Any],
    retrieval: dict[str, Any] | None = None,
    dispatch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """集中拦截客户回复中的安全和保修硬风险。"""
    guarded = dict(action or {})
    reply = str(guarded.get("customer_reply", "") or "").strip()
    if safety.get("risk_level") in {"p0_emergency", "p1_high"}:
        reply = build_high_risk_reply(case, safety)
    else:
        reply = _replace_free_promises(reply)
        reply = _replace_dangerous_actions(reply)
        if _has_free_promise(action.get("customer_reply", "")) and not _has_warranty_caution(reply):
            reply += " 是否可以免费处理或换新，需要结合售后政策、购买/安装凭证、设备状态和人工核验结果确认。"

    guarded["customer_reply"] = reply
    guarded.setdefault("internal_advice", "")
    guarded["dispatch"] = dispatch or guarded.get("dispatch", {})
    return ChargerActionResult(
        customer_reply=guarded.get("customer_reply", ""),
        internal_advice=guarded.get("internal_advice", ""),
        dispatch=guarded.get("dispatch", {}),
    ).to_dict()


def enforce_diagnosis_grounding(
    diagnosis: dict[str, Any],
    case: dict[str, Any],
    retrieval: dict[str, Any],
) -> dict[str, Any]:
    """无知识库依据时清空具体原因和故障码解释，避免确定性编造。"""
    if retrieval.get("results"):
        return diagnosis

    guarded = dict(diagnosis or {})
    brand = case.get("brand") or "待确认品牌"
    model = case.get("charger_model") or "待确认型号"
    issue = case.get("issue_description") or "问题描述待补充"
    guarded["summary"] = f"{brand} {model}，客户问题：{issue}。当前知识库依据不足，不能自动判断具体原因或处理结论。"
    guarded["evidence_status"] = "insufficient"
    guarded["likely_issue_areas"] = []
    guarded["fault_code_interpretation"] = []
    guarded["evidence_sources"] = []
    guarded["suggested_next_step"] = "请补充充电桩知识库依据，或转人工/电工核验后再给出具体处理方案。"
    return guarded


def merge_with_local_audit(
    llm_audit: dict[str, Any],
    case: dict[str, Any],
    diagnosis: dict[str, Any],
    action: dict[str, Any],
    safety: dict[str, Any],
    warranty: dict[str, Any],
    retrieval: dict[str, Any],
    input_safety: dict[str, Any] | None = None,
    memory_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """合并 LLM 审核和不可删除的本地审核 warning。"""
    local = local_audit(case, diagnosis, action, safety, warranty, retrieval, input_safety, memory_context)
    warnings = list(local.get("warnings", []))
    llm_warnings = llm_audit.get("warnings", [])
    if isinstance(llm_warnings, list):
        warnings.extend(str(item) for item in llm_warnings if str(item).strip())

    risk_level = local.get("risk_level", "unknown")
    if risk_level not in {"p0_emergency", "p1_high"} and str(llm_audit.get("risk_level", "")).strip():
        risk_level = str(llm_audit["risk_level"]).strip()

    return ChargerAuditResult(
        passed=not warnings and bool(llm_audit.get("passed", True)),
        warnings=list(dict.fromkeys(warnings)),
        final_note=str(llm_audit.get("final_note") or local.get("final_note") or ""),
        risk_level=risk_level,
    ).to_dict()


def local_audit(
    case: dict[str, Any],
    diagnosis: dict[str, Any],
    action: dict[str, Any],
    safety: dict[str, Any],
    warranty: dict[str, Any],
    retrieval: dict[str, Any],
    input_safety: dict[str, Any] | None = None,
    memory_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """生成本地确定性审核结果。"""
    warnings = []
    input_safety = input_safety or {}
    memory_context = memory_context or {}
    for warning in input_safety.get("warnings", []):
        if str(warning).strip():
            warnings.append(str(warning).strip())

    isolation = memory_context.get("isolation", {}) if isinstance(memory_context, dict) else {}
    if isolation and isolation.get("used_as_diagnostic_evidence") is not False:
        warnings.append("记忆上下文未明确隔离为非诊断证据，需人工复核。")

    evidence_sources = diagnosis.get("evidence_sources", [])
    if any(_looks_like_memory_source(item) for item in evidence_sources if str(item).strip()):
        warnings.append("诊断证据来源疑似引用会话或长期记忆，记忆只能作为历史摘要，不能替代 RAG 知识库依据。")

    missing = case.get("missing_info", [])
    if missing:
        warnings.append(f"缺少充电桩工单关键信息：{'、'.join(missing)}。")

    if not retrieval.get("results"):
        warnings.append("没有检索到充电桩知识库依据，诊断不能给出具体故障原因。")

    risk_level = str(safety.get("risk_level", "unknown"))
    if risk_level in {"p0_emergency", "p1_high"}:
        warnings.append("高风险充电桩安全问题需要人工或上门电工介入。")

    reply = action.get("customer_reply", "")
    if risk_level in {"p0_emergency", "p1_high"} and not has_required_safety_reply(reply):
        warnings.append("高风险场景回复必须包含停止充电/暂停使用、远离风险源、条件安全时断电、不要自行拆修和人工/上门处理。")

    dangerous_hits = _dangerous_action_hits(reply)
    if dangerous_hits:
        warnings.append(f"回复包含危险电气操作建议（{'、'.join(dangerous_hits)}），必须删除或改为专业人员处理。")

    if _has_free_promise(reply):
        warnings.append("回复存在保修、换新或免费处理的过度承诺风险。")

    if not retrieval.get("results") and any(word in reply for word in ["就是", "确定是", "一定是", "主板", "漏保模块", "交流接触器"]):
        warnings.append("缺少知识库依据时不得给出确定故障部件或具体原因。")

    if not evidence_text_from_retrieval(retrieval) and any(word in reply for word in ["秒", "分钟", "电压", "V", "千瓦", "kW"]):
        warnings.append("回复包含具体操作参数但缺少知识库依据，建议人工复核。")

    if _has_warranty_review_context(case, warranty) and not (
        case.get("purchase_or_install_time") or case.get("warranty_or_order_evidence")
    ):
        warnings.append("保修或免费处理判断缺少购买/安装时间、订单记录或凭证，不能直接承诺免费。")

    return ChargerAuditResult(
        passed=not warnings,
        warnings=warnings,
        final_note="可直接回复客户。" if not warnings else "建议人工确认后再回复客户。",
        risk_level=risk_level,
    ).to_dict()


def build_high_risk_reply(case: dict[str, Any], safety: dict[str, Any]) -> str:
    """构造分段式高风险客户回复。"""
    charger = _charger_label(case)
    matched = _string_list(safety.get("matched_safety_signals"))
    source_text = " ".join([
        str(case.get("raw_text", "") or ""),
        " ".join(_string_list(case.get("fault_codes"))),
        " ".join(_string_list(case.get("safety_signals"))),
        " ".join(matched),
    ])

    immediate_actions = _unique_strings([
        "请立即停止充电，暂停使用这台充电桩。",
        "请先远离充电桩、枪线、车辆充电口和配电箱等风险源。",
        "在确认人身安全且不需要接触异常部位的前提下，切断充电桩或上级空开电源。",
    ])
    forbidden_actions = _unique_strings([
        "不要自行拆修、开盖或拆改配电箱。",
        *_specific_forbidden_actions(source_text),
    ])
    follow_up = "我会按高优先级转人工，并建议安排上门电工或工程师到现场核验处理。"
    if safety.get("risk_level") == "p0_emergency":
        follow_up = "如果现场仍有明火、持续冒烟、触电或人员受伤，请先远离现场并优先联系当地应急救援；随后我们再安排人工和上门电工跟进。"

    return (
        f"您好，{charger} 当前描述已经涉及充电桩电气安全风险，建议先按高风险处理。\n\n"
        f"立即动作：{' '.join(immediate_actions)}\n"
        f"禁止动作：{' '.join(forbidden_actions)}\n"
        f"后续处理：{follow_up}"
    )


def has_required_safety_reply(reply: str) -> bool:
    return all([
        any(word in reply for word in ["停止充电", "暂停使用", "停止使用"]),
        "远离" in reply,
        any(word in reply for word in ["断电", "切断", "空开电源", "上级空开"]),
        any(word in reply for word in ["不要自行拆修", "不要自行", "不要拆修"]),
        any(word in reply for word in ["人工", "上门", "电工", "工程师"]),
    ])


def _replace_free_promises(reply: str) -> str:
    guarded = reply
    for phrase in FREE_PROMISES:
        guarded = guarded.replace(phrase, "是否免费或换新需核验后确认")
    return guarded


def _replace_dangerous_actions(reply: str) -> str:
    guarded = reply
    for phrase in DANGEROUS_ACTIONS:
        guarded = guarded.replace(phrase, "等待专业人员核验")
    return guarded


def _dangerous_action_hits(reply: str) -> list[str]:
    hits = []
    for phrase in DANGEROUS_ACTIONS:
        if phrase in reply and not _is_negated_action(reply, phrase):
            hits.append(phrase)
    return hits


def _is_negated_action(reply: str, phrase: str) -> bool:
    markers = ["不要", "不得", "禁止", "避免", "不能", "请勿"]
    start = 0
    while True:
        index = reply.find(phrase, start)
        if index < 0:
            return False
        window = reply[max(0, index - 12):index]
        if any(marker in window for marker in markers):
            return True
        start = index + len(phrase)


def _has_free_promise(reply: str) -> bool:
    return any(phrase in str(reply or "") for phrase in FREE_PROMISES)


def _has_warranty_caution(reply: str) -> bool:
    has_evidence = "凭证" in reply or "订单" in reply or "发票" in reply
    has_verification = "核验" in reply or "确认" in reply or "核实" in reply
    return has_evidence and has_verification


def _has_warranty_review_context(case: dict[str, Any], warranty: dict[str, Any]) -> bool:
    return bool(
        case.get("issue_type") == "warranty"
        or case.get("purchase_or_install_time")
        or case.get("warranty_or_order_evidence")
        or warranty.get("status") in {"possibly_in_warranty", "possibly_out_of_warranty"}
    )


def _charger_label(case: dict[str, Any]) -> str:
    parts = [case.get("brand", ""), case.get("charger_model", "")]
    label = " ".join(str(part).strip() for part in parts if str(part).strip())
    return label or "您的家用充电桩"


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _specific_forbidden_actions(text: str) -> list[str]:
    actions = []
    if any(word in text for word in ["C-RCD-04", "漏保", "跳闸", "空开"]):
        actions.extend([
            "不要重新合上漏保或空开再试。",
            "不要反复复位漏保或空开。",
            "不要继续充电观察。",
        ])
    if any(word in text for word in ["C-GND-01", "接地异常", "接地故障"]):
        actions.extend([
            "不要绕过接地、私拉乱接或继续充电。",
        ])
    if any(word in text for word in ["枪头发热", "枪线破皮", "枪线破损", "烧焦味", "焦糊味", "积水", "进水", "雨水倒灌"]):
        actions.extend([
            "不要触摸枪头、枪线、发热或破损部位。",
            "请远离积水和设备区域。",
        ])
    if any(word in text for word in ["冒烟", "明火", "触电", "起火", "着火"]):
        actions.append("如有冒烟、明火、触电或人员受伤，请按紧急情况处理并优先联系应急救援。")
    return actions


def _unique_strings(items: list[str]) -> list[str]:
    result = []
    seen = set()
    for item in items:
        value = str(item or "").strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _looks_like_memory_source(value: Any) -> bool:
    text = str(value or "").lower()
    return any(token in text for token in [
        "memory",
        "session",
        "customer_memory",
        "charger_memory",
        "site_memory",
        "ticket_memory",
        "会话记忆",
        "客户记忆",
        "设备记忆",
        "场地记忆",
        "工单记忆",
        "历史记忆",
    ])
