from __future__ import annotations

"""充电桩派工草稿、证据清单和派工建议规则。"""

from typing import Any

from backend.schemas import DispatchDraft


def build_dispatch(
    case: dict[str, Any],
    diagnosis: dict[str, Any],
    warranty: dict[str, Any],
    safety: dict[str, Any],
) -> dict[str, Any]:
    """根据 case、诊断、保修和安全结果生成派工草稿。"""
    need_human = bool(safety.get("need_human") or diagnosis.get("priority") in {"p0_emergency", "p1_high"})
    need_onsite = bool(safety.get("need_onsite") or diagnosis.get("onsite_reasons"))
    need_electrician = bool(safety.get("need_electrician") or need_onsite)

    return DispatchDraft(
        customer_problem=case.get("raw_text", ""),
        brand=case.get("brand") or "待补充",
        charger_model=case.get("charger_model") or "待补充",
        serial_number=case.get("serial_number") or "待补充",
        fault_codes=_string_list(case.get("fault_codes")),
        observed_symptoms=_string_list(case.get("observed_symptoms")),
        safety_level=safety.get("risk_level", "unknown"),
        site_environment=_string_list(case.get("environment_factors")),
        city=case.get("city") or "待补充",
        contact_name=case.get("contact_name") or "待补充",
        contact_phone=case.get("contact_phone") or "待补充",
        contact_address=case.get("contact_address") or "待补充",
        evidence_needed=_build_evidence_needed(case, diagnosis, safety),
        suggested_dispatch=_suggest_dispatch(safety, diagnosis),
        need_electrician=need_electrician,
        need_onsite=need_onsite,
        priority=diagnosis.get("priority") or safety.get("risk_level", "normal"),
        missing_info=_string_list(case.get("missing_info")),
        internal_note=_internal_note(need_human, need_onsite, need_electrician, warranty),
    ).to_dict()


def _build_evidence_needed(
    case: dict[str, Any],
    diagnosis: dict[str, Any],
    safety: dict[str, Any],
) -> list[str]:
    evidence = [
        "设备铭牌或 App 设备页截图",
        "故障码、屏幕提示或 App 报错截图",
        "安装环境、配电箱外观、枪线和车辆充电口照片",
        "订单、发票或安装记录",
    ]
    if safety.get("matched_safety_signals"):
        evidence.append("安全风险现场照片或视频，确保拍摄前已远离风险源")
    if case.get("missing_info"):
        evidence.append(f"补充缺失信息：{'、'.join(_string_list(case.get('missing_info')))}")
    if diagnosis.get("evidence_status") == "insufficient":
        evidence.append("补充知识库或人工/电工核验记录")
    return list(dict.fromkeys(evidence))


def _suggest_dispatch(safety: dict[str, Any], diagnosis: dict[str, Any]) -> str:
    risk_level = safety.get("risk_level", "unknown")
    if risk_level == "p0_emergency":
        return "紧急安全事件，优先人工接入；存在明火、触电或人员受伤时优先当地应急救援，随后安排上门电工/工程师。"
    if risk_level == "p1_high":
        return "高风险安全事件，停止远程普通排障，优先人工接入并安排上门电工或工程师核验。"
    if diagnosis.get("evidence_status") == "insufficient":
        return "知识库依据不足，先补充资料或人工核验，再决定是否派工。"
    return "可先按知识库进行安全远程核验；若复现或客户要求服务，再创建上门/人工跟进工单。"


def _internal_note(
    need_human: bool,
    need_onsite: bool,
    need_electrician: bool,
    warranty: dict[str, Any],
) -> str:
    parts = []
    if need_human:
        parts.append("安全风险需优先人工介入")
    if need_onsite:
        parts.append("建议上门核验")
    if need_electrician:
        parts.append("建议具备电工资质人员处理")
    parts.append(f"保修状态：{warranty.get('status', 'unknown')}")
    return "；".join(parts)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
