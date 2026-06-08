from __future__ import annotations

"""家用充电桩电气安全信号、风险分级和诊断覆盖规则。"""

import re
from typing import Any

from backend.schemas import SafetyResult


EMERGENCY_SIGNALS = [
    "明火",
    "起火",
    "着火",
    "火苗",
    "冒烟",
    "配电箱冒烟",
    "触电",
    "电到人",
    "人员受伤",
]

HIGH_RISK_SIGNALS = [
    "火花",
    "打火",
    "烧焦味",
    "焦糊味",
    "漏电",
    "麻手",
    "漏保频繁跳闸",
    "漏电保护频繁跳闸",
    "空开跳闸",
    "枪线破皮",
    "枪线破损",
    "枪头发热",
    "车辆充电口发热",
    "充电口发热",
    "过热",
    "进水",
    "雨水倒灌",
    "积水",
    "接地异常",
    "接地故障",
    "私拉乱接",
    "自行拆盖",
    "私自拆开",
]

SAFETY_FAULT_CODES = {
    "C-GND-01": "接地异常",
    "C-RCD-04": "漏保自检失败",
    "C-TEMP-09": "枪头温度过高",
}

FORBIDDEN_ACTIONS = [
    "不要开盖检修或拆改充电桩外壳。",
    "不要带电测量、触碰内部端子或拆改配电箱。",
    "不要绕过漏保、接地或空开继续充电。",
    "不要触摸发热、破损、进水的枪线或枪头。",
    "不要在冒烟、异味、进水或跳闸后继续充电观察。",
]

HIGH_RISK_REQUIRED_ACTIONS = [
    "立即停止充电并暂停使用充电桩。",
    "远离充电桩、枪线、车辆充电口和配电箱等风险源。",
    "在确保自身安全的前提下，切断充电桩或上级空开电源。",
    "不要自行拆修，等待人工客服、电工或上门工程师处理。",
]

EMERGENCY_REQUIRED_ACTIONS = [
    "立即停止充电并远离现场。",
    "如存在明火、持续冒烟、触电或人员受伤，请优先联系当地应急救援。",
    "在确保自身安全的前提下，切断充电桩或上级空开电源。",
    "不要自行拆修或继续靠近设备，等待人工客服、电工或上门工程师处理。",
]


def find_charger_safety_signals(*texts: str) -> list[str]:
    """从文本中识别充电桩安全风险信号。"""
    combined_text = " ".join(text for text in texts if text)
    matched: list[str] = []
    for signal in [*EMERGENCY_SIGNALS, *HIGH_RISK_SIGNALS]:
        if signal in combined_text and signal not in matched:
            matched.append(signal)

    for code, meaning in SAFETY_FAULT_CODES.items():
        if re.search(rf"\b{re.escape(code)}\b", combined_text, re.I):
            matched.append(code)
            if meaning not in matched:
                matched.append(meaning)

    return _unique(matched)


def evaluate_charger_safety(case: dict[str, Any], raw_text: str = "") -> dict[str, Any]:
    """根据结构化案例和原始文本生成本地安全分级结果。"""
    source_text = " ".join([
        raw_text,
        str(case.get("raw_text", "") or ""),
        " ".join(_string_list(case.get("safety_signals"))),
        " ".join(_string_list(case.get("fault_codes"))),
    ])
    matched = find_charger_safety_signals(source_text)

    if not matched:
        return SafetyResult(
            risk_level="p3_low",
            need_human=False,
            need_onsite=False,
            need_electrician=False,
            reason="未命中本地充电桩高风险安全信号，可继续按知识库证据进行安全远程核验。",
            matched_safety_signals=[],
            forbidden_actions=FORBIDDEN_ACTIONS,
            required_customer_actions=[],
        ).to_dict()

    if any(signal in matched for signal in EMERGENCY_SIGNALS):
        return SafetyResult(
            risk_level="p0_emergency",
            need_human=True,
            need_onsite=True,
            need_electrician=True,
            reason=f"命中紧急安全信号：{'、'.join(matched)}。",
            matched_safety_signals=matched,
            forbidden_actions=FORBIDDEN_ACTIONS,
            required_customer_actions=EMERGENCY_REQUIRED_ACTIONS,
        ).to_dict()

    return SafetyResult(
        risk_level="p1_high",
        need_human=True,
        need_onsite=True,
        need_electrician=True,
        reason=f"命中高风险充电桩安全信号：{'、'.join(matched)}。",
        matched_safety_signals=matched,
        forbidden_actions=FORBIDDEN_ACTIONS,
        required_customer_actions=HIGH_RISK_REQUIRED_ACTIONS,
    ).to_dict()


def enforce_diagnosis(diagnosis: dict[str, Any], safety: dict[str, Any]) -> dict[str, Any]:
    """把高风险安全结论显式覆盖到诊断结果中。"""
    risk_level = safety.get("risk_level", "unknown")
    if risk_level not in {"p0_emergency", "p1_high"}:
        return diagnosis

    guarded = dict(diagnosis)
    guarded["priority"] = risk_level
    guarded["risk_flags"] = _string_list(safety.get("matched_safety_signals"))
    guarded["onsite_reasons"] = _unique([
        *_string_list(guarded.get("onsite_reasons")),
        str(safety.get("reason", "") or ""),
    ])
    guarded["safe_remote_checks"] = _unique([
        *_string_list(safety.get("required_customer_actions")),
        "请保留故障码、现场照片、视频和订单/安装凭证，等待人工或上门工程师核验。",
    ])
    guarded["suggested_next_step"] = "立即按安全护栏停止充电、远离风险源，并转人工或上门电工处理。"
    if "安全风险" not in guarded.get("summary", ""):
        guarded["summary"] = f"{guarded.get('summary', '')} 当前描述涉及充电桩安全风险。".strip()
    return guarded


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _unique(items: list[str]) -> list[str]:
    result = []
    seen = set()
    for item in items:
        value = str(item or "").strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result
