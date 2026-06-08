from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


CHARGER_TRIAGE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是新能源家用充电桩售后安全分诊 Agent。"
        "请理解客户真实诉求，并优先识别电气安全、火灾、触电、过热、漏保、接地、进水等风险。"
        "只输出 JSON，不要输出 Markdown。不要只靠关键词，要结合上下文判断。",
    ),
    (
        "human",
        """客户问题：
{user_input}

可选 intent：
- safety_emergency：客户描述冒烟、明火、触电、人员受伤、严重过热、配电箱异常等紧急安全问题。
- fault_diagnosis：客户描述充电桩故障、故障码、无法充电、充到一半停止、APP 离线、刷卡无效等，需要排障。
- warranty_consultation：客户主要询问保修、质保、费用、是否免费、是否换新。
- service_dispatch：客户明确要求上门、派工、电工处理、建工单、预约。
- usage_or_policy_lookup：客户询问使用方法、安装要求、保修政策或资料说明。
- human_handoff：客户要求人工、投诉、主管，或分诊无法安全自动处理。
- unknown：无法判断。

请输出 JSON 字段：
intent, confidence, reason

confidence 只能是 high、medium、low。

输出 JSON：""",
    ),
])
