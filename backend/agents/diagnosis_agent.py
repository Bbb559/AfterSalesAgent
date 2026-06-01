from __future__ import annotations

from typing import Any

from backend.rules.fault_rules import FAULT_RULES
from backend.rules.risk_rules import RISK_KEYWORDS
from backend.schemas import DiagnosisResult


class DiagnosisAgent:
    """根据病因字段和RAG命中创建初步有据诊断."""

    def diagnose(self, case: dict[str, Any], retrieval: dict[str, Any]) -> dict[str, Any]:
        results = retrieval.get("results", [])  # 取前3条检索结果作为诊断依据
        evidence_text = "\n".join(item.get("text", "") for item in results[:3]) # 拼接检索结果文本作为诊断参考
        fault_code = case.get("fault_code", "") # 获取故障码
        symptoms = case.get("symptoms", []) # 获取故障现象
        symptom_text = "、".join(symptoms) # 将故障现象用顿号连接成一段文本，方便后续关键词匹配

        possible_causes = [] # 可能的故障原因
        remote_steps = [] # 远程排查步骤
        priority = "normal" # 故障优先级，默认为normal

        fault_rule = FAULT_RULES.get(fault_code) 
        if fault_rule:
            possible_causes.extend(fault_rule["possible_causes"])
            remote_steps.extend(fault_rule["suggested_actions"])
            priority = fault_rule["priority"]
        elif fault_code:
            possible_causes.append(f"知识库中需要优先核对故障码 {fault_code} 的定义和处理流程。")

        if any(word in symptom_text for word in ["出水慢", "出水变慢", "不出水"]):
            possible_causes.extend(["进水压力不足", "前置滤芯堵塞", "进水阀未完全打开"])
        if any(word in symptom_text for word in ["漏水", "渗水", "插座打湿"]):
            possible_causes.extend(["接头松动", "滤芯或管路密封异常"])
        if not possible_causes:
            possible_causes.append("客户描述仍不完整，需要补充现象、故障码和设备状态。")

        risk_flags = [word for word in RISK_KEYWORDS if word in case.get("raw_text", "")]
        urgency = "normal"
        if any(RISK_KEYWORDS[word] == "high" for word in risk_flags):
            urgency = "high"
            priority = "high"
        elif any(RISK_KEYWORDS[word] == "medium" for word in risk_flags):
            priority = "medium"

        remote_steps.extend(self._build_steps(case, evidence_text))

        return DiagnosisResult(
            summary=self._build_summary(case, possible_causes), # 初步诊断总结
            possible_causes=list(dict.fromkeys(possible_causes)),# 可能的故障原因并去重
            remote_steps=list(dict.fromkeys(remote_steps)),# 远程排查步骤然后去重
            urgency=urgency, # 紧急程度
            priority=priority, # 故障优先级
            suggested_action=self._suggest_action(urgency, case),   # 建议处理方式
            evidence_sources=retrieval.get("sources", []),   # 诊断依据的来源
            risk_flags=risk_flags, # 诊断过程中识别出的风险关键词
        ).to_dict()

    def _build_summary(self, case: dict[str, Any], causes: list[str]) -> str:
        model = case.get("product_model") or "未知型号"
        fault = case.get("fault_code") or "未提供故障码"
        symptoms = case.get("symptoms") or []
        symptom = "、".join(symptoms) if symptoms else "现象待补充"
        return f"{model}，{fault}，客户现象：{symptom}。初步关注：{'、'.join(causes[:3])}。"

    def _build_steps(self, case: dict[str, Any], evidence_text: str) -> list[str]:
        steps = [
            "确认设备型号、故障码、购买时间、门店地址和联系电话。",
            "让客户拍摄设备屏幕、进水阀、水压状态和滤芯状态。",
        ]

        symptom = "、".join(case.get("symptoms", []))
        if any(word in symptom for word in ["出水慢", "出水变慢", "不出水"]):
            steps.extend([
                "检查进水阀是否完全打开。",
                "检查前置滤芯是否到期或堵塞。",
                "确认同一水路其他设备是否也水压偏低。",
            ])
        if any(word in symptom for word in ["漏水", "渗水", "插座打湿"]):
            steps.extend([
                "确认漏水位置：进水口、滤芯仓、排水管或机身底部。",
                "建议客户先断电并关闭进水阀，避免扩大损失。",
            ])
        if evidence_text:
            steps.append("对照知识库检索依据执行对应 SOP。")

        return steps

    def _suggest_action(self, urgency: str, case: dict[str, Any]) -> str: 
        if urgency == "high":
            return "建议立即升级人工并创建上门工单。"
        if case.get("fault_code") or case.get("symptoms"):
            return "建议先远程排查；若无法恢复，再创建上门工单。"
        return "建议先补充关键信息，再进入诊断流程。"
