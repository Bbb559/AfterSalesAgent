from __future__ import annotations

import json
from typing import Any

from backend.agents.llm_utils import invoke_json
from backend.prompts.diagnosis import CHARGER_DIAGNOSIS_PROMPT
from backend.schemas import ChargerDiagnosisResult


class ChargerDiagnosisAgent:
    """使用 LLM JSON 链生成充电桩安全诊断。"""

    PRIORITY_VALUES = {"p0_emergency", "p1_high", "p2_medium", "p3_low", "normal"}

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm

    def diagnose(
        self,
        case: dict[str, Any],
        retrieval: dict[str, Any],
        tools: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        fallback = self._fallback_diagnose(case, retrieval)
        llm_result = invoke_json(
            self.llm,
            CHARGER_DIAGNOSIS_PROMPT,
            {
                "case": json.dumps(case, ensure_ascii=False),
                "retrieval": json.dumps(self._compact_retrieval(retrieval), ensure_ascii=False),
                "tools": json.dumps(tools or {}, ensure_ascii=False),
            },
        )
        return self._normalize_diagnosis(llm_result, fallback, retrieval) if llm_result else fallback

    def _normalize_diagnosis(
        self,
        llm_result: dict[str, Any],
        fallback: dict[str, Any],
        retrieval: dict[str, Any],
    ) -> dict[str, Any]:
        normalized = dict(fallback)
        for key in ["summary", "suggested_next_step"]:
            value = self._clean_text(llm_result.get(key))
            if value:
                normalized[key] = value

        evidence_status = self._clean_text(llm_result.get("evidence_status")).lower()
        if evidence_status in {"grounded", "partial", "insufficient"}:
            normalized["evidence_status"] = evidence_status

        priority = self._clean_text(llm_result.get("priority"))
        if priority in self.PRIORITY_VALUES:
            normalized["priority"] = priority

        for key in [
            "likely_issue_areas",
            "fault_code_interpretation",
            "safe_remote_checks",
            "onsite_reasons",
            "risk_flags",
            "evidence_sources",
        ]:
            values = self._string_list(llm_result.get(key))
            if values:
                normalized[key] = values

        if not normalized.get("evidence_sources"):
            normalized["evidence_sources"] = retrieval.get("sources", [])
        return normalized

    def _fallback_diagnose(self, case: dict[str, Any], retrieval: dict[str, Any]) -> dict[str, Any]:
        brand = case.get("brand") or "待确认品牌"
        model = case.get("charger_model") or "待确认型号"
        issue = case.get("issue_description") or "问题描述待补充"
        fault_codes = self._string_list(case.get("fault_codes"))
        safe_checks = []
        if fault_codes:
            safe_checks.append(f"请先留存故障码 {'、'.join(fault_codes)} 的屏幕、App 报错或语音播报记录。")
        else:
            safe_checks.append("请补充故障发生时间、触发条件、是否可复现，以及 App/屏幕提示截图。")
        safe_checks.append("请拍摄设备铭牌、安装环境、配电箱外观、枪线和车辆充电口状态照片。")
        safe_checks.append("请补充订单或安装凭证、联系电话和安装地址，便于后续派工核验。")

        if retrieval.get("results"):
            summary = f"{brand} {model}，客户问题：{issue}。已检索到知识库资料，但当前未获得可用 LLM 诊断，需按资料继续核验。"
            evidence_status = "partial"
            suggested = "按知识库证据采集要求补充截图、铭牌和现场照片；仍异常或涉及安全信号时转人工/电工处理。"
        else:
            summary = f"{brand} {model}，客户问题：{issue}。当前知识库依据不足，不能自动判断具体原因或处理结论。"
            evidence_status = "insufficient"
            suggested = "请补充充电桩知识库依据，或转人工/电工核验后再给出具体处理方案。"

        return ChargerDiagnosisResult(
            summary=summary,
            evidence_status=evidence_status,
            likely_issue_areas=[],
            fault_code_interpretation=[],
            safe_remote_checks=safe_checks,
            onsite_reasons=[],
            priority="normal",
            suggested_next_step=suggested,
            evidence_sources=retrieval.get("sources", []),
            risk_flags=[],
        ).to_dict()

    def _compact_retrieval(self, retrieval: dict[str, Any]) -> dict[str, Any]:
        return {
            "sources": retrieval.get("sources", []),
            "results": [
                {
                    "file_name": item.get("file_name", ""),
                    "page": item.get("page", ""),
                    "text": item.get("text", "")[:700],
                }
                for item in retrieval.get("results", [])[:5]
            ],
        }

    def _clean_text(self, value: Any) -> str:
        return str(value or "").strip()

    def _string_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]
