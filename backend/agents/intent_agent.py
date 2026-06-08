from __future__ import annotations

from typing import Any

from backend.agents.llm_utils import invoke_json
from backend.prompts.intent import CHARGER_TRIAGE_PROMPT
from backend.schemas import TriageResult


class ChargerTriageAgent:
    """使用 LLM 优先的 JSON 链识别充电桩安全分诊意图。"""

    VALID_INTENTS = {
        "safety_emergency", # 有效意图
        "fault_diagnosis", # 故障诊断
        "warranty_consultation", # 保修咨询
        "service_dispatch", # 服务派遣
        "usage_or_policy_lookup", # 使用或政策查询
        "human_handoff", # 人工转接
        "unknown", # 未知意图
    }
    VALID_CONFIDENCE = {"high", "medium", "low"}

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm

    def triage(self, text: str) -> dict[str, Any]:
        normalized = (text or "").strip()
        if not normalized:
            return self._result("unknown", "low", "客户未提供充电桩售后问题。")

        llm_result = invoke_json(self.llm, CHARGER_TRIAGE_PROMPT, {"user_input": normalized})
        if llm_result:
            return self._normalize_llm_result(llm_result, normalized)

        return self._fallback_triage(normalized)

    def _normalize_llm_result(self, payload: dict[str, Any], text: str) -> dict[str, Any]:
        intent = str(payload.get("intent") or payload.get("name") or "").strip()
        if intent not in self.VALID_INTENTS:
            return self._fallback_triage(text)

        confidence = str(payload.get("confidence") or "medium").strip().lower()
        if confidence not in self.VALID_CONFIDENCE:
            confidence = "medium"

        reason = str(payload.get("reason") or "").strip()
        return self._result(intent, confidence, reason)

    def _fallback_triage(self, text: str) -> dict[str, Any]:
        return self._result("unknown", "low", "LLM 不可用，未做本地业务语义推断。")

    def _result(self, intent: str, confidence: str, reason: str = "") -> dict[str, Any]:
        return TriageResult(intent=intent, confidence=confidence, reason=reason).to_dict()
