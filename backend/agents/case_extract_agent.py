from __future__ import annotations

from typing import Any

from backend.agents.llm_utils import invoke_json
from backend.prompts.case_extract import CHARGER_CASE_EXTRACT_PROMPT
from backend.schemas import ChargerCase


class ChargerCaseExtractAgent:
    """用 LLM JSON 链抽取充电桩安全诊断字段。"""

    TEXT_FIELDS = [
        "brand",
        "charger_model",
        "charger_series",
        "serial_number",
        "charger_type",
        "installation_type",
        "rated_power_kw",
        "connector_type",
        "power_supply_phase",
        "breaker_or_rcd_info",
        "grounding_status",
        "vehicle_brand_model",
        "issue_type",
        "issue_description",
        "purchase_or_install_time",
        "warranty_or_order_evidence",
        "city",
        "contact_name",
        "contact_phone",
        "contact_address",
    ]
    LIST_FIELDS = [
        "fault_codes",
        "observed_symptoms",
        "safety_signals",
        "environment_factors",
        "installation_or_recent_changes",
        "customer_actions",
        "customer_requests",
    ]

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm

    def extract(self, text: str) -> dict[str, Any]:
        llm_case = invoke_json(self.llm, CHARGER_CASE_EXTRACT_PROMPT, {"user_input": text or ""})
        return self._normalize_case(llm_case, text or "") if llm_case else ChargerCase(raw_text=text or "").to_dict()

    def _normalize_case(self, llm_case: dict[str, Any], raw_text: str) -> dict[str, Any]:
        normalized = ChargerCase(raw_text=raw_text).to_dict()
        for key in self.TEXT_FIELDS:
            normalized[key] = self._clean_text(llm_case.get(key))
        for key in self.LIST_FIELDS:
            normalized[key] = self._unique_list(self._string_list(llm_case.get(key)))
        return normalized

    def _clean_text(self, value: Any) -> str:
        cleaned = str(value or "").strip()
        if cleaned in {"无", "未知", "未提供", "待补充", "空", "null", "None"}:
            return ""
        return cleaned

    def _string_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    def _unique_list(self, items: list[Any]) -> list[str]:
        result = []
        seen = set()
        for item in items:
            value = str(item or "").strip()
            if value and value not in seen:
                seen.add(value)
                result.append(value)
        return result
