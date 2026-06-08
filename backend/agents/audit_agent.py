from __future__ import annotations

import json
from typing import Any

from backend.agents.llm_utils import invoke_json
from backend.prompts.audit import CHARGER_AUDIT_PROMPT
from backend.schemas import ChargerAuditResult


class ChargerAuditAgent:
    """使用 LLM JSON 链审核充电桩安全诊断回复。"""

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm

    def audit(
        self,
        case: dict[str, Any],
        diagnosis: dict[str, Any],
        retrieval: dict[str, Any],
        action: dict[str, Any],
        safety: dict[str, Any] | None = None,
        warranty: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        llm_audit = invoke_json(
            self.llm,
            CHARGER_AUDIT_PROMPT,
            {
                "case": json.dumps(case, ensure_ascii=False),
                "safety": json.dumps(safety or {}, ensure_ascii=False),
                "diagnosis": json.dumps(diagnosis, ensure_ascii=False),
                "retrieval": json.dumps(
                    {"sources": retrieval.get("sources", []), "result_count": len(retrieval.get("results", []))},
                    ensure_ascii=False,
                ),
                "action": json.dumps(action, ensure_ascii=False),
            },
        )
        return self._normalize_audit(llm_audit) if llm_audit else ChargerAuditResult().to_dict()

    def _normalize_audit(self, payload: dict[str, Any]) -> dict[str, Any]:
        warnings = payload.get("warnings", [])
        if not isinstance(warnings, list):
            warnings = []
        return ChargerAuditResult(
            passed=bool(payload.get("passed", True)),
            warnings=[str(item).strip() for item in warnings if str(item).strip()],
            final_note=str(payload.get("final_note") or "可直接回复客户。"),
            risk_level=str(payload.get("risk_level") or "unknown"),
        ).to_dict()
