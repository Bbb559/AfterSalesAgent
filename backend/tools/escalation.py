from __future__ import annotations

from typing import Any

from backend.rules.escalation_rules import ESCALATION_REASONS
from backend.rules.risk_rules import RISK_KEYWORDS
from backend.schemas import EscalationResult
from backend.tools.base import BaseTool


class EscalationTool(BaseTool):
    name = "escalation_check"
    description = "判断售后案例是否需要升级至人工支持."

    def run(self, **kwargs: Any) -> dict[str, Any]:
        case = kwargs.get("case", {}) or {}
        text = case.get("raw_text") or " ".join(str(value) for value in case.values())

        matched = [word for word in RISK_KEYWORDS if word in text]
        high_matched = [word for word in matched if RISK_KEYWORDS[word] == "high"]
        medium_matched = [word for word in matched if RISK_KEYWORDS[word] == "medium"]

        if high_matched:
            return EscalationResult(
                need_escalation=True,
                level="high",
                reason=f"{ESCALATION_REASONS['high']}命中关键词：{', '.join(high_matched)}。",
                matched_keywords=high_matched,
            ).to_dict()

        if medium_matched or case.get("complaint_intent") or case.get("refund_intent"):
            return EscalationResult(
                need_escalation=True,
                level="medium",
                reason=f"{ESCALATION_REASONS['medium']}命中关键词：{', '.join(medium_matched) or '投诉/退款意图'}。",
                matched_keywords=medium_matched,
            ).to_dict()

        if case.get("fault_code") or case.get("symptoms"):
            return EscalationResult(
                need_escalation=False,
                level="normal",
                reason=ESCALATION_REASONS["normal"],
            ).to_dict()

        return EscalationResult(
            need_escalation=True,
            level="medium",
            reason="客户信息不足，需要人工补充关键字段。",
        ).to_dict()
