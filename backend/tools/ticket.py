from __future__ import annotations

from typing import Any

from backend.schemas import TicketDraft
from backend.tools.base import BaseTool


class TicketTool(BaseTool):
    name = "ticket_draft"
    description = "Create a structured after-sales ticket draft."

    def run(self, **kwargs: Any) -> dict[str, Any]:
        case = kwargs.get("case", {}) or {}
        diagnosis = kwargs.get("diagnosis", {}) or {}
        warranty = kwargs.get("warranty", {}) or {}
        escalation = kwargs.get("escalation", {}) or {}

        need_onsite = escalation.get("need_escalation") or diagnosis.get("urgency") == "high"
        return TicketDraft(
            customer_problem=case.get("raw_text", ""),
            product_model=case.get("product_model") or "待补充",
            fault_code=case.get("fault_code") or "无/待确认",
            symptoms=case.get("symptoms", []),
            purchase_time=case.get("purchase_time") or "待补充",
            city=case.get("city") or "待补充",
            phone=case.get("phone") or "待补充",
            address=case.get("address") or "待补充",
            initial_diagnosis=diagnosis.get("summary", ""),
            suggested_action=diagnosis.get("suggested_action", "先远程排查，必要时转人工或派单。"),
            warranty_result=warranty.get("status", "unknown"),
            need_onsite_service=bool(need_onsite),
            priority=diagnosis.get("priority", diagnosis.get("urgency", "normal")),
            missing_info=case.get("missing_info", []),
            internal_note="高风险问题需优先人工介入。" if need_onsite else "可先远程排查，必要时转人工或派单。",
        ).to_dict()
