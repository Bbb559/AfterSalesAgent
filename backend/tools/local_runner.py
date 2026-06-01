from __future__ import annotations

from typing import Any

from backend.tools.escalation import EscalationTool
from backend.tools.ticket import TicketTool
from backend.tools.warranty import WarrantyTool


class LocalToolRunner:
    """统一封装本地售后工具调用，便于后续替换为 MCP 或外部服务。"""

    def __init__(self) -> None:
        self.warranty_tool = WarrantyTool()
        self.escalation_tool = EscalationTool()
        self.ticket_tool = TicketTool()

    def run(self, case: dict[str, Any], diagnosis: dict[str, Any]) -> dict[str, Any]:
        warranty_result = self.warranty_tool.execute(
            purchase_time=case.get("purchase_time"),
            raw_text=case.get("raw_text", ""),
        )
        escalation_result = self.escalation_tool.execute(case=case)
        ticket_result = self.ticket_tool.execute(
            case=case,
            diagnosis=diagnosis,
            warranty=warranty_result.data,
            escalation=escalation_result.data,
        )

        results = [warranty_result, escalation_result, ticket_result]

        return {
            "warranty": warranty_result.data,
            "escalation": escalation_result.data,
            "ticket": ticket_result.data,
            "tool_results": {
                item.tool_name: {
                    "success": item.success,
                    "data": item.data,
                    "error": item.error,
                    "tool_name": item.tool_name,
                    "execution_time": item.execution_time,
                }
                for item in results
            },
            "tool_history": [
                {
                    "call_type": "local_python",
                    "tool_name": item.tool_name,
                    "input": _tool_input(item.tool_name, case, diagnosis, warranty_result.data, escalation_result.data),
                    "output": item.data,
                    "status": "success" if item.success else "failed",
                    "error": item.error,
                    "latency_ms": int(item.execution_time * 1000),
                }
                for item in results
            ],
            "errors": [
                item.error
                for item in results
                if not item.success and item.error
            ],
        }


def run_after_sales_tools_sync(
    case: dict[str, Any],
    diagnosis: dict[str, Any],
) -> dict[str, Any]:
    return LocalToolRunner().run(case, diagnosis)


def _tool_input(
    tool_name: str,
    case: dict[str, Any],
    diagnosis: dict[str, Any],
    warranty: dict[str, Any],
    escalation: dict[str, Any],
) -> dict[str, Any]:
    if tool_name == "warranty_check":
        return {"case": case}
    if tool_name == "escalation_check":
        return {"case": case}
    if tool_name == "ticket_draft":
        return {
            "case": case,
            "diagnosis": diagnosis,
            "warranty": warranty,
            "escalation": escalation,
        }
    return {}
