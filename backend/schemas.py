from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AfterSalesCase:
    product_model: str = ""
    fault_code: str = ""
    symptoms: list[str] = field(default_factory=list)
    purchase_time: str = ""
    city: str = ""
    phone: str = ""
    address: str = ""
    has_water_leak: bool = False
    has_power_issue: bool = False
    has_restarted: bool = False
    complaint_intent: bool = False
    refund_intent: bool = False
    missing_info: list[str] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DiagnosisResult:
    summary: str = ""
    possible_causes: list[str] = field(default_factory=list)
    remote_steps: list[str] = field(default_factory=list)
    urgency: str = "normal"
    priority: str = "normal"
    suggested_action: str = ""
    evidence_sources: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WarrantyResult:
    status: str = "unknown"
    reason: str = ""
    need_evidence: bool = True
    policy_months: int = 12

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EscalationResult:
    need_escalation: bool = False
    level: str = "normal"
    reason: str = ""
    matched_keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TicketDraft:
    customer_problem: str = ""
    product_model: str = "待补充"
    fault_code: str = "无/待确认"
    symptoms: list[str] = field(default_factory=list)
    purchase_time: str = "待补充"
    city: str = "待补充"
    phone: str = "待补充"
    address: str = "待补充"
    initial_diagnosis: str = ""
    suggested_action: str = ""
    warranty_result: str = "unknown"
    need_onsite_service: bool = False
    priority: str = "normal"
    missing_info: list[str] = field(default_factory=list)
    internal_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        title_parts = [
            data["product_model"],
            data["fault_code"] if data["fault_code"] != "无/待确认" else "",
            "、".join(data["symptoms"]) if data["symptoms"] else "售后问题",
        ]
        data["title"] = " - ".join(part for part in title_parts if part and part != "待补充")
        return data


@dataclass
class AuditResult:
    passed: bool = True
    warnings: list[str] = field(default_factory=list)
    final_note: str = "可直接回复客户。"
    risk_level: str = "normal"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TraceItem:
    node: str
    title: str
    status: str
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    duration: float | None = None
    timestamp: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolHistoryItem:
    tool_name: str
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    execution_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorkflowResult:
    intent: dict[str, Any] = field(default_factory=dict)
    case: dict[str, Any] = field(default_factory=dict)
    retrieval: dict[str, Any] = field(default_factory=dict)
    diagnosis: dict[str, Any] = field(default_factory=dict)
    warranty: dict[str, Any] = field(default_factory=dict)
    escalation: dict[str, Any] = field(default_factory=dict)
    action: dict[str, Any] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)
    tool_history: list[dict[str, Any]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
