from __future__ import annotations

import re
from typing import Any

from backend.rules.warranty_rules import DEFAULT_WARRANTY_MONTHS, WARRANTY_CAUTION
from backend.schemas import WarrantyResult
from backend.tools.base import BaseTool


class WarrantyTool(BaseTool):
    name = "warranty_check"
    description = "Estimate whether an after-sales case may be under warranty."

    def run(self, **kwargs: Any) -> dict[str, Any]:
        purchase_time = str(kwargs.get("purchase_time", "")).strip()
        text = purchase_time or str(kwargs.get("raw_text", ""))

        months = self._extract_months(text)
        if months is None:
            return WarrantyResult(
                status="unknown",
                reason=f"缺少明确购买时间，需要补充购买日期或凭证。{WARRANTY_CAUTION}",
                need_evidence=True,
            ).to_dict()

        if months <= DEFAULT_WARRANTY_MONTHS:
            return WarrantyResult(
                status="possibly_in_warranty",
                reason=f"客户描述购买约 {months} 个月，可能仍在 {DEFAULT_WARRANTY_MONTHS} 个月质保范围内。{WARRANTY_CAUTION}",
                need_evidence=True,
                policy_months=DEFAULT_WARRANTY_MONTHS,
            ).to_dict()

        return WarrantyResult(
            status="possibly_out_of_warranty",
            reason=f"客户描述购买约 {months} 个月，可能超过常规 {DEFAULT_WARRANTY_MONTHS} 个月质保期。{WARRANTY_CAUTION}",
            need_evidence=True,
            policy_months=DEFAULT_WARRANTY_MONTHS,
        ).to_dict()

    def _extract_months(self, text: str) -> int | None:
        year_match = re.search(r"(\d+(?:\.\d+)?)\s*(年|year)", text, re.I)
        if year_match:
            return int(float(year_match.group(1)) * 12)

        month_match = re.search(r"(\d+)\s*(个月|月|month)", text, re.I)
        if month_match:
            return int(month_match.group(1))

        if "半年" in text:
            return 6
        if "一年半" in text:
            return 18
        if "一年" in text:
            return 12
        if "两年" in text or "二年" in text:
            return 24
        return None
