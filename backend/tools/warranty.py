from __future__ import annotations

from datetime import date, datetime
import re
from typing import Any

from backend.schemas import WarrantyResult
from backend.tools.base import BaseTool


class DurationMonthsCalculator:
    def calculate(self, value: str) -> int | None:
        normalized = value.strip().lower().replace(" ", "")
        if not normalized:
            return None

        month_match = re.search(r"(\d+(?:\.\d+)?)(?:个月|月|months?|mos?|m)\b", normalized)
        if month_match:
            return max(0, int(round(float(month_match.group(1)))))

        year_match = re.search(r"(\d+(?:\.\d+)?)(?:年|years?|yrs?|y)\b", normalized)
        if year_match:
            return max(0, int(round(float(year_match.group(1)) * 12)))

        return None


class PurchaseDateMonthsCalculator:
    DATE_PATTERNS = [
        r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})",
        r"(\d{4})年(\d{1,2})月(\d{1,2})日?",
    ]
    MONTH_PATTERNS = [
        r"(\d{4})[-/.](\d{1,2})",
        r"(\d{4})年(\d{1,2})月",
    ]

    def calculate(self, value: str, today: date | None = None) -> int | None:
        normalized = value.strip()
        if not normalized:
            return None

        purchase_date = self._parse_date(normalized)
        if purchase_date is None:
            return None

        return self._months_between(purchase_date, today or date.today())

    def _parse_date(self, value: str) -> date | None:
        for pattern in self.DATE_PATTERNS:
            match = re.search(pattern, value)
            if match:
                return self._safe_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))

        for pattern in self.MONTH_PATTERNS:
            match = re.search(pattern, value)
            if match:
                return self._safe_date(int(match.group(1)), int(match.group(2)), 1)

        return None

    def _safe_date(self, year: int, month: int, day: int) -> date | None:
        try:
            return date(year, month, day)
        except ValueError:
            return None

    def _months_between(self, start: date, end: date) -> int:
        months = (end.year - start.year) * 12 + end.month - start.month
        if end.day < start.day:
            months -= 1
        return max(0, months)


class WarrantyPolicyExtractor:
    POLICY_PATTERNS = [
        re.compile(
            r"(?:保修期|质保期|保障期|服务保障期|warranty\s*period|warranty)"
            r"[^0-9一二三四五六七八九十两]{0,20}"
            r"(\d+(?:\.\d+)?)\s*(个月|月|年|months?|mos?|years?|yrs?)",
            re.I,
        ),
        re.compile(
            r"(\d+(?:\.\d+)?)\s*(个月|月|年|months?|mos?|years?|yrs?)"
            r"[^。；;\n]{0,20}"
            r"(?:保修|质保|保障|warranty)",
            re.I,
        ),
    ]

    def __init__(self) -> None:
        self.duration_calculator = DurationMonthsCalculator()

    def extract(self, retrieval: dict[str, Any] | None) -> tuple[int | None, list[str]]:
        if not isinstance(retrieval, dict):
            return None, []

        for item in retrieval.get("results", []) or []:
            text = str(item.get("text", "") or "")
            months = self._extract_from_text(text)
            if months is not None:
                return months, [self._format_source(item)]

        return None, []

    def _extract_from_text(self, text: str) -> int | None:
        for pattern in self.POLICY_PATTERNS:
            match = pattern.search(text)
            if match:
                return self.duration_calculator.calculate(f"{match.group(1)}{match.group(2)}")
        return None

    def _format_source(self, item: dict[str, Any]) -> str:
        file_name = str(item.get("file_name") or item.get("source") or "知识库结果")
        page = item.get("page")
        return f"{file_name} 第{page}页" if page not in (None, "") else file_name


class WarrantyTool(BaseTool):
    name = "warranty_check"
    description = "根据知识库保修期限和结构化购买/安装时间估算充电桩是否可能在保修期内。"

    def __init__(self) -> None:
        self.duration_calculator = DurationMonthsCalculator()
        self.date_calculator = PurchaseDateMonthsCalculator()
        self.policy_extractor = WarrantyPolicyExtractor()

    def run(self, **kwargs: Any) -> dict[str, Any]:
        purchase_or_install_time = str(kwargs.get("purchase_or_install_time") or "").strip()
        policy_months, policy_sources = self.policy_extractor.extract(kwargs.get("retrieval"))
        if policy_months is None:
            return WarrantyResult(
                status="unknown",
                reason="知识库未提供可计算的充电桩保修期限，需要补充保修政策依据。",
                need_evidence=True,
            ).to_dict()

        months = self._calculate_months(
            purchase_or_install_time,
            self._coerce_today(kwargs.get("today") or kwargs.get("current_date")),
        )
        if months is None:
            return WarrantyResult(
                status="unknown",
                reason="缺少可计算的结构化购买/安装时间，需要补充购买日期、安装日期或凭证。",
                need_evidence=True,
                policy_months=policy_months,
                policy_sources=policy_sources,
            ).to_dict()

        if months <= policy_months:
            return WarrantyResult(
                status="possibly_in_warranty",
                reason=f"客户描述购买/安装约 {months} 个月，知识库政策显示保障期限约 {policy_months} 个月，可能仍在保障范围内。",
                need_evidence=True,
                policy_months=policy_months,
                policy_sources=policy_sources,
            ).to_dict()

        return WarrantyResult(
            status="possibly_out_of_warranty",
            reason=f"客户描述购买/安装约 {months} 个月，可能超过知识库政策中的 {policy_months} 个月保障期。",
            need_evidence=True,
            policy_months=policy_months,
            policy_sources=policy_sources,
        ).to_dict()

    def _calculate_months(self, purchase_or_install_time: str, today: date | None = None) -> int | None:
        months = self.date_calculator.calculate(purchase_or_install_time, today)
        if months is not None:
            return months
        return self.duration_calculator.calculate(purchase_or_install_time)

    def _coerce_today(self, value: Any) -> date | None:
        if isinstance(value, date):
            return value
        if not value:
            return None
        try:
            return datetime.strptime(str(value), "%Y-%m-%d").date()
        except ValueError:
            return None
