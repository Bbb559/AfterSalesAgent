from __future__ import annotations

from typing import Any

from backend.rules.risk_rules import STOP_USE_KEYWORDS
from backend.schemas import AuditResult


class AuditAgent:
    """检查生成的答案是否需要警告或人工审核."""

    def audit(
        self,
        case: dict[str, Any],
        diagnosis: dict[str, Any],
        retrieval: dict[str, Any],
        action: dict[str, Any],
    ) -> dict[str, Any]:
        warnings = []
        missing = case.get("missing_info", [])
        if missing:
            warnings.append(f"缺少关键信息：{'、'.join(missing)}。")

        if not retrieval.get("results"):
            warnings.append("没有检索到知识库依据，回答只能作为流程建议。")

        if diagnosis.get("urgency") == "high":
            warnings.append("高风险问题需要人工介入，不建议只给自动回复。")

        if "质保" in action.get("internal_advice", "") and not case.get("purchase_time"):
            warnings.append("质保判断缺少购买时间或凭证，不能直接承诺免费。")

        reply = action.get("customer_reply", "")
        if any(word in reply for word in ["肯定免费", "一定免费", "保证免费", "绝对免费"]):
            warnings.append("回复存在过度承诺免费维修风险。")

        raw_text = case.get("raw_text", "")
        if any(word in raw_text for word in STOP_USE_KEYWORDS) and not any(
            word in reply for word in ["停止使用", "断开电源", "关闭进水阀"]
        ):
            warnings.append("危险场景下必须提示停止使用、断电或关闭进水阀。")

        risk_level = "high" if diagnosis.get("urgency") == "high" else "normal"
        return AuditResult(
            passed=not warnings,
            warnings=warnings,
            final_note="可直接回复客户。" if not warnings else "建议人工确认后再回复客户。",
            risk_level=risk_level,
        ).to_dict()
