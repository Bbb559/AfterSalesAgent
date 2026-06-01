from __future__ import annotations

from typing import Any


class ActionAgent:
    """生成面向客户和内部的售后输出."""

    def generate(
        self,
        case: dict[str, Any], # 用户/设备信息
        diagnosis: dict[str, Any], # 诊断结果
        warranty: dict[str, Any], # 质保判断
        escalation: dict[str, Any], # 升级信息
        ticket: dict[str, Any], # 升级判断
    ) -> dict[str, Any]:
        customer_reply = self._customer_reply(case, diagnosis) 
        internal_advice = self._internal_advice(diagnosis, warranty, escalation)

        return {
            "customer_reply": customer_reply,  # 客户回复
            "internal_advice": internal_advice, # 内部建议
            "ticket": ticket, # 工单信息
        }

    def _customer_reply(self, case: dict[str, Any], diagnosis: dict[str, Any]) -> str:
        steps = "\n".join(f"{idx}. {step}" for idx, step in enumerate(diagnosis.get("remote_steps", []), start=1))
        model = case.get("product_model") or "您的设备"
        if diagnosis.get("urgency") == "high":
            return (
                f"您好，{model} 当前描述涉及安全风险。请您先停止使用设备，断开电源，"
                "关闭进水阀，并避免继续触碰潮湿插座或设备内部部件。我们会优先转人工核实并安排后续处理。"
            )
        return (
            f"您好，关于 {model} 的问题我们先帮您做远程排查。\n"
            f"{steps}\n"
            "如果以上步骤仍无法恢复，请补充设备照片、购买凭证和门店地址，我们会继续安排处理。"
        )

    def _internal_advice(
        self,
        diagnosis: dict[str, Any], # 诊断结果
        warranty: dict[str, Any], # 质保判断
        escalation: dict[str, Any], # 升级判断
    ) -> str:
        lines = [
            f"初步判断：{diagnosis.get('summary')}",
            f"建议动作：{diagnosis.get('suggested_action')}",
            f"质保判断：{warranty.get('status')}，{warranty.get('reason')}",
            f"升级判断：{escalation.get('level')}，{escalation.get('reason')}",
        ]
        return "\n".join(lines)
