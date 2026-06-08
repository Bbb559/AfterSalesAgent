from __future__ import annotations

import json
from typing import Any

from backend.agents.llm_utils import invoke_json
from backend.prompts.action import CHARGER_ACTION_PROMPT
from backend.schemas import ChargerActionResult


class ChargerActionAgent:
    """生成充电桩安全诊断的客户回复和内部建议。"""

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm

    def generate(
        self,
        case: dict[str, Any],
        diagnosis: dict[str, Any],
        warranty: dict[str, Any],
        safety: dict[str, Any],
        dispatch: dict[str, Any],
        retrieval: dict[str, Any] | None = None,
        triage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        llm_action = invoke_json(
            self.llm,
            CHARGER_ACTION_PROMPT,
            {
                "triage": json.dumps(triage or {}, ensure_ascii=False),
                "case": json.dumps(case, ensure_ascii=False),
                "safety": json.dumps(safety, ensure_ascii=False),
                "diagnosis": json.dumps(diagnosis, ensure_ascii=False),
                "warranty": json.dumps(warranty, ensure_ascii=False),
                "dispatch": json.dumps(dispatch, ensure_ascii=False),
                "retrieval": json.dumps(retrieval or {}, ensure_ascii=False),
            },
        )
        if llm_action.get("customer_reply"):
            return ChargerActionResult(
                customer_reply=str(llm_action.get("customer_reply", "")).strip(),
                internal_advice=str(
                    llm_action.get("internal_advice") or "LLM 已生成客户回复，内部建议需结合 workflow 工具结果复核。"
                ),
                dispatch=dispatch,
            ).to_dict()

        return ChargerActionResult(
            customer_reply="您好，当前智能回复生成暂不可用。我们已记录您的充电桩问题，会转人工结合安全分级、知识库依据、保修和派工信息核验后回复。",
            internal_advice="LLM 不可用，ActionAgent 仅返回最小默认输出；安全、保修和危险动作拦截由 workflow 的 rules 层处理。",
            dispatch=dispatch,
        ).to_dict()
