from __future__ import annotations


class IntentAgent:
    """将售后请求分类为少数几种业务意图."""

    INTENTS = {
        "diagnosis": ["故障", "报错", "故障码", "显示", "不出水", "出水慢", "漏水", "异常", "维修"], # 诊断相关的关键词
        "warranty_consultation": ["质保", "保修", "在保", "过保", "收费", "免费", "费用"], # 质保咨询相关的关键词
        "ticket_creation": ["工单", "派单", "上门", "维修单", "安排师傅"], # 工单创建相关的关键词
        "knowledge_lookup": ["是什么", "怎么用", "多久", "说明", "政策", "流程", "标准"], # 知识查询相关的关键词
        "human_handoff": ["人工", "主管", "投诉", "升级", "紧急"], # 人工转接相关的关键词
    }

    def determine_intent(self, text: str) -> dict:
        normalized = (text or "").strip()
        if not normalized:
            return {"name": "unknown", "confidence": "low", "matched_keywords": []}

        scores = {
            intent: sum(1 for keyword in keywords if keyword in normalized)
            for intent, keywords in self.INTENTS.items()
        }
        matched = {
            intent: [keyword for keyword in keywords if keyword in normalized]
            for intent, keywords in self.INTENTS.items()
        }

        strong_diagnosis_keywords = ["故障", "报错", "故障码", "显示", "不出水", "出水慢", "漏水", "异常"]

        if scores["human_handoff"] > 0:
            name = "human_handoff"
        elif scores["ticket_creation"] > 0 and (scores["diagnosis"] > 0 or "上门" in normalized):
            name = "ticket_creation"
        elif scores["warranty_consultation"] > 0:
            name = "warranty_consultation"
        elif any(keyword in normalized for keyword in strong_diagnosis_keywords):
            name = "diagnosis"
        elif scores["diagnosis"] > 0:
            name = "diagnosis"
        elif scores["knowledge_lookup"] > 0:
            name = "knowledge_lookup"
        else:
            name = "knowledge_lookup"

        return {
            "name": name,
            "confidence": "high" if scores.get(name, 0) > 0 else "medium",
            "matched_keywords": matched.get(name, []),
        }
