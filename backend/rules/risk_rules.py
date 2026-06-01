from __future__ import annotations

"""风险规则定义"""

RISK_KEYWORDS = {
    "漏电": "high",
    "插座打湿": "high",
    "冒烟": "high",
    "烧焦味": "high",
    "严重漏水": "high",
    "跳闸": "high",
    "反复故障": "medium",
    "多次维修": "medium",
    "退款": "medium",
    "投诉": "medium",
    "无法营业": "medium",
}


STOP_USE_KEYWORDS = ["漏电", "插座打湿", "冒烟", "烧焦味", "严重漏水", "跳闸"]
