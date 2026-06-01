from __future__ import annotations

"""升级规则配置文件"""

ESCALATION_REASONS = {
    "high": "命中安全风险，需要立即升级人工处理。",
    "medium": "存在投诉、退款或反复故障风险，建议人工跟进。",
    "normal": "可先按知识库流程远程排查。",
}
