from __future__ import annotations

"""定义故障规则"""

FAULT_RULES = {
    "E01": {
        "possible_causes": ["水箱缺水", "进水异常"],
        "suggested_actions": ["检查进水阀", "确认水源是否正常", "重启设备后观察"],
        "priority": "medium",
    },
    "E02": {
        "possible_causes": ["加热模块异常", "温控保护触发"],
        "suggested_actions": ["暂停使用加热功能", "断电重启", "必要时安排上门检测"],
        "priority": "medium",
    },
    "E03": {
        "possible_causes": ["滤芯堵塞", "进水压力异常", "流量传感器异常"],
        "suggested_actions": ["检查进水阀", "重启设备", "查看滤芯寿命"],
        "priority": "medium",
    },
}
