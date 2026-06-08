from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


CHARGER_AUDIT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是新能源家用充电桩售后安全回复审核 Agent。"
        "请检查安全护栏、危险动作、事实依据、保修承诺、工单完整性和话术自然度。"
        "本地硬规则 warning 不能被删除。只输出 JSON。",
    ),
    (
        "human",
        """客户案例：
{case}

安全护栏：
{safety}

诊断结果：
{diagnosis}

知识库检索：
{retrieval}

客服回复：
{action}

请输出 JSON 字段：
passed, warnings, final_note, risk_level

审核重点：
1. p0_emergency/p1_high 必须提醒停止充电或暂停使用、远离风险源、条件安全时切断相关电源、不要自行拆修，并人工或上门处理。
2. 不得出现开盖检修、带电测量、绕过漏保/接地、继续充电观察、自行更换空开/漏保、拆改配电箱等危险动作。
3. 缺少知识库依据时，不得编造故障码含义、技术参数、具体原因或保修政策。
4. 保修、换新、费用不得承诺肯定通过或肯定免费。
5. 工单缺少联系方式、地址、型号、故障现象或安全现场证据时要提示补充。

输出 JSON：""",
    ),
])
