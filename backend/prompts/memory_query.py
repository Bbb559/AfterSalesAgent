from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


MEMORY_QUERY_PARSE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是新能源家用充电桩售后 Agent 的记忆查询解析器。"
        "你的唯一职责是：判断用户问题是否在询问当前会话中已记录的历史信息，并解析查询意图。"
        "只输出 JSON，不要输出 Markdown，不要输出解释。",
    ),
    (
        "human",
        """用户问题：
{user_input}

你需要判断用户是否在询问当前会话中已经记录过的历史信息（如型号、城市、风险等级、缺失信息等）。

如果是在询问历史记忆信息，设置 is_memory_query 为 true，并选择对应的 target_fields。
如果不是在询问历史记忆（如新的故障报告、新的保修咨询、新的使用问题等），设置 is_memory_query 为 false，target_fields 为空数组。

target_fields 只能是以下字段（按类别分组）。LLM 不能发明此列表之外的字段：

设备信息（来源：ChargerCase）：
brand, charger_model, charger_series, rated_power_kw,
charger_type, connector_type, serial_number

现场信息（来源：ChargerCase）：
city, contact_address, installation_type,
purchase_or_install_time

故障信息（来源：ChargerCase）：
fault_codes, observed_symptoms, safety_signals,
environment_factors, trip_status, indicator_status

安全与诊断（来源：SafetyResult / ChargerDiagnosisResult）：
risk_level, need_onsite, need_electrician,
diagnosis_summary, suggested_next_step

工单信息（来源：DispatchDraft / ChargerCase）：
ticket_id, ticket_title, ticket_priority, missing_info

对话与回复（来源：SessionMemory）：
last_customer_reply, last_user_message, customer_request

query_scope 取值（必选其一）：
- recent：用户问的是本轮对话中刚刚提到的信息（默认，绝大多数情况选此项）
- session：用户问的是当前会话中较早的信息
- cross_session：用户明确问的是更早的历史记录或跨会话信息

answer_style 取值（必选其一）：
- precise：用户期望精确的字段值回答（如"型号是什么"）
- summary：用户期望汇总性的回答（如"还缺哪些信息""记住了哪些"）

entities：用户问题中提到的实体关键词，用于后续 FTS5 搜索。如果没有明确实体，用空数组。

示例1：
用户："刚才我说的是什么品牌和功率？"
输出：{{"is_memory_query": true, "target_fields": ["brand", "rated_power_kw"], "query_scope": "recent", "entities": [], "answer_style": "precise"}}

示例2：
用户："刚才那个型号是什么？"
输出：{{"is_memory_query": true, "target_fields": ["charger_model"], "query_scope": "recent", "entities": [], "answer_style": "precise"}}

示例3：
用户："现在还缺哪些信息？"
输出：{{"is_memory_query": true, "target_fields": ["missing_info"], "query_scope": "recent", "entities": [], "answer_style": "summary"}}

示例4：
用户："充电桩无法启动，屏幕显示 C-RCD-04，漏保频繁跳闸"
输出：{{"is_memory_query": false, "target_fields": [], "query_scope": "recent", "entities": ["C-RCD-04"], "answer_style": "precise"}}

示例5：
用户："你好"
输出：{{"is_memory_query": false, "target_fields": [], "query_scope": "recent", "entities": [], "answer_style": "precise"}}

输出 JSON：""",
    ),
])


# ---------------------------------------------------------------------------
# memory_answer v2：FTS5 候选片段 -> 字段值抽取 Prompt
# ---------------------------------------------------------------------------

FTS5_FIELD_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是新能源家用充电桩售后 Agent 的记忆字段抽取器。"
        "你的唯一职责是：从候选对话片段中提取指定的字段值。"
        "只输出 JSON，不要输出 Markdown，不要输出解释。"
        "如果某个字段无法从候选片段中可靠提取，不要猜测，将其放在 missing_fields 中。"
        "对于每个提取到的字段，要标注其来源于候选片段的第几条消息（source_index，从 0 开始编号），"
        "放入 extracted_sources 中。",
    ),
    (
        "human",
        """需要提取的字段：
{target_fields}

候选对话片段（来自当前会话的历史消息）：
{candidate_evidence}

对每个 target_field，判断候选片段中是否包含其值：
- 如果能提取到值，放入 extracted_values（字段名 -> 提取的值），并在 extracted_sources 中标记来源于候选片段的第几条消息（source_index，从 0 开始）
- 如果无法提取，放入 missing_fields（字段名列表）

示例1：
target_fields: ["brand", "rated_power_kw"]
candidate_evidence:
  [0][user] 我家华为 7kW 家充桩不能充电，屏幕不亮
  [1][assistant] 好的，已记录。请问漏保是否跳闸？
输出：{{"extracted_values": {{"brand": "华为", "rated_power_kw": "7kW"}}, "extracted_sources": {{"brand": 0, "rated_power_kw": 0}}, "missing_fields": []}}

示例2：
target_fields: ["city", "serial_number"]
candidate_evidence:
  [0][user] 我在杭州，充电桩在车库
输出：{{"extracted_values": {{"city": "杭州"}}, "extracted_sources": {{"city": 0}}, "missing_fields": ["serial_number"]}}

输出 JSON：""",
    ),
])


# ---------------------------------------------------------------------------
# memory_answer v2：Answer LLM — 基于 resolver 输出生成自然语言回答
# ---------------------------------------------------------------------------

_MEMORY_ANSWER_FIELD_LABEL: dict[str, str] = {
    "brand": "品牌", "charger_model": "型号", "charger_series": "系列",
    "rated_power_kw": "额定功率", "charger_type": "充电桩类型",
    "connector_type": "连接器类型", "serial_number": "序列号",
    "city": "城市", "contact_address": "联系地址",
    "installation_type": "安装类型", "purchase_or_install_time": "购买/安装时间",
    "fault_codes": "故障码", "observed_symptoms": "观察到的问题",
    "safety_signals": "安全信号", "environment_factors": "环境因素",
    "trip_status": "跳闸状态", "indicator_status": "指示灯状态",
    "risk_level": "风险等级", "need_onsite": "是否需要现场",
    "need_electrician": "是否需要电工",
    "diagnosis_summary": "诊断小结", "suggested_next_step": "建议下一步",
    "ticket_id": "工单ID", "ticket_title": "工单标题",
    "ticket_priority": "工单优先级", "missing_info": "缺失信息",
    "last_customer_reply": "上一次回复", "last_user_message": "上一次用户问题",
    "customer_request": "客户诉求",
}

MEMORY_ANSWER_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是新能源家用充电桩售后 Agent 的记忆回答生成器。"
        "你的唯一职责是：根据已解析出的会话记忆字段值，生成面向客户的自然语言回答。"
        "只输出回答文本，不要输出 JSON，不要输出 Markdown，不要输出技术标记。",
    ),
    (
        "human",
        """用户问题：
{user_input}

查询风格：{answer_style}（precise=精确回答每个字段 / summary=汇总描述）

已找到的信息（可靠性={confidence}）：
{resolved_text}

未找到的信息：
{missing_text}

回答约束（必须遵守）：
1. 只能根据"已找到的信息"回答，不能编造任何当前会话记忆中未记录的值。
2. "未找到的信息"必须明确告知用户，使用句式如"当前会话记忆中没有找到XX信息"。
3. 如果可靠性为 medium：使用"从当前会话记录中看""根据历史消息推断""目前记录显示"等措辞，不要说得过满（不要用"已确认""确定是"）。
4. 如果可靠性为 high：可以正常表述为"之前记录的XX是YY"。
5. 不要输出任何技术标记（如 confidence、source、字段名、[confidence: high] 等），这是面向客户的对话文本。
6. 这些信息来自当前会话记忆，不是售后诊断结论。不要追问用户补充信息，用户只是在回顾已经说过的话。
7. 如果用户问题本身就是新的故障报告或保修咨询（is_memory_query=false），简短引导用户描述问题，不要展开。

回答：""",
    ),
])
