from __future__ import annotations

"""定义家用充电桩安全诊断流程使用的数据结构。"""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TriageResult:
    intent: str = "unknown" 
    confidence: str = "low" 
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChargerCase:
    brand: str = ""  # 品牌
    charger_model: str = "" # 充电桩型号
    charger_series: str = "" # 充电桩系列
    serial_number: str = "" # 充电桩序列号
    charger_type: str = "" # 充电桩类型（壁挂式、立式、便携式等）
    installation_type: str = "" # 安装类型（户内、户外、车库等）
    rated_power_kw: str = "" # 额定功率（kW）
    connector_type: str = "" # 连接器类型（Type1、Type2、CCS、CHAdeMO等）
    power_supply_phase: str = "" # 电源相数（单相、三相）
    breaker_or_rcd_info: str = "" # 断路器或漏电保护器信息
    grounding_status: str = "" # 接地状态
    vehicle_brand_model: str = "" # 车辆品牌型号
    issue_type: str = "" # 问题类型
    issue_description: str = "" # 问题描述
    fault_codes: list[str] = field(default_factory=list) # 故障代码
    observed_symptoms: list[str] = field(default_factory=list) # 观察到的症状
    safety_signals: list[str] = field(default_factory=list)   # 安全信号
    environment_factors: list[str] = field(default_factory=list) # 环境因素
    installation_or_recent_changes: list[str] = field(default_factory=list) # 安装或近期变更
    customer_actions: list[str] = field(default_factory=list) # 客户行动
    customer_requests: list[str] = field(default_factory=list) # 客户请求
    purchase_or_install_time: str = "" # 购买或安装时间
    warranty_or_order_evidence: str = "" # 保修或订单凭证
    city: str = "" # 城市
    contact_name: str = "" # 联系人姓名
    contact_phone: str = "" # 联系电话
    contact_address: str = "" # 联系地址
    missing_info: list[str] = field(default_factory=list) # 缺失信息
    raw_text: str = "" # 原本文

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SafetyResult:
    risk_level: str = "unknown"  # 风险等级
    need_human: bool = False  # 是否需要人工介入
    need_onsite: bool = False # 是否需要现场服务
    need_electrician: bool = False # 是否需要电工
    reason: str = "" # 评估理由
    matched_safety_signals: list[str] = field(default_factory=list) # 匹配的已确认安全信号
    forbidden_actions: list[str] = field(default_factory=list) # 禁止行动
    required_customer_actions: list[str] = field(default_factory=list) # 要求客户采取的行动
    # v2 debug 字段 — 仅用于调试/提示，不参与 risk_level 判定
    negated_safety_signals: list[str] = field(default_factory=list) # 被否定词覆盖的安全信号
    uncertain_safety_mentions: list[str] = field(default_factory=list) # 假设/询问/不确定语义中的安全词

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChargerDiagnosisResult:
    summary: str = "" # 初步诊断
    evidence_status: str = "insufficient" # 证据状态（grounded / partial / insufficient）
    likely_issue_areas: list[str] = field(default_factory=list) # 可能包含的问题领域
    fault_code_interpretation: list[str] = field(default_factory=list) # 故障代码
    safe_remote_checks: list[str] = field(default_factory=list) # 可安全远程检查的项目
    onsite_reasons: list[str] = field(default_factory=list) # 需要现场服务的理由
    priority: str = "normal" # 优先级
    suggested_next_step: str = "" # 下一步行动
    evidence_sources: list[str] = field(default_factory=list) # 证据来源
    risk_flags: list[str] = field(default_factory=list) # 风险标记

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WarrantyResult:
    status: str = "unknown" 
    reason: str = "" 
    need_evidence: bool = True 
    policy_months: int | None = None  
    policy_sources: list[str] = field(default_factory=list) 

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DispatchDraft:
    title: str = "" # 工单标题
    customer_problem: str = "" # 客户问题
    brand: str = "待补充" # 品牌
    charger_model: str = "待补充" # 充电桩型号
    serial_number: str = "待补充" # 充电桩序列号
    fault_codes: list[str] = field(default_factory=list) # 故障代码
    observed_symptoms: list[str] = field(default_factory=list) # 观察到的症状
    safety_level: str = "unknown" # 安全等级
    site_environment: list[str] = field(default_factory=list) # 环境因素
    city: str = "待补充" # 城市
    contact_name: str = "待补充" # 联系人姓名
    contact_phone: str = "待补充" # 联系电话
    contact_address: str = "待补充" # 联系地址
    evidence_needed: list[str] = field(default_factory=list) # 需要的证据
    suggested_dispatch: str = "" # 建议的派遣方案
    need_electrician: bool = False # 是否需要电工
    need_onsite: bool = False # 是否需要现场服务
    priority: str = "normal" # 优先级
    missing_info: list[str] = field(default_factory=list) # 缺失信息
    internal_note: str = "" # 内部备注

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if not data["title"]:
            title_parts = [
                data["brand"] if data["brand"] != "待补充" else "",
                data["charger_model"] if data["charger_model"] != "待补充" else "",
                data["customer_problem"] or "充电桩安全诊断工单",
            ]
            data["title"] = " - ".join(part for part in title_parts if part) or "充电桩安全诊断工单"
        return data


@dataclass
class ChargerActionResult:
    customer_reply: str = "" 
    internal_advice: str = "" 
    dispatch: dict[str, Any] = field(default_factory=dict) # 派遣信息

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChargerAuditResult:
    passed: bool = True # 是否通过审核
    warnings: list[str] = field(default_factory=list) # 警告信息
    final_note: str = "可直接回复客户。" # 最终备注
    risk_level: str = "unknown" # 风险等级

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TraceItem:
    node: str 
    title: str 
    status: str
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    duration: float | None = None 
    timestamp: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolHistoryItem:
    call_type: str = "local_python"
    tool_name: str = ""
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    status: str = ""
    error: str = ""
    latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChargerWorkflowResult:
    input_safety: dict[str, Any] = field(default_factory=dict) # 输入安全扫描
    triage: dict[str, Any] = field(default_factory=dict) # 分类判断
    case: dict[str, Any] = field(default_factory=dict) # 案例抽取
    memory_context: dict[str, Any] = field(default_factory=dict) # 分层记忆上下文摘要
    retrieval: dict[str, Any] = field(default_factory=dict) # 相关案例检索
    safety: dict[str, Any] = field(default_factory=dict) # 安全评估
    diagnosis: dict[str, Any] = field(default_factory=dict) # 诊断分析
    warranty: dict[str, Any] = field(default_factory=dict) # 保修评估
    dispatch: dict[str, Any] = field(default_factory=dict) # 派遣信息
    action: dict[str, Any] = field(default_factory=dict) # 处理动作
    audit: dict[str, Any] = field(default_factory=dict) # 审核信息
    governance: dict[str, Any] = field(default_factory=dict) # 安全治理汇总
    tool_history: list[dict[str, Any]] = field(default_factory=list) # 工具调用历史
    trace: list[dict[str, Any]] = field(default_factory=list) # 流程追踪

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# memory_answer v2：parse_memory_query 受控字段枚举
# 字段名与下游 dataclass 的原始字段名对齐（→ 表示映射关系）。
# resolver 必须按"来源 dataclass"路由读取，不能假设所有字段都在 last_case 里。
# ---------------------------------------------------------------------------

MEMORY_QUERY_TARGET_FIELDS: list[str] = [
    # === 设备信息（来源：ChargerCase）===
    "brand",                          # ChargerCase.brand
    "charger_model",                  # ChargerCase.charger_model
    "charger_series",                 # ChargerCase.charger_series
    "rated_power_kw",                 # ChargerCase.rated_power_kw
    "charger_type",                   # ChargerCase.charger_type
    "connector_type",                 # ChargerCase.connector_type
    "serial_number",                  # ChargerCase.serial_number
    # === 现场信息（来源：ChargerCase）===
    "city",                           # ChargerCase.city
    "contact_address",                # ChargerCase.contact_address
    "installation_type",              # ChargerCase.installation_type
    "purchase_or_install_time",       # ChargerCase.purchase_or_install_time
    # === 故障信息（来源：ChargerCase）===
    "fault_codes",                    # ChargerCase.fault_codes
    "observed_symptoms",              # ChargerCase.observed_symptoms
    "safety_signals",                 # ChargerCase.safety_signals
    "environment_factors",            # ChargerCase.environment_factors
    "trip_status",                    # ChargerCase 扩展字段（11.3.5 计划写入）
    "indicator_status",               # ChargerCase 扩展字段（11.3.5 计划写入）
    # === 安全与诊断（来源：SafetyResult / ChargerDiagnosisResult）===
    "risk_level",                     # SafetyResult.risk_level
    "need_onsite",                    # SafetyResult.need_onsite
    "need_electrician",               # SafetyResult.need_electrician
    "diagnosis_summary",              # → ChargerDiagnosisResult.summary
    "suggested_next_step",            # ChargerDiagnosisResult.suggested_next_step
    # === 工单信息（来源：DispatchDraft / ChargerCase）===
    "ticket_id",                      # 派生字段，DispatchDraft 序列化时生成
    "ticket_title",                   # → DispatchDraft.title
    "ticket_priority",                # → DispatchDraft.priority
    "missing_info",                   # ChargerCase.missing_info（DispatchDraft.missing_info 同源）
    # === 对话与回复（来源：SessionMemory）===
    "last_customer_reply",            # SessionMemory.recent_messages 中最近一次 assistant 回复
    "last_user_message",              # SessionMemory.recent_user_messages 中最近非 memory 用户消息
    "customer_request",               # → ChargerCase.customer_requests（取第一条非空）
]

# parse_memory_query 允许的枚举值
MEMORY_QUERY_SCOPE_VALUES: set[str] = {"recent", "session", "cross_session"}
MEMORY_ANSWER_STYLE_VALUES: set[str] = {"precise", "summary"}


@dataclass
class MemoryQueryResult:
    """parse_memory_query LLM 调用的结构化输出。"""

    is_memory_query: bool = False
    target_fields: list[str] = field(default_factory=list)
    query_scope: str = "recent"
    entities: list[str] = field(default_factory=list)
    answer_style: str = "precise"
    fallback_reason: str = ""  # 非空表示解析失败/回退（parse_failed / llm_unavailable / empty_response）

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate_fields(self) -> list[str]:
        """返回 target_fields 中不在允许列表内的非法字段名。"""
        allowed = set(MEMORY_QUERY_TARGET_FIELDS)
        return [f for f in self.target_fields if f not in allowed]

    def validate_query_scope(self) -> bool:
        """query_scope 是否在允许值内。"""
        return self.query_scope in MEMORY_QUERY_SCOPE_VALUES

    def validate_answer_style(self) -> bool:
        """answer_style 是否在允许值内。"""
        return self.answer_style in MEMORY_ANSWER_STYLE_VALUES

    def clean_fields(self) -> None:
        """丢弃 target_fields 中的非法字段，原地修改。"""
        allowed = set(MEMORY_QUERY_TARGET_FIELDS)
        self.target_fields = [f for f in self.target_fields if f in allowed]

    def normalize_scope(self) -> None:
        """query_scope 不在允许值时回退为 'recent'，原地修改。"""
        if self.query_scope not in MEMORY_QUERY_SCOPE_VALUES:
            self.query_scope = "recent"

    def normalize_answer_style(self) -> None:
        """answer_style 不在允许值时回退为 'precise'，原地修改。"""
        if self.answer_style not in MEMORY_ANSWER_STYLE_VALUES:
            self.answer_style = "precise"


# ---------------------------------------------------------------------------
# memory_answer v2：field resolver 输出
# ---------------------------------------------------------------------------


@dataclass
class MemoryFieldResolution:
    """field_resolver_v1 的单次解析结果。"""

    resolved_values: dict[str, Any] = field(default_factory=dict)  # field → value
    missing_fields: list[str] = field(default_factory=list)  # 未找到值的字段
    confidence: str = "low"  # high / medium / low
    resolver_sources: dict[str, str] = field(default_factory=dict)  # field → "last_case.brand"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# memory_answer v2：额定功率格式归一化（deterministic，本地 Python）
#
# 职责：仅把已抽取出的 rated_power_kw 值统一成 "XkW" 格式。
# 不做：从用户句子里猜功率、品牌/型号/场景判断。
# ---------------------------------------------------------------------------


def normalize_power_kw(value: Any) -> str:
    """将 rated_power_kw 统一为 "XkW" 格式。

    >>> normalize_power_kw("7")       → "7kW"
    >>> normalize_power_kw("7.5")     → "7.5kW"
    >>> normalize_power_kw("7kW")     → "7kW"
    >>> normalize_power_kw("7kw")     → "7kW"
    >>> normalize_power_kw("7 KW")    → "7kW"
    >>> normalize_power_kw("")        → ""
    >>> normalize_power_kw(None)      → ""
    """
    # deterministic: 纯字符串处理，不依赖 LLM
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""

    import re

    # 已是 "X kW" / "Xkw" 格式 → 统一为 "XkW"
    m = re.match(r"^(\d+(?:\.\d+)?)\s*[kK][wW]$", text)
    if m:
        return f"{m.group(1)}kW"

    # 纯数字 → 追加 kW
    if re.match(r"^\d+(?:\.\d+)?$", text):
        return f"{text}kW"

    # 非标准格式 → 原样返回，不猜测
    return text
