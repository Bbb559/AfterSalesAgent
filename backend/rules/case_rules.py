from __future__ import annotations

"""充电桩案例字段规整、结构化兜底和缺失信息计算。"""

import json
import re
from pathlib import Path
from typing import Any

from backend.schemas import ChargerCase


# ---------------------------------------------------------------------------
# 品牌/型号识别配置加载
# ---------------------------------------------------------------------------

_BRAND_PATTERNS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "rules" / "brand_patterns.json"

def _load_brand_patterns() -> dict[str, Any]:
    """从 data/rules/brand_patterns.json 加载品牌识别配置，JSON 缺失时返回空字典。"""
    try:
        if _BRAND_PATTERNS_PATH.is_file():
            return json.loads(_BRAND_PATTERNS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    return {}

# 模块加载时缓存，避免每次调用时重复读磁盘
_BRAND_PATTERNS_CACHE: dict[str, Any] | None = None

def _get_brand_patterns() -> dict[str, Any]:
    global _BRAND_PATTERNS_CACHE
    if _BRAND_PATTERNS_CACHE is None:
        _BRAND_PATTERNS_CACHE = _load_brand_patterns()
    return _BRAND_PATTERNS_CACHE

def _build_model_regexes_from_config(config: dict[str, Any]) -> list[str]:
    """从配置中提取所有品牌的型号正则模式，失败时返回空列表。"""
    patterns: list[str] = []
    for brand_entry in config.get("brands", []):
        patterns.extend(brand_entry.get("model_patterns", []))
    return patterns

def _build_brand_names_from_config(config: dict[str, Any]) -> list[str]:
    """从配置中提取所有品牌的名称和别名，用于品牌文本检测。"""
    names: list[str] = []
    for brand_entry in config.get("brands", []):
        name = brand_entry.get("name", "")
        if name:
            names.append(name)
        names.extend(brand_entry.get("aliases", []))
    return names


TEXT_FIELDS = [
    "brand", # 充电桩品牌
    "charger_model", # 充电桩型号
    "charger_series", # 充电桩系列
    "serial_number", # 充电桩序列号
    "charger_type", # 充电桩类型（交流/直流/换电）
    "installation_type", # 安装类型（户内/户外/车位）
    "rated_power_kw", # 充电桩额定功率（kW）
    "connector_type", # 充电接口类型（国标/欧标/美标/特斯拉等）
    "power_supply_phase", # 供电相数
    "breaker_or_rcd_info", # 断路器或漏电保护器信息
    "grounding_status", # 接地状态
    "vehicle_brand_model", # 车辆品牌型号
    "issue_type", # 问题类型
    "issue_description", # 问题描述
    "purchase_or_install_time", # 购买或安装时间
    "warranty_or_order_evidence", # 保修或订单凭证
    "city", # 城市
    "contact_name", # 联系人姓名
    "contact_phone", # 联系电话
    "contact_address", # 联系地址
]

LIST_FIELDS = [
    "fault_codes", # 故障码
    "observed_symptoms", # 观察到的故障现象
    "safety_signals", # 现场安全信号（如冒烟、异味等）
    "environment_factors", # 环境因素
    "installation_or_recent_changes", # 安装或近期变更
    "customer_actions", # 客户行为
    "customer_requests", # 客户请求
]


def normalize_charger_case(payload: dict[str, Any] | None, raw_text: str) -> dict[str, Any]:
    """把 LLM 输出和必要结构化兜底合并为稳定 ChargerCase。"""
    payload = payload or {}
    fallback = _structured_fallback(raw_text or "")
    normalized = ChargerCase(raw_text=raw_text or "").to_dict()

    for key in TEXT_FIELDS:
        normalized[key] = _clean_text(payload.get(key)) or fallback.get(key, "")
    for key in LIST_FIELDS:
        normalized[key] = _unique_list(_string_list(payload.get(key)) or fallback.get(key, []))

    if not normalized["issue_description"]:
        normalized["issue_description"] = raw_text.strip()
    normalized["raw_text"] = raw_text or ""
    normalized["missing_info"] = missing_info(normalized)
    return normalized


def merge_case_with_memory(case: dict[str, Any] | None, memory_context: dict[str, Any] | None) -> dict[str, Any]:
    """从同一会话的上一个案例中填充缺失的当前案例字段.

    这是会话连续性，而不是诊断证据：合并的值是先前用户提供的上下文.
    并且标记了，以便 downstream 的界面/测试可以将它们与本轮提取和RAG证据区分开来.
    """
    merged = dict(case or {})
    memory_context = memory_context or {}
    last_case = _last_case_from_memory(memory_context)
    merge_meta = {
        "applied": False,
        "source": "session.last_case",
        "filled_fields": [],
        "merged_list_fields": [],
        "used_as_diagnostic_evidence": False,
    }
    if not last_case or not _should_merge_with_last_case(merged, last_case):
        merged["missing_info"] = missing_info(merged)
        merged["_memory_merge"] = merge_meta
        return merged

    for field_name in TEXT_FIELDS:
        if field_name in {"issue_description", "issue_type", "raw_text"}:
            continue
        if _is_empty_case_value(merged.get(field_name)) and not _is_empty_case_value(last_case.get(field_name)):
            merged[field_name] = last_case.get(field_name)
            merge_meta["filled_fields"].append(field_name)

    for field_name in LIST_FIELDS:
        current_items = _string_list(merged.get(field_name))
        remembered_items = _string_list(last_case.get(field_name))
        if remembered_items and not current_items:
            merged[field_name] = remembered_items
            merge_meta["filled_fields"].append(field_name)
            merge_meta["merged_list_fields"].append(field_name)
        elif remembered_items and field_name in {"fault_codes", "safety_signals", "environment_factors"}:
            combined = _unique_list(current_items + remembered_items)
            if combined != current_items:
                merged[field_name] = combined
                merge_meta["merged_list_fields"].append(field_name)

    merged["missing_info"] = missing_info(merged)
    merge_meta["applied"] = bool(merge_meta["filled_fields"] or merge_meta["merged_list_fields"])
    merged["_memory_merge"] = merge_meta
    return merged


def missing_info(case: dict[str, Any]) -> list[str]:
    """计算充电桩诊断和派工需要补齐的字段。"""
    missing = []
    if not case.get("charger_model"):
        missing.append("充电桩型号或铭牌照片")
    if not case.get("issue_description") and not case.get("observed_symptoms") and not case.get("fault_codes"):
        missing.append("故障现象或故障码")
    if not case.get("safety_signals") and not case.get("environment_factors"):
        missing.append("现场安全环境照片")
    if not case.get("purchase_or_install_time") and not case.get("warranty_or_order_evidence"):
        missing.append("购买/安装/订单凭证")
    if not case.get("contact_phone"):
        missing.append("联系电话")
    if not case.get("contact_address"):
        missing.append("安装地址")
    return missing


def _last_case_from_memory(memory_context: dict[str, Any]) -> dict[str, Any]:
    last_case = memory_context.get("last_case")
    if isinstance(last_case, dict) and last_case:
        return last_case
    session = memory_context.get("session")
    if isinstance(session, dict):
        for key in ["last_case", "recent_case"]:
            value = session.get(key)
            if isinstance(value, dict) and value:
                return value
    return {}


def _should_merge_with_last_case(case: dict[str, Any], last_case: dict[str, Any]) -> bool:
    raw_text = str(case.get("raw_text") or case.get("issue_description") or "")
    if _has_followup_marker(raw_text):
        return True
    if _identity_overlaps(case, last_case):
        return True
    has_current_identity = any(
        not _is_empty_case_value(case.get(field_name))
        for field_name in ["charger_model", "serial_number", "fault_codes", "observed_symptoms", "safety_signals"]
    )
    return not has_current_identity and 0 < len(raw_text.strip()) <= 80


def _has_followup_marker(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    markers = [
        "刚才",
        "之前",
        "上一轮",
        "上次",
        "这个",
        "那个",
        "现在",
        "继续",
        "还",
        "够不够",
        "缺哪些",
        "还缺",
        "开工单",
        "派什么工单",
        "风险高不高",
        "高不高",
        "下一步",
        "优先派",
        "应该先",
    ]
    return any(marker in compact for marker in markers)


def _identity_overlaps(case: dict[str, Any], last_case: dict[str, Any]) -> bool:
    for field_name in ["serial_number", "charger_model", "contact_phone", "contact_address"]:
        current = str(case.get(field_name) or "").strip().lower()
        remembered = str(last_case.get(field_name) or "").strip().lower()
        if current and remembered and current == remembered:
            return True
    current_codes = set(_string_list(case.get("fault_codes")))
    remembered_codes = set(_string_list(last_case.get("fault_codes")))
    return bool(current_codes and remembered_codes and current_codes.intersection(remembered_codes))


def _is_empty_case_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, list):
        return not _string_list(value)
    cleaned = str(value).strip()
    return cleaned in {"", "无", "未知", "未提供", "待补充", "空", "null", "None"}


def _structured_fallback(text: str) -> dict[str, Any]:
    fault_codes = _extract_fault_codes(text)

    # 品牌识别：优先从 data/rules/brand_patterns.json 加载，缺失时使用 VoltGate 硬编码兜底
    brand = ""
    cfg = _get_brand_patterns()
    brand_names = _build_brand_names_from_config(cfg) if cfg else []
    if not brand_names:
        # JSON 缺失或为空时的硬编码 fallback
        brand = "VoltGate" if "VoltGate" in text else ""
    else:
        for name in brand_names:
            if name in text:
                brand = name
                break

    return ChargerCase(
        brand=brand,
        charger_model=_extract_charger_model(text, fault_codes),
        serial_number=_first_match(text, r"(?:SN|S/N|序列号|桩号|设备编号)[:：\s]*([A-Za-z0-9-]{6,})"),
        fault_codes=fault_codes,
        city=_extract_city(text),
        contact_phone=_first_match(text, r"1[3-9]\d{9}"),
        issue_description=text.strip(),
        raw_text=text,
    ).to_dict()


def _extract_charger_model(text: str, fault_codes: list[str]) -> str:
    fault_set = {code.upper() for code in fault_codes}

    # 型号正则：优先从 brand_patterns.json 加载，缺失时使用 VG-* 硬编码兜底
    cfg = _get_brand_patterns()
    model_patterns = _build_model_regexes_from_config(cfg) if cfg else []
    if not model_patterns:
        # JSON 缺失或为空时的硬编码 fallback
        model_patterns = [
            r"\b(VG-[A-Z0-9]+(?:-[A-Z0-9]+)*)\b",
            r"\b(VG-[A-Za-z]+[A-Za-z0-9]*)\b",
        ]

    candidates: list[str] = []
    for pattern in model_patterns:
        candidates.extend(re.findall(pattern, text, re.I))

    seen = set()
    for candidate in candidates:
        key = candidate.upper()
        if key in seen:
            continue
        seen.add(key)
        if key not in fault_set:
            return candidate
    explicit = _extract_explicit_model(text)
    if explicit and explicit.upper() not in fault_set:
        return explicit
    return ""


def _extract_explicit_model(text: str) -> str:
    match = re.search(r"(?:型号|款型|机型|model)[:：\s]*([^，。,；;！？!\?\n]+)", text, re.I)
    if not match:
        return ""
    value = match.group(1).strip()
    value = re.sub(r"\s+", " ", value)
    # 品牌名来自 JSON 配置或硬编码兜底
    cfg = _get_brand_patterns()
    brand_names = _build_brand_names_from_config(cfg) if cfg else []
    if not brand_names:
        brand_names = ["VoltGate"]
    brand_joined = "|".join(re.escape(n) for n in brand_names)
    if not re.search(rf"\d|kw|kW|KW|{brand_joined}|VG-", value):
        return ""
    return value[:48].strip()


def _extract_city(text: str) -> str:
    patterns = [
        r"(?:我在|位于)([\u4e00-\u9fa5]{2,8})(?:市)?(?=[，。,；;！!\?？\s]|$)",
        r"(?:城市|所在城市)[:：\s]*([\u4e00-\u9fa5]{2,8})(?:市)?(?=[，。,；;！!\?？\s]|$)",
        r"(?<!现)在([\u4e00-\u9fa5]{2,8})(?:市)?(?=[，。,；;！!\?？\s]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        city = match.group(1).strip()
        if city:
            return city[:-1] if city.endswith("市") and len(city) > 2 else city
    return ""


def _extract_fault_codes(text: str) -> list[str]:
    context_codes = re.findall(
        r"(?:故障码|错误码|报错|显示|屏幕显示|App显示|APP显示)[:：\s]*([A-Z]{1,3}(?:-[A-Z0-9]{2,8}){1,3})",
        text,
        re.I,
    )
    charger_codes = re.findall(r"\b(C-[A-Z]{2,8}-\d{2,4})\b", text, re.I)
    return _unique_list(context_codes + charger_codes)


def _first_match(text: str, pattern: str) -> str:
    match = re.search(pattern, text, re.I)
    if not match:
        return ""
    return (match.group(1) if match.groups() else match.group(0)).strip()


def _clean_text(value: Any) -> str:
    cleaned = str(value or "").strip()
    if cleaned in {"无", "未知", "未提供", "待补充", "空", "null", "None"}:
        return ""
    return cleaned


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _unique_list(items: list[Any]) -> list[str]:
    result = []
    seen = set()
    for item in items:
        raw = str(item or "").strip()
        value = raw.replace(" ", "-").upper() if re.fullmatch(r"[A-Za-z0-9 _-]+", raw) else raw
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result
