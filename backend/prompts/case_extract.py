from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


CHARGER_CASE_EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是新能源家用充电桩售后安全诊断的信息抽取 Agent。"
        "请从客户原话中抽取充电桩、安装现场、车辆、故障、安全风险和联系信息。"
        "只输出 JSON，不要输出 Markdown。缺失字段用空字符串或空数组。字段名必须使用英文。",
    ),
    (
        "human",
        """客户问题：
{user_input}

必须输出这些字段：
brand, charger_model, charger_series, serial_number,
charger_type, installation_type, rated_power_kw, connector_type,
power_supply_phase, breaker_or_rcd_info, grounding_status, vehicle_brand_model,
issue_type, issue_description, fault_codes, observed_symptoms, safety_signals,
environment_factors, installation_or_recent_changes, customer_actions, customer_requests,
purchase_or_install_time, warranty_or_order_evidence, city, contact_name,
contact_phone, contact_address

字段说明：
- charger_model：例如 VG-7KW-AC、VG-11KW-Pro、VG-WallBox2、VG-CloudMini。
- issue_type：可用 malfunction、fault_code、charging_failed、app_offline、auth_failed、overheat、rcd_or_grounding、warranty、dispatch、usage_question、unknown。
- fault_codes：只记录客户明确提到的充电桩故障码，例如 C-GND-01、C-RCD-04、C-TEMP-09、C-COM-12、C-AUTH-20、C-LOCK-08、C-POWER-16。
- safety_signals：记录冒烟、明火、火花、烧焦味、触电、麻手、漏电、跳闸、枪线破皮、枪头发热、进水、雨水倒灌、积水、接地异常、私拉乱接、自行拆盖等安全信号。
- environment_factors：记录雨天、室外、地下车库、潮湿、积水、配电箱、充电口、安装位置等现场因素。
- customer_actions：记录客户已经做过的动作，例如重启、复位、重新插拔、换车尝试、联系物业、自行拆开。
- customer_requests：记录客户明确诉求，例如排障、上门、电工、换新、保修、费用、人工。

输出 JSON：""",
    ),
])
