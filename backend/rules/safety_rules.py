from __future__ import annotations

"""家用充电桩电气安全信号、风险分级和诊断覆盖规则。

v2 语义分类：
- confirmed_safety_signals  ：确认发生的安全信号，允许提升 risk_level。
- negated_safety_signals   ：被否定词覆盖的安全信号，仅用于 debug，不提升 risk_level。
- uncertain_safety_mentions ：假设/询问/不确定语义中的安全词，只用于提示，不直接提升到 p0/p1。
"""

import re
from typing import Any

from backend.schemas import SafetyResult


EMERGENCY_SIGNALS = [
    "明火",
    "起火",
    "着火",
    "火苗",
    "冒烟",
    "配电箱冒烟",
    "触电",
    "电到人",
    "人员受伤",
    # v2 扩展：烧焦/烧坏类
    "烧了",
    "烧坏",
    "烧糊",
]

HIGH_RISK_SIGNALS = [
    "火花",
    "打火",
    "电火花",
    "烧焦味",
    "焦糊味",
    "漏电",
    # v2 扩展：麻手类 — 覆盖 "手麻" "手发麻" "手有点麻" "发麻" "麻了一下" 等自然表达
    "麻手",
    "手麻",
    "手发麻",
    "手有点麻",
    "发麻",
    "麻了一下",
    # v2 扩展：跳闸类 — 覆盖 "漏保跳闸" "漏保跳了" "漏保跳了两次" "频繁跳闸" 等
    "漏保频繁跳闸",
    "漏保跳闸",
    "漏保跳了",
    "频繁跳闸",
    "漏电保护频繁跳闸",
    "空开跳闸",
    # v2 扩展：发热类 — 覆盖 "烫" "发烫" "很烫" "枪头很烫" "充电口很烫" 等
    "枪头发热",
    "枪头很烫",
    "车辆充电口发热",
    "充电口发热",
    "充电口很烫",
    "发烫",
    "很烫",
    "过热",
    # v2 扩展：配电异常类
    "滋滋响",
    "枪线破皮",
    "枪线破损",
    "进水",
    "雨水倒灌",
    "积水",
    "接地异常",
    "接地故障",
    "私拉乱接",
    "自行拆盖",
    "私自拆开",
]

# ---------------------------------------------------------------------------
# 否定词列表 — 出现在安全信号之前的否定表达
# 示例："没有烧焦味" "暂时没有发热" "未发现漏电" "不伴随跳闸"
# ---------------------------------------------------------------------------
NEGATION_PATTERNS = [
    "没有",
    "暂时没有",
    "暂时没",
    "暂无",
    "没有明显",
    "无明显",
    "未见明显",
    "未发现",
    "没发现",
    "没有发现",
    "不伴随",
    "不伴有",
    "不伴",
    "没出现",
    "没有出现",
    "未出现",
    "不是",
    "并非",
    "未",
    "无",
    "没",
    "不存在",
    "排除",
    "否认",
    "不会",
]

# ---------------------------------------------------------------------------
# 不确定语义标记 — 出现在安全信号附近的假设/询问/猜测表达
# 示例："是不是漏电了？" "请问这个故障码危险吗？" "好像有点发热"
# ---------------------------------------------------------------------------
UNCERTAIN_MARKERS = [
    # 显式疑问标记（紧邻信号前才触发）
    "是不是",
    "会不会",
    "有没有可能",
    "不清楚是不是",
    "不太确定是不是",
    # 假设/条件标记（紧邻信号前才触发）
    "如果",
    "假如",
    "假设",
    "万一",
    # 听说的/未证实标记
    "听说",
    "据说",
    "好像",
    "感觉像",
]

SAFETY_FAULT_CODES = {
    "C-GND-01": "接地异常",
    "C-RCD-04": "漏保自检失败",
    "C-TEMP-09": "枪头温度过高",
}

FORBIDDEN_ACTIONS = [
    "不要开盖检修或拆改充电桩外壳。",
    "不要带电测量、触碰内部端子或拆改配电箱。",
    "不要绕过漏保、接地或空开继续充电。",
    "不要触摸发热、破损、进水的枪线或枪头。",
    "不要在冒烟、异味、进水或跳闸后继续充电观察。",
]

HIGH_RISK_REQUIRED_ACTIONS = [
    "立即停止充电并暂停使用充电桩。",
    "远离充电桩、枪线、车辆充电口和配电箱等风险源。",
    "在确保自身安全的前提下，切断充电桩或上级空开电源。",
    "不要自行拆修，等待人工客服、电工或上门工程师处理。",
]

EMERGENCY_REQUIRED_ACTIONS = [
    "立即停止充电并远离现场。",
    "如存在明火、持续冒烟、触电或人员受伤，请优先联系当地应急救援。",
    "在确保自身安全的前提下，切断充电桩或上级空开电源。",
    "不要自行拆修或继续靠近设备，等待人工客服、电工或上门工程师处理。",
]


def find_charger_safety_signals(*texts: str) -> list[str]:
    """从文本中识别充电桩安全风险信号（所有原始命中，不做语义分类）。

    语义分类（confirmed / negated / uncertain）由 evaluate_charger_safety 完成。
    """
    combined_text = " ".join(text for text in texts if text)
    matched: list[str] = []

    # 文本信号：substring 匹配（从长到短避免短词优先命中后长词被跳过）
    all_text_signals = sorted(
        [*EMERGENCY_SIGNALS, *HIGH_RISK_SIGNALS],
        key=len, reverse=True,
    )
    for signal in all_text_signals:
        if signal in combined_text and signal not in matched:
            matched.append(signal)

    # 安全故障码：确定性匹配（故障码出现在屏幕上 = 已确认事实，不受否定词影响）
    for code, meaning in SAFETY_FAULT_CODES.items():
        if re.search(rf"\b{re.escape(code)}\b", combined_text, re.I):
            if code not in matched:
                matched.append(code)
            if meaning not in matched:
                matched.append(meaning)

    return _unique(matched)


def _classify_text_signals(
    raw_matches: list[str], combined_text: str
) -> tuple[list[str], list[str], list[str]]:
    """将文本信号分类为 confirmed / negated / uncertain。

    规则：
    1. SAFETY_FAULT_CODES 中的故障码/含义 → 始终 confirmed（屏幕上显示故障码是事实）。
    2. 匹配到的文本信号，如果在信号前 N 个字符内命中否定词 → negated。
    3. 否则如果信号处于疑问/假设/不确定语境 → uncertain。
    4. 否则 → confirmed。
    5. 同一信号多次出现时，confirmed 优先（安全兜底）。
    """
    fault_code_set: set[str] = set(SAFETY_FAULT_CODES.keys()) | set(SAFETY_FAULT_CODES.values())

    confirmed: list[str] = []
    negated: list[str] = []
    uncertain: list[str] = []

    for signal in raw_matches:
        # 故障码 → 始终 confirmed
        if signal in fault_code_set:
            confirmed.append(signal)
            continue

        classification = _classify_single_signal(signal, combined_text)
        if classification == "confirmed":
            confirmed.append(signal)
        elif classification == "negated":
            negated.append(signal)
        else:
            uncertain.append(signal)

    return confirmed, negated, uncertain


def _classify_single_signal(signal: str, text: str) -> str:
    """对单个文本信号的每次出现做语义分类。

    对 signal 在 text 中的所有出现位置逐一检查：
    - 只要有一次是 confirmed → 返回 "confirmed"（安全兜底）
    - 否则如果有 uncertain → 返回 "uncertain"
    - 否则 → "negated"
    """
    best: str = "negated"

    for m in re.finditer(re.escape(signal), text):
        pos = m.start()
        # 取信号前最多 20 个字符作为否定判断窗口
        before_start = max(0, pos - 20)
        before = text[before_start:pos]

        if _has_negation_before(before):
            classification = "negated"
        elif _has_uncertainty_around(text, pos, len(signal)):
            classification = "uncertain"
        else:
            classification = "confirmed"

        if classification == "confirmed":
            return "confirmed"
        if classification == "uncertain":
            best = "uncertain"

    return best


def _has_negation_before(before_context: str) -> bool:
    """检查 before_context 末尾是否以否定词结尾。

    匹配模式：
    - before_context 以否定词结尾（如 "没有" → "没有烧焦味"）
    - before_context 以 "否定词 + 的/了/过/着/到/见/呢/吗/啊/呀" 结尾
    - before_context 包含 "否定词 + 连接词" 结构（如 "没有发热、跳闸或者"）

    排除规则：否定词如果前面紧接 "是/会/有" 形成 "是不是/会不会/有没有" 等疑问形式，
    则不算否定（疑问由 _has_uncertainty_around 处理）。
    """
    before = before_context.rstrip()
    if not before:
        return False

    # 复合疑问形式映射：疑问标记 → 其中包含的否定词
    # "是不是" 包含 "不是"；"会不会" 包含 "不会"；"有没有" 包含 "没有"
    INTERROGATIVE_WRAPPERS = {
        "不是": "是不是",
        "不会": "会不会",
        "没有": "有没有",
    }

    for neg in sorted(NEGATION_PATTERNS, key=len, reverse=True):
        # 直接以否定词结尾
        if before.endswith(neg):
            if not _neg_is_wrapped_by_interrogative(before, neg, INTERROGATIVE_WRAPPERS):
                return True
        # 否定词 + 少量虚词后结尾
        m = re.search(rf"{re.escape(neg)}[\s的了吧呢吗啊呀着到见]{{0,3}}$", before)
        if m:
            if not _neg_is_wrapped_by_interrogative(before, neg, INTERROGATIVE_WRAPPERS):
                return True
        # 否定词 + 连接词（、或者 等），即否定作用域延伸
        # 如 "没有发热、跳闸或者烧焦味" → 否定词作用域覆盖枚举中所有信号
        # [^。！？；\n] 允许任意非句尾字符，确保枚举中的中间信号词不阻断否定链
        if re.search(
            rf"{re.escape(neg)}[^。！？；\n]{{0,30}}$", before
        ):
            if not _neg_is_wrapped_by_interrogative(before, neg, INTERROGATIVE_WRAPPERS):
                return True

    return False


def _neg_is_wrapped_by_interrogative(
    before: str, neg: str, wrappers: dict[str, str]
) -> bool:
    """检查否定词 neg 是否被疑问前缀包裹（如 "是不是" 包裹 "不是"）。"""
    if neg not in wrappers:
        return False
    interr = wrappers[neg]
    # 找到 neg 在 before 中的最后位置
    neg_pos = before.rfind(neg)
    if neg_pos < 0:
        return False
    interr_start = neg_pos - (len(interr) - len(neg))
    if interr_start >= 0 and before[interr_start:interr_start + len(interr)] == interr:
        return True
    return False


def _has_uncertainty_around(text: str, pos: int, length: int) -> bool:
    """检查信号是否紧邻显式不确定/疑问标记。

    保守策略：只在以下情况判为 uncertain：
    1. 信号前 10 字符内出现显式不确定标记（如 "是不是" "会不会" "如果" "好像" 等）。
    2. 信号后紧跟疑问标点且中间无逗号/分号隔开（如 "漏电？" "危险吗？"）。

    整个输入的末尾问号（如 "烧焦味，现在怎么办？"）不会误判前面的安全信号，
    因为逗号隔开了不同从句。
    """
    # ── 1. 显式不确定标记在信号前 10 字符内 ──
    before_start = max(0, pos - 10)
    before = text[before_start:pos]
    for marker in sorted(UNCERTAIN_MARKERS, key=len, reverse=True):
        if marker in before:
            marker_pos = before.rfind(marker)
            # marker 末尾到信号开头不超过 5 个字符
            gap = len(before) - marker_pos - len(marker)
            if 0 <= gap <= 5:
                return True

    # ── 2. 信号后紧跟疑问标点（？吗呢吧），且中间无逗号/分号 ──
    after_start = pos + length
    after = text[after_start:min(len(text), after_start + 6)]
    for i, ch in enumerate(after):
        if ch in ("，", "、", "；", "。", "\n"):
            break  # 从句分隔符 → 疑问不属于该信号
        if ch in ("？", "吗", "呢", "吧"):
            return True

    return False


def evaluate_charger_safety(case: dict[str, Any], raw_text: str = "") -> dict[str, Any]:
    """根据结构化案例和原始文本生成本地安全分级结果。

    v2 语义分类：
    - 只将 confirmed_safety_signals 用于风险等级判定。
    - negated_safety_signals 和 uncertain_safety_mentions 仅输出为 debug 字段，
      不提升 risk_level。
    """
    source_text = " ".join([
        raw_text,
        str(case.get("raw_text", "") or ""),
        " ".join(_string_list(case.get("safety_signals"))),
        " ".join(_string_list(case.get("fault_codes"))),
    ])
    raw_matches = find_charger_safety_signals(source_text)

    confirmed, negated, uncertain = _classify_text_signals(raw_matches, source_text)

    # ── 只使用 confirmed 信号做风险判定 ──
    if not confirmed:
        return SafetyResult(
            risk_level="p3_low",
            need_human=False,
            need_onsite=False,
            need_electrician=False,
            reason="未命中本地充电桩高风险安全信号，可继续按知识库证据进行安全远程核验。",
            matched_safety_signals=[],
            forbidden_actions=FORBIDDEN_ACTIONS,
            required_customer_actions=[],
            negated_safety_signals=negated,
            uncertain_safety_mentions=uncertain,
        ).to_dict()

    if any(signal in confirmed for signal in EMERGENCY_SIGNALS):
        return SafetyResult(
            risk_level="p0_emergency",
            need_human=True,
            need_onsite=True,
            need_electrician=True,
            reason=f"命中紧急安全信号：{'、'.join(confirmed)}。",
            matched_safety_signals=confirmed,
            forbidden_actions=FORBIDDEN_ACTIONS,
            required_customer_actions=EMERGENCY_REQUIRED_ACTIONS,
            negated_safety_signals=negated,
            uncertain_safety_mentions=uncertain,
        ).to_dict()

    return SafetyResult(
        risk_level="p1_high",
        need_human=True,
        need_onsite=True,
        need_electrician=True,
        reason=f"命中高风险充电桩安全信号：{'、'.join(confirmed)}。",
        matched_safety_signals=confirmed,
        forbidden_actions=FORBIDDEN_ACTIONS,
        required_customer_actions=HIGH_RISK_REQUIRED_ACTIONS,
        negated_safety_signals=negated,
        uncertain_safety_mentions=uncertain,
    ).to_dict()


def enforce_diagnosis(diagnosis: dict[str, Any], safety: dict[str, Any]) -> dict[str, Any]:
    """把高风险安全结论显式覆盖到诊断结果中。"""
    risk_level = safety.get("risk_level", "unknown")
    if risk_level not in {"p0_emergency", "p1_high"}:
        return diagnosis

    guarded = dict(diagnosis)
    guarded["priority"] = risk_level
    guarded["risk_flags"] = _string_list(safety.get("matched_safety_signals"))
    guarded["onsite_reasons"] = _unique([
        *_string_list(guarded.get("onsite_reasons")),
        str(safety.get("reason", "") or ""),
    ])
    guarded["safe_remote_checks"] = _unique([
        *_string_list(safety.get("required_customer_actions")),
        "请保留故障码、现场照片、视频和订单/安装凭证，等待人工或上门工程师核验。",
    ])
    guarded["suggested_next_step"] = "立即按安全护栏停止充电、远离风险源，并转人工或上门电工处理。"
    if "安全风险" not in guarded.get("summary", ""):
        guarded["summary"] = f"{guarded.get('summary', '')} 当前描述涉及充电桩安全风险。".strip()
    return guarded


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _unique(items: list[str]) -> list[str]:
    result = []
    seen = set()
    for item in items:
        value = str(item or "").strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result
