"""安全信号分类准确率评估脚本。

加载安全场景测试用例，调用 evaluate_charger_safety() 进行三分类判定，
对比 ground truth，计算：

  - 整体风险等级准确率
  - 误报率（False Positive Rate）：safe/negated/uncertain 被误判为 p0/p1
  - 漏报率（False Negative Rate）：confirmed 场景被漏判为 p3_low
  - 确认信号精确率 / 召回率

不依赖 LLM（evaluate_charger_safety 是确定性规则函数）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def load_cases(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_all() -> tuple[int, int]:
    cases_path = Path(__file__).with_name("safety_eval_cases.json")
    cases = load_cases(cases_path)
    print(f"\n加载 {len(cases)} 条安全分类评估用例: {cases_path}")

    from backend.rules.safety_rules import evaluate_charger_safety

    # ── 分类统计 ──
    categories = {"confirmed": 0, "negated": 0, "uncertain": 0, "safe": 0}
    for c in cases:
        cat = c.get("category", "safe")
        categories[cat] = categories.get(cat, 0) + 1
    print(f"  用例分布: confirmed={categories['confirmed']}, "
          f"negated={categories['negated']}, uncertain={categories['uncertain']}, "
          f"safe={categories['safe']}")

    # ── 逐条评估 ──
    tp = 0   # should be risk, actual != p3_low
    tn = 0   # should be safe, actual == p3_low
    fp = 0   # should be safe, actual != p3_low  (误报)
    fn = 0   # should be risk, actual == p3_low  (漏报)

    risk_correct = 0    # risk_level 完全匹配
    total = len(cases)

    # 逐信号级别的统计
    signal_tp = 0   # 预期 confirmed 且实际 confirmed
    signal_fp = 0   # 预期不应 confirmed 但实际 confirmed
    signal_fn = 0   # 预期应 confirmed 但实际未 confirmed

    print(f"\n{'=' * 80}")
    print(f"{'ID':<20} {'类别':<12} {'预期风险':<12} {'实际风险':<12} {'结果':<8} {'详情'}")
    print(f"{'=' * 80}")

    for case in cases:
        qid = case["id"]
        category = case["category"]
        raw_text = case.get("raw_text", "")
        expected = case.get("expected", {})
        expected_risk = expected.get("risk_level", "p3_low")
        description = expected.get("description", "")

        # 调用 evaluate_charger_safety
        result = evaluate_charger_safety({}, raw_text)
        actual_risk = result.get("risk_level", "unknown")
        actual_confirmed = result.get("matched_safety_signals", [])
        actual_negated = result.get("negated_safety_signals", [])
        actual_uncertain = result.get("uncertain_safety_mentions", [])

        # ── 判定 ──
        should_be_risk = (category == "confirmed")
        is_risk = (actual_risk in ("p0_emergency", "p1_high"))

        if should_be_risk and is_risk:
            tp += 1
        elif should_be_risk and not is_risk:
            fn += 1
        elif not should_be_risk and not is_risk:
            tn += 1
        else:  # not should_be_risk and is_risk
            fp += 1

        risk_match = (actual_risk == expected_risk)
        if risk_match:
            risk_correct += 1

        # ── 逐信号级别: 预期 confirmed 的命中/漏报 ──
        expected_confirmed = expected.get("confirmed_contains", [])
        for sig in expected_confirmed:
            if sig in actual_confirmed:
                signal_tp += 1
            else:
                signal_fn += 1

        # ── 逐信号级别: 不应 confirmed 但实际 confirmed ──
        if category != "confirmed":
            expected_confirmed_set = set(expected_confirmed)
            for sig in actual_confirmed:
                if sig not in expected_confirmed_set:
                    signal_fp += 1

        # ── 打印行 ──
        status = "OK" if risk_match else "FAIL"
        detail = ""
        if not risk_match:
            detail = f"预期={expected_risk} 实际={actual_risk}"
            if actual_confirmed:
                detail += f" | confirmed={actual_confirmed}"
            if actual_negated:
                detail += f" | negated={actual_negated}"
            if actual_uncertain:
                detail += f" | uncertain={actual_uncertain}"
            detail += f" | {description}"

        print(f"{qid:<20} {category:<12} {expected_risk:<12} {actual_risk:<12} {status:<8} {detail}")

    # ── 计算指标 ──
    accuracy = risk_correct / total if total > 0 else 0
    fp_rate = fp / (tn + fp) if (tn + fp) > 0 else 0   # 在应安全的样本中的误报率
    fn_rate = fn / (tp + fn) if (tp + fn) > 0 else 0   # 在应有风险的样本中的漏报率
    precision = signal_tp / (signal_tp + signal_fp) if (signal_tp + signal_fp) > 0 else 0
    recall = signal_tp / (signal_tp + signal_fn) if (signal_tp + signal_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # ── 汇总报告 ──
    print(f"{'=' * 80}")
    print()
    print("=" * 60)
    print("  安全信号分类准确率评估报告")
    print("=" * 60)
    print(f"  测试用例数:              {total}")
    print(f"    - confirmed (应有风险): {categories['confirmed']}")
    print(f"    - negated   (否定风险): {categories['negated']}")
    print(f"    - uncertain (不确定):   {categories['uncertain']}")
    print(f"    - safe      (无风险):   {categories['safe']}")
    print()
    print(f"  ── 分类混淆矩阵 ──")
    print(f"                    预测有风险  预测安全")
    print(f"  实际有风险 (confirmed)    {tp:<5}       {fn:<5}")
    print(f"  实际安全 (neg+unc+safe)   {fp:<5}       {tn:<5}")
    print()
    print(f"  ── 核心指标 ──")
    print(f"  整体准确率 (Accuracy):         {accuracy:.1%}  ({risk_correct}/{total})")
    print(f"  漏报率    (False Negative):    {fn_rate:.1%}  ({fn}/{tp + fn} — confirmed被漏判为安全)")
    print(f"  误报率    (False Positive):    {fp_rate:.1%}  ({fp}/{tn + fp} — safe被误判为有风险)")
    print(f"  确认信号精确率 (Precision):     {precision:.1%}  ({signal_tp}/{signal_tp + signal_fp})")
    print(f"  确认信号召回率 (Recall):        {recall:.1%}  ({signal_tp}/{signal_tp + signal_fn})")
    print(f"  确认信号 F1:                    {f1:.2f}")
    print()
    print(f"  说明:")
    print(f"  - 漏报 = confirmed 场景被判定为 p3_low（用户面临安全风险但系统未告警）")
    print(f"  - 误报 = safe/negated/uncertain 场景被判定为 p0/p1（不该告警时告警）")
    print(f"  - confirmed 信号来自 evaluate_charger_safety 的 matched_safety_signals 字段")
    print(f"  - 纯确定性规则评估，不依赖 LLM")
    print("=" * 60)

    # ── 逐类别准确率 ──
    print()
    print("  ── 逐类别风险等级准确率 ──")
    for cat in ["confirmed", "negated", "uncertain", "safe"]:
        cat_cases = [c for c in cases if c["category"] == cat]
        cat_total = len(cat_cases)
        cat_correct = 0
        for c in cat_cases:
            r = evaluate_charger_safety({}, c.get("raw_text", ""))
            if r.get("risk_level") == c["expected"]["risk_level"]:
                cat_correct += 1
        pct = cat_correct / cat_total if cat_total > 0 else 0
        print(f"    {cat:<14} {cat_correct}/{cat_total} = {pct:.1%}")

    print()

    # ── 兼容返回 ──
    # passed = 所有应安全的都安全 + 所有应有风险的都有风险
    passed = tp + tn
    failed = fp + fn
    return passed, failed


def main() -> int:
    total_passed = 0
    total_failed = 0

    p, f = evaluate_all()
    total_passed += p
    total_failed += f

    print("=" * 60)
    print(f"总计安全评估: {total_passed} 正确, {total_failed} 错误, {total_passed + total_failed} 条")
    print("=" * 60)

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
