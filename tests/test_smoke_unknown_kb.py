"""未知知识场景全链路测试（需要真实 LLM + 真实知识库）。

使用当前知识库 PDF 中不存在的品牌、型号、故障码（NeoCharge / BluePile / StarDock），
验证 RAG 低命中或无命中时：

- diagnosis.evidence_status 必须 = insufficient
- 不编造具体故障原因
- 不假装引用资料
- safety 确定性匹配仍正常工作（如 "烧焦味"）

⚠️ 集成测试：默认跳过，需设置 RUN_INTEGRATION_SMOKE=1 才会执行：

    $env:RUN_INTEGRATION_SMOKE="1"
    python -m pytest tests/test_smoke_unknown_kb.py -v
"""

from __future__ import annotations

import os
import unittest
from typing import Any

import pytest
from backend.graph_workflow import ChargerDiagnosisWorkflow


# ---------------------------------------------------------------------------
# 集成测试守卫
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_SMOKE") != "1",
    reason="integration smoke tests require RUN_INTEGRATION_SMOKE=1；"
           "需要真实 LLM + 已构建的知识库",
)


# ---------------------------------------------------------------------------
# 测试类
# ---------------------------------------------------------------------------

class UnknownKnowledgeSmokeTest(unittest.TestCase):
    """U 类：未知知识场景"""

    @classmethod
    def setUpClass(cls) -> None:
        cls._workflow = ChargerDiagnosisWorkflow()

    def _run(self, user_input: str, session_id: str = "") -> dict[str, Any]:
        sid = session_id or f"smoke-uk-{self._testMethodName}"
        return self._workflow.run(user_input, session_id=sid)

    # ------------------------------------------------------------------
    # U1. NeoCharge — 完全未知品牌，期望零 RAG 命中
    # ------------------------------------------------------------------
    def test_u1_neocharge_unknown_brand_yields_insufficient(self) -> None:
        result = self._run(
            "我家用的是 NeoCharge NC-5000E 充电桩，屏幕完全黑屏不亮，"
            "指示灯也不亮，用了8个月左右，不知道怎么处理",
        )

        retrieval = result["retrieval"]
        result_count = len(retrieval.get("results", []))
        # 未知品牌，期望 RAG 命中极低或为零
        if result_count > 0:
            # 如果有命中，不应该包含 NeoCharge 相关文件名
            file_names = [r.get("file_name", "") for r in retrieval.get("results", [])]
            self._skip_if_neo_charge_in_kb(file_names)

        diagnosis = result["diagnosis"]
        evidence = diagnosis.get("evidence_status", "")
        self.assertEqual(
            evidence, "insufficient",
            f"未知品牌 NeoCharge 应 evidence_status=insufficient，实际: {evidence}。"
            f"summary={str(diagnosis.get('summary', ''))[:200]}"
        )

        # 不应假装引用资料
        action = result["action"]
        reply = action.get("customer_reply", "")
        self._assert_no_fabricated_reference(reply, "NeoCharge")
        self._assert_no_fabricated_reference(
            str(diagnosis.get("summary", "")), "NeoCharge"
        )

    # ------------------------------------------------------------------
    # U2. BluePile — 未知品牌 + 安全风险词 "烧焦味"
    # ------------------------------------------------------------------
    def test_u2_bluepile_unknown_brand_with_safety_signal(self) -> None:
        result = self._run(
            "BluePile BP-30A 充电桩用了两年，今天充电时闻到烧焦味，"
            "配电箱里面滋滋响，不敢用了",
        )

        # 🔴 safety 匹配 "烧焦味" 是确定性规则，应正确触发
        safety = result["safety"]
        self.assertEqual(
            safety.get("risk_level"), "p1_high",
            f"'烧焦味' 应触发 p1_high（确定性安全匹配），实际: {safety.get('risk_level')}"
        )
        self.assertIn(
            "烧焦味", safety.get("matched_safety_signals", []),
            f"matched_safety_signals 应包含 '烧焦味'，实际: {safety.get('matched_safety_signals')}"
        )

        # 🔴 但 diagnosis 仍应 insufficient — 知识库没有 BluePile 数据
        diagnosis = result["diagnosis"]
        self.assertEqual(
            diagnosis.get("evidence_status"), "insufficient",
            f"未知品牌 even with safety signal 仍应 insufficient，"
            f"实际: {diagnosis.get('evidence_status')}。"
            f"summary={str(diagnosis.get('summary', ''))[:200]}"
        )

        # 🔴 客服回复必须包含安全指令，但不应编造 BluePile 的故障原因
        action = result["action"]
        reply = action.get("customer_reply", "")
        self._assert_contains_safety_instruction(reply)
        self._assert_no_fabricated_reference(reply, "BluePile")

    # ------------------------------------------------------------------
    # U3. StarDock — 未知品牌 + 未知故障码 E-0521
    # ------------------------------------------------------------------
    def test_u3_stardock_unknown_fault_code_yields_p3_low(self) -> None:
        result = self._run(
            "StarDock SD-SmartCharge 报故障码 E-0521，"
            "安装在上海浦东，买了3个月",
        )

        # safety: E-0521 不在 SAFETY_FAULT_CODES 中 → p3_low
        safety = result["safety"]
        self.assertEqual(
            safety.get("risk_level"), "p3_low",
            f"未知故障码 E-0521 应为 p3_low，实际: {safety.get('risk_level')}。"
            f"matched={safety.get('matched_safety_signals')}"
        )

        # diagnosis: insufficient
        diagnosis = result["diagnosis"]
        self.assertEqual(
            diagnosis.get("evidence_status"), "insufficient",
            f"未知品牌/故障码应 insufficient，实际: {diagnosis.get('evidence_status')}"
        )

        # 不应编造 E-0521 的含义
        diag_text = str(diagnosis.get("summary", "")) + str(
            diagnosis.get("fault_code_interpretation", "")
        )
        self._assert_no_fabricated_fault_code_meaning(diag_text, "E-0521")

        # action 回复应建议转人工 / 补充资料
        action = result["action"]
        reply = action.get("customer_reply", "")
        has_fallback_guidance = any(
            phrase in reply
            for phrase in ["依据不足", "补充", "转人工", "核验", "建议"]
        )
        self.assertTrue(
            has_fallback_guidance,
            f"未知故障码时应引导补充资料或转人工，实际回复: {reply[:200]}"
        )

    # ------------------------------------------------------------------
    # 辅助断言
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_no_fabricated_reference(text: str, brand: str) -> None:
        """确保文本不假装引用关于某个品牌的资料。"""
        fabricated_markers = [
            f"根据{brand}",
            f"据{brand}资料",
            f"{brand}的保修",
            f"{brand}手册",
            f"{brand}说明",
            f"查阅{brand}",
        ]
        for marker in fabricated_markers:
            if marker in text:
                raise AssertionError(
                    f"回复疑似编造 {brand} 引用: '{marker}'。回复: {text[:200]}"
                )

    @staticmethod
    def _assert_contains_safety_instruction(reply: str) -> None:
        """确保客服回复包含安全指令。"""
        keywords = ["停止充电", "停止使用", "暂停使用", "远离", "不要"]
        has_any = any(kw in reply for kw in keywords)
        if not has_any:
            raise AssertionError(
                f"p1_high 回复应含安全指令，实际: {reply[:200]}"
            )

    @staticmethod
    def _assert_no_fabricated_fault_code_meaning(text: str, code: str) -> None:
        """确保不编造未知故障码的含义。"""
        # 如果 LLM 编造了类似 "E-0521 表示 XXX" 的内容，检测出来
        fabricated_patterns = [
            f"{code} 表示",
            f"{code} 代表",
            f"{code} 是",
            f"{code} 意为",
            f"{code} 含义",
        ]
        for pattern in fabricated_patterns:
            if pattern in text:
                raise AssertionError(
                    f"疑似编造故障码 {code} 含义: '{pattern}'。文本: {text[:300]}"
                )

    @staticmethod
    def _skip_if_neo_charge_in_kb(file_names: list[str]) -> None:
        """如果知识库中意外包含 NeoCharge，跳过测试并给出清晰原因。"""
        for name in file_names:
            if "neocharge" in name.lower() or "neo_charge" in name.lower():
                raise unittest.SkipTest(
                    f"知识库意外包含 NeoCharge 文件 ({name})，"
                    "本测试假设 NeoCharge 不在知识库中，跳过。"
                )


if __name__ == "__main__":
    unittest.main()
