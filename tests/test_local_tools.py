from __future__ import annotations

import unittest

from backend.rules import dispatch_rules, safety_rules
from backend.tools.warranty import WarrantyTool


class LocalRulesAndWarrantyToolTest(unittest.TestCase):
    def test_warranty_tool_calculates_structured_purchase_or_install_time(self) -> None:
        tool = WarrantyTool()
        retrieval = {"results": [{"text": "家用充电桩保修期为24个月。", "file_name": "policy.pdf", "page": 1}]}

        in_warranty = tool.execute(purchase_or_install_time="6个月", retrieval=retrieval)
        out_of_warranty = tool.execute(purchase_or_install_time="30 months", retrieval=retrieval)
        unstructured = tool.execute(purchase_or_install_time="半年", retrieval=retrieval)

        self.assertTrue(in_warranty.success)
        self.assertEqual(in_warranty.data["status"], "possibly_in_warranty")
        self.assertEqual(in_warranty.data["policy_months"], 24)
        self.assertNotIn("不能直接承诺免费", in_warranty.data["reason"])
        self.assertTrue(out_of_warranty.success)
        self.assertEqual(out_of_warranty.data["status"], "possibly_out_of_warranty")
        self.assertTrue(unstructured.success)
        self.assertEqual(unstructured.data["status"], "unknown")

    def test_warranty_tool_calculates_from_purchase_or_install_date(self) -> None:
        tool = WarrantyTool()

        result = tool.execute(
            purchase_or_install_time="2025-01-15",
            current_date="2026-06-03",
            retrieval={"results": [{"text": "本充电桩的保修期为24个月，以安装或购买凭证日期起算。", "file_name": "policy.pdf", "page": 3}]},
        )

        self.assertTrue(result.success)
        self.assertEqual(result.data["status"], "possibly_in_warranty")
        self.assertEqual(result.data["policy_months"], 24)
        self.assertEqual(result.data["policy_sources"], ["policy.pdf 第3页"])

    def test_warranty_tool_requires_knowledge_base_policy_period(self) -> None:
        tool = WarrantyTool()

        result = tool.execute(purchase_or_install_time="6个月", retrieval={"results": []})

        self.assertTrue(result.success)
        self.assertEqual(result.data["status"], "unknown")
        self.assertIsNone(result.data["policy_months"])
        self.assertNotIn("不能直接承诺免费", result.data["reason"])

    def test_safety_rules_classify_charger_risk_without_tool_wrapper(self) -> None:
        emergency = safety_rules.evaluate_charger_safety({"raw_text": "充电桩配电箱冒烟，家里有人被电到"})
        cable_damage = safety_rules.evaluate_charger_safety({"raw_text": "充电桩枪线破皮，还有烧焦味"})
        fault_code = safety_rules.evaluate_charger_safety({"fault_codes": ["C-RCD-04"], "raw_text": "屏幕显示 C-RCD-04"})
        low_risk = safety_rules.evaluate_charger_safety({"raw_text": "VG-CloudMini APP 离线，预约失败"})

        self.assertEqual(emergency["risk_level"], "p0_emergency")
        self.assertIn("冒烟", emergency["matched_safety_signals"])
        self.assertEqual(cable_damage["risk_level"], "p1_high")
        self.assertIn("枪线破皮", cable_damage["matched_safety_signals"])
        self.assertEqual(fault_code["risk_level"], "p1_high")
        self.assertIn("C-RCD-04", fault_code["matched_safety_signals"])
        self.assertEqual(low_risk["risk_level"], "p3_low")

    def test_dispatch_rules_build_charger_ticket_draft(self) -> None:
        case = {
            "raw_text": "VoltGate VG-11KW-Pro 无法启动充电，屏幕显示 C-RCD-04，安装 6个月",
            "brand": "VoltGate",
            "charger_model": "VG-11KW-Pro",
            "issue_description": "无法启动充电",
            "fault_codes": ["C-RCD-04"],
            "observed_symptoms": ["无法启动充电"],
            "purchase_or_install_time": "6个月",
            "environment_factors": ["地下车库"],
            "missing_info": ["联系电话", "安装地址"],
        }
        diagnosis = {
            "summary": "需按充电桩手册核对漏保自检失败相关证据。",
            "suggested_next_step": "按知识库核验，必要时创建派工。",
            "priority": "p1_high",
            "evidence_status": "grounded",
        }
        safety = safety_rules.evaluate_charger_safety(case)
        warranty = {"status": "possibly_in_warranty"}

        dispatch = dispatch_rules.build_dispatch(case, diagnosis, warranty, safety)

        self.assertTrue(dispatch["title"])
        self.assertEqual(dispatch["charger_model"], "VG-11KW-Pro")
        self.assertTrue(dispatch["need_onsite"])
        self.assertTrue(dispatch["need_electrician"])
        self.assertIn("设备铭牌", " ".join(dispatch["evidence_needed"]))
        self.assertIn("保修状态：possibly_in_warranty", dispatch["internal_note"])
        self.assertNotIn("model_or_sku", dispatch)

    def test_dispatch_rules_handle_emergency_and_insufficient_evidence(self) -> None:
        emergency_safety = safety_rules.evaluate_charger_safety({"raw_text": "充电桩配电箱冒烟，家里有人被电到"})
        emergency = dispatch_rules.build_dispatch(
            {"raw_text": "充电桩配电箱冒烟，家里有人被电到", "safety_signals": ["配电箱冒烟", "触电"]},
            {"summary": "存在紧急安全风险。", "priority": "p0_emergency", "evidence_status": "grounded"},
            {"status": "unknown"},
            emergency_safety,
        )
        insufficient = dispatch_rules.build_dispatch(
            {"raw_text": "APP 离线", "missing_info": ["联系电话", "安装地址"]},
            {"summary": "依据不足。", "priority": "normal", "evidence_status": "insufficient"},
            {"status": "unknown"},
            safety_rules.evaluate_charger_safety({"raw_text": "APP 离线"}),
        )

        self.assertTrue(emergency["need_onsite"])
        self.assertTrue(emergency["need_electrician"])
        self.assertIn("应急救援", emergency["suggested_dispatch"])
        self.assertFalse(insufficient["need_onsite"])
        self.assertIn("知识库依据不足", insufficient["suggested_dispatch"])
        self.assertIn("联系电话", " ".join(insufficient["evidence_needed"]))


if __name__ == "__main__":
    unittest.main()
