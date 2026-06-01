from __future__ import annotations

import unittest

from backend.tools.local_runner import LocalToolRunner, run_after_sales_tools_sync


class LocalToolRunnerTest(unittest.TestCase):
    def test_local_runner_generates_warranty_escalation_and_ticket(self) -> None:
        case = {
            "raw_text": "QY-320 显示 E03，出水变慢，买了半年",
            "product_model": "QY-320",
            "fault_code": "E03",
            "symptoms": ["出水变慢"],
            "purchase_time": "半年",
            "missing_info": ["联系方式", "地址"],
        }
        diagnosis = {
            "summary": "可能与滤芯堵塞、进水压力异常或流量传感器异常有关。",
            "suggested_action": "先远程排查。",
            "priority": "medium",
        }

        bundle = LocalToolRunner().run(case, diagnosis)

        self.assertEqual(bundle["warranty"]["status"], "possibly_in_warranty")
        self.assertFalse(bundle["escalation"]["need_escalation"])
        self.assertTrue(bundle["ticket"]["title"])
        self.assertEqual([item["call_type"] for item in bundle["tool_history"]], ["local_python"] * 3)
        self.assertEqual(bundle["errors"], [])

    def test_legacy_function_delegates_to_local_runner(self) -> None:
        bundle = run_after_sales_tools_sync(
            {"raw_text": "机器漏水把插座打湿了", "symptoms": ["漏水", "插座打湿"]},
            {"summary": "存在安全风险。", "priority": "high"},
        )

        self.assertTrue(bundle["escalation"]["need_escalation"])
        self.assertEqual(bundle["escalation"]["level"], "high")


if __name__ == "__main__":
    unittest.main()
