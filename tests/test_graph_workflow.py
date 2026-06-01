from __future__ import annotations

import unittest
from typing import Any

from backend.graph_workflow import AfterSalesGraphWorkflow


def fake_retrieval(question: str, **_: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return [
        {
            "file_name": "QY-320故障码手册.pdf",
            "page": 3,
            "text": "E03 表示进水压力异常或滤芯堵塞，建议检查进水阀、前置滤芯和流量状态。",
            "score": 0.91,
            "doc_type": "fault_code_table",
        }
    ], {"mode": "fake", "queries": [question]}


class AfterSalesGraphWorkflowTest(unittest.TestCase):
    def test_e03_case_returns_customer_reply_and_tool_outputs(self) -> None:
        workflow = AfterSalesGraphWorkflow(retrieval_func=fake_retrieval)

        result = workflow.run("QY-320 显示 E03，出水变慢，买了半年")

        self.assertEqual(result["case"]["product_model"], "QY-320")
        self.assertEqual(result["case"]["fault_code"], "E03")
        self.assertTrue(result["action"]["customer_reply"])
        self.assertEqual(result["warranty"]["status"], "possibly_in_warranty")
        self.assertFalse(result["escalation"]["need_escalation"])
        self.assertTrue(result["action"]["ticket"])
        self.assertEqual([item["call_type"] for item in result["tool_history"]], ["local_python"] * 3)
        self.assertIn("QY-320故障码手册.pdf 第3页", result["retrieval"]["sources"])

    def test_wet_socket_case_requires_high_risk_escalation(self) -> None:
        workflow = AfterSalesGraphWorkflow(retrieval_func=fake_retrieval)

        result = workflow.run("机器漏水把插座打湿了")

        self.assertTrue(result["escalation"]["need_escalation"])
        self.assertEqual(result["escalation"]["level"], "high")
        self.assertFalse(result["audit"]["passed"])
        self.assertEqual(result["audit"]["risk_level"], "high")
        self.assertIn("停止使用", result["action"]["customer_reply"])

    def test_empty_knowledge_base_returns_structured_warning(self) -> None:
        workflow = AfterSalesGraphWorkflow()

        result = workflow.run("QY-320 显示 E03")

        self.assertEqual(result["retrieval"]["results"], [])
        self.assertFalse(result["audit"]["passed"])
        self.assertIn("没有检索到知识库依据", " ".join(result["audit"]["warnings"]))
        self.assertEqual(
            set(result),
            {
                "intent",
                "case",
                "retrieval",
                "diagnosis",
                "warranty",
                "escalation",
                "action",
                "audit",
                "tool_history",
                "trace",
            },
        )


if __name__ == "__main__":
    unittest.main()
