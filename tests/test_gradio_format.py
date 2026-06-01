from __future__ import annotations

import unittest

from gradio_app import format_agent_response


class GradioFormatTest(unittest.TestCase):
    def test_format_agent_response_exposes_customer_reply_first(self) -> None:
        payload = {
            "success": True,
            "data": {
                "action": {"customer_reply": "您好，请先停止使用设备。", "ticket": {"title": "测试工单"}},
                "warranty": {"status": "unknown"},
                "escalation": {"need_escalation": True},
                "audit": {"passed": False},
                "retrieval": {
                    "sources": ["测试手册 第1页"],
                    "results": [{"file_name": "测试手册", "page": 1, "text": "E03"}],
                    "trace": {"mode": "hybrid"},
                },
                "tool_history": [{"tool_name": "ticket_draft"}],
                "trace": [{"node": "final"}],
            },
        }

        customer_reply, debug_info, tool_history, trace = format_agent_response(payload)

        self.assertIn("停止使用", customer_reply)
        self.assertEqual(debug_info["工单草稿"]["title"], "测试工单")
        self.assertEqual(debug_info["知识库检索结果"][0]["text"], "E03")
        self.assertEqual(tool_history["tool_history"][0]["tool_name"], "ticket_draft")
        self.assertEqual(trace["trace"][0]["node"], "final")


if __name__ == "__main__":
    unittest.main()
