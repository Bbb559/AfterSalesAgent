from __future__ import annotations

import unittest

try:
    from fastapi.testclient import TestClient
    from api import app
except ModuleNotFoundError:
    TestClient = None
    app = None


@unittest.skipIf(TestClient is None, "当前环境未安装 fastapi，安装 requirements.txt 后会执行该测试。")
class ApiContractTest(unittest.TestCase):
    def test_health_endpoint_returns_ok(self) -> None:
        client = TestClient(app)

        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_kb_list_and_status_endpoints_return_structured_payloads(self) -> None:
        client = TestClient(app)

        list_response = client.get("/api/kb/list")
        status_response = client.get("/api/kb/status")

        self.assertEqual(list_response.status_code, 200)
        self.assertTrue(list_response.json()["success"])
        self.assertIn("items", list_response.json()["data"])
        self.assertEqual(status_response.status_code, 200)
        self.assertTrue(status_response.json()["success"])
        self.assertIn("loaded", status_response.json()["data"])

    def test_after_sales_run_endpoint_returns_workflow_result(self) -> None:
        client = TestClient(app)

        response = client.post(
            "/api/aftersales/run",
            json={"user_input": "机器漏水把插座打湿了", "retrieval_options": {}},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["data"]["escalation"]["level"], "high")
        self.assertIn("停止使用", payload["data"]["action"]["customer_reply"])


if __name__ == "__main__":
    unittest.main()
