"""Memory 调试 API 测试。

覆盖阶段 4 需求：
1. SQLite 双写后能通过 API 查到 session
2. 能查到 messages
3. session 不存在时返回 404
4. MEMORY_SQLITE_DUAL_WRITE=false 时接口能优雅提示 SQLite 未启用
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
    from api import app, get_memory_manager
    from backend.memory import MemoryManager
except ModuleNotFoundError:
    TestClient = None
    app = None
    get_memory_manager = None
    MemoryManager = None


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _make_messages(count: int = 2) -> list[dict]:
    return [
        {
            "role": "user",
            "content": f"测试用户消息 {i} — VG-11KW-Pro 故障码 C-RCD-04",
            "timestamp": _now(),
            "metadata": {"source": "workflow"},
        }
        for i in range(count)
    ] + [
        {
            "role": "assistant",
            "content": f"测试助手回复 {i} — 已记录，建议暂停使用并联系电工。",
            "timestamp": _now(),
            "metadata": {"source": "workflow"},
        }
        for i in range(count)
    ]


def _make_case() -> dict:
    return {
        "brand": "VoltGate",
        "charger_model": "VG-11KW-Pro",
        "rated_power_kw": "11kW",
        "city": "广州",
        "contact_address": "天河区测试路88号地下车库",
        "install_time": "2024-03",
        "fault_codes": ["C-RCD-04", "C-COM-12"],
        "observed_symptoms": ["漏保频繁跳闸", "无法启动充电"],
        "safety_signals": ["漏保频繁跳闸"],
        "missing_info": ["联系电话", "序列号"],
        "issue_description": "VG-11KW-Pro 无法启动充电，屏幕显示 C-RCD-04",
    }


def _make_ticket(ticket_id: str = "ticket_api_001") -> dict:
    return {
        "ticket_id": ticket_id,
        "title": "VoltGate VG-11KW-Pro 无法启动充电",
        "priority": "p1_high",
        "status": "draft",
        "created_at": _now(),
        "dispatch": {"title": "测试派工"},
    }


@unittest.skipIf(TestClient is None, "当前环境未安装 fastapi，安装 requirements.txt 后会执行该测试。")
class MemoryApiSessionTest(unittest.TestCase):
    """测试 GET /api/memory/sessions/{session_id}"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.manager = MemoryManager(memory_dir=Path(self.tmp.name))
        self.session_id = "session_api_test"

        # 通过 SQLite 双写写入测试数据
        self.manager.sqlite_store.write_workflow_result(
            session_id=self.session_id,
            created_at=_now(),
            updated_at=_now(),
            messages=_make_messages(3),  # 3 user + 3 assistant = 6
            case=_make_case(),
            ticket_id="ticket_api_001",
            ticket=_make_ticket("ticket_api_001"),
            triage={"intent": "fault_diagnosis", "confidence": "high"},
            safety={"risk_level": "p1_high"},
            diagnosis={"summary": "需排查 RCD 模块。"},
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_get_session_returns_session_data(self) -> None:
        """SQLite 双写后能通过 API 查到 session。"""
        with patch("api.get_memory_manager", return_value=self.manager):
            client = TestClient(app)
            response = client.get(f"/api/memory/sessions/{self.session_id}")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["data"]["session_id"], self.session_id)
        self.assertEqual(payload["data"]["message_count"], 6)
        self.assertEqual(payload["data"]["status"], "active")

    def test_get_session_missing_returns_404(self) -> None:
        """session 不存在时返回 404。"""
        with patch("api.get_memory_manager", return_value=self.manager):
            client = TestClient(app)
            response = client.get("/api/memory/sessions/nonexistent_session")

        self.assertEqual(response.status_code, 404)
        payload = response.json()
        self.assertFalse(payload["success"])
        self.assertIn("不存在", payload["error"])
        self.assertIn("nonexistent_session", payload["error"])

    def test_get_session_sqlite_disabled_returns_503(self) -> None:
        """MEMORY_SQLITE_DUAL_WRITE=false 时接口优雅提示 SQLite 未启用。"""
        # 模拟 feature flag 关闭：sqlite_store 为 None
        self.manager.sqlite_store = None

        with patch("api.get_memory_manager", return_value=self.manager):
            client = TestClient(app)
            response = client.get(f"/api/memory/sessions/{self.session_id}")

        self.assertEqual(response.status_code, 503)
        payload = response.json()
        self.assertFalse(payload["success"])
        self.assertIn("MEMORY_SQLITE_DUAL_WRITE", payload["error"])

    def test_get_session_sqlite_unavailable_returns_503(self) -> None:
        """SQLite 存储不可用时返回 503。"""
        self.manager.sqlite_store.available = False
        self.manager.sqlite_store.error = "磁盘空间不足"

        with patch("api.get_memory_manager", return_value=self.manager):
            client = TestClient(app)
            response = client.get(f"/api/memory/sessions/{self.session_id}")

        self.assertEqual(response.status_code, 503)
        payload = response.json()
        self.assertFalse(payload["success"])
        self.assertIn("磁盘空间不足", payload["error"])


@unittest.skipIf(TestClient is None, "当前环境未安装 fastapi，安装 requirements.txt 后会执行该测试。")
class MemoryApiMessagesTest(unittest.TestCase):
    """测试 GET /api/memory/sessions/{session_id}/messages"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.manager = MemoryManager(memory_dir=Path(self.tmp.name))
        self.session_id = "session_api_msgs"

        # 写入 4 条消息（2 user + 2 assistant）
        self.msgs = _make_messages(2)
        self.manager.sqlite_store.write_workflow_result(
            session_id=self.session_id,
            created_at=_now(),
            updated_at=_now(),
            messages=self.msgs,
            case=_make_case(),
            ticket_id="ticket_api_msgs_001",
            ticket=_make_ticket("ticket_api_msgs_001"),
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_get_messages_returns_message_list(self) -> None:
        """SQLite 双写后能通过 API 查到 messages。"""
        with patch("api.get_memory_manager", return_value=self.manager):
            client = TestClient(app)
            response = client.get(f"/api/memory/sessions/{self.session_id}/messages")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["data"]["session_id"], self.session_id)
        self.assertEqual(payload["data"]["message_count"], 4)
        self.assertEqual(len(payload["data"]["messages"]), 4)
        # 验证消息内容
        first_msg = payload["data"]["messages"][0]
        self.assertEqual(first_msg["role"], "user")
        self.assertEqual(first_msg["turn_index"], 0)
        self.assertIn("VG-11KW-Pro", first_msg["content"])
        # 验证角色分布
        roles = [m["role"] for m in payload["data"]["messages"]]
        self.assertEqual(roles.count("user"), 2)
        self.assertEqual(roles.count("assistant"), 2)

    def test_get_messages_respects_limit(self) -> None:
        """limit 参数能限制返回条数。"""
        with patch("api.get_memory_manager", return_value=self.manager):
            client = TestClient(app)
            response = client.get(
                f"/api/memory/sessions/{self.session_id}/messages?limit=1"
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertEqual(len(payload["data"]["messages"]), 1)
        self.assertEqual(payload["data"]["limit"], 1)

    def test_get_messages_session_missing_returns_404(self) -> None:
        """session 不存在时返回 404（不返回空列表）。"""
        with patch("api.get_memory_manager", return_value=self.manager):
            client = TestClient(app)
            response = client.get("/api/memory/sessions/nonexistent/messages")

        self.assertEqual(response.status_code, 404)
        payload = response.json()
        self.assertFalse(payload["success"])
        self.assertIn("不存在", payload["error"])

    def test_get_messages_sqlite_disabled_returns_503(self) -> None:
        """MEMORY_SQLITE_DUAL_WRITE=false 时 messages 接口也返回 503。"""
        self.manager.sqlite_store = None

        with patch("api.get_memory_manager", return_value=self.manager):
            client = TestClient(app)
            response = client.get(f"/api/memory/sessions/{self.session_id}/messages")

        self.assertEqual(response.status_code, 503)
        payload = response.json()
        self.assertFalse(payload["success"])
        self.assertIn("MEMORY_SQLITE_DUAL_WRITE", payload["error"])

    def test_get_messages_empty_session_returns_empty_list(self) -> None:
        """有 session 但无消息时返回空列表（非 404）。"""
        empty_session_id = "session_empty_msgs"
        self.manager.sqlite_store.write_workflow_result(
            session_id=empty_session_id,
            created_at=_now(),
            updated_at=_now(),
            messages=[],
            case={},
            ticket_id="",
            ticket={},
        )

        with patch("api.get_memory_manager", return_value=self.manager):
            client = TestClient(app)
            response = client.get(f"/api/memory/sessions/{empty_session_id}/messages")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["data"]["message_count"], 0)
        self.assertEqual(payload["data"]["messages"], [])


@unittest.skipIf(TestClient is None, "当前环境未安装 fastapi，安装 requirements.txt 后会执行该测试。")
class MemoryApiCrossCheckTest(unittest.TestCase):
    """端到端验证：MemoryManager.remember_workflow_result → SQLite 双写 → API 查询"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.manager = MemoryManager(memory_dir=Path(self.tmp.name))
        self.session_id = "session_e2e"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_workflow_result_writes_to_sqlite_and_api_returns_it(self) -> None:
        """模拟完整 workflow 双写链路后，API 能查到完整数据。"""
        workflow_result = {
            "triage": {"intent": "fault_diagnosis", "confidence": "high"},
            "case": {
                "brand": "VoltGate",
                "charger_model": "VG-11KW-Pro",
                "rated_power_kw": "11kW",
                "city": "深圳",
                "contact_address": "南山区科技园",
                "install_time": "2024-06",
                "fault_codes": ["C-RCD-04"],
                "observed_symptoms": ["漏保频繁跳闸"],
                "safety_signals": ["漏保频繁跳闸"],
                "missing_info": ["序列号"],
                "issue_description": "无法启动充电",
            },
            "safety": {"risk_level": "p1_high"},
            "diagnosis": {"summary": "需排查 RCD 模块。"},
            "dispatch": {"title": "深圳派工"},
            "action": {"customer_reply": "请暂停使用。"},
            "audit": {"passed": True},
            "tool_history": [],
            "trace": [],
        }

        ids = self.manager.remember_workflow_result(
            "VG-11KW-Pro 无法启动充电",
            workflow_result,
            session_id=self.session_id,
        )
        self.assertEqual(ids["session_id"], self.session_id)

        # 通过 API 查询 session
        with patch("api.get_memory_manager", return_value=self.manager):
            client = TestClient(app)
            session_resp = client.get(f"/api/memory/sessions/{self.session_id}")
            msgs_resp = client.get(f"/api/memory/sessions/{self.session_id}/messages")

        # 验证 session
        self.assertEqual(session_resp.status_code, 200)
        session_data = session_resp.json()["data"]
        self.assertEqual(session_data["session_id"], self.session_id)
        self.assertGreater(session_data["message_count"], 0)

        # 验证 messages
        self.assertEqual(msgs_resp.status_code, 200)
        msgs_data = msgs_resp.json()["data"]
        self.assertGreater(msgs_data["message_count"], 0)
        # 应包含用户消息和助手回复
        contents = " ".join(m["content"] for m in msgs_data["messages"])
        self.assertIn("无法启动充电", contents)

        # 验证 SQLite trace 被记录
        trace_entries = [
            t for t in workflow_result.get("trace", [])
            if t.get("node") == "memory_sqlite"
        ]
        self.assertTrue(
            any(t.get("node") == "memory_sqlite" for t in workflow_result.get("trace", [])),
            "workflow_result trace 中应包含 memory_sqlite 记录",
        )


if __name__ == "__main__":
    unittest.main()
