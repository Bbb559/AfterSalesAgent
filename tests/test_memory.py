from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.memory import MemoryManager, SessionMemory


class MemoryModuleTest(unittest.TestCase):
    def test_session_memory_persists_messages_and_charger_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session = SessionMemory(session_id="session_demo", root=Path(temp_dir) / "sessions")
            session.add_message("user", "充电桩枪线破皮")
            session.update_context("last_triage", "safety_emergency")

            reloaded = SessionMemory(session_id="session_demo", root=Path(temp_dir) / "sessions")

            self.assertEqual(reloaded.messages[0].content, "充电桩枪线破皮")
            self.assertEqual(reloaded.get_context("last_triage"), "safety_emergency")
            self.assertEqual(reloaded.get_status()["message_count"], 1)

    def test_session_memory_persists_structured_last_workflow_result(self) -> None:
        workflow_result = {
            "triage": {"intent": "fault_diagnosis"},
            "case": {
                "charger_model": "VG-7KW-AC",
                "city": "惠州",
                "issue_description": "昨晚跳闸",
                "missing_info": ["联系电话", "安装地址"],
            },
            "safety": {"risk_level": "p1_high"},
            "diagnosis": {"summary": "需要排查跳闸原因。"},
            "dispatch": {"title": "跳闸派工", "priority": "p1_high"},
            "action": {"customer_reply": "请先暂停使用。"},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            session = SessionMemory(session_id="session_demo", root=Path(temp_dir) / "sessions")
            session.remember_workflow_result("我家充电桩昨晚跳闸，型号 VG-7KW-AC，在惠州。", workflow_result)

            reloaded = SessionMemory(session_id="session_demo", root=Path(temp_dir) / "sessions")

            self.assertEqual(reloaded.get_context("last_case")["charger_model"], "VG-7KW-AC")
            self.assertEqual(reloaded.get_context("last_intent"), "fault_diagnosis")
            self.assertEqual(reloaded.get_context("last_safety")["risk_level"], "p1_high")
            self.assertEqual(reloaded.get_context("last_diagnosis")["summary"], "需要排查跳闸原因。")
            self.assertEqual(reloaded.get_context("last_dispatch")["priority"], "p1_high")
            self.assertEqual(reloaded.get_context("last_customer_reply"), "请先暂停使用。")
            self.assertEqual(reloaded.get_context("recent_case")["charger_model"], "VG-7KW-AC")
            self.assertEqual(reloaded.get_context("recent_dispatch")["priority"], "p1_high")
            self.assertEqual(reloaded.get_status()["last_model"], "VG-7KW-AC")
            self.assertEqual(reloaded.get_status()["last_risk_level"], "p1_high")

    def test_manager_remembers_customer_charger_site_and_ticket_snapshots(self) -> None:
        workflow_result = {
            "triage": {"intent": "fault_diagnosis"},
            "case": {
                "brand": "VoltGate",
                "charger_model": "VG-11KW-Pro",
                "charger_series": "VG",
                "serial_number": "SN123",
                "issue_description": "无法启动充电",
                "fault_codes": ["C-RCD-04"],
                "safety_signals": ["漏保频繁跳闸"],
                "environment_factors": ["地下车库"],
                "contact_name": "张三",
                "contact_phone": "13800000000",
                "city": "广州",
                "contact_address": "天河区测试路地下车库",
                "missing_info": [],
            },
            "safety": {"risk_level": "p1_high"},
            "diagnosis": {"summary": "需按知识库排查。", "suggested_next_step": "创建派工。"},
            "warranty": {"status": "unknown"},
            "dispatch": {"title": "VoltGate - VG-11KW-Pro - 无法启动充电", "customer_problem": "无法启动充电"},
            "action": {
                "customer_reply": "您好，已为您创建充电桩安全诊断记录。",
                "dispatch": {"title": "VoltGate - VG-11KW-Pro - 无法启动充电"},
            },
            "audit": {"passed": True},
            "tool_history": [],
            "trace": [],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MemoryManager(memory_dir=temp_dir)
            ids = manager.remember_workflow_result("VG-11KW-Pro 无法启动充电", workflow_result)

            reloaded = MemoryManager(memory_dir=temp_dir)
            customer = reloaded.customer_memory.get(ids["customer_id"])
            charger = reloaded.charger_memory.get(ids["charger_id"])
            site = reloaded.site_memory.get(ids["site_id"])
            ticket = reloaded.ticket_memory.get(ids["ticket_id"])

            self.assertEqual(ids["customer_id"], "13800000000")
            self.assertEqual(customer["contact_name"], "张三")
            self.assertEqual(charger["serial_number"], "SN123")
            self.assertEqual(site["city"], "广州")
            self.assertEqual(ticket["title"], "VoltGate - VG-11KW-Pro - 无法启动充电")
            self.assertEqual(reloaded.ticket_memory.get_status()["total_tickets"], 1)

    def test_manager_recall_context_returns_isolated_layered_summary(self) -> None:
        workflow_result = {
            "case": {
                "brand": "VoltGate",
                "charger_model": "VG-11KW-Pro",
                "serial_number": "SN999",
                "issue_description": "漏保跳闸",
                "contact_phone": "13900000000",
                "city": "深圳",
                "contact_address": "南山区测试车库",
            },
            "safety": {"risk_level": "p1_high"},
            "diagnosis": {"suggested_next_step": "上门电工核验"},
            "dispatch": {"title": "历史派工", "customer_problem": "漏保跳闸"},
            "action": {"customer_reply": "历史回复"},
            "audit": {"passed": True},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MemoryManager(memory_dir=temp_dir)
            manager.remember_workflow_result("SN999 漏保跳闸", workflow_result, session_id="session_demo")

            context = manager.recall_context(workflow_result["case"], session_id="session_demo")

            self.assertEqual(context["session"]["session_id"], "session_demo")
            self.assertEqual(context["last_case"]["serial_number"], "SN999")
            self.assertEqual(context["recent_safety"]["risk_level"], "p1_high")
            self.assertEqual(context["recent_ticket"]["title"], "历史派工")
            self.assertEqual(context["last_customer_reply"], "历史回复")
            self.assertEqual(context["charger"]["serial_number"], "SN999")
            self.assertEqual(context["customer"]["contact_phone"], "13900000000")
            self.assertEqual(context["site"]["city"], "深圳")
            self.assertTrue(context["ticket"]["title"])
            self.assertTrue(context["isolation"]["session_isolated"])
            self.assertFalse(context["isolation"]["used_as_diagnostic_evidence"])

    def test_manager_recall_context_includes_sqlite_fts5_session_search(self) -> None:
        workflow_result = {
            "case": {
                "brand": "VoltGate",
                "charger_model": "VG-FTS",
                "issue_description": "C-RCD-04 keeps tripping the RCD",
                "fault_codes": ["C-RCD-04"],
                "missing_info": [],
            },
            "safety": {"risk_level": "p1_high"},
            "diagnosis": {"summary": "RCD trip needs safe onsite verification"},
            "dispatch": {"title": "FTS session ticket"},
            "action": {"customer_reply": "Please stop charging and wait for verification."},
            "audit": {"passed": True},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MemoryManager(memory_dir=temp_dir)
            manager.remember_workflow_result(
                "VoltGate VG-FTS C-RCD-04 keeps tripping",
                workflow_result,
                session_id="session_fts",
            )

            context = manager.recall_context(
                {"raw_text": "C-RCD-04 risk", "charger_model": "VG-FTS"},
                session_id="session_fts",
            )

            self.assertTrue(context["session_search"]["available"])
            self.assertTrue(context["session_search"]["matches"])
            self.assertIn("C-RCD-04", context["session_search"]["summary"])
            self.assertEqual(context["isolation"]["session_search_store"], "sqlite_fts5")

    def test_manager_status_exposes_charger_memory_modules(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MemoryManager(memory_dir=temp_dir)
            manager.create_session("session_demo")

            status = manager.get_status()

            self.assertEqual(status["current_session_id"], "session_demo")
            self.assertIn("customers", status)
            self.assertIn("chargers", status)
            self.assertIn("sites", status)
            self.assertIn("tickets", status)


if __name__ == "__main__":
    unittest.main()
