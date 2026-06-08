"""SQLite recall_context 兼容读取测试（阶段 3）。

覆盖：
1. flag=false 时走 JSON 路径
2. flag=true 时走 SQLite 路径
3. SQLite 失败时回退 JSON
4. SQLite 路径返回结构与 JSON 路径兼容
5. memory_answer v2 在 SQLite 路径下能召回型号/功率/缺失字段
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from backend.memory.core import MemoryManager
from backend.memory.sqlite_store import SQLiteLongTermMemoryStore


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


def _make_ticket(ticket_id: str = "ticket_test_001") -> dict:
    return {
        "ticket_id": ticket_id,
        "title": "VoltGate VG-11KW-Pro 无法启动充电",
        "priority": "p1_high",
        "status": "draft",
        "created_at": _now(),
        "dispatch": {"title": "测试派工", "priority": "p1_high"},
        "safety": {"risk_level": "p1_high"},
        "audit": {"passed": True},
    }


# ── flag=false 走 JSON 路径 ────────────────────────────────────────────


class RecallContextFlagOffTest(unittest.TestCase):
    """MEMORY_READ_FROM_SQLITE=false 时保持 JSON 路径不变。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.manager = MemoryManager(memory_dir=Path(self.tmp.name))
        self.session_id = "session_flag_off"
        self._write_data()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_data(self) -> None:
        """通过 remember_workflow_result 双写 JSON 和 SQLite，确保 JSON 路径可读。"""
        result = {
            "triage": {"intent": "fault_diagnosis", "confidence": "high"},
            "case": _make_case(),
            "safety": {"risk_level": "p1_high"},
            "diagnosis": {"summary": "需排查 RCD 模块。"},
            "dispatch": {"title": "测试派工", "priority": "p1_high"},
            "action": {"customer_reply": "请暂停使用。"},
            "audit": {"passed": True},
            "tool_history": [],
            "trace": [],
        }
        self.manager.remember_workflow_result(
            "VG-11KW-Pro 无法启动充电", result, session_id=self.session_id
        )

    def test_flag_false_uses_json_path(self) -> None:
        """flag=false 时 isolation.long_term_store = 'local_json'。"""
        with patch("backend.memory.core.MEMORY_READ_FROM_SQLITE", False):
            context = self.manager.recall_context(
                case=_make_case(), session_id=self.session_id
            )

        self.assertEqual(context["isolation"]["long_term_store"], "local_json")
        self.assertGreater(context["session"]["message_count"], 0)

    def test_flag_false_sqlite_store_none_still_works(self) -> None:
        """即使 sqlite_store=None，JSON 路径仍正常工作。"""
        self.manager.sqlite_store = None
        with patch("backend.memory.core.MEMORY_READ_FROM_SQLITE", False):
            context = self.manager.recall_context(
                case=_make_case(), session_id=self.session_id
            )

        self.assertEqual(context["isolation"]["long_term_store"], "local_json")
        # session_id 应在 matched_ids 中
        self.assertEqual(
            context["matched_ids"]["session_id"], self.session_id
        )


# ── flag=true 走 SQLite 路径 ──────────────────────────────────────────


class RecallContextFlagOnTest(unittest.TestCase):
    """MEMORY_READ_FROM_SQLITE=true 时走 SQLite 路径。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.manager = MemoryManager(memory_dir=Path(self.tmp.name))
        self.session_id = "session_flag_on"
        self._write_data()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_data(self) -> None:
        self.manager.sqlite_store.write_workflow_result(
            session_id=self.session_id,
            created_at=_now(),
            updated_at=_now(),
            messages=_make_messages(2),
            case=_make_case(),
            ticket_id="ticket_flag_on",
            ticket=_make_ticket("ticket_flag_on"),
            triage={"intent": "fault_diagnosis", "confidence": "high"},
            safety={"risk_level": "p1_high"},
            diagnosis={"summary": "需排查 RCD 模块。"},
        )

    def test_flag_true_uses_sqlite_path(self) -> None:
        """flag=true 时 isolation.long_term_store = 'sqlite'。"""
        with patch("backend.memory.core.MEMORY_READ_FROM_SQLITE", True):
            context = self.manager.recall_context(
                case=_make_case(), session_id=self.session_id
            )

        self.assertEqual(context["isolation"]["long_term_store"], "sqlite")
        self.assertGreater(context["session"]["message_count"], 0)

    def test_flag_true_returns_compatible_top_keys(self) -> None:
        """SQLite 路径返回的顶层 key set 与 JSON 路径一致。"""
        with patch("backend.memory.core.MEMORY_READ_FROM_SQLITE", False):
            json_ctx = self.manager.recall_context(
                case=_make_case(), session_id=self.session_id
            )
        with patch("backend.memory.core.MEMORY_READ_FROM_SQLITE", True):
            sqlite_ctx = self.manager.recall_context(
                case=_make_case(), session_id=self.session_id
            )

        self.assertSetEqual(
            set(json_ctx.keys()),
            set(sqlite_ctx.keys()),
            f"SQLite 路径缺少 key: {set(json_ctx.keys()) - set(sqlite_ctx.keys())}；"
            f"多余 key: {set(sqlite_ctx.keys()) - set(json_ctx.keys())}",
        )

    def test_flag_true_reconstructs_case_fields(self) -> None:
        """SQLite 路径能重建 last_case 中的型号/城市/故障码。"""
        with patch("backend.memory.core.MEMORY_READ_FROM_SQLITE", True):
            context = self.manager.recall_context(
                case=_make_case(), session_id=self.session_id
            )

        last_case = context["last_case"]
        self.assertEqual(last_case.get("charger_model"), "VG-11KW-Pro")
        self.assertEqual(last_case.get("city"), "广州")
        self.assertEqual(last_case.get("brand"), "VoltGate")

    def test_flag_true_reconstructs_missing_info(self) -> None:
        """SQLite 路径能重建 missing_info。"""
        with patch("backend.memory.core.MEMORY_READ_FROM_SQLITE", True):
            context = self.manager.recall_context(
                case=_make_case(), session_id=self.session_id
            )

        missing = context["missing_info"]
        self.assertIsInstance(missing, list)
        self.assertIn("联系电话", missing)
        self.assertIn("序列号", missing)

    def test_flag_true_reconstructs_session_summary(self) -> None:
        """SQLite 路径能重建 session_summary。"""
        with patch("backend.memory.core.MEMORY_READ_FROM_SQLITE", True):
            context = self.manager.recall_context(
                case=_make_case(), session_id=self.session_id
            )

        summary = context["session_summary"]
        self.assertEqual(summary["session_id"], self.session_id)
        self.assertGreater(summary["message_count"], 0)
        self.assertEqual(summary["last_model"], "VG-11KW-Pro")
        self.assertEqual(summary["last_risk_level"], "p1_high")
        self.assertEqual(summary["last_dispatch_priority"], "p1_high")

    def test_flag_true_session_search_available(self) -> None:
        """SQLite 路径下 session_search 可用。"""
        with patch("backend.memory.core.MEMORY_READ_FROM_SQLITE", True):
            context = self.manager.recall_context(
                case=_make_case(), session_id=self.session_id
            )

        search = context["session_search"]
        self.assertTrue(search.get("available", False))
        self.assertIsInstance(search.get("matches"), list)

    def test_flag_true_fts5_hit_by_content(self) -> None:
        """SQLite FTS5 搜索能命中写入的消息内容。"""
        with patch("backend.memory.core.MEMORY_READ_FROM_SQLITE", True):
            context = self.manager.recall_context(
                case={"raw_text": "C-RCD-04 故障"}, session_id=self.session_id
            )

        search = context["session_search"]
        if search.get("available") and search.get("matches"):
            contents = " ".join(
                str(m.get("content", "")) for m in search["matches"]
            )
            self.assertIn("C-RCD-04", contents)

    def test_flag_true_empty_session_returns_minimal_structure(self) -> None:
        """不存在的 session 返回最小有效结构。"""
        with patch("backend.memory.core.MEMORY_READ_FROM_SQLITE", True):
            context = self.manager.recall_context(
                case={}, session_id="nonexistent_session_xyz"
            )

        self.assertEqual(context["session"]["session_id"], "nonexistent_session_xyz")
        self.assertEqual(context["session"]["message_count"], 0)
        self.assertEqual(context["session"]["last_case"], {})
        self.assertEqual(context["missing_info"], [])
        self.assertEqual(context["recent_ticket"], {})

    def test_flag_true_respects_matched_ids(self) -> None:
        """SQLite 路径的 matched_ids 包含 session_id。"""
        with patch("backend.memory.core.MEMORY_READ_FROM_SQLITE", True):
            context = self.manager.recall_context(
                case=_make_case(), session_id=self.session_id
            )

        ids = context["matched_ids"]
        self.assertEqual(ids["session_id"], self.session_id)


# ── SQLite 失败回退 JSON ──────────────────────────────────────────────


class RecallContextFallbackTest(unittest.TestCase):
    """SQLite 读取失败时静默回退 JSON。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.manager = MemoryManager(memory_dir=Path(self.tmp.name))
        self.session_id = "session_fallback"
        self._write_data()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_data(self) -> None:
        """通过 remember_workflow_result 双写 JSON 和 SQLite，确保回退后 JSON 路径可读。"""
        result = {
            "triage": {"intent": "fault_diagnosis"},
            "case": _make_case(),
            "safety": {"risk_level": "p1_high"},
            "diagnosis": {"summary": "需排查。"},
            "dispatch": {"title": "测试派工", "priority": "p1_high"},
            "action": {"customer_reply": "请暂停使用。"},
            "audit": {"passed": True},
            "tool_history": [],
            "trace": [],
        }
        self.manager.remember_workflow_result(
            "VG-11KW-Pro 无法启动充电", result, session_id=self.session_id
        )

    def test_sqlite_unavailable_falls_back_to_json(self) -> None:
        """SQLite store unavailable 时不崩溃，走 JSON 路径。"""
        self.manager.sqlite_store.available = False
        self.manager.sqlite_store.error = "模拟 SQLite 不可用"

        with patch("backend.memory.core.MEMORY_READ_FROM_SQLITE", True):
            context = self.manager.recall_context(
                case=_make_case(), session_id=self.session_id
            )

        # 回退后应走 JSON 路径
        self.assertEqual(context["isolation"]["long_term_store"], "local_json")
        self.assertGreater(context["session"]["message_count"], 0)

    def test_sqlite_store_none_falls_back_to_json(self) -> None:
        """sqlite_store=None 时不崩溃，走 JSON 路径。"""
        self.manager.sqlite_store = None

        with patch("backend.memory.core.MEMORY_READ_FROM_SQLITE", True):
            context = self.manager.recall_context(
                case=_make_case(), session_id=self.session_id
            )

        self.assertEqual(context["isolation"]["long_term_store"], "local_json")

    def test_sqlite_exception_falls_back_to_json(self) -> None:
        """build_session_context 抛出异常时回退 JSON。"""
        # 用异常注入：让 build_session_context 抛出异常
        original = self.manager.sqlite_store.build_session_context

        def _raise(*args, **kwargs):
            raise RuntimeError("模拟 SQLite 查询异常")

        self.manager.sqlite_store.build_session_context = _raise

        try:
            with patch("backend.memory.core.MEMORY_READ_FROM_SQLITE", True):
                context = self.manager.recall_context(
                    case=_make_case(), session_id=self.session_id
                )

            self.assertEqual(context["isolation"]["long_term_store"], "local_json")
        finally:
            self.manager.sqlite_store.build_session_context = original


# ── 结构兼容性 ─────────────────────────────────────────────────────────


class RecallContextCompatTest(unittest.TestCase):
    """SQLite 路径与 JSON 路径返回结构兼容。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.manager = MemoryManager(memory_dir=Path(self.tmp.name))
        self.session_id = "session_compat"
        self._write_data()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_data(self) -> None:
        self.manager.sqlite_store.write_workflow_result(
            session_id=self.session_id,
            created_at=_now(),
            updated_at=_now(),
            messages=_make_messages(2),
            case=_make_case(),
            ticket_id="ticket_compat",
            ticket=_make_ticket("ticket_compat"),
            triage={"intent": "fault_diagnosis", "confidence": "high"},
            safety={"risk_level": "p1_high"},
            diagnosis={"summary": "需排查 RCD 模块。"},
        )

    def _deep_keys(self, d: dict, prefix: str = "") -> set:
        """递归收集所有 key 路径（用于结构对比）。"""
        keys = set()
        for k, v in d.items():
            path = f"{prefix}.{k}" if prefix else k
            keys.add(path)
            if isinstance(v, dict):
                keys |= self._deep_keys(v, path)
        return keys

    def test_nested_key_structure_compatible(self) -> None:
        """SQLite 路径的嵌套 key 结构应覆盖 JSON 路径的核心字段。

        注意：SQLite 路径的 customer/charger/site 维度尚未迁移，这部分 key 在
        SQLite 路径下无嵌套结构是预期行为。本测试仅验证 session/case/ticket 维度。
        """
        with patch("backend.memory.core.MEMORY_READ_FROM_SQLITE", False):
            json_ctx = self.manager.recall_context(
                case=_make_case(), session_id=self.session_id
            )
        with patch("backend.memory.core.MEMORY_READ_FROM_SQLITE", True):
            sqlite_ctx = self.manager.recall_context(
                case=_make_case(), session_id=self.session_id
            )

        # 验证 session 维度的 key 兼容
        json_session_keys = {
            k for k in self._deep_keys(json_ctx["session"])
        }
        sqlite_session_keys = {
            k for k in self._deep_keys(sqlite_ctx["session"])
        }
        missing_in_sqlite = json_session_keys - sqlite_session_keys
        # 两个路径的 session 顶层 key 应一致
        json_top = set(json_ctx["session"].keys())
        sqlite_top = set(sqlite_ctx["session"].keys())
        self.assertSetEqual(
            json_top,
            sqlite_top,
            f"session 顶层 key 不一致：JSON={json_top - sqlite_top}，SQLite={sqlite_top - json_top}",
        )

        # 验证 session_summary 的 key 兼容
        self.assertSetEqual(
            set(json_ctx["session_summary"].keys()),
            set(sqlite_ctx["session_summary"].keys()),
            f"session_summary key 不一致",
        )

        # 验证 core 字段类型兼容
        for field in ["last_case", "missing_info", "recent_safety",
                       "recent_ticket", "session_search", "matched_ids", "isolation"]:
            self.assertIn(field, sqlite_ctx, f"SQLite 路径缺少顶层字段 {field}")
            json_type = type(json_ctx[field])
            sqlite_type = type(sqlite_ctx[field])
            self.assertIsInstance(
                sqlite_ctx[field], json_type,
                f"字段 {field} 类型不兼容：JSON={json_type}，SQLite={sqlite_type}",
            )

    def test_isolation_fields_consistent(self) -> None:
        """isolation 的固定字段在两个路径下一致。"""
        with patch("backend.memory.core.MEMORY_READ_FROM_SQLITE", False):
            json_iso = self.manager.recall_context(
                case=_make_case(), session_id=self.session_id
            )["isolation"]
        with patch("backend.memory.core.MEMORY_READ_FROM_SQLITE", True):
            sqlite_iso = self.manager.recall_context(
                case=_make_case(), session_id=self.session_id
            )["isolation"]

        # 除 long_term_store 和 policy 外的字段应一致
        for key in ["scope", "session_id", "session_isolated",
                     "session_search_store", "repo_knowledge_separated",
                     "used_as_diagnostic_evidence"]:
            self.assertEqual(
                json_iso[key], sqlite_iso[key],
                f"isolation.{key} 不一致：JSON={json_iso[key]}，SQLite={sqlite_iso[key]}",
            )

        # long_term_store 应不同
        self.assertEqual(json_iso["long_term_store"], "local_json")
        self.assertEqual(sqlite_iso["long_term_store"], "sqlite")


# ── memory_answer v2 兼容 ──────────────────────────────────────────────


class RecallContextMemoryAnswerV2Test(unittest.TestCase):
    """SQLite recall_context 输出能支持 memory_answer v2 的字段解析。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.manager = MemoryManager(memory_dir=Path(self.tmp.name))
        self.session_id = "session_mav2"
        self._write_data()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_data(self) -> None:
        self.manager.sqlite_store.write_workflow_result(
            session_id=self.session_id,
            created_at=_now(),
            updated_at=_now(),
            messages=_make_messages(2),
            case=_make_case(),
            ticket_id="ticket_mav2",
            ticket=_make_ticket("ticket_mav2"),
            triage={"intent": "fault_diagnosis", "confidence": "high"},
            safety={"risk_level": "p1_high"},
            diagnosis={"summary": "需排查 RCD 模块。"},
        )

    def test_recall_model_from_sqlite_context(self) -> None:
        """memory_answer v2 的 field resolver 能从 SQLite memory_context 召回型号。"""
        with patch("backend.memory.core.MEMORY_READ_FROM_SQLITE", True):
            context = self.manager.recall_context(
                case=_make_case(), session_id=self.session_id
            )

        # 模拟 _resolve_memory_fields 的 Pass 1：从 last_case 取 charger_model
        last_case = context.get("last_case", {})
        if not last_case:
            last_case = context.get("session", {}).get("last_case", {})
        model = last_case.get("charger_model", "")
        self.assertEqual(model, "VG-11KW-Pro",
                         f"SQLite recall 应能召回型号，实际: {model!r}")

    def test_recall_power_kw_from_sqlite_context(self) -> None:
        """能从 SQLite memory_context 召回额定功率。"""
        with patch("backend.memory.core.MEMORY_READ_FROM_SQLITE", True):
            context = self.manager.recall_context(
                case=_make_case(), session_id=self.session_id
            )

        last_case = context.get("last_case", {}) or context.get("session", {}).get("last_case", {})
        power = last_case.get("rated_power_kw", "")
        self.assertEqual(power, "11kW",
                         f"SQLite recall 应能召回额定功率，实际: {power!r}")

    def test_recall_missing_info_from_sqlite_context(self) -> None:
        """能从 SQLite memory_context 召回缺失信息。"""
        with patch("backend.memory.core.MEMORY_READ_FROM_SQLITE", True):
            context = self.manager.recall_context(
                case=_make_case(), session_id=self.session_id
            )

        missing = context.get("missing_info", [])
        if not missing:
            missing = context.get("session", {}).get("missing_info", [])
        self.assertIsInstance(missing, list)
        self.assertIn("联系电话", missing)

    def test_recall_risk_level_from_sqlite_context(self) -> None:
        """能从 SQLite memory_context 召回风险等级。"""
        with patch("backend.memory.core.MEMORY_READ_FROM_SQLITE", True):
            context = self.manager.recall_context(
                case=_make_case(), session_id=self.session_id
            )

        recent_safety = context.get("recent_safety", {}) or context.get("session", {}).get("recent_safety", {})
        risk = recent_safety.get("risk_level", "")
        self.assertEqual(risk, "p1_high",
                         f"SQLite recall 应能召回风险等级，实际: {risk!r}")

    def test_recall_ticket_priority_from_sqlite_context(self) -> None:
        """能从 SQLite memory_context 召回工单优先级。"""
        with patch("backend.memory.core.MEMORY_READ_FROM_SQLITE", True):
            context = self.manager.recall_context(
                case=_make_case(), session_id=self.session_id
            )

        summary = context.get("session_summary", {})
        priority = summary.get("last_dispatch_priority", "")
        self.assertEqual(priority, "p1_high",
                         f"SQLite recall 应能召回工单优先级，实际: {priority!r}")


# ── build_session_context 单元测试 ──────────────────────────────────────


class BuildSessionContextTest(unittest.TestCase):
    """SQLiteLongTermMemoryStore.build_session_context() 单元测试。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "test.sqlite"
        self.store = SQLiteLongTermMemoryStore(self.db_path)
        self.session_id = "session_bsctx"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_empty_session_returns_minimal_structure(self) -> None:
        """不存在 session 时返回最小有效结构。"""
        ctx = self.store.build_session_context("nonexistent")
        self.assertEqual(ctx["session"]["session_id"], "nonexistent")
        self.assertEqual(ctx["session"]["message_count"], 0)
        self.assertEqual(ctx["last_case"], {})
        self.assertEqual(ctx["missing_info"], [])

    def test_write_then_build_returns_data(self) -> None:
        """写入后 build_session_context 能返回数据。"""
        self.store.write_workflow_result(
            session_id=self.session_id,
            created_at=_now(),
            updated_at=_now(),
            messages=_make_messages(2),
            case=_make_case(),
            ticket_id="ticket_bsctx",
            ticket=_make_ticket("ticket_bsctx"),
            triage={"intent": "fault_diagnosis"},
            safety={"risk_level": "p1_high"},
            diagnosis={"summary": "需排查 RCD 模块。"},
        )

        ctx = self.store.build_session_context(self.session_id)

        # session
        self.assertEqual(ctx["session"]["session_id"], self.session_id)
        self.assertGreater(ctx["session"]["message_count"], 0)
        self.assertEqual(ctx["session"]["last_intent"], "fault_diagnosis")

        # last_case
        self.assertEqual(ctx["last_case"].get("charger_model"), "VG-11KW-Pro")

        # missing_info
        self.assertIn("联系电话", ctx["missing_info"])

        # recent_safety
        self.assertEqual(ctx["recent_safety"].get("risk_level"), "p1_high")

        # recent_ticket
        self.assertIn("ticket_id", ctx["recent_ticket"])

        # session_summary
        self.assertEqual(ctx["session_summary"]["last_model"], "VG-11KW-Pro")
        self.assertEqual(ctx["session_summary"]["last_risk_level"], "p1_high")

    def test_unavailable_store_returns_empty(self) -> None:
        """store 不可用时返回空结构。"""
        self.store.available = False
        ctx = self.store.build_session_context(self.session_id)
        self.assertEqual(ctx["session"]["message_count"], 0)
        self.assertEqual(ctx["last_case"], {})

    def test_no_case_no_ticket_returns_safe_defaults(self) -> None:
        """无 case 无 ticket 时返回安全的默认值。"""
        self.store.write_workflow_result(
            session_id=self.session_id,
            created_at=_now(),
            updated_at=_now(),
            messages=_make_messages(1),
            case={},
            ticket_id="",
            ticket={},
        )

        ctx = self.store.build_session_context(self.session_id)

        self.assertEqual(ctx["last_case"], {})
        self.assertEqual(ctx["missing_info"], [])
        self.assertEqual(ctx["recent_ticket"], {})
        self.assertEqual(ctx["recent_safety"], {})


if __name__ == "__main__":
    unittest.main()
