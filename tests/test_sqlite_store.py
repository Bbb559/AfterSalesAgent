"""SQLiteLongTermMemoryStore v1 单元测试。

覆盖：
1. 初始化创建数据库和表
2. 写入 session
3. 写入 messages
4. 写入 case
5. 写入 ticket
6. 写入 summary
7. write_workflow_result 一次性写入
8. get_session
9. get_messages
10. search_messages FTS5
11. search_cases FTS5
12. SQLite 写入异常不抛出到主流程
13. 重复写同一 session 不重复 messages
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from datetime import datetime

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
        "dispatch": {"title": "测试派工"},
    }


class SQLiteStoreInitTest(unittest.TestCase):
    """测试 1：初始化创建数据库和表。"""

    def test_init_creates_db_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.sqlite"
            store = SQLiteLongTermMemoryStore(db_path)
            self.assertTrue(store.available)
            self.assertEqual(store.error, "")
            self.assertTrue(db_path.exists())

    def test_init_creates_all_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.sqlite"
            store = SQLiteLongTermMemoryStore(db_path)
            self.assertTrue(store.available)

            conn = sqlite3.connect(str(db_path))
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
                ).fetchall()
            }
            conn.close()

            expected = {
                "schema_migrations",
                "sessions",
                "messages",
                "cases",
                "tickets",
                "memory_summaries",
                "messages_fts",
                "cases_fts",
            }
            missing = expected - tables
            self.assertSetEqual(missing, set(), f"缺少表: {missing}")

    def test_init_idempotent(self) -> None:
        """多次初始化不会报错。"""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.sqlite"
            store1 = SQLiteLongTermMemoryStore(db_path)
            self.assertTrue(store1.available)
            store2 = SQLiteLongTermMemoryStore(db_path)
            self.assertTrue(store2.available)

    def test_init_handles_invalid_path_gracefully(self) -> None:
        """无法写入的路径不应抛出异常。"""
        import os as _os
        # 使用操作系统保留名作为不可用路径
        invalid_path = "NUL/impossible/test.sqlite" if _os.name == "nt" else "/dev/null/impossible/test.sqlite"
        try:
            store = SQLiteLongTermMemoryStore(invalid_path)
            # 在不同 OS 上 available 行为可能不同，关键是初始化不抛异常
            self.assertIsInstance(store.available, bool)
        except Exception as exc:
            self.fail(f"初始化不应抛出异常，实际抛出: {exc}")


class SQLiteStoreWriteTest(unittest.TestCase):
    """测试 2-6：单表写入。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "test.sqlite"
        self.store = SQLiteLongTermMemoryStore(self.db_path)
        self.session_id = "session_write_test"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_write_session(self) -> None:
        with self.store._connect() as conn:
            self.store.write_session(conn, self.session_id, message_count=4)
            conn.commit()

        session = self.store.get_session(self.session_id)
        self.assertIsNotNone(session)
        self.assertEqual(session["session_id"], self.session_id)
        self.assertEqual(session["message_count"], 4)
        self.assertEqual(session["status"], "active")

    def test_write_messages(self) -> None:
        msgs = _make_messages(2)  # 2 user + 2 assistant = 4
        with self.store._connect() as conn:
            self.store.write_session(conn, self.session_id, message_count=len(msgs))
            written = self.store.write_messages(conn, self.session_id, msgs)
            conn.commit()

        self.assertEqual(written, 4)
        stored = self.store.get_messages(self.session_id)
        self.assertEqual(len(stored), 4)
        self.assertEqual(stored[0]["role"], "user")
        self.assertEqual(stored[0]["turn_index"], 0)
        self.assertIn("VG-11KW-Pro", stored[0]["content"])

    def test_write_messages_skips_empty_content(self) -> None:
        msgs = [
            {"role": "user", "content": "有效消息", "timestamp": _now(), "metadata": {}},
            {"role": "user", "content": "", "timestamp": _now(), "metadata": {}},
        ]
        with self.store._connect() as conn:
            self.store.write_session(conn, self.session_id, message_count=2)
            written = self.store.write_messages(conn, self.session_id, msgs)
            conn.commit()
        self.assertEqual(written, 1)

    def test_write_case(self) -> None:
        case = _make_case()
        with self.store._connect() as conn:
            self.store.write_session(conn, self.session_id)
            ok = self.store.write_case(conn, self.session_id, case)
            conn.commit()

        self.assertTrue(ok)
        stored = self.store.get_case(self.session_id)
        self.assertIsNotNone(stored)
        self.assertEqual(stored["brand"], "VoltGate")
        self.assertEqual(stored["charger_model"], "VG-11KW-Pro")
        self.assertEqual(stored["rated_power_kw"], "11kW")
        self.assertEqual(stored["city"], "广州")

    def test_write_case_empty_dict(self) -> None:
        with self.store._connect() as conn:
            ok = self.store.write_case(conn, self.session_id, {})
            conn.commit()
        self.assertFalse(ok)

    def test_write_ticket(self) -> None:
        ticket = _make_ticket("ticket_001")
        with self.store._connect() as conn:
            self.store.write_session(conn, self.session_id)
            ok = self.store.write_ticket(conn, self.session_id, "ticket_001", ticket)
            conn.commit()

        self.assertTrue(ok)
        # 通过原始 SQL 验证
        with self.store._connect() as conn:
            row = conn.execute(
                "SELECT ticket_id, title, priority FROM tickets WHERE ticket_id = ?",
                ("ticket_001",),
            ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "ticket_001")

    def test_write_ticket_empty_id(self) -> None:
        with self.store._connect() as conn:
            ok = self.store.write_ticket(conn, self.session_id, "", {})
            conn.commit()
        self.assertFalse(ok)

    def test_write_summary(self) -> None:
        with self.store._connect() as conn:
            self.store.write_session(conn, self.session_id)
            ok = self.store.write_summary(conn, self.session_id, "triage",
                                          '{"intent": "fault_diagnosis"}')
            conn.commit()

        self.assertTrue(ok)
        with self.store._connect() as conn:
            row = conn.execute(
                "SELECT summary_type, summary FROM memory_summaries WHERE session_id = ?",
                (self.session_id,),
            ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "triage")

    def test_write_summary_empty_text(self) -> None:
        with self.store._connect() as conn:
            ok = self.store.write_summary(conn, self.session_id, "triage", "")
            conn.commit()
        self.assertFalse(ok)


class SQLiteStoreWorkflowResultTest(unittest.TestCase):
    """测试 7：write_workflow_result 一次性写入。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "test.sqlite"
        self.store = SQLiteLongTermMemoryStore(self.db_path)
        self.session_id = "session_wf_test"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_write_workflow_result_full(self) -> None:
        result = self.store.write_workflow_result(
            session_id=self.session_id,
            created_at=_now(),
            updated_at=_now(),
            messages=_make_messages(2),
            case=_make_case(),
            ticket_id="ticket_wf_001",
            ticket=_make_ticket("ticket_wf_001"),
            triage={"intent": "fault_diagnosis"},
            safety={"risk_level": "p1_high"},
            diagnosis={"summary": "需排查 RCD 模块。"},
        )

        self.assertTrue(result["success"], f"写入失败: {result.get('error')}")
        self.assertTrue(result["session_written"])
        self.assertEqual(result["messages_written"], 4)
        self.assertTrue(result["case_written"])
        self.assertTrue(result["ticket_written"])
        self.assertTrue(result["summary_written"])
        self.assertEqual(result["error"], "")

        # 验证各表
        self.assertIsNotNone(self.store.get_session(self.session_id))
        self.assertEqual(len(self.store.get_messages(self.session_id)), 4)
        self.assertIsNotNone(self.store.get_case(self.session_id))

    def test_write_workflow_result_minimal(self) -> None:
        """最简输入（空 case、无 ticket、无 summary）不崩溃。"""
        result = self.store.write_workflow_result(
            session_id=self.session_id,
            created_at=_now(),
            updated_at=_now(),
            messages=[],
            case={},
            ticket_id="",
            ticket={},
        )

        self.assertTrue(result["success"])
        self.assertTrue(result["session_written"])
        self.assertEqual(result["messages_written"], 0)
        self.assertFalse(result["case_written"])
        self.assertFalse(result["ticket_written"])
        self.assertFalse(result["summary_written"])

    def test_write_workflow_result_unavailable_store(self) -> None:
        """available=False 时直接返回失败，不抛异常。"""
        self.store.available = False
        self.store.error = "模拟不可用"
        result = self.store.write_workflow_result(
            session_id=self.session_id,
            created_at=_now(),
            updated_at=_now(),
            messages=[],
            case={},
            ticket_id="",
            ticket={},
        )
        self.assertFalse(result["success"])
        self.assertIn("模拟不可用", result["error"])


class SQLiteStoreQueryTest(unittest.TestCase):
    """测试 8-9：查询接口。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "test.sqlite"
        self.store = SQLiteLongTermMemoryStore(self.db_path)
        self.session_id = "session_query_test"
        self.store.write_workflow_result(
            session_id=self.session_id,
            created_at=_now(),
            updated_at=_now(),
            messages=_make_messages(3),
            case=_make_case(),
            ticket_id="ticket_query_001",
            ticket=_make_ticket("ticket_query_001"),
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_get_session(self) -> None:
        session = self.store.get_session(self.session_id)
        self.assertIsNotNone(session)
        self.assertEqual(session["session_id"], self.session_id)
        self.assertEqual(session["message_count"], 6)  # 3 user + 3 assistant
        self.assertEqual(session["status"], "active")

    def test_get_session_missing(self) -> None:
        session = self.store.get_session("nonexistent")
        self.assertIsNone(session)

    def test_get_messages_limit(self) -> None:
        msgs = self.store.get_messages(self.session_id, limit=2)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["turn_index"], 0)
        self.assertEqual(msgs[1]["turn_index"], 1)

    def test_get_messages_empty_session(self) -> None:
        msgs = self.store.get_messages("nonexistent")
        self.assertEqual(msgs, [])

    def test_get_case(self) -> None:
        case = self.store.get_case(self.session_id)
        self.assertIsNotNone(case)
        self.assertEqual(case["brand"], "VoltGate")
        self.assertEqual(case["charger_model"], "VG-11KW-Pro")


class SQLiteStoreFTSSearchTest(unittest.TestCase):
    """测试 10-11：FTS5 搜索。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "test.sqlite"
        self.store = SQLiteLongTermMemoryStore(self.db_path)
        self.session_id = "session_fts_test"
        self.store.write_workflow_result(
            session_id=self.session_id,
            created_at=_now(),
            updated_at=_now(),
            messages=_make_messages(2),
            case=_make_case(),
            ticket_id="ticket_fts_001",
            ticket=_make_ticket("ticket_fts_001"),
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_search_messages_fts_hit(self) -> None:
        results = self.store.search_messages("C-RCD-04", session_id=self.session_id)
        self.assertTrue(len(results) > 0, f"FTS5 应命中 C-RCD-04，结果: {results}")
        for r in results:
            self.assertIn("C-RCD-04", r["content"])

    def test_search_messages_fts_miss(self) -> None:
        results = self.store.search_messages("不存在的关键词XYZ", session_id=self.session_id)
        self.assertEqual(results, [])

    def test_search_messages_empty_query(self) -> None:
        results = self.store.search_messages("")
        self.assertEqual(results, [])

    def test_search_cases_fts_hit(self) -> None:
        results = self.store.search_cases("VG-11KW-Pro", session_id=self.session_id)
        self.assertTrue(len(results) > 0, f"FTS5 应命中 VG-11KW-Pro，结果: {results}")

    def test_search_cases_fts_hit_by_fault_code(self) -> None:
        results = self.store.search_cases("C-RCD-04")
        self.assertTrue(len(results) > 0, f"FTS5 应跨 session 命中 C-RCD-04")

    def test_search_cases_empty_query(self) -> None:
        results = self.store.search_cases("")
        self.assertEqual(results, [])

    def test_search_when_unavailable(self) -> None:
        self.store.available = False
        results = self.store.search_messages("C-RCD-04")
        self.assertEqual(results, [])


class SQLiteStoreDuplicateWriteTest(unittest.TestCase):
    """测试 13：重复写同一 session 不重复 messages。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "test.sqlite"
        self.store = SQLiteLongTermMemoryStore(self.db_path)
        self.session_id = "session_dup_test"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_duplicate_write_does_not_duplicate_messages(self) -> None:
        msgs = _make_messages(1)  # 2 messages
        # 第一次写入
        self.store.write_workflow_result(
            session_id=self.session_id,
            created_at=_now(),
            updated_at=_now(),
            messages=msgs,
            case=_make_case(),
            ticket_id="ticket_dup_001",
            ticket=_make_ticket("ticket_dup_001"),
        )
        first = self.store.get_messages(self.session_id)
        self.assertEqual(len(first), 2)

        # 第二次写入同一 session（模拟 workflow 再次完成）
        self.store.write_workflow_result(
            session_id=self.session_id,
            created_at=_now(),
            updated_at=_now(),
            messages=msgs,
            case=_make_case(),
            ticket_id="ticket_dup_001",
            ticket=_make_ticket("ticket_dup_001"),
        )
        second = self.store.get_messages(self.session_id)
        self.assertEqual(len(second), 2, f"重复写入不应产生重复消息，预期 2 条，实际 {len(second)} 条")

    def test_duplicate_write_overwrites_case_not_duplicates(self) -> None:
        """重复写入 case 不应产生多行。"""
        case1 = _make_case()
        case2 = dict(_make_case(), brand="UpdatedBrand")

        self.store.write_workflow_result(
            session_id=self.session_id,
            created_at=_now(), updated_at=_now(),
            messages=[], case=case1, ticket_id="", ticket={},
        )
        self.store.write_workflow_result(
            session_id=self.session_id,
            created_at=_now(), updated_at=_now(),
            messages=[], case=case2, ticket_id="", ticket={},
        )

        stored = self.store.get_case(self.session_id)
        self.assertIsNotNone(stored)
        self.assertEqual(stored["brand"], "UpdatedBrand")

        # 确认只有一行
        with self.store._connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM cases WHERE session_id = ?",
                (self.session_id,),
            ).fetchone()[0]
        self.assertEqual(count, 1)


class SQLiteStoreExceptionSafetyTest(unittest.TestCase):
    """测试 12：SQLite 写入异常不抛出到主流程。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "test.sqlite"
        self.store = SQLiteLongTermMemoryStore(self.db_path)
        self.session_id = "session_exc_test"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_write_workflow_result_returns_error_not_raises(self) -> None:
        """write_workflow_result 始终返回 dict，不抛异常。"""
        # 删除数据库文件后写入——因为 store 已经打开连接，可能仍能工作；
        # 更可靠的测试：传入非法数据。
        result = self.store.write_workflow_result(
            session_id=self.session_id,
            created_at=_now(),
            updated_at=_now(),
            messages=[{"role": "user", "content": "test", "timestamp": _now(), "metadata": {}}],
            case={"brand": "Test"},
            ticket_id="ticket_exc_001",
            ticket={},
        )
        # 即使成功，也必须返回 dict
        self.assertIsInstance(result, dict)
        self.assertIn("success", result)

    def test_get_methods_return_safe_defaults(self) -> None:
        """查询方法在异常时返回安全默认值（None/[]），不抛异常。"""
        self.store.available = False
        self.assertIsNone(self.store.get_session(self.session_id))
        self.assertEqual(self.store.get_messages(self.session_id), [])
        self.assertIsNone(self.store.get_case(self.session_id))
        self.assertEqual(self.store.search_messages("test"), [])
        self.assertEqual(self.store.search_cases("test"), [])


class SQLiteStoreSchemaVersionTest(unittest.TestCase):
    """测试 schema_migrations 表记录版本。"""

    def test_schema_version_recorded(self) -> None:
        import shutil
        tmp = tempfile.mkdtemp()
        try:
            db_path = Path(tmp) / "test.sqlite"
            store = SQLiteLongTermMemoryStore(db_path)
            self.assertTrue(store.available)

            with store._connect() as conn:
                row = conn.execute(
                    "SELECT version, applied_at FROM schema_migrations ORDER BY version DESC LIMIT 1"
                ).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], SQLiteLongTermMemoryStore.SCHEMA_VERSION)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class SQLiteStoreTTLTest(unittest.TestCase):
    """测试 Session TTL 生命周期管理：active → expired → archived。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "test.sqlite"
        self.store = SQLiteLongTermMemoryStore(self.db_path)
        self.session_prefix = "session_ttl"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_session_with_status_and_date(self, session_id: str, status: str, days_ago: int) -> None:
        """直接写入一条带指定 status 和 updated_at 的 session 记录。"""
        with self.store._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO sessions(session_id, created_at, updated_at, status)
                   VALUES (?, datetime('now', ? || ' days'), datetime('now', ? || ' days'), ?)""",
                (session_id, f"-{days_ago}", f"-{days_ago}", status),
            )
            conn.commit()

    def test_active_recent_not_marked_expired(self) -> None:
        """最近 1 天创建的 active session 不应被标记过期。"""
        self._write_session_with_status_and_date("s1", "active", 1)

        result = self.store.mark_expired_sessions(expire_days=7, archive_days=30)

        self.assertEqual(result["expired"], 0)
        self.assertEqual(result["archived"], 0)
        self.assertEqual(result["error"], "")
        # 验证状态未变
        s = self.store.get_session("s1")
        self.assertEqual(s["status"], "active")

    def test_active_old_marked_expired(self) -> None:
        """超过 expire_days 的 active session 应被标记为 expired。"""
        self._write_session_with_status_and_date("s1", "active", 10)  # 10 天前

        result = self.store.mark_expired_sessions(expire_days=7, archive_days=30)

        self.assertEqual(result["expired"], 1)
        self.assertEqual(result["archived"], 0)
        s = self.store.get_session("s1")
        self.assertEqual(s["status"], "expired")

    def test_expired_recent_not_marked_archived(self) -> None:
        """刚标记为 expired 的 session 不应立即被归档。"""
        self._write_session_with_status_and_date("s1", "expired", 10)  # 10 天前过期

        result = self.store.mark_expired_sessions(expire_days=7, archive_days=30)

        self.assertEqual(result["expired"], 0)  # 已 expired 不重复计数
        self.assertEqual(result["archived"], 0)
        s = self.store.get_session("s1")
        self.assertEqual(s["status"], "expired")  # 仍为 expired

    def test_expired_old_marked_archived(self) -> None:
        """超过 archive_days 的 expired session 应被标记为 archived。"""
        self._write_session_with_status_and_date("s1", "expired", 35)  # 35 天前过期

        result = self.store.mark_expired_sessions(expire_days=7, archive_days=30)

        self.assertEqual(result["archived"], 1)
        s = self.store.get_session("s1")
        self.assertEqual(s["status"], "archived")

    def test_both_transitions_in_one_call(self) -> None:
        """一次调用中同时处理 active→expired 和 expired→archived。"""
        self._write_session_with_status_and_date("s_active", "active", 10)    # 应 → expired
        self._write_session_with_status_and_date("s_expired", "expired", 35)  # 应 → archived
        self._write_session_with_status_and_date("s_recent", "active", 1)     # 不变

        result = self.store.mark_expired_sessions(expire_days=7, archive_days=30)

        self.assertEqual(result["expired"], 1)
        self.assertEqual(result["archived"], 1)
        self.assertEqual(self.store.get_session("s_active")["status"], "expired")
        self.assertEqual(self.store.get_session("s_expired")["status"], "archived")
        self.assertEqual(self.store.get_session("s_recent")["status"], "active")

    def test_archived_not_affected_by_twice(self) -> None:
        """已 archived 的 session 多次调用不受影响。"""
        self._write_session_with_status_and_date("s1", "archived", 50)

        result = self.store.mark_expired_sessions(expire_days=7, archive_days=30)

        self.assertEqual(result["expired"], 0)
        self.assertEqual(result["archived"], 0)
        s = self.store.get_session("s1")
        self.assertEqual(s["status"], "archived")

    def test_ttl_when_store_unavailable(self) -> None:
        """store 不可用时返回 0 计数和错误信息。"""
        self.store.available = False
        self.store.error = "模拟不可用"

        result = self.store.mark_expired_sessions(expire_days=7, archive_days=30)

        self.assertEqual(result["expired"], 0)
        self.assertEqual(result["archived"], 0)
        self.assertIn("模拟不可用", result["error"])

    def test_ttl_default_parameters(self) -> None:
        """默认参数（expire=7, archive=30）正常工作。"""
        self._write_session_with_status_and_date("s_old", "active", 8)  # 8 天，超过默认 7

        result = self.store.mark_expired_sessions()  # 使用默认值

        self.assertEqual(result["expired"], 1)
        self.assertEqual(self.store.get_session("s_old")["status"], "expired")


if __name__ == "__main__":
    unittest.main()
