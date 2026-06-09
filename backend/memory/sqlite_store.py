"""SQLite 长期记忆存储 v1。

基于项目阶段计划（CLAUDE.md §10）：
- 第一版只处理 sessions / messages / cases / tickets / memory_summaries。
- FTS5 先覆盖 messages 和 cases。
- 仅做追加写入，不删除 JSON 主存储。
- 写入失败不阻塞主流程。

读取策略（阶段 3 灰度中）：
- 默认：JSON 为主读路径，SQLite 仅用于调试 API（GET /api/memory/sessions/...）。
- 设置 MEMORY_READ_FROM_SQLITE=true 后，recall_context() 优先从 SQLite
  读取 session/case/ticket 维度数据；读取失败时静默回退 JSON。
- Customer/Charger/Site 维度仍走 JSON（SQLite 尚无对应表）。
- build_session_context() 方法实现了从 SQLite 多表联合查询重建
  memory_context 结构的兼容层。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class SQLiteLongTermMemoryStore:
    """SQLite 长期记忆存储。

    职责：
    1. 初始化 SQLite 数据库并建表。
    2. 写入 session / message / case / ticket / summary。
    3. 查询当前 session 记忆。
    4. 提供 FTS5 搜索接口。

    JSON 仍然是 Source of Truth。本类只做追加写入，读取优先走 JSON。
    """

    # 当前 schema 版本，用于 schema_migrations 表。
    SCHEMA_VERSION = 1

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.available = True
        self.error = ""
        try:
            self._ensure_schema()
        except Exception as exc:
            self.available = False
            self.error = str(exc)

    # ------------------------------------------------------------------
    # 数据库连接
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=OFF")
        return conn

    # ------------------------------------------------------------------
    # Schema 初始化
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        """建表，幂等。"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(self._schema_sql())
            # 记录 schema 版本（幂等）
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                (self.SCHEMA_VERSION, _now()),
            )

    def _schema_sql(self) -> str:
        return f"""
        -- 版本管理
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version     INTEGER PRIMARY KEY,
            applied_at  TEXT NOT NULL
        );

        -- 会话表
        CREATE TABLE IF NOT EXISTS sessions (
            session_id      TEXT PRIMARY KEY,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'active',
            source          TEXT NOT NULL DEFAULT 'web',
            message_count   INTEGER NOT NULL DEFAULT 0,
            metadata_json   TEXT NOT NULL DEFAULT '{{}}'
        );

        -- 消息表
        CREATE TABLE IF NOT EXISTS messages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT NOT NULL,
            turn_index      INTEGER NOT NULL DEFAULT 0,
            role            TEXT NOT NULL,
            content         TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            metadata_json   TEXT NOT NULL DEFAULT '{{}}'
        );
        CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages(session_id, turn_index);

        -- Case 表（展开高频查询字段 + 保留原始 JSON）
        CREATE TABLE IF NOT EXISTS cases (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id          TEXT NOT NULL,
            brand               TEXT NOT NULL DEFAULT '',
            charger_model       TEXT NOT NULL DEFAULT '',
            rated_power_kw      TEXT NOT NULL DEFAULT '',
            city                TEXT NOT NULL DEFAULT '',
            contact_address     TEXT NOT NULL DEFAULT '',
            install_time        TEXT NOT NULL DEFAULT '',
            fault_codes_json    TEXT NOT NULL DEFAULT '[]',
            symptoms_json       TEXT NOT NULL DEFAULT '[]',
            missing_info_json   TEXT NOT NULL DEFAULT '[]',
            raw_case_json       TEXT NOT NULL DEFAULT '{{}}',
            created_at          TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_cases_session
            ON cases(session_id);

        -- 工单表
        CREATE TABLE IF NOT EXISTS tickets (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT NOT NULL,
            ticket_id       TEXT NOT NULL UNIQUE,
            title           TEXT NOT NULL DEFAULT '',
            priority        TEXT NOT NULL DEFAULT '',
            status          TEXT NOT NULL DEFAULT 'draft',
            raw_ticket_json TEXT NOT NULL DEFAULT '{{}}',
            created_at      TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_tickets_session
            ON tickets(session_id);

        -- 记忆摘要表
        CREATE TABLE IF NOT EXISTS memory_summaries (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT NOT NULL,
            summary_type    TEXT NOT NULL,
            summary         TEXT NOT NULL,
            source_refs_json TEXT NOT NULL DEFAULT '[]',
            created_at      TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_summaries_session
            ON memory_summaries(session_id, summary_type);

        -- FTS5：消息全文搜索
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
            session_id UNINDEXED,
            turn_index UNINDEXED,
            role UNINDEXED,
            content,
            created_at UNINDEXED
        );

        -- FTS5：Case 全文搜索
        CREATE VIRTUAL TABLE IF NOT EXISTS cases_fts USING fts5(
            session_id UNINDEXED,
            brand,
            charger_model,
            fault_codes,
            symptoms,
            city
        );
        """

    # ------------------------------------------------------------------
    # 单表写入
    # ------------------------------------------------------------------

    def write_session(self, conn: sqlite3.Connection, session_id: str,
                      created_at: str = "", updated_at: str = "",
                      message_count: int = 0, metadata_json: str = "{}") -> bool:
        """写入 / 更新 session 记录。"""
        now = _now()
        conn.execute(
            """
            INSERT OR REPLACE INTO sessions
                (session_id, created_at, updated_at, status, source, message_count, metadata_json)
            VALUES (?, ?, ?, 'active', 'web', ?, ?)
            """,
            (
                session_id,
                created_at or now,
                updated_at or now,
                message_count,
                metadata_json,
            ),
        )
        return True

    def write_messages(self, conn: sqlite3.Connection, session_id: str,
                       messages: list[dict[str, Any]]) -> int:
        """清空并重建当前 session 的 messages 和 messages_fts 索引。

        策略：先 DELETE 再 INSERT，避免重复写入同一 session 的消息。
        """
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM messages_fts WHERE session_id = ?", (session_id,))
        written = 0
        for i, msg in enumerate(messages):
            role = str(msg.get("role", "") or "")
            content = str(msg.get("content", "") or "")
            if not content:
                continue
            timestamp = str(msg.get("timestamp", "") or _now())
            metadata = msg.get("metadata", {}) or {}
            metadata_str = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
            conn.execute(
                """
                INSERT INTO messages(session_id, turn_index, role, content, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, i, role, content, timestamp, metadata_str),
            )
            conn.execute(
                """
                INSERT INTO messages_fts(session_id, turn_index, role, content, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, i, role, content, timestamp),
            )
            written += 1
        return written

    def write_case(self, conn: sqlite3.Connection, session_id: str,
                   case: dict[str, Any]) -> bool:
        """写入 / 替换当前 session 的 case 记录和 cases_fts 索引。"""
        if not case:
            return False
        conn.execute("DELETE FROM cases WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM cases_fts WHERE session_id = ?", (session_id,))

        now = _now()
        brand = str(case.get("brand", "") or "")
        charger_model = str(case.get("charger_model", "") or "")
        rated_power_kw = str(case.get("rated_power_kw", "") or "")
        city = str(case.get("city", "") or "")
        contact_address = str(case.get("contact_address", "") or "")
        install_time = str(case.get("install_time", "") or case.get("purchase_or_install_time", "") or "")
        fault_codes_json = json.dumps(case.get("fault_codes", []) or [], ensure_ascii=False)
        symptoms_json = json.dumps(case.get("observed_symptoms", []) or [], ensure_ascii=False)
        missing_info_json = json.dumps(case.get("missing_info", []) or [], ensure_ascii=False)
        raw_case_json = json.dumps(case, ensure_ascii=False, sort_keys=True)

        conn.execute(
            """
            INSERT INTO cases(session_id, brand, charger_model, rated_power_kw,
                               city, contact_address, install_time,
                               fault_codes_json, symptoms_json, missing_info_json,
                               raw_case_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id, brand, charger_model, rated_power_kw,
                city, contact_address, install_time,
                fault_codes_json, symptoms_json, missing_info_json,
                raw_case_json, now,
            ),
        )

        # FTS5 索引：拼接可搜索文本
        fts_fault_codes = " ".join(case.get("fault_codes", []) or [])
        fts_symptoms = " ".join(case.get("observed_symptoms", []) or [])
        conn.execute(
            """
            INSERT INTO cases_fts(session_id, brand, charger_model, fault_codes, symptoms, city)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, brand, charger_model, fts_fault_codes, fts_symptoms, city),
        )
        return True

    def write_ticket(self, conn: sqlite3.Connection, session_id: str,
                     ticket_id: str, ticket: dict[str, Any]) -> bool:
        """写入 / 替换工单记录。"""
        if not ticket_id:
            return False
        title = str(ticket.get("title", "") or "")
        priority = str(ticket.get("priority", "") or "")
        status = str(ticket.get("status", "") or "draft")
        raw_ticket_json = json.dumps(ticket, ensure_ascii=False, sort_keys=True)
        created_at = str(ticket.get("created_at", "") or _now())
        conn.execute(
            """
            INSERT OR REPLACE INTO tickets(session_id, ticket_id, title, priority, status,
                                            raw_ticket_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, ticket_id, title, priority, status, raw_ticket_json, created_at),
        )
        return True

    def write_summary(self, conn: sqlite3.Connection, session_id: str,
                      summary_type: str, summary_text: str,
                      source_refs: list[str] | None = None) -> bool:
        """追加一条记忆摘要。"""
        if not summary_text:
            return False
        source_refs_json = json.dumps(source_refs or [], ensure_ascii=False)
        conn.execute(
            """
            INSERT INTO memory_summaries(session_id, summary_type, summary, source_refs_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, summary_type, summary_text, source_refs_json, _now()),
        )
        return True

    # ------------------------------------------------------------------
    # 批量写入（workflow 结果一次性落库）
    # ------------------------------------------------------------------

    def write_workflow_result(
        self,
        session_id: str,
        created_at: str,
        updated_at: str,
        messages: list[dict[str, Any]],
        case: dict[str, Any],
        ticket_id: str,
        ticket: dict[str, Any],
        triage: dict[str, Any] | None = None,
        safety: dict[str, Any] | None = None,
        diagnosis: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """将一次 workflow 的完整结果写入 SQLite。

        所有写入在一个事务内完成；任一子步骤失败则整体回滚，不落部分数据。
        返回结构化结果，调用方不需要解析异常。
        """
        if not self.available:
            return {
                "success": False,
                "session_written": False,
                "messages_written": 0,
                "case_written": False,
                "ticket_written": False,
                "summary_written": False,
                "error": self.error or "SQLite store unavailable",
            }

        try:
            with self._connect() as conn:
                # 1. session
                self.write_session(
                    conn, session_id,
                    created_at=created_at,
                    updated_at=updated_at,
                    message_count=len(messages),
                )

                # 2. messages（清空后重建）
                messages_written = self.write_messages(conn, session_id, messages)

                # 3. case
                case_written = self.write_case(conn, session_id, case)

                # 4. ticket
                ticket_written = self.write_ticket(conn, session_id, ticket_id, ticket)

                # 5. summaries（每类一条）
                summary_count = 0
                if triage:
                    triage_text = json.dumps(triage, ensure_ascii=False, sort_keys=True)
                    if self.write_summary(conn, session_id, "triage", triage_text):
                        summary_count += 1
                if safety:
                    safety_text = json.dumps(safety, ensure_ascii=False, sort_keys=True)
                    if self.write_summary(conn, session_id, "safety", safety_text):
                        summary_count += 1
                if diagnosis:
                    diagnosis_text = json.dumps(diagnosis, ensure_ascii=False, sort_keys=True)
                    if self.write_summary(conn, session_id, "diagnosis", diagnosis_text):
                        summary_count += 1

                conn.commit()

            return {
                "success": True,
                "session_written": True,
                "messages_written": messages_written,
                "case_written": case_written,
                "ticket_written": ticket_written,
                "summary_written": summary_count > 0,
                "summary_count": summary_count,
                "error": "",
            }
        except Exception as exc:
            return {
                "success": False,
                "session_written": False,
                "messages_written": 0,
                "case_written": False,
                "ticket_written": False,
                "summary_written": False,
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # 查询接口（只读）
    # ------------------------------------------------------------------

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """查询单个 session 记录。"""
        if not self.available:
            return None
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT session_id, created_at, updated_at, status, source, message_count, metadata_json "
                    "FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
            if row is None:
                return None
            return {
                "session_id": row[0],
                "created_at": row[1],
                "updated_at": row[2],
                "status": row[3],
                "source": row[4],
                "message_count": row[5],
                "metadata_json": row[6],
            }
        except Exception:
            return None

    def get_messages(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """查询某个 session 的消息列表（按 turn_index 排序）。"""
        if not self.available:
            return []
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT id, session_id, turn_index, role, content, created_at, metadata_json "
                    "FROM messages WHERE session_id = ? ORDER BY turn_index ASC LIMIT ?",
                    (session_id, limit),
                ).fetchall()
            return [
                {
                    "id": row[0],
                    "session_id": row[1],
                    "turn_index": row[2],
                    "role": row[3],
                    "content": row[4],
                    "created_at": row[5],
                    "metadata_json": row[6],
                }
                for row in rows
            ]
        except Exception:
            return []

    def get_case(self, session_id: str) -> dict[str, Any] | None:
        """查询某个 session 的 case 记录。"""
        if not self.available:
            return None
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT id, session_id, brand, charger_model, rated_power_kw, "
                    "city, contact_address, install_time, fault_codes_json, symptoms_json, "
                    "missing_info_json, raw_case_json, created_at "
                    "FROM cases WHERE session_id = ? ORDER BY id DESC LIMIT 1",
                    (session_id,),
                ).fetchone()
            if row is None:
                return None
            return {
                "id": row[0],
                "session_id": row[1],
                "brand": row[2],
                "charger_model": row[3],
                "rated_power_kw": row[4],
                "city": row[5],
                "contact_address": row[6],
                "install_time": row[7],
                "fault_codes_json": row[8],
                "symptoms_json": row[9],
                "missing_info_json": row[10],
                "raw_case_json": row[11],
                "created_at": row[12],
            }
        except Exception:
            return None

    def search_messages(self, query: str, session_id: str = "",
                        limit: int = 5) -> list[dict[str, Any]]:
        """FTS5 搜索 messages。"""
        if not self.available or not query.strip():
            return []
        try:
            with self._connect() as conn:
                terms = self._build_fts_query(query)
                if not terms:
                    return []
                where = "messages_fts MATCH ?"
                params: list[Any] = [terms]
                if session_id:
                    where += " AND session_id = ?"
                    params.append(session_id)
                params.append(limit)
                rows = conn.execute(
                    f"SELECT session_id, turn_index, role, content, created_at "
                    f"FROM messages_fts WHERE {where} LIMIT ?",
                    params,
                ).fetchall()
            return [
                {
                    "session_id": row[0],
                    "turn_index": row[1],
                    "role": row[2],
                    "content": row[3],
                    "created_at": row[4],
                }
                for row in rows
            ]
        except Exception:
            return []

    def search_cases(self, query: str, session_id: str = "",
                     limit: int = 5) -> list[dict[str, Any]]:
        """FTS5 搜索 cases。"""
        if not self.available or not query.strip():
            return []
        try:
            with self._connect() as conn:
                terms = self._build_fts_query(query)
                if not terms:
                    return []
                where = "cases_fts MATCH ?"
                params: list[Any] = [terms]
                if session_id:
                    where += " AND session_id = ?"
                    params.append(session_id)
                params.append(limit)
                rows = conn.execute(
                    f"SELECT session_id, brand, charger_model, fault_codes, symptoms, city "
                    f"FROM cases_fts WHERE {where} LIMIT ?",
                    params,
                ).fetchall()
            return [
                {
                    "session_id": row[0],
                    "brand": row[1],
                    "charger_model": row[2],
                    "fault_codes": row[3],
                    "symptoms": row[4],
                    "city": row[5],
                }
                for row in rows
            ]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # memory_context 重建（阶段 3：SQLite 主读兼容层）
    # ------------------------------------------------------------------

    def build_session_context(self, session_id: str) -> dict[str, Any]:
        """从 SQLite 多表联合查询重建 session/case/ticket 维度的 memory_context。

        返回结构与 MemoryManager._session_summary() + recall_context() 的
        session/case/ticket 维度兼容。
        Customer/Charger/Site 维度不在本方法范围内（SQLite 尚无对应表）。
        """
        if not self.available:
            return self._empty_session_context(session_id)

        try:
            with self._connect() as conn:
                # 1. session 基本信息
                session_row = conn.execute(
                    "SELECT session_id, created_at, updated_at, message_count, metadata_json "
                    "FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()

                if session_row is None:
                    return self._empty_session_context(session_id)

                message_count = int(session_row[3] or 0)
                updated_at = str(session_row[2] or "")

                # 2. 最近 3 条用户消息
                user_rows = conn.execute(
                    "SELECT content FROM messages "
                    "WHERE session_id = ? AND role = 'user' "
                    "ORDER BY turn_index DESC LIMIT 3",
                    (session_id,),
                ).fetchall()
                recent_user_messages = [str(row[0] or "") for row in reversed(user_rows)]

                # 3. case（最近一条）
                case_row = conn.execute(
                    "SELECT raw_case_json, missing_info_json FROM cases "
                    "WHERE session_id = ? ORDER BY id DESC LIMIT 1",
                    (session_id,),
                ).fetchone()
                last_case: dict[str, Any] = {}
                missing_info: list[str] = []
                if case_row is not None:
                    try:
                        last_case = json.loads(str(case_row[0] or "{}"))
                    except Exception:
                        last_case = {}
                    try:
                        parsed_missing = json.loads(str(case_row[1] or "[]"))
                        missing_info = parsed_missing if isinstance(parsed_missing, list) else []
                    except Exception:
                        missing_info = []

                # 4. memory_summaries（triage / safety / diagnosis）
                summary_rows = conn.execute(
                    "SELECT summary_type, summary FROM memory_summaries "
                    "WHERE session_id = ? AND summary_type IN ('triage', 'safety', 'diagnosis') "
                    "ORDER BY id DESC",
                    (session_id,),
                ).fetchall()

                triage_summary: dict[str, Any] = {}
                safety_summary: dict[str, Any] = {}
                diagnosis_summary: dict[str, Any] = {}
                seen_types: set[str] = set()
                for s_type, s_text in summary_rows:
                    s_type = str(s_type or "")
                    if s_type in seen_types:
                        continue
                    seen_types.add(s_type)
                    try:
                        parsed = json.loads(str(s_text or "{}"))
                    except Exception:
                        parsed = {}
                    if s_type == "triage":
                        triage_summary = parsed
                    elif s_type == "safety":
                        safety_summary = parsed
                    elif s_type == "diagnosis":
                        diagnosis_summary = parsed

                last_intent = str(triage_summary.get("intent", "") or "")

                # 4b. 最近一条 assistant 回复作为 last_customer_reply
                assistant_row = conn.execute(
                    "SELECT content FROM messages "
                    "WHERE session_id = ? AND role = 'assistant' "
                    "ORDER BY turn_index DESC LIMIT 1",
                    (session_id,),
                ).fetchone()
                last_customer_reply = str(assistant_row[0] or "") if assistant_row else ""

                # 5. ticket（最近一条）
                ticket_row = conn.execute(
                    "SELECT ticket_id, title, priority, status, raw_ticket_json, created_at "
                    "FROM tickets WHERE session_id = ? ORDER BY id DESC LIMIT 1",
                    (session_id,),
                ).fetchone()

                last_ticket_id = ""
                recent_ticket: dict[str, Any] = {}
                ticket_raw: dict[str, Any] = {}
                if ticket_row is not None:
                    last_ticket_id = str(ticket_row[0] or "")
                    try:
                        ticket_raw = json.loads(str(ticket_row[4] or "{}"))
                    except Exception:
                        ticket_raw = {}
                    recent_ticket = {
                        "ticket_id": last_ticket_id,
                        "title": str(ticket_row[1] or ""),
                        "created_at": str(ticket_row[5] or ""),
                        "safety": ticket_raw.get("safety", {}),
                        "dispatch": ticket_raw.get("dispatch", {}),
                        "audit": ticket_raw.get("audit", {}),
                    }
                    recent_ticket = {
                        k: v for k, v in recent_ticket.items()
                        if v is not None and v != "" and v != []
                    }

                # 从 ticket dispatch 提取 last_dispatch
                dispatch_from_ticket = ticket_raw.get("dispatch", {}) if isinstance(ticket_raw, dict) else {}
                last_dispatch: dict[str, Any] = {}
                if isinstance(dispatch_from_ticket, dict) and dispatch_from_ticket:
                    last_dispatch = dispatch_from_ticket

            # ---- 组装返回 ----
            # session 维度（兼容 _session_summary 结构）
            session_summary = {
                "session_id": session_id,
                "message_count": message_count,
                "recent_user_messages": recent_user_messages,
                "last_intent": last_intent,
                "last_case": last_case,
                "recent_case": last_case,
                "missing_info": missing_info,
                "recent_safety": safety_summary,
                "last_diagnosis": diagnosis_summary,
                "last_dispatch": last_dispatch,
                "recent_dispatch": last_dispatch,
                "last_customer_reply": last_customer_reply,
                "last_ticket_id": last_ticket_id,
                "updated_at": updated_at,
            }

            return {
                "session": session_summary,
                "last_case": last_case,
                "missing_info": missing_info,
                "recent_safety": safety_summary,
                "recent_ticket": recent_ticket,
                "last_customer_reply": last_customer_reply,
                "last_ticket_id": last_ticket_id,
                "session_summary": {
                    "session_id": session_id,
                    "message_count": message_count,
                    "last_intent": last_intent,
                    "last_model": last_case.get("charger_model", ""),
                    "last_risk_level": safety_summary.get("risk_level", ""),
                    "last_dispatch_priority": last_dispatch.get("priority", ""),
                },
            }
        except Exception:
            return self._empty_session_context(session_id)

    @staticmethod
    def _empty_session_context(session_id: str) -> dict[str, Any]:
        """返回与 JSON 路径无记录时兼容的最小结构。"""
        empty_session = {
            "session_id": session_id,
            "message_count": 0,
            "recent_user_messages": [],
            "last_intent": "",
            "last_case": {},
            "recent_case": {},
            "missing_info": [],
            "recent_safety": {},
            "last_diagnosis": {},
            "last_dispatch": {},
            "recent_dispatch": {},
            "last_customer_reply": "",
            "last_ticket_id": "",
            "updated_at": "",
        }
        return {
            "session": empty_session,
            "last_case": {},
            "missing_info": [],
            "recent_safety": {},
            "recent_ticket": {},
            "last_customer_reply": "",
            "last_ticket_id": "",
            "session_summary": {
                "session_id": session_id,
                "message_count": 0,
                "last_intent": "",
                "last_model": "",
                "last_risk_level": "",
                "last_dispatch_priority": "",
            },
        }

    # ------------------------------------------------------------------
    # Session TTL 生命周期管理
    # ------------------------------------------------------------------

    def mark_expired_sessions(self, expire_days: int = 7, archive_days: int = 30) -> dict[str, Any]:
        """标记过期和归档 session（不执行物理删除）。

        - active + updated_at < now - expire_days → expired
        - expired + updated_at < now - archive_days → archived

        返回 {"expired": int, "archived": int, "error": str}。
        """
        if not self.available:
            return {"expired": 0, "archived": 0, "error": self.error or "SQLite store unavailable"}

        try:
            with self._connect() as conn:
                # 1. active → expired
                cur = conn.execute(
                    """UPDATE sessions SET status = 'expired'
                       WHERE status = 'active'
                         AND updated_at < datetime('now', ? || ' days')""",
                    (f"-{expire_days}",),
                )
                expired_count = cur.rowcount

                # 2. expired → archived
                cur = conn.execute(
                    """UPDATE sessions SET status = 'archived'
                       WHERE status = 'expired'
                         AND updated_at < datetime('now', ? || ' days')""",
                    (f"-{archive_days}",),
                )
                archived_count = cur.rowcount

                conn.commit()

            return {"expired": expired_count, "archived": archived_count, "error": ""}
        except Exception as exc:
            return {"expired": 0, "archived": 0, "error": str(exc)}

    def _build_fts_query(self, query: str) -> str:
        """将用户输入的查询文本转换为 FTS5 查询语法。

        对每个有效词使用双引号包裹以支持精确匹配。
        """
        import re
        tokens = re.findall(r"[\w一-鿿]+", query)
        if not tokens:
            return ""
        # 去重，取前 8 个
        seen = set()
        unique = []
        for t in tokens:
            if t.lower() not in seen:
                seen.add(t.lower())
                unique.append(t)
        parts = [f'"{t}"' for t in unique[:8]]
        return " OR ".join(parts)
