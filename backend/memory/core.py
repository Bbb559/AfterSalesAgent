from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.config import MEMORY_DIR, MEMORY_READ_FROM_SQLITE, MEMORY_SQLITE_DUAL_WRITE


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _stable_key(*parts: str) -> str:
    raw = "|".join(part.strip() for part in parts if part and part.strip())
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16] if raw else ""


def _append_unique(items: list[dict[str, Any]], item: dict[str, Any], limit: int = 20) -> list[dict[str, Any]]:
    compact = json.dumps(item, ensure_ascii=False, sort_keys=True)
    existing = {json.dumps(value, ensure_ascii=False, sort_keys=True) for value in items}
    if compact not in existing:
        items.append(item)
    return items[-limit:]


@dataclass
class MemoryEntry:
    role: str
    content: str
    timestamp: str = field(default_factory=_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEntry":
        return cls(
            role=str(data.get("role", "")),
            content=str(data.get("content", "")),
            timestamp=str(data.get("timestamp", "")) or _now(),
            metadata=dict(data.get("metadata", {}) or {}),
        )


class JsonMemoryStore:
    """提供本地 JSON 记忆存储的基础读写能力。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _read_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default

    def _write_json(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def clear(self) -> bool:
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        return True


class SessionMemory(JsonMemoryStore):
    """保存单次充电桩诊断会话的消息和最近上下文。"""

    def __init__(self, session_id: str | None = None, root: str | Path | None = None) -> None:
        self.session_id = session_id or _new_id("session")
        self.base_root = Path(root or MEMORY_DIR / "sessions")
        super().__init__(self.base_root / self.session_id)
        self.messages: list[MemoryEntry] = []
        self.context: dict[str, Any] = {
            "last_triage": None,
            "last_intent": None,
            "last_case": {},
            "recent_case": {},
            "missing_info": [],
            "last_safety": {},
            "last_diagnosis": {},
            "last_dispatch": {},
            "recent_dispatch": {},
            "last_customer_reply": "",
            "last_ticket_id": "",
        }
        self.created_at = _now()
        self.updated_at = self.created_at
        self.load()

    def add_message(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        if not content:
            return
        self.messages.append(MemoryEntry(role=role, content=content, metadata=metadata or {}))
        self.messages = self.messages[-50:]
        self.updated_at = _now()
        self.save()

    def update_context(self, key: str, value: Any) -> None:
        self.context[key] = value
        self.updated_at = _now()
        self.save()

    def get_context(self, key: str, default: Any = None) -> Any:
        return self.context.get(key, default)

    def remember_workflow_result(self, user_input: str, result: dict[str, Any]) -> None:
        self.add_message("user", user_input, {"source": "workflow"})
        reply = result.get("action", {}).get("customer_reply", "")
        if reply:
            self.add_message("assistant", reply, {"source": "workflow"})
        intent = result.get("triage", {}).get("intent")
        case = result.get("case", {}) or {}
        safety = result.get("safety", {}) or {}
        diagnosis = result.get("diagnosis", {}) or {}
        dispatch = result.get("dispatch", {}) or result.get("action", {}).get("dispatch", {}) or {}
        self.context.update({
            "last_triage": intent,
            "last_intent": intent,
            "last_case": case,
            "recent_case": case,
            "missing_info": case.get("missing_info", []),
            "last_safety": safety,
            "last_diagnosis": diagnosis,
            "last_dispatch": dispatch,
            "recent_dispatch": dispatch,
            "last_customer_reply": reply,
        })
        self.updated_at = _now()
        self.save()

    def save(self) -> bool:
        self._write_json(
            self.root / "conversation.json",
            {
                "session_id": self.session_id,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "messages": [message.to_dict() for message in self.messages],
            },
        )
        self._write_json(self.root / "context.json", {"session_id": self.session_id, "context": self.context})
        return True

    def load(self) -> bool:
        conversation = self._read_json(self.root / "conversation.json", {})
        context = self._read_json(self.root / "context.json", {})
        if conversation:
            self.created_at = conversation.get("created_at", self.created_at)
            self.updated_at = conversation.get("updated_at", self.updated_at)
            self.messages = [MemoryEntry.from_dict(item) for item in conversation.get("messages", [])]
        if context:
            self.context.update(context.get("context", {}) or {})
            self._ensure_context_aliases()
        return True

    def get_status(self) -> dict[str, Any]:
        last_case = self.context.get("last_case") or self.context.get("recent_case") or {}
        last_safety = self.context.get("last_safety") or {}
        last_dispatch = self.context.get("last_dispatch") or self.context.get("recent_dispatch") or {}
        return {
            "session_id": self.session_id, # 会话 ID
            "message_count": len(self.messages),  # 当前会话消息总数
            "last_triage": self.context.get("last_triage"),
            "last_intent": self.context.get("last_intent") or self.context.get("last_triage"), # 最近一次相关工单的初步诊断意图
            "last_model": last_case.get("charger_model", ""), # 最近一次相关工单的充电桩型号
            "last_risk_level": last_safety.get("risk_level", ""), # 最近安全评估风险等级
            "last_dispatch_priority": last_dispatch.get("priority", ""), # 最近派工优先级
            "last_customer_reply": self.context.get("last_customer_reply", ""), # 最近客户回复
            "missing_info": self.context.get("missing_info", []), # 当前会话中缺失信息
            "updated_at": self.updated_at, # 会话最后更新时间
        }

    def _ensure_context_aliases(self) -> None:
        if not self.context.get("last_case") and self.context.get("recent_case"):
            self.context["last_case"] = self.context.get("recent_case", {})
        if not self.context.get("recent_case") and self.context.get("last_case"):
            self.context["recent_case"] = self.context.get("last_case", {})
        if not self.context.get("last_dispatch") and self.context.get("recent_dispatch"):
            self.context["last_dispatch"] = self.context.get("recent_dispatch", {})
        if not self.context.get("recent_dispatch") and self.context.get("last_dispatch"):
            self.context["recent_dispatch"] = self.context.get("last_dispatch", {})
        if not self.context.get("last_intent") and self.context.get("last_triage"):
            self.context["last_intent"] = self.context.get("last_triage")
        if not self.context.get("last_triage") and self.context.get("last_intent"):
            self.context["last_triage"] = self.context.get("last_intent")


class CustomerMemory(JsonMemoryStore):
    """按客户联系方式沉淀售后历史。"""

    def __init__(self, root: str | Path | None = None) -> None:
        super().__init__(root or MEMORY_DIR / "customers")
        self.index_path = self.root / "index.json"
        self.records: dict[str, dict[str, Any]] = self._read_json(self.index_path, {})

    def upsert_from_case(self, case: dict[str, Any], result: dict[str, Any] | None = None) -> str:
        key = self._build_key(case)
        if not key:
            return ""
        record = self.records.get(key) or {"customer_id": key, "created_at": _now(), "history_summaries": []}
        for field_name in ["contact_name", "contact_phone", "city", "contact_address"]:
            value = case.get(field_name)
            if value:
                record[field_name] = value
        dispatch = (result or {}).get("dispatch", {}) or (result or {}).get("action", {}).get("dispatch", {})
        summary = case.get("issue_description") or dispatch.get("customer_problem")
        if summary:
            record["history_summaries"] = _append_unique(
                record.get("history_summaries", []),
                {"summary": summary, "timestamp": _now()},
            )
        record["updated_at"] = _now()
        self.records[key] = record
        self.save()
        return key

    def get(self, customer_id: str) -> dict[str, Any] | None:
        return self.records.get(customer_id)

    def list_all(self) -> list[dict[str, Any]]:
        return sorted(self.records.values(), key=lambda item: item.get("updated_at", ""), reverse=True)

    def save(self) -> bool:
        self._write_json(self.index_path, self.records)
        return True

    def clear(self) -> bool:
        super().clear()
        self.records = {}
        self.index_path = self.root / "index.json"
        return True

    def get_status(self) -> dict[str, Any]:
        return {"total_customers": len(self.records), "updated_at": _now()}

    def _build_key(self, case: dict[str, Any]) -> str:
        explicit_id = str(case.get("customer_id", "")).strip()
        if explicit_id:
            return explicit_id
        phone = str(case.get("contact_phone", "")).strip()
        name = str(case.get("contact_name", "")).strip()
        return phone or _stable_key(name, str(case.get("city", "")), str(case.get("contact_address", "")))


class ChargerMemory(JsonMemoryStore):
    """按充电桩型号、序列号或系列沉淀历史问题。"""

    def __init__(self, root: str | Path | None = None) -> None:
        super().__init__(root or MEMORY_DIR / "chargers")
        self.index_path = self.root / "index.json"
        self.records: dict[str, dict[str, Any]] = self._read_json(self.index_path, {})

    def upsert_from_case(self, case: dict[str, Any], result: dict[str, Any] | None = None) -> str:
        key = self._build_key(case)
        if not key:
            return ""
        record = self.records.get(key) or {"charger_id": key, "created_at": _now(), "issue_history": []}
        for field_name in [
            "brand",
            "charger_model",
            "charger_series",
            "serial_number",
            "charger_type",
            "rated_power_kw",
            "connector_type",
            "vehicle_brand_model",
        ]:
            value = case.get(field_name)
            if value:
                record[field_name] = value
        dispatch = (result or {}).get("dispatch", {}) or (result or {}).get("action", {}).get("dispatch", {})
        history_item = {
            "issue_summary": case.get("issue_description") or dispatch.get("customer_problem", ""),
            "safety_level": (result or {}).get("safety", {}).get("risk_level", ""),
            "suggested_action": (result or {}).get("diagnosis", {}).get("suggested_next_step", ""),
            "dispatch_title": dispatch.get("title", ""),
            "timestamp": _now(),
        }
        if history_item["issue_summary"] or history_item["dispatch_title"]:
            record["issue_history"] = _append_unique(record.get("issue_history", []), history_item)
        record["updated_at"] = _now()
        self.records[key] = record
        self.save()
        return key

    def get(self, item_id: str) -> dict[str, Any] | None:
        return self.records.get(item_id)

    def list_all(self) -> list[dict[str, Any]]:
        return sorted(self.records.values(), key=lambda item: item.get("updated_at", ""), reverse=True)

    def save(self) -> bool:
        self._write_json(self.index_path, self.records)
        return True

    def clear(self) -> bool:
        super().clear()
        self.records = {}
        self.index_path = self.root / "index.json"
        return True

    def get_status(self) -> dict[str, Any]:
        return {"total_chargers": len(self.records), "updated_at": _now()}

    def _build_key(self, case: dict[str, Any]) -> str:
        serial = str(case.get("serial_number", "")).strip()
        if serial:
            return serial
        model = str(case.get("charger_model", "")).strip()
        brand = str(case.get("brand", "")).strip()
        if model:
            return _stable_key(brand, model)
        return _stable_key(brand, str(case.get("charger_series", "")), str(case.get("charger_type", "")))


class SiteMemory(JsonMemoryStore):
    """按城市和安装地址沉淀充电桩现场风险历史。"""

    def __init__(self, root: str | Path | None = None) -> None:
        super().__init__(root or MEMORY_DIR / "sites")
        self.index_path = self.root / "index.json"
        self.records: dict[str, dict[str, Any]] = self._read_json(self.index_path, {})

    def upsert_from_case(self, case: dict[str, Any], result: dict[str, Any] | None = None) -> str:
        key = self._build_key(case)
        if not key:
            return ""
        record = self.records.get(key) or {"site_id": key, "created_at": _now(), "risk_history": []}
        for field_name in ["city", "contact_address", "installation_type", "power_supply_phase", "breaker_or_rcd_info", "grounding_status"]:
            value = case.get(field_name)
            if value:
                record[field_name] = value
        risk_item = {
            "safety_level": (result or {}).get("safety", {}).get("risk_level", ""),
            "safety_signals": case.get("safety_signals", []),
            "environment_factors": case.get("environment_factors", []),
            "timestamp": _now(),
        }
        if risk_item["safety_level"] or risk_item["safety_signals"] or risk_item["environment_factors"]:
            record["risk_history"] = _append_unique(record.get("risk_history", []), risk_item)
        record["updated_at"] = _now()
        self.records[key] = record
        self.save()
        return key

    def get(self, site_id: str) -> dict[str, Any] | None:
        return self.records.get(site_id)

    def list_all(self) -> list[dict[str, Any]]:
        return sorted(self.records.values(), key=lambda item: item.get("updated_at", ""), reverse=True)

    def save(self) -> bool:
        self._write_json(self.index_path, self.records)
        return True

    def clear(self) -> bool:
        super().clear()
        self.records = {}
        self.index_path = self.root / "index.json"
        return True

    def get_status(self) -> dict[str, Any]:
        return {"total_sites": len(self.records), "updated_at": _now()}

    def _build_key(self, case: dict[str, Any]) -> str:
        address = str(case.get("contact_address", "")).strip()
        city = str(case.get("city", "")).strip()
        return _stable_key(city, address) if address or city else ""


class TicketMemory(JsonMemoryStore):
    """保存每次充电桩诊断流程的派工快照。"""

    def __init__(self, root: str | Path | None = None) -> None:
        super().__init__(root or MEMORY_DIR / "tickets")
        self.index_path = self.root / "index.json"
        self.index: list[dict[str, Any]] = self._read_json(self.index_path, [])

    def create_from_workflow(self, result: dict[str, Any], user_input: str = "") -> str:
        ticket_id = _new_id("ticket")
        dispatch = result.get("dispatch", {}) or result.get("action", {}).get("dispatch", {})
        snapshot = {
            "ticket_id": ticket_id,
            "title": dispatch.get("title", ""),
            "user_input": user_input,
            "case": result.get("case", {}),
            "safety": result.get("safety", {}),
            "diagnosis": result.get("diagnosis", {}),
            "warranty": result.get("warranty", {}),
            "dispatch": dispatch,
            "action": result.get("action", {}),
            "audit": result.get("audit", {}),
            "created_at": _now(),
        }
        self._write_json(self.root / f"{ticket_id}.json", snapshot)
        self.index.insert(0, {"ticket_id": ticket_id, "title": snapshot["title"], "created_at": snapshot["created_at"]})
        self.index = self.index[:200]
        self.save()
        return ticket_id

    def get(self, ticket_id: str) -> dict[str, Any] | None:
        return self._read_json(self.root / f"{ticket_id}.json", None)

    def list_all(self) -> list[dict[str, Any]]:
        return list(self.index)

    def save(self) -> bool:
        self._write_json(self.index_path, self.index)
        return True

    def clear(self) -> bool:
        super().clear()
        self.index = []
        self.index_path = self.root / "index.json"
        return True

    def get_status(self) -> dict[str, Any]:
        return {"total_tickets": len(self.index), "updated_at": _now()}


class SessionSearchIndex:
    """SQLite FTS5 index for session messages.

    The JSON memory files remain the source of truth. This index is rebuilt from
    session messages on write/read so it stays compatible with a later SQLite
    migration while remaining optional when FTS5 is unavailable.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.available = True
        self.error = ""
        self._ensure_schema()

    def index_session(self, session: SessionMemory) -> dict[str, Any]:
        if not self.available:
            return {"available": False, "indexed": 0, "error": self.error}
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM session_messages_fts WHERE session_id = ?", (session.session_id,))
                for message in session.messages:
                    conn.execute(
                        """
                        INSERT INTO session_messages_fts(session_id, role, content, timestamp, metadata)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            session.session_id,
                            message.role,
                            message.content,
                            message.timestamp,
                            json.dumps(message.metadata, ensure_ascii=False, sort_keys=True),
                        ),
                    )
            return {"available": True, "indexed": len(session.messages), "error": ""}
        except Exception as exc:
            self.error = str(exc)
            return {"available": False, "indexed": 0, "error": self.error}

    def search(self, query: str, session_id: str = "", limit: int = 5) -> dict[str, Any]:
        query = str(query or "").strip()
        if not self.available:
            return self._empty_result(query, available=False, error=self.error)
        try:
            matches = self._match_search(query, session_id=session_id, limit=limit)
            if not matches:
                matches = self._like_search(query, session_id=session_id, limit=limit)
            return {
                "available": True,
                "query": query,
                "matches": matches,
                "summary": self._summarize(matches),
                "summary_method": "sqlite_fts5_extractive",
                "error": "",
            }
        except Exception as exc:
            return self._empty_result(query, available=False, error=str(exc))

    def _ensure_schema(self) -> None:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS session_messages_fts USING fts5(
                        session_id UNINDEXED,
                        role UNINDEXED,
                        content,
                        timestamp UNINDEXED,
                        metadata UNINDEXED
                    )
                    """
                )
        except Exception as exc:
            self.available = False
            self.error = str(exc)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _match_search(self, query: str, session_id: str, limit: int) -> list[dict[str, Any]]:
        terms = self._search_terms(query)
        if not terms:
            return []
        match_query = " OR ".join(self._quote_fts_term(term) for term in terms[:8])
        params: list[Any] = [match_query]
        where = "session_messages_fts MATCH ?"
        if session_id:
            where += " AND session_id = ?"
            params.append(session_id)
        params.append(limit)
        sql = f"""
            SELECT session_id, role, content, timestamp, metadata, bm25(session_messages_fts) AS score
            FROM session_messages_fts
            WHERE {where}
            ORDER BY score
            LIMIT ?
        """
        return self._rows_to_matches(sql, params)

    def _like_search(self, query: str, session_id: str, limit: int) -> list[dict[str, Any]]:
        terms = self._search_terms(query) or ([query] if query else [])
        params: list[Any] = []
        where_parts = []
        if session_id:
            where_parts.append("session_id = ?")
            params.append(session_id)
        if terms:
            where_parts.append("(" + " OR ".join("content LIKE ?" for _ in terms[:5]) + ")")
            params.extend([f"%{term}%" for term in terms[:5]])
        where = " AND ".join(where_parts) if where_parts else "1 = 1"
        params.append(limit)
        sql = f"""
            SELECT session_id, role, content, timestamp, metadata, 0.0 AS score
            FROM session_messages_fts
            WHERE {where}
            ORDER BY rowid DESC
            LIMIT ?
        """
        return self._rows_to_matches(sql, params)

    def _rows_to_matches(self, sql: str, params: list[Any]) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        matches = []
        for session_id, role, content, timestamp, metadata, score in rows:
            try:
                parsed_metadata = json.loads(metadata or "{}")
            except Exception:
                parsed_metadata = {}
            matches.append({
                "session_id": session_id,
                "role": role,
                "content": content,
                "timestamp": timestamp,
                "metadata": parsed_metadata,
                "score": float(score or 0.0),
            })
        return matches

    def _search_terms(self, query: str) -> list[str]:
        terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{1,}", query or "")
        if not terms:
            terms = [
                item.strip()
                for item in re.split(r"[\s,，。；;？?！!]+", query or "")
                if len(item.strip()) >= 2
            ]
        deduped = []
        seen = set()
        for term in terms:
            normalized = term.strip()
            key = normalized.lower()
            if normalized and key not in seen:
                seen.add(key)
                deduped.append(normalized)
        return deduped

    def _quote_fts_term(self, term: str) -> str:
        return '"' + term.replace('"', '""') + '"'

    def _summarize(self, matches: list[dict[str, Any]]) -> str:
        if not matches:
            return ""
        snippets = []
        for item in matches[:3]:
            content = str(item.get("content") or "").strip()
            if content:
                snippets.append(f"{item.get('role', 'message')}: {content[:160]}")
        return " | ".join(snippets)

    def _empty_result(self, query: str, available: bool, error: str = "") -> dict[str, Any]:
        return {
            "available": available,
            "query": query,
            "matches": [],
            "summary": "",
            "summary_method": "sqlite_fts5_extractive",
            "error": error,
        }


class MemoryManager:
    """统一管理充电桩诊断会话、客户、设备、现场和派工记忆。"""

    def __init__(self, memory_dir: str | Path | None = None) -> None:
        self.memory_dir = Path(memory_dir or MEMORY_DIR)
        self.sessions_root = self.memory_dir / "sessions"
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        self.sessions: dict[str, SessionMemory] = {}
        self.current_session_id = ""
        self.customer_memory = CustomerMemory(self.memory_dir / "customers")
        self.charger_memory = ChargerMemory(self.memory_dir / "chargers")
        self.site_memory = SiteMemory(self.memory_dir / "sites")
        self.ticket_memory = TicketMemory(self.memory_dir / "tickets")
        self.session_search = SessionSearchIndex(self.memory_dir / "session_search.sqlite")
        self.sqlite_store = None
        if MEMORY_SQLITE_DUAL_WRITE:
            from backend.memory.sqlite_store import SQLiteLongTermMemoryStore
            self.sqlite_store = SQLiteLongTermMemoryStore(self.memory_dir / "long_term.sqlite")

    def create_session(self, session_id: str | None = None) -> SessionMemory:
        session = SessionMemory(session_id=session_id, root=self.sessions_root)
        self.sessions[session.session_id] = session
        self.current_session_id = session.session_id
        return session

    def get_or_create_session(self, session_id: str | None = None) -> SessionMemory:
        target_id = session_id or self.current_session_id
        if target_id:
            session = self.sessions.get(target_id)
            if session is None:
                session = SessionMemory(session_id=target_id, root=self.sessions_root)
                self.sessions[target_id] = session
            self.current_session_id = session.session_id
            return session
        return self.create_session()

    def remember_workflow_result(
        self,
        user_input: str,
        result: dict[str, Any],
        session_id: str | None = None,
    ) -> dict[str, str]:
        session = self.get_or_create_session(session_id)
        session.remember_workflow_result(user_input, result)
        self.session_search.index_session(session)
        ticket_id = self.ticket_memory.create_from_workflow(result, user_input=user_input)
        session.update_context("last_ticket_id", ticket_id)
        case = result.get("case", {})
        customer_id = self.customer_memory.upsert_from_case(case, result)
        charger_id = self.charger_memory.upsert_from_case(case, result)
        site_id = self.site_memory.upsert_from_case(case, result)

        # SQLite 双写（阶段 2）：JSON 为主，SQLite 追加写入，失败不阻塞。
        if self.sqlite_store is not None:
            try:
                ticket_data = self.ticket_memory.get(ticket_id) or {}
                messages_data = [m.to_dict() for m in session.messages]
                sqlite_result = self.sqlite_store.write_workflow_result(
                    session_id=session.session_id,
                    created_at=session.created_at,
                    updated_at=session.updated_at,
                    messages=messages_data,
                    case=case,
                    ticket_id=ticket_id,
                    ticket=ticket_data,
                    triage=result.get("triage"),
                    safety=result.get("safety"),
                    diagnosis=result.get("diagnosis"),
                )
                result.setdefault("trace", []).append({
                    "node": "memory_sqlite",
                    "title": "SQLite 长期记忆双写",
                    "status": "completed" if sqlite_result.get("success") else "warning",
                    "output": sqlite_result,
                    "timestamp": round(time.time(), 3),
                })
            except Exception as exc:
                result.setdefault("trace", []).append({
                    "node": "memory_sqlite",
                    "title": "SQLite 长期记忆双写",
                    "status": "failed",
                    "output": {"success": False, "error": str(exc)},
                    "timestamp": round(time.time(), 3),
                })

        return {
            "session_id": session.session_id,
            "customer_id": customer_id,
            "charger_id": charger_id,
            "site_id": site_id,
            "ticket_id": ticket_id,
        }

    def recall_context(self, case: dict[str, Any] | None = None, session_id: str | None = None) -> dict[str, Any]:
        """读取分层记忆摘要，不把记忆内容作为诊断证据。

        MEMORY_READ_FROM_SQLITE=false（默认）：走 JSON 路径（现有行为不变）。
        MEMORY_READ_FROM_SQLITE=true ：优先从 SQLite 读取 session/case/ticket 维度，
                                     失败时回退 JSON。
        """
        if MEMORY_READ_FROM_SQLITE and self.sqlite_store is not None and self.sqlite_store.available:
            try:
                return self._recall_context_from_sqlite(case, session_id)
            except Exception:
                # SQLite 路径异常 → 静默回退 JSON
                pass
        return self._recall_context_from_json(case, session_id)

    def _recall_context_from_json(
        self, case: dict[str, Any] | None = None, session_id: str | None = None
    ) -> dict[str, Any]:
        """现有 JSON 路径（MEMORY_READ_FROM_SQLITE=false 时使用）。"""
        case = case or {}
        session = self.get_or_create_session(session_id)
        self.session_search.index_session(session)
        customer_id = self.customer_memory._build_key(case)
        charger_id = self.charger_memory._build_key(case)
        site_id = self.site_memory._build_key(case)
        last_ticket_id = str(session.get_context("last_ticket_id", "") or "")

        customer = self.customer_memory.get(customer_id) if customer_id else None
        charger = self.charger_memory.get(charger_id) if charger_id else None
        # 保证 charger 始终包含 serial_number 字段，避免下游 KeyError
        charger = dict(charger) if charger else {}
        charger.setdefault("serial_number", "")
        site = self.site_memory.get(site_id) if site_id else None
        ticket = self.ticket_memory.get(last_ticket_id) if last_ticket_id else None
        session_summary = self._session_summary(session)
        session_search = self.session_search.search(self._memory_search_query(case), session_id=session.session_id)
        recent_ticket = self._record_summary(ticket, ["ticket_id", "title", "created_at", "safety", "dispatch", "audit"])
        if not recent_ticket:
            last_dispatch = session.get_context("last_dispatch", {}) or session.get_context("recent_dispatch", {}) or {}
            recent_ticket = {
                key: value
                for key, value in {
                    "title": last_dispatch.get("title"),
                    "priority": last_dispatch.get("priority"),
                    "need_onsite": last_dispatch.get("need_onsite"),
                    "need_electrician": last_dispatch.get("need_electrician"),
                }.items()
                if value is not None and value != "" and value != []
            }

        return {
            "session": session_summary,
            "last_case": session_summary.get("last_case", {}),
            "missing_info": session_summary.get("missing_info", []),
            "recent_safety": session_summary.get("recent_safety", {}),
            "recent_ticket": recent_ticket,
            "session_summary": {
                "session_id": session.session_id,
                "message_count": len(session.messages),
                "last_intent": session_summary.get("last_intent", ""),
                "last_model": session_summary.get("last_case", {}).get("charger_model", ""),
                "last_risk_level": session_summary.get("recent_safety", {}).get("risk_level", ""),
                "last_dispatch_priority": session_summary.get("last_dispatch", {}).get("priority", ""),
            },
            "session_search": session_search,
            "last_customer_reply": session.get_context("last_customer_reply", ""),
            "customer": self._record_summary(customer, ["customer_id", "contact_name", "contact_phone", "city", "contact_address", "history_summaries"]),
            "charger": self._record_summary(charger, ["charger_id", "brand", "charger_model", "serial_number", "issue_history"]),
            "site": self._record_summary(site, ["site_id", "city", "contact_address", "installation_type", "risk_history"]),
            "ticket": recent_ticket,
            "matched_ids": {
                "session_id": session.session_id,
                "customer_id": customer_id if customer else "",
                "charger_id": charger_id if charger else "",
                "site_id": site_id if site else "",
                "ticket_id": last_ticket_id if ticket else "",
            },
            "isolation": {
                "scope": "session/customer/charger/site/ticket/repo",
                "session_id": session.session_id,
                "session_isolated": True,
                "long_term_store": "local_json",
                "session_search_store": "sqlite_fts5",
                "repo_knowledge_separated": True,
                "used_as_diagnostic_evidence": False,
                "policy": "记忆只提供历史摘要和追问上下文，诊断依据仍以当前用户输入和 RAG 知识库为准。",
            },
        }

    def _recall_context_from_sqlite(
        self, case: dict[str, Any] | None = None, session_id: str | None = None
    ) -> dict[str, Any]:
        """SQLite 优先路径：从 long_term.sqlite 重建 session/case/ticket 维度。

        Customer/Charger/Site 维度暂不在 SQLite 中，返回空结构（与 JSON 无记录时行为一致）。
        失败时由调用方 recall_context() 的 try/except 回退 JSON。
        """
        case = case or {}

        # 解析 session_id（不加载 JSON）
        target_id = session_id or self.current_session_id
        if not target_id:
            target_id = _new_id("session")
            self.current_session_id = target_id

        sqlite_ctx = self.sqlite_store.build_session_context(target_id)
        session_summary = sqlite_ctx["session"]

        # Customer / Charger / Site ID（沿用 JSON key-building 逻辑）
        customer_id = self.customer_memory._build_key(case)
        charger_id = self.charger_memory._build_key(case)
        site_id = self.site_memory._build_key(case)

        # Customer / Charger / Site 数据 — SQLite 尚无对应表，从当前 case 提取稳定结构
        customer: dict[str, Any] = {
            "customer_id": customer_id,
            "contact_name": case.get("contact_name", ""),
            "contact_phone": case.get("contact_phone", ""),
            "city": case.get("city", ""),
            "contact_address": case.get("contact_address", ""),
        }
        charger: dict[str, Any] = {
            "charger_id": charger_id,
            "serial_number": case.get("serial_number", ""),
            "brand": case.get("brand", ""),
            "charger_model": case.get("charger_model", ""),
        }
        site: dict[str, Any] = {
            "site_id": site_id,
            "city": case.get("city", ""),
            "contact_address": case.get("contact_address", ""),
            "installation_type": case.get("installation_type", ""),
        }

        # Session search：复用 sqlite_store 的 FTS5 搜索，适配返回格式
        search_query = self._memory_search_query(case)
        fts5_matches = self.sqlite_store.search_messages(search_query, session_id=target_id)
        session_search = self._format_sqlite_fts5_result(search_query, fts5_matches)

        recent_ticket = sqlite_ctx.get("recent_ticket", {})
        last_ticket_id = sqlite_ctx.get("last_ticket_id", "")

        has_customer = bool(customer)
        has_charger = bool(charger)
        has_site = bool(site)
        has_ticket = bool(recent_ticket)

        return {
            "session": session_summary,
            "last_case": session_summary.get("last_case", {}),
            "missing_info": session_summary.get("missing_info", []),
            "recent_safety": session_summary.get("recent_safety", {}),
            "recent_ticket": recent_ticket,
            "session_summary": sqlite_ctx.get("session_summary", {}),
            "session_search": session_search,
            "last_customer_reply": sqlite_ctx.get("last_customer_reply", ""),
            "customer": customer,
            "charger": charger,
            "site": site,
            "ticket": recent_ticket,
            "matched_ids": {
                "session_id": target_id,
                "customer_id": customer_id if has_customer else "",
                "charger_id": charger_id if has_charger else "",
                "site_id": site_id if has_site else "",
                "ticket_id": last_ticket_id if has_ticket else "",
            },
            "isolation": {
                "scope": "session/customer/charger/site/ticket/repo",
                "session_id": target_id,
                "session_isolated": True,
                "long_term_store": "sqlite",
                "session_search_store": "sqlite_fts5",
                "repo_knowledge_separated": True,
                "used_as_diagnostic_evidence": False,
                "policy": "记忆只提供历史摘要和追问上下文，诊断依据仍以当前用户输入和 RAG 知识库为准。（SQLite 主读）",
            },
        }

    @staticmethod
    def _format_sqlite_fts5_result(query: str, matches: list[dict[str, Any]]) -> dict[str, Any]:
        """将 SQLiteLongTermMemoryStore.search_messages() 的返回适配为 session_search 格式。"""
        if not matches:
            return {
                "available": True,
                "query": query,
                "matches": [],
                "summary": "",
                "summary_method": "sqlite_fts5_extractive",
                "error": "",
            }
        formatted = []
        snippets: list[str] = []
        for idx, m in enumerate(matches[:5]):
            role = str(m.get("role", "") or "")
            content = str(m.get("content", "") or "")
            formatted.append({
                "session_id": str(m.get("session_id", "") or ""),
                "role": role,
                "content": content,
                "timestamp": str(m.get("created_at", "") or ""),
                "metadata": {},
                "score": float(idx == 0),
            })
            if content:
                snippets.append(f"{role}: {content[:160]}")
        return {
            "available": True,
            "query": query,
            "matches": formatted,
            "summary": " | ".join(snippets[:3]),
            "summary_method": "sqlite_fts5_extractive",
            "error": "",
        }

    def get_status(self) -> dict[str, Any]:
        current_session = self.sessions.get(self.current_session_id)
        return {
            "current_session_id": self.current_session_id,
            "total_sessions": len(self.sessions),
            "current_session": current_session.get_status() if current_session else None,
            "customers": self.customer_memory.get_status(),
            "chargers": self.charger_memory.get_status(),
            "sites": self.site_memory.get_status(),
            "tickets": self.ticket_memory.get_status(),
        }

    def enforce_session_ttl(self) -> dict[str, Any]:
        """手动触发 session TTL 标记（不自动，不接 workflow）。

        将过期的 active session 标记为 expired，
        将过期的 expired session 标记为 archived。
        不执行物理删除。
        """
        if self.sqlite_store is None:
            return {"expired": 0, "archived": 0, "error": "SQLite 未启用"}
        from backend.config import (
            MEMORY_SESSION_EXPIRE_AFTER_DAYS,
            MEMORY_SESSION_ARCHIVE_AFTER_DAYS,
        )
        return self.sqlite_store.mark_expired_sessions(
            expire_days=MEMORY_SESSION_EXPIRE_AFTER_DAYS,
            archive_days=MEMORY_SESSION_ARCHIVE_AFTER_DAYS,
        )

    def _session_summary(self, session: SessionMemory) -> dict[str, Any]:
        recent_user_messages = [
            message.content
            for message in session.messages
            if message.role == "user"
        ][-3:]
        return {
            "session_id": session.session_id,
            "message_count": len(session.messages),
            "recent_user_messages": recent_user_messages,
            "last_intent": session.get_context("last_intent") or session.get_context("last_triage"),
            "last_case": session.get_context("last_case", {}) or session.get_context("recent_case", {}),
            "recent_case": session.get_context("recent_case", {}) or session.get_context("last_case", {}),
            "missing_info": session.get_context("missing_info", []),
            "recent_safety": session.get_context("last_safety", {}),
            "last_diagnosis": session.get_context("last_diagnosis", {}),
            "last_dispatch": session.get_context("last_dispatch", {}) or session.get_context("recent_dispatch", {}),
            "recent_dispatch": session.get_context("recent_dispatch", {}) or session.get_context("last_dispatch", {}),
            "last_customer_reply": session.get_context("last_customer_reply", ""),
            "last_ticket_id": session.get_context("last_ticket_id", ""),
            "updated_at": session.updated_at,
        }

    def _memory_search_query(self, case: dict[str, Any]) -> str:
        parts: list[str] = []
        for key in ["raw_text", "issue_description", "charger_model", "serial_number", "contact_phone", "city"]:
            value = case.get(key)
            if value:
                parts.append(str(value))
        for key in ["fault_codes", "observed_symptoms", "safety_signals", "customer_requests"]:
            value = case.get(key)
            if isinstance(value, list):
                parts.extend(str(item) for item in value if str(item).strip())
        return " ".join(parts).strip()

    def _record_summary(self, record: dict[str, Any] | None, fields: list[str]) -> dict[str, Any]:
        if not record:
            return {}
        summary = {}
        for field in fields:
            value = record.get(field)
            if value is None or value == "" or value == []:
                continue
            summary[field] = value
        for list_field in ["history_summaries", "issue_history", "risk_history"]:
            if isinstance(summary.get(list_field), list):
                summary[list_field] = summary[list_field][-3:]
        return summary


_memory_manager: MemoryManager | None = None


def get_memory_manager() -> MemoryManager:
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
