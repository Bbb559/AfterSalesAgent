from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from pathlib import Path
import time
import uuid
from typing import Any

from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.graph_workflow import ChargerDiagnosisWorkflow
from backend.memory import get_memory_manager
from backend.rag.rag_service import RAGService


app = FastAPI(title="ChargerSafetyDiagnosis API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
rag_service = RAGService()
workflow = ChargerDiagnosisWorkflow(rag_service=rag_service)


class AsyncRunManager:
    """使用 asyncio task 管理前端轮询需要的充电桩诊断运行状态。"""

    def __init__(self, timeout_seconds: float = 180.0, log_dir: str | Path = Path("data") / "run_logs") -> None:
        self._lock = asyncio.Lock()
        self._runs: dict[str, dict[str, Any]] = {}
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self.timeout_seconds = timeout_seconds
        self.log_dir = Path(log_dir)

    async def create(self, session_id: str = "") -> str:
        run_id = uuid.uuid4().hex
        now = round(time.time(), 3)
        async with self._lock:
            self._runs[run_id] = {
                "run_id": run_id,
                "session_id": session_id,
                "status": "running",
                "events": [],
                "node_statuses": {},
                "trace": [],
                "tool_history": [],
                "result": {},
                "error": "",
                "created_at": now,
                "updated_at": now,
                "timeout_seconds": self.timeout_seconds,
                "debug_log_path": str(self._debug_log_path(run_id)),
            }
        return run_id

    def start_task(self, run_id: str, payload: "ChargerDiagnosisRunRequest", session_id: str) -> None:
        self._loop = asyncio.get_running_loop()
        task = asyncio.create_task(self._run_workflow(run_id, payload, session_id))
        self._tasks[run_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(run_id, None))

    async def add_event(self, run_id: str, event: dict[str, Any]) -> None:
        async with self._lock:
            run = self._runs.get(run_id)
            if not run or run.get("status") != "running":
                return
            event_copy = deepcopy(event)
            run["events"].append(event_copy)
            node = event_copy.get("node")
            if node:
                run["node_statuses"][node] = event_copy
            run["updated_at"] = round(time.time(), 3)

    def add_event_from_worker(self, run_id: str, event: dict[str, Any]) -> None:
        if self._loop is None or not self._loop.is_running():
            return
        future = asyncio.run_coroutine_threadsafe(self.add_event(run_id, event), self._loop)
        try:
            future.result(timeout=2)
        except Exception:
            pass

    async def finish(self, run_id: str, result: dict[str, Any]) -> None:
        async with self._lock:
            run = self._runs.get(run_id)
            if not run or run.get("status") != "running":
                return
            run["status"] = "completed"
            run["result"] = deepcopy(result)
            run["trace"] = deepcopy(result.get("trace", []))
            run["tool_history"] = deepcopy(result.get("tool_history", []))
            run["updated_at"] = round(time.time(), 3)
            self._write_debug_log_locked(run)

    async def fail(self, run_id: str, error: str) -> None:
        async with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return
            self._mark_failed_locked(run, error)
            self._write_debug_log_locked(run)

    async def get(self, run_id: str) -> dict[str, Any] | None:
        async with self._lock:
            run = self._runs.get(run_id)
            if run and run.get("status") == "running" and self._is_timed_out(run):
                self._mark_failed_locked(
                    run,
                    f"充电桩安全诊断 Agent 运行超时（超过 {self.timeout_seconds:g} 秒），请稍后重试或检查后端服务。",
                )
                self._write_debug_log_locked(run)
            return deepcopy(run) if run else None

    async def _run_workflow(self, run_id: str, payload: "ChargerDiagnosisRunRequest", session_id: str) -> None:
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(_run_charger_diagnosis_sync, run_id, payload, session_id, self),
                timeout=self.timeout_seconds,
            )
            await self.finish(run_id, result)
        except asyncio.TimeoutError:
            await self.fail(
                run_id,
                f"充电桩安全诊断 Agent 运行超时（超过 {self.timeout_seconds:g} 秒），请稍后重试或检查后端服务。",
            )
        except Exception as exc:  # pragma: no cover - 后台 task 最外层防御边界
            await self.fail(run_id, f"充电桩安全诊断 Agent 运行失败：{exc}")

    def _is_timed_out(self, run: dict[str, Any]) -> bool:
        created_at = run.get("created_at")
        if not isinstance(created_at, (int, float)):
            return False
        return time.time() - float(created_at) > self.timeout_seconds

    def _mark_failed_locked(self, run: dict[str, Any], error: str) -> None:
        run["status"] = "failed"
        run["error"] = error
        run["updated_at"] = round(time.time(), 3)
        final_event = {
            "node": "final",
            "title": "结果汇总",
            "status": "failed",
            "input": {},
            "output": {"error": error},
            "timestamp": round(time.time(), 3),
        }
        run.setdefault("events", []).append(final_event)
        run.setdefault("node_statuses", {})["final"] = final_event

    def _debug_log_path(self, run_id: str) -> Path:
        return self.log_dir / f"{run_id}.json"

    def _write_debug_log_locked(self, run: dict[str, Any]) -> None:
        run_id = str(run.get("run_id") or "")
        if not run_id:
            return
        path = self._debug_log_path(run_id)
        run["debug_log_path"] = str(path)
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(deepcopy(run), ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:  # pragma: no cover - 日志失败不能影响主链路
            run["debug_log_error"] = str(exc)


run_store = AsyncRunManager()


SUMMARY_OUTPUT_KEYS = {
    "answer_type",
    "charger_model",
    "error",
    "has_customer_reply",
    "has_dispatch",
    "intent",
    "keys",
    "message",
    "missing_info",
    "passed",
    "priority",
    "reason",
    "result_count",
    "risk_level",
    "status",
    "summary",
    "warnings",
}


def summarize_run_for_frontend(run: dict[str, Any]) -> dict[str, Any]:
    result = run.get("result") if isinstance(run.get("result"), dict) else {}
    action = result.get("action") if isinstance(result.get("action"), dict) else {}
    return {
        "run_id": run.get("run_id", ""),
        "session_id": run.get("session_id", ""),
        "status": run.get("status", ""),
        "error": run.get("error", ""),
        "node_statuses_compact": _compact_node_statuses(run.get("node_statuses", {})),
        "customer_reply": action.get("customer_reply", ""),
        "debug_log_path": run.get("debug_log_path", ""),
        "updated_at": run.get("updated_at", ""),
    }


def _compact_node_statuses(node_statuses: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(node_statuses, dict):
        return {}
    compact: dict[str, dict[str, Any]] = {}
    for node_id, event in node_statuses.items():
        if not isinstance(event, dict):
            continue
        compact[str(node_id)] = {
            "node": event.get("node", node_id),
            "title": event.get("title", ""),
            "status": event.get("status", "pending"),
            "duration": event.get("duration", 0),
            "timestamp": event.get("timestamp", ""),
            "input": _compact_event_value(event.get("input")),
            "output": _compact_event_value(event.get("output")),
        }
    return compact


def _compact_event_value(value: Any, max_length: int = 220) -> Any:
    if not value:
        return {}
    if isinstance(value, dict):
        selected = {key: value.get(key) for key in SUMMARY_OUTPUT_KEYS if key in value}
        if selected:
            return selected
        return {"summary": _compact_text(value, max_length=max_length)}
    if isinstance(value, list):
        return {"count": len(value), "summary": _compact_text(value[:3], max_length=max_length)}
    return _compact_text(value, max_length=max_length)


def _compact_text(value: Any, max_length: int = 220) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
        except TypeError:
            text = str(value)
    text = text.strip()
    return text if len(text) <= max_length else text[:max_length].rstrip() + "..."


class ChargerDiagnosisRunRequest(BaseModel):
    user_input: str = Field(..., min_length=1, description="客户充电桩售后安全问题")
    retrieval_options: dict[str, Any] = Field(default_factory=dict, description="RAG 检索参数")
    database_id: str | None = Field(default=None, description="知识库编号，第一版可以为空")
    session_id: str | None = Field(default=None, description="会话编号，用于会话级记忆回忆")


class ApiResponse(BaseModel):
    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class LoadKnowledgeBaseRequest(BaseModel):
    database_id: str = Field(..., min_length=1, description="知识库编号")


class RagSearchRequest(BaseModel):
    question: str = Field(..., min_length=1, description="检索问题")
    retrieval_options: dict[str, Any] = Field(default_factory=dict, description="RAG 检索参数")
    database_id: str | None = Field(default=None, description="知识库编号")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "ChargerSafetyDiagnosis API"}


@app.post("/api/memory/sessions", response_model=ApiResponse)
def create_memory_session() -> ApiResponse:
    session = get_memory_manager().create_session()
    return ApiResponse(
        success=True,
        data={
            "session_id": session.session_id,
            "message_count": len(session.messages),
            "created_at": session.created_at,
            "current_session_id": get_memory_manager().current_session_id,
        },
    )


@app.get("/api/kb/list", response_model=ApiResponse)
def list_knowledge_bases() -> ApiResponse:
    return ApiResponse(success=True, data={"items": rag_service.list_knowledge_bases()})


@app.get("/api/kb/status", response_model=ApiResponse)
def knowledge_base_status() -> ApiResponse:
    return ApiResponse(success=True, data=rag_service.status())


@app.post("/api/kb/build", response_model=ApiResponse)
def build_knowledge_base(
    files: list[UploadFile] = File(...),
    parser_name: str = Form("pypdf"),
    splitter_name: str = Form("recursive"),
    chunk_size: int = Form(700),
    chunk_overlap: int = Form(80),
    display_name: str = Form(""),
    doc_type: str = Form(""),
    product_line: str = Form(""),
    item_identifier: str = Form(""),
    version: str = Form(""),
) -> ApiResponse:
    result = rag_service.build_knowledge_base(
        uploaded_files=files,
        parser_name=parser_name,
        splitter_name=splitter_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        display_name=display_name,
        doc_type=doc_type,
        product_line=product_line,
        item_identifier=item_identifier,
        version=version,
    )
    return ApiResponse(success=bool(result.get("success")), data=result if result.get("success") else {}, error=result.get("error", ""))


@app.post("/api/kb/load", response_model=ApiResponse)
def load_knowledge_base(payload: LoadKnowledgeBaseRequest) -> ApiResponse:
    result = rag_service.load_knowledge_base(payload.database_id)
    return ApiResponse(success=bool(result.get("success")), data=result if result.get("success") else {}, error=result.get("error", ""))


@app.delete("/api/kb/{database_id}", response_model=ApiResponse)
def delete_knowledge_base(database_id: str) -> ApiResponse:
    result = rag_service.delete_knowledge_base(database_id)
    return ApiResponse(success=bool(result.get("success")), data=result if result.get("success") else {}, error=result.get("error", ""))


@app.post("/api/rag/search", response_model=ApiResponse)
def search_rag(payload: RagSearchRequest) -> ApiResponse:
    options = dict(payload.retrieval_options)
    if payload.database_id:
        options["database_id"] = payload.database_id
    result = rag_service.retrieve(payload.question, options)
    return ApiResponse(success=not bool(result.get("error")), data=result, error=result.get("error", ""))


def _run_charger_diagnosis_sync(
    run_id: str,
    payload: ChargerDiagnosisRunRequest,
    session_id: str,
    manager: AsyncRunManager,
) -> dict[str, Any]:
    options = dict(payload.retrieval_options)
    if payload.database_id:
        options["database_id"] = payload.database_id
    return workflow.run(
        payload.user_input,
        options,
        progress_callback=lambda event: manager.add_event_from_worker(run_id, event),
        session_id=session_id,
    )


@app.post("/api/charger-diagnosis/start", response_model=ApiResponse)
async def start_charger_diagnosis_agent(payload: ChargerDiagnosisRunRequest) -> ApiResponse:
    session_id = payload.session_id or get_memory_manager().get_or_create_session().session_id
    run_id = await run_store.create(session_id=session_id)
    run_store.start_task(run_id, payload, session_id)
    return ApiResponse(success=True, data=await run_store.get(run_id) or {"run_id": run_id, "status": "running"})


@app.get("/api/charger-diagnosis/runs/{run_id}", response_model=ApiResponse)
async def get_charger_diagnosis_run(run_id: str, view: str = Query("full")) -> ApiResponse:
    run = await run_store.get(run_id)
    if not run:
        return ApiResponse(success=False, error=f"运行记录不存在：{run_id}")
    if view == "summary":
        return ApiResponse(success=True, data=summarize_run_for_frontend(run))
    return ApiResponse(success=True, data=run)


@app.post("/api/charger-diagnosis/run", response_model=ApiResponse)
def run_charger_diagnosis_agent(payload: ChargerDiagnosisRunRequest) -> ApiResponse:
    try:
        options = dict(payload.retrieval_options)
        if payload.database_id:
            options["database_id"] = payload.database_id
        result = workflow.run(payload.user_input, options, session_id=payload.session_id)
        return ApiResponse(success=True, data=result)
    except Exception as exc:  # pragma: no cover - API 最外层防御边界
        return ApiResponse(success=False, error=f"充电桩安全诊断 Agent 运行失败：{exc}")
