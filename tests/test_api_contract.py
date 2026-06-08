from __future__ import annotations

import inspect
import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
    from api import AsyncRunManager, app, create_memory_session, get_charger_diagnosis_run, start_charger_diagnosis_agent
    from backend.graph_workflow import ChargerDiagnosisWorkflow
    from backend.memory import MemoryManager
except ModuleNotFoundError:
    TestClient = None
    app = None
    AsyncRunManager = None
    get_charger_diagnosis_run = None
    create_memory_session = None
    start_charger_diagnosis_agent = None
    ChargerDiagnosisWorkflow = None
    MemoryManager = None


class QueueLLM:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = [json.dumps(item, ensure_ascii=False) for item in responses]
        self.calls: list[Any] = []

    def __call__(self, prompt_value: Any) -> str:
        self.calls.append(prompt_value)
        return self.responses.pop(0)


def charger_retrieval(question: str, **_: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return [
        {
            "file_name": "04_新能源家用充电桩售后运维与安全指南.pdf",
            "page": 2,
            "text": "C-RCD-04 漏保自检失败。漏保频繁跳闸需要停止充电，采集现场照片并转人工或电工核验。",
            "score": 0.91,
        }
    ], {"mode": "fake", "queries": [question]}


@unittest.skipIf(TestClient is None, "当前环境未安装 fastapi，安装 requirements.txt 后会执行该测试。")
class ApiContractTest(unittest.TestCase):
    def test_charger_diagnosis_start_and_status_are_real_async_endpoints(self) -> None:
        self.assertTrue(inspect.iscoroutinefunction(start_charger_diagnosis_agent))
        self.assertTrue(inspect.iscoroutinefunction(get_charger_diagnosis_run))

    def test_create_memory_session_endpoint_returns_unique_backend_sessions(self) -> None:
        client = TestClient(app)

        first = client.post("/api/memory/sessions")
        second = client.post("/api/memory/sessions")

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertTrue(first.json()["success"])
        self.assertTrue(second.json()["success"])
        first_session = first.json()["data"]["session_id"]
        second_session = second.json()["data"]["session_id"]
        self.assertTrue(first_session.startswith("session_"))
        self.assertTrue(second_session.startswith("session_"))
        self.assertNotEqual(first_session, second_session)
        self.assertEqual(second.json()["data"]["current_session_id"], second_session)
        self.assertEqual(first.json()["data"]["message_count"], 0)

    def test_created_memory_session_can_be_used_by_async_run_summary(self) -> None:
        class FakeWorkflow:
            def run(self, user_input, retrieval_options=None, progress_callback=None, session_id=None, memory_manager=None):
                if progress_callback:
                    progress_callback({"node": "final", "title": "结果汇总", "status": "completed", "output": {"keys": ["action"]}})
                return {
                    "action": {"customer_reply": f"session={session_id}"},
                    "trace": [],
                    "tool_history": [],
                }

        with TestClient(app) as client:
            session_id = client.post("/api/memory/sessions").json()["data"]["session_id"]

            with patch("api.workflow", FakeWorkflow()):
                with patch("api.run_store", AsyncRunManager(timeout_seconds=5)):
                    start_response = client.post(
                        "/api/charger-diagnosis/start",
                        json={"user_input": "充电桩异常", "retrieval_options": {}, "session_id": session_id},
                    )
                    run_id = start_response.json()["data"]["run_id"]
                    for _ in range(20):
                        status_response = client.get(f"/api/charger-diagnosis/runs/{run_id}?view=summary")
                        payload = status_response.json()["data"]
                        if payload["status"] == "completed":
                            break
                        time.sleep(0.05)

            self.assertEqual(payload["session_id"], session_id)
            self.assertEqual(payload["status"], "completed")
            self.assertIn(session_id, payload["customer_reply"])

    def test_health_endpoint_returns_ok(self) -> None:
        client = TestClient(app)

        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertEqual(response.json()["service"], "ChargerSafetyDiagnosis API")

    def test_react_dev_origin_is_allowed_by_cors(self) -> None:
        client = TestClient(app)

        response = client.options(
            "/api/kb/list",
            headers={
                "Origin": "http://127.0.0.1:5173",
                "Access-Control-Request-Method": "GET",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://127.0.0.1:5173")

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

    def test_charger_diagnosis_run_endpoint_returns_workflow_result(self) -> None:
        class FakeWorkflow:
            def __init__(self) -> None:
                self.session_id = ""

            def run(self, user_input, retrieval_options=None, progress_callback=None, session_id=None, memory_manager=None):
                self.session_id = session_id or ""
                return {
                    "triage": {"intent": "safety_emergency"},
                    "case": {},
                    "retrieval": {},
                    "safety": {"risk_level": "p0_emergency", "matched_safety_signals": ["配电箱冒烟"]},
                    "diagnosis": {},
                    "warranty": {},
                    "dispatch": {"title": "高风险派工"},
                    "action": {"customer_reply": "请立即停止充电，远离风险源。"},
                    "audit": {"passed": False},
                    "tool_history": [],
                    "trace": [],
                }

        client = TestClient(app)
        fake_workflow = FakeWorkflow()

        with patch("api.workflow", fake_workflow):
            response = client.post(
                "/api/charger-diagnosis/run",
                json={"user_input": "充电桩配电箱冒烟，有烧焦味", "retrieval_options": {}, "session_id": "sync_session"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["data"]["safety"]["risk_level"], "p0_emergency")
        self.assertIn("停止充电", payload["data"]["action"]["customer_reply"])
        self.assertIn("matched_safety_signals", payload["data"]["safety"])
        self.assertIn("dispatch", payload["data"])
        self.assertNotIn("escalation", payload["data"])
        self.assertEqual(fake_workflow.session_id, "sync_session")

    def test_charger_diagnosis_async_run_exposes_progress_and_final_result(self) -> None:
        class FakeWorkflow:
            def __init__(self) -> None:
                self.session_id = ""

            def run(self, user_input, retrieval_options=None, progress_callback=None, session_id=None, memory_manager=None):
                self.session_id = session_id or ""
                if progress_callback:
                    progress_callback({
                        "node": "triage",
                        "title": "安全分诊",
                        "status": "running",
                        "input": {"user_input": user_input},
                        "output": {},
                        "timestamp": 1.0,
                    })
                    progress_callback({
                        "node": "triage",
                        "title": "安全分诊",
                        "status": "completed",
                        "input": {"user_input": user_input},
                        "output": {"intent": "fault_diagnosis"},
                        "timestamp": 2.0,
                        "duration": 0.1,
                    })
                return {
                    "triage": {"intent": "fault_diagnosis"},
                    "case": {},
                    "retrieval": {},
                    "safety": {"risk_level": "p3_low"},
                    "diagnosis": {},
                    "warranty": {},
                    "dispatch": {"title": "测试派工"},
                    "action": {"customer_reply": "已生成充电桩安全诊断回复"},
                    "audit": {},
                    "tool_history": [{"tool_name": "warranty_check"}],
                    "trace": [{"node": "triage", "status": "completed"}],
                }

        fake_workflow = FakeWorkflow()

        with TestClient(app) as client:
            with patch("api.workflow", fake_workflow):
                start_response = client.post(
                    "/api/charger-diagnosis/start",
                    json={"user_input": "VG-WallBox2 充到一半停止", "retrieval_options": {}, "session_id": "async_session"},
                )

                self.assertEqual(start_response.status_code, 200)
                run_id = start_response.json()["data"]["run_id"]
                self.assertEqual(start_response.json()["data"]["session_id"], "async_session")
                status_payload = {}
                for _ in range(20):
                    status_response = client.get(f"/api/charger-diagnosis/runs/{run_id}")
                    status_payload = status_response.json()["data"]
                    if status_payload["status"] == "completed":
                        break
                    time.sleep(0.05)

        self.assertEqual(status_payload["status"], "completed")
        self.assertEqual(status_payload["node_statuses"]["triage"]["status"], "completed")
        self.assertEqual(status_payload["result"]["action"]["customer_reply"], "已生成充电桩安全诊断回复")
        self.assertEqual(status_payload["tool_history"][0]["tool_name"], "warranty_check")
        self.assertEqual(status_payload["session_id"], "async_session")
        self.assertEqual(fake_workflow.session_id, "async_session")

    def test_async_run_summary_view_is_light_and_writes_debug_log(self) -> None:
        class FakeWorkflow:
            def run(self, user_input, options, progress_callback=None, session_id: str = ""):
                if progress_callback:
                    progress_callback({
                        "node": "retrieval",
                        "title": "知识库检索",
                        "status": "completed",
                        "input": {"query": user_input},
                        "output": {"result_count": 1, "message": "检索完成"},
                    })
                return {
                    "action": {"customer_reply": "已生成轻量前端回复"},
                    "retrieval": {"results": [{"text": "large evidence text" * 200}], "trace": {"mode": "hybrid"}},
                    "trace": [{"node": "retrieval", "status": "completed"}],
                    "tool_history": [{"tool_name": "memory_context_read"}],
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "run_logs"
            with TestClient(app) as client:
                with patch("api.workflow", FakeWorkflow()):
                    with patch("api.run_store", AsyncRunManager(timeout_seconds=5, log_dir=log_dir)):
                        start_response = client.post(
                            "/api/charger-diagnosis/start",
                            json={"user_input": "充电桩异常", "retrieval_options": {}, "session_id": "summary_session"},
                        )
                        run_id = start_response.json()["data"]["run_id"]
                        full_payload = {}
                        for _ in range(20):
                            full_payload = client.get(f"/api/charger-diagnosis/runs/{run_id}").json()["data"]
                            if full_payload["status"] == "completed":
                                break
                            time.sleep(0.05)
                        summary_payload = client.get(f"/api/charger-diagnosis/runs/{run_id}?view=summary").json()["data"]

            self.assertEqual(summary_payload["status"], "completed")
            self.assertEqual(summary_payload["customer_reply"], "已生成轻量前端回复")
            self.assertIn("node_statuses_compact", summary_payload)
            self.assertIn("debug_log_path", summary_payload)
            self.assertNotIn("result", summary_payload)
            self.assertNotIn("trace", summary_payload)
            self.assertNotIn("tool_history", summary_payload)
            self.assertNotIn("large evidence text", json.dumps(summary_payload, ensure_ascii=False))
            self.assertTrue(full_payload["result"]["retrieval"]["results"])
            self.assertTrue(full_payload["trace"])
            self.assertTrue(full_payload["tool_history"])
            log_path = Path(full_payload["debug_log_path"])
            self.assertTrue(log_path.exists())
            log_payload = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertEqual(log_payload["result"]["action"]["customer_reply"], "已生成轻量前端回复")
            self.assertIn("large evidence text", json.dumps(log_payload, ensure_ascii=False))

    def test_async_run_manager_writes_failed_and_timeout_debug_logs(self) -> None:
        import asyncio

        async def scenario(log_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
            failed_store = AsyncRunManager(timeout_seconds=5, log_dir=log_dir / "failed")
            failed_run_id = await failed_store.create(session_id="failed_session")
            await failed_store.fail(failed_run_id, "boom")
            failed_run = await failed_store.get(failed_run_id)

            timeout_store = AsyncRunManager(timeout_seconds=0.01, log_dir=log_dir / "timeout")
            timeout_run_id = await timeout_store.create(session_id="timeout_session")
            await asyncio.sleep(0.03)
            timeout_run = await timeout_store.get(timeout_run_id)
            return failed_run, timeout_run

        with tempfile.TemporaryDirectory() as temp_dir:
            failed_run, timeout_run = asyncio.run(scenario(Path(temp_dir)))

            self.assertEqual(failed_run["status"], "failed")
            self.assertEqual(timeout_run["status"], "failed")
            self.assertTrue(Path(failed_run["debug_log_path"]).exists())
            self.assertTrue(Path(timeout_run["debug_log_path"]).exists())
            self.assertIn("boom", Path(failed_run["debug_log_path"]).read_text(encoding="utf-8"))
            self.assertIn("超时", Path(timeout_run["debug_log_path"]).read_text(encoding="utf-8"))

    def test_async_endpoint_runs_real_workflow_agent_rag_and_memory_chain(self) -> None:
        llm = QueueLLM([
            {"intent": "fault_diagnosis", "confidence": "high", "reason": "客户反馈漏保跳闸。"},
            {
                "brand": "VoltGate",
                "charger_model": "VG-11KW-Pro",
                "issue_type": "fault",
                "issue_description": "无法启动充电，屏幕显示 C-RCD-04，漏保频繁跳闸",
                "fault_codes": ["C-RCD-04"],
                "observed_symptoms": ["漏保频繁跳闸"],
                "city": "东莞",
            },
            {
                "summary": "C-RCD-04 与漏保自检失败相关，需要结合知识库和现场照片安全核验。",
                "evidence_status": "grounded",
                "safe_remote_checks": ["拍摄报错截图和配电箱外观。"],
                "onsite_reasons": ["漏保频繁跳闸需要电工核验。"],
                "priority": "p1_high",
                "suggested_next_step": "停止充电并转人工/电工处理。",
                "evidence_sources": ["04_新能源家用充电桩售后运维与安全指南.pdf 第2页"],
            },
            {
                "customer_reply": "您好，请先停止充电并远离风险源，漏保频繁跳闸需要人工或电工核验。",
                "internal_advice": "安排安全核验。",
            },
            {"passed": True, "warnings": [], "final_note": "可回复", "risk_level": "p1_high"},
        ])

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MemoryManager(memory_dir=Path(temp_dir))
            real_workflow = ChargerDiagnosisWorkflow(retrieval_func=charger_retrieval, llm=llm, memory_manager=manager)

            with TestClient(app) as client:
                with patch("api.workflow", real_workflow):
                    start_response = client.post(
                        "/api/charger-diagnosis/start",
                        json={
                            "user_input": "VoltGate VG-11KW-Pro 无法启动充电，屏幕显示 C-RCD-04，漏保频繁跳闸，东莞。",
                            "retrieval_options": {},
                            "session_id": "api_main_chain",
                        },
                    )
                    self.assertEqual(start_response.status_code, 200)
                    run_id = start_response.json()["data"]["run_id"]
                    status_payload = {}
                    for _ in range(40):
                        status_response = client.get(f"/api/charger-diagnosis/runs/{run_id}")
                        status_payload = status_response.json()["data"]
                        if status_payload["status"] == "completed":
                            break
                        time.sleep(0.05)

        self.assertEqual(status_payload["status"], "completed")
        for key in ["run_id", "session_id", "status", "node_statuses", "result", "error", "trace", "tool_history"]:
            self.assertIn(key, status_payload)
        for node in ["input_guard", "memory_context", "retrieval", "diagnosis", "action", "audit", "final"]:
            self.assertIn(node, status_payload["node_statuses"])
        result = status_payload["result"]
        self.assertIn("停止充电", result["action"]["customer_reply"])
        self.assertTrue(result["retrieval"]["results"])
        self.assertEqual(result["memory_context"]["isolation"]["used_as_diagnostic_evidence"], False)
        self.assertEqual(manager.get_or_create_session("api_main_chain").get_context("last_case")["charger_model"], "VG-11KW-Pro")
        tool_names = [item["tool_name"] for item in status_payload["tool_history"]]
        self.assertIn("memory_context_read", tool_names)
        self.assertIn("warranty_check", tool_names)
        self.assertIn("memory_workflow_write", tool_names)
        self.assertEqual(len(llm.calls), 5)

    def test_async_endpoint_marks_background_exceptions_failed_with_stable_shape(self) -> None:
        class ExplodingWorkflow:
            def run(self, *args, **kwargs):
                raise RuntimeError("boom")

        with TestClient(app) as client:
            with patch("api.workflow", ExplodingWorkflow()):
                start_response = client.post(
                    "/api/charger-diagnosis/start",
                    json={"user_input": "充电桩异常", "retrieval_options": {}, "session_id": "failed_session"},
                )
                run_id = start_response.json()["data"]["run_id"]
                status_payload = {}
                for _ in range(20):
                    status_payload = client.get(f"/api/charger-diagnosis/runs/{run_id}").json()["data"]
                    if status_payload["status"] == "failed":
                        break
                    time.sleep(0.05)

        self.assertEqual(status_payload["status"], "failed")
        self.assertIn("boom", status_payload["error"])
        for key in ["run_id", "session_id", "status", "node_statuses", "result", "error", "trace", "tool_history"]:
            self.assertIn(key, status_payload)

    def test_async_run_manager_times_out_running_runs(self) -> None:
        import asyncio

        async def scenario() -> dict[str, Any]:
            store = AsyncRunManager(timeout_seconds=0.01)
            run_id = await store.create(session_id="timeout_session")
            await asyncio.sleep(0.03)
            return await store.get(run_id)

        run = asyncio.run(scenario())

        self.assertEqual(run["status"], "failed")
        self.assertIn("超时", run["error"])
        for key in ["run_id", "session_id", "status", "node_statuses", "result", "error", "trace", "tool_history"]:
            self.assertIn(key, run)

    def test_async_endpoint_timeout_does_not_block_status_polling(self) -> None:
        class SlowWorkflow:
            def run(self, *args, **kwargs):
                time.sleep(0.2)
                return {"action": {"customer_reply": "too late"}, "trace": [], "tool_history": []}

        with TestClient(app) as client:
            with patch("api.workflow", SlowWorkflow()):
                with patch("api.run_store", AsyncRunManager(timeout_seconds=0.05)):
                    start_response = client.post(
                        "/api/charger-diagnosis/start",
                        json={"user_input": "充电桩异常", "retrieval_options": {}, "session_id": "timeout_session"},
                    )
                    run_id = start_response.json()["data"]["run_id"]
                    status_payload = {}
                    for _ in range(20):
                        status_payload = client.get(f"/api/charger-diagnosis/runs/{run_id}").json()["data"]
                        if status_payload["status"] == "failed":
                            break
                        time.sleep(0.02)

        self.assertEqual(status_payload["status"], "failed")
        self.assertIn("超时", status_payload["error"])

    def _old_deleted_block_marker(self) -> None:
        return

        with patch("api.workflow", fake_workflow):
            start_response = client.post(
                "/api/charger-diagnosis/start",
                json={"user_input": "VG-WallBox2 充到一半停止", "retrieval_options": {}, "session_id": "async_session"},
            )

            self.assertEqual(start_response.status_code, 200)
            run_id = start_response.json()["data"]["run_id"]
            self.assertEqual(start_response.json()["data"]["session_id"], "async_session")
            status_payload = {}
            for _ in range(20):
                status_response = client.get(f"/api/charger-diagnosis/runs/{run_id}")
                status_payload = status_response.json()["data"]
                if status_payload["status"] == "completed":
                    break
                time.sleep(0.05)

        self.assertEqual(status_payload["status"], "completed")
        self.assertEqual(status_payload["node_statuses"]["triage"]["status"], "completed")
        self.assertEqual(status_payload["result"]["action"]["customer_reply"], "已生成充电桩安全诊断回复")
        self.assertEqual(status_payload["tool_history"][0]["tool_name"], "warranty_check")
        self.assertEqual(status_payload["session_id"], "async_session")
        self.assertEqual(fake_workflow.session_id, "async_session")

    def legacy_async_endpoint_runs_real_workflow_agent_rag_and_memory_chain(self) -> None:
        llm = QueueLLM([
            {"intent": "fault_diagnosis", "confidence": "high", "reason": "客户反馈漏保跳闸。"},
            {
                "brand": "VoltGate",
                "charger_model": "VG-11KW-Pro",
                "issue_type": "fault",
                "issue_description": "无法启动充电，屏幕显示 C-RCD-04，漏保频繁跳闸",
                "fault_codes": ["C-RCD-04"],
                "observed_symptoms": ["漏保频繁跳闸"],
                "city": "东莞",
            },
            {
                "summary": "C-RCD-04 与漏保自检失败相关，需要结合知识库和现场照片安全核验。",
                "evidence_status": "grounded",
                "safe_remote_checks": ["拍摄报错截图和配电箱外观。"],
                "onsite_reasons": ["漏保频繁跳闸需要电工核验。"],
                "priority": "p1_high",
                "suggested_next_step": "停止充电并转人工/电工处理。",
                "evidence_sources": ["04_新能源家用充电桩售后运维与安全指南.pdf 第2页"],
            },
            {
                "customer_reply": "您好，请先停止充电并远离风险源，漏保频繁跳闸需要人工或电工核验。",
                "internal_advice": "安排安全核验。",
            },
            {"passed": True, "warnings": [], "final_note": "可回复", "risk_level": "p1_high"},
        ])

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MemoryManager(memory_dir=Path(temp_dir))
            real_workflow = ChargerDiagnosisWorkflow(retrieval_func=charger_retrieval, llm=llm, memory_manager=manager)
            client = TestClient(app)

            with patch("api.workflow", real_workflow):
                start_response = client.post(
                    "/api/charger-diagnosis/start",
                    json={
                        "user_input": "VoltGate VG-11KW-Pro 无法启动充电，屏幕显示 C-RCD-04，漏保频繁跳闸，东莞。",
                        "retrieval_options": {},
                        "session_id": "api_main_chain",
                    },
                )
                self.assertEqual(start_response.status_code, 200)
                run_id = start_response.json()["data"]["run_id"]
                status_payload = {}
                for _ in range(40):
                    status_response = client.get(f"/api/charger-diagnosis/runs/{run_id}")
                    status_payload = status_response.json()["data"]
                    if status_payload["status"] == "completed":
                        break
                    time.sleep(0.05)

        self.assertEqual(status_payload["status"], "completed")
        for key in ["run_id", "session_id", "status", "node_statuses", "result", "error", "trace", "tool_history"]:
            self.assertIn(key, status_payload)
        for node in ["input_guard", "memory_context", "retrieval", "diagnosis", "action", "audit", "final"]:
            self.assertIn(node, status_payload["node_statuses"])
        result = status_payload["result"]
        self.assertIn("停止充电", result["action"]["customer_reply"])
        self.assertTrue(result["retrieval"]["results"])
        self.assertEqual(result["memory_context"]["isolation"]["used_as_diagnostic_evidence"], False)
        self.assertEqual(manager.get_or_create_session("api_main_chain").get_context("last_case")["charger_model"], "VG-11KW-Pro")
        tool_names = [item["tool_name"] for item in status_payload["tool_history"]]
        self.assertIn("memory_context_read", tool_names)
        self.assertIn("warranty_check", tool_names)
        self.assertIn("memory_workflow_write", tool_names)
        self.assertEqual(len(llm.calls), 5)

    def legacy_async_endpoint_marks_background_exceptions_failed_with_stable_shape(self) -> None:
        class ExplodingWorkflow:
            def run(self, *args, **kwargs):
                raise RuntimeError("boom")

        client = TestClient(app)
        with patch("api.workflow", ExplodingWorkflow()):
            start_response = client.post(
                "/api/charger-diagnosis/start",
                json={"user_input": "充电桩异常", "retrieval_options": {}, "session_id": "failed_session"},
            )
            run_id = start_response.json()["data"]["run_id"]
            status_payload = {}
            for _ in range(20):
                status_payload = client.get(f"/api/charger-diagnosis/runs/{run_id}").json()["data"]
                if status_payload["status"] == "failed":
                    break
                time.sleep(0.05)

        self.assertEqual(status_payload["status"], "failed")
        self.assertIn("boom", status_payload["error"])
        for key in ["run_id", "session_id", "status", "node_statuses", "result", "error", "trace", "tool_history"]:
            self.assertIn(key, status_payload)

    def legacy_async_run_manager_times_out_running_runs(self) -> None:
        import asyncio

        async def scenario() -> dict[str, Any]:
            store = AsyncRunManager(timeout_seconds=0.01)
            run_id = await store.create(session_id="timeout_session")
            await asyncio.sleep(0.03)
            return await store.get(run_id)

        run = asyncio.run(scenario())

        self.assertEqual(run["status"], "failed")
        self.assertIn("超时", run["error"])
        for key in ["run_id", "session_id", "status", "node_statuses", "result", "error", "trace", "tool_history"]:
            self.assertIn(key, run)

    def legacy_async_endpoint_timeout_does_not_block_status_polling(self) -> None:
        class SlowWorkflow:
            def run(self, *args, **kwargs):
                time.sleep(0.2)
                return {"action": {"customer_reply": "too late"}, "trace": [], "tool_history": []}

        client = TestClient(app)
        with patch("api.workflow", SlowWorkflow()):
            with patch("api.run_store", AsyncRunManager(timeout_seconds=0.05)):
                start_response = client.post(
                    "/api/charger-diagnosis/start",
                    json={"user_input": "充电桩异常", "retrieval_options": {}, "session_id": "timeout_session"},
                )
                run_id = start_response.json()["data"]["run_id"]
                status_payload = {}
                for _ in range(20):
                    status_payload = client.get(f"/api/charger-diagnosis/runs/{run_id}").json()["data"]
                    if status_payload["status"] == "failed":
                        break
                    time.sleep(0.02)

        self.assertEqual(status_payload["status"], "failed")
        self.assertIn("超时", status_payload["error"])


if __name__ == "__main__":
    unittest.main()
