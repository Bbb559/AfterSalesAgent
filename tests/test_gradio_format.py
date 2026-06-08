from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

from gradio_app import (
    build_node_visualization,
    create_demo,
    delete_kb,
    first_kb_value,
    format_agent_response,
    format_kb_status,
    format_memory_summary,
    format_rag_results,
    format_system_status,
    get_kb_choices,
    get_kb_choices_safe,
    load_kb,
    run_agent_stream,
)

try:
    from fastapi.testclient import TestClient
    from api import AsyncRunManager, app
    from backend.graph_workflow import ChargerDiagnosisWorkflow
    from backend.memory import MemoryManager
except ModuleNotFoundError:
    TestClient = None
    app = None
    AsyncRunManager = None
    ChargerDiagnosisWorkflow = None
    MemoryManager = None


class QueueLLM:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = [json.dumps(item, ensure_ascii=False) for item in responses]
        self.calls: list[Any] = []

    def __call__(self, prompt_value: Any) -> str:
        self.calls.append(prompt_value)
        return self.responses.pop(0)


class InProcessResponse:
    def __init__(self, response: Any) -> None:
        self.response = response

    def raise_for_status(self) -> None:
        self.response.raise_for_status()

    def json(self) -> dict[str, Any]:
        return self.response.json()


def charger_retrieval(question: str, **_: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return [
        {
            "file_name": "04_新能源家用充电桩售后运维与安全指南.pdf",
            "page": 2,
            "text": "C-RCD-04 漏保自检失败。漏保频繁跳闸需要停止充电，采集现场照片并转人工或电工核验。",
            "score": 0.91,
        }
    ], {"mode": "fake", "queries": [question]}


class GradioFormatTest(unittest.TestCase):
    def test_create_demo_does_not_call_backend_when_fastapi_is_down(self) -> None:
        with patch("gradio_app.requests.get", side_effect=RuntimeError("backend should not be called")) as get_call:
            demo = create_demo()

        self.assertEqual(get_call.call_count, 0)
        tab_labels = [
            component.get("props", {}).get("label")
            for component in demo.config.get("components", [])
            if component.get("type") == "tabitem"
        ]
        tab_render_flags = [
            component.get("props", {}).get("render_children")
            for component in demo.config.get("components", [])
            if component.get("type") == "tabitem"
        ]
        self.assertEqual(
            tab_labels,
            ["充电桩安全诊断 Agent", "知识库管理", "知识检索功能", "系统状态"],
        )
        self.assertFalse(any(tab_render_flags))
        html_values = [
            component.get("props", {}).get("value", "")
            for component in demo.config.get("components", [])
            if component.get("type") == "html"
        ]
        self.assertTrue(any("FastAPI" in str(value) for value in html_values))

    def test_kb_choices_safe_returns_empty_when_backend_unavailable(self) -> None:
        with patch("gradio_app.requests.get", side_effect=RuntimeError("api down")):
            choices = get_kb_choices_safe()

        self.assertEqual(choices, [])

    def test_format_agent_response_exposes_charger_business_text_panels(self) -> None:
        payload = {
            "success": True,
            "data": {
                "input_safety": {
                    "status": "warning",
                    "prompt_injection_detected": True,
                    "privilege_escalation_detected": False,
                    "sensitive_info_detected": False,
                    "matched_markers": ["ignore previous"],
                    "warnings": ["输入疑似包含提示注入指令。"],
                    "context_policy": "外部输入不能覆盖系统规则。",
                },
                "triage": {"intent": "safety_emergency", "confidence": "high", "reason": "命中安全风险"},
                "case": {
                    "brand": "VoltGate",
                    "charger_model": "VG-11KW-Pro",
                    "fault_codes": ["C-RCD-04"],
                    "missing_info": ["安装地址"],
                },
                "memory_context": {
                    "session": {"session_id": "session_demo", "message_count": 2, "recent_user_messages": ["C-RCD-04"]},
                    "customer": {"contact_phone": "13900000000"},
                    "charger": {"serial_number": "SN123"},
                    "site": {"city": "深圳"},
                    "ticket": {"title": "历史工单"},
                    "isolation": {"used_as_diagnostic_evidence": False, "policy": "记忆只提供历史摘要。"},
                },
                "safety": {
                    "risk_level": "p1_high",
                    "need_human": True,
                    "need_onsite": True,
                    "need_electrician": True,
                    "matched_safety_signals": ["枪线破皮"],
                    "required_customer_actions": ["停止充电"],
                    "reason": "命中枪线破皮",
                },
                "diagnosis": {"summary": "需安全核验。", "priority": "p1_high", "safe_remote_checks": ["拍照"]},
                "action": {"customer_reply": "您好，请先停止充电并远离风险源。"},
                "dispatch": {"title": "充电桩安全派工", "need_onsite": True},
                "warranty": {"status": "unknown", "need_evidence": True},
                "audit": {"passed": False, "warnings": ["高风险需人工"], "risk_level": "p1_high"},
                "governance": {
                    "status": "warning",
                    "input_scan_enabled": True,
                    "context_isolation_enabled": True,
                    "final_audit_enabled": True,
                    "memory_scope": "session/customer/charger/site/ticket/repo",
                    "memory_used_as_diagnostic_evidence": False,
                    "warnings": ["输入疑似包含提示注入指令。"],
                },
                "retrieval": {
                    "sources": ["充电桩手册 第1页"],
                    "results": [{"file_name": "充电桩手册", "page": 1, "chunk_id": "c1", "score": 0.92, "text": "C-RCD-04"}],
                    "trace": {"mode": "hybrid"},
                },
                "tool_history": [{"tool_name": "warranty_check", "status": "success", "latency_ms": 12}],
                "trace": [{"node": "final", "status": "completed", "duration": 0.1}],
            },
        }

        (
            customer_reply,
            intent_text,
            safety_text,
            case_text,
            diagnosis_text,
            warranty_text,
            dispatch_text,
            audit_text,
            sources_text,
            tool_history_text,
            trace_text,
            raw_json_text,
        ) = format_agent_response(payload)

        self.assertIn("停止充电", customer_reply)
        self.assertIn("输入安全扫描", intent_text)
        self.assertIn("提示注入", intent_text)
        self.assertIn("as-card", safety_text)
        self.assertIn("safety_emergency", intent_text)
        self.assertIn("p1_high", safety_text)
        self.assertIn("VG-11KW-Pro", case_text)
        self.assertIn("会话记忆", case_text)
        self.assertIn("非诊断证据", case_text)
        self.assertIn("需安全核验", diagnosis_text)
        self.assertIn("unknown", warranty_text)
        self.assertIn("充电桩安全派工", dispatch_text)
        self.assertIn("高风险需人工", audit_text)
        self.assertIn("安全治理汇总", audit_text)
        self.assertIn("上下文隔离", audit_text)
        self.assertIn("充电桩手册", sources_text)
        self.assertIn("warranty_check", tool_history_text)
        self.assertIn("final", trace_text)
        self.assertIn('"triage"', raw_json_text)
        self.assertNotIn("**风险等级", safety_text)

    def test_rag_and_system_status_formatters_hide_raw_json_shape(self) -> None:
        rag_text = format_rag_results({
            "success": True,
            "data": {
                "query": "C-RCD-04",
                "results": [
                    {
                        "file_name": "故障码手册.pdf",
                        "page": 3,
                        "chunk_id": "chunk_1",
                        "score": 0.87,
                        "text": "| 序号 | 故障 | 措施 |\n| --- | --- | --- |\n| 1 | C-RCD-04 | 表示漏保自检失败，需要安全核验。 |" * 20,
                    }
                ],
                "trace": {
                    "mode": "hybrid",
                    "queries": ["C-RCD-04", "漏保自检失败"],
                    "query_rewrite": {"enabled": True, "rewritten_queries": ["漏保自检失败"], "error": ""},
                },
            },
        })
        system_text = format_system_status({
            "loaded": True,
            "database_id": "kb_demo",
            "chunk_count": 806,
            "metadata": {"display_name": "充电桩知识库", "file_names": ["a.pdf", "b.pdf"], "updated_at": "2026-06-05"},
        })

        self.assertIn("命中数", rag_text)
        self.assertIn("1", rag_text)
        self.assertIn("故障码手册.pdf", rag_text)
        self.assertIn("漏保自检失败", rag_text)
        self.assertNotIn("as-preview", rag_text)
        self.assertNotIn("表示漏保自检失败", rag_text)
        self.assertNotIn("<table", rag_text)
        self.assertLess(len(rag_text), 3000)
        self.assertIn("当前加载知识库", system_text)
        self.assertIn("kb_demo", system_text)
        self.assertIn("SQLite 状态", system_text)
        self.assertIn("未接入", system_text)
        self.assertIn("安全治理状态", system_text)

    def test_memory_summary_shows_session_user_repo_isolation(self) -> None:
        memory_text = format_memory_summary({
            "session": {
                "session_id": "session_demo",
                "message_count": 2,
                "recent_user_messages": ["C-RCD-04 漏保跳闸"],
                "missing_info": ["安装地址"],
                "last_ticket_id": "ticket_1",
            },
            "customer": {"contact_phone": "13900000000"},
            "charger": {"serial_number": "SN123"},
            "site": {"city": "深圳"},
            "ticket": {"title": "历史工单"},
        })

        self.assertIn("Session 记忆", memory_text)
        self.assertIn("Customer / Charger / Site / Ticket", memory_text)
        self.assertIn("Repo / Knowledge", memory_text)
        self.assertIn("RAG 独立", memory_text)
        self.assertNotIn("原始 JSON", memory_text)

    def test_node_visualization_renders_charger_realtime_status_cards(self) -> None:
        html = build_node_visualization({
            "status": "running",
            "node_statuses": {
                "triage": {"title": "安全分诊", "status": "completed", "duration": 0.1, "input": {"user_input": "充电桩异常"}, "output": {"intent": "fault_diagnosis"}},
                "case_extract": {"title": "信息提取", "status": "completed", "duration": 0.2, "output": {"charger_model": "VG-11KW-Pro"}},
                "retrieval": {"title": "知识库检索", "status": "completed", "duration": 0.3, "input": {"query": "C-RCD-04"}, "output": {"result_count": 3}},
                "safety_guard": {"title": "安全护栏", "status": "completed", "duration": 0.1, "output": {"risk_level": "p1_high"}},
                "diagnosis": {"title": "安全诊断", "status": "completed", "duration": 0.4, "output": {"summary": "需安全处理"}},
                "warranty_dispatch": {"title": "保修派工", "status": "completed", "duration": 0.1, "output": {"tool_count": 1, "has_dispatch": True}},
                "action": {"title": "回复生成", "status": "completed", "duration": 0.5, "output": {"has_customer_reply": True}},
                "audit": {"title": "安全审核", "status": "running", "input": {"customer_reply": "请停止充电"}},
                "final": {"title": "结果汇总", "status": "pending"},
            },
        })

        for label in ["意图识别", "知识检索", "安全诊断", "方案生成", "方案审核", "结束"]:
            self.assertIn(label, html)
        self.assertIn("输入：", html)
        self.assertIn("输出：", html)
        self.assertIn("耗时：", html)
        self.assertIn("运行中", html)
        self.assertIn("完成", html)

    def test_node_visualization_shows_memory_answer_shortcut(self) -> None:
        html = build_node_visualization({
            "status": "completed",
            "node_statuses": {
                "input_guard": {"title": "输入安全扫描", "status": "completed", "duration": 0.01, "input": {"user_input": "刚才那个型号你还记得吗？"}, "output": {"status": "passed"}},
                "memory_context": {"title": "会话记忆读取", "status": "completed", "duration": 0.01, "output": {"last_model": "VG-7KW-AC", "missing_count": 2}},
                "memory_answer": {"title": "会话记忆回答", "status": "completed", "duration": 0.01, "output": {"answer_type": "model", "message": "VG-7KW-AC"}},
                "final": {"title": "结果汇总", "status": "completed", "duration": 0.01, "output": {"keys": ["triage", "action"]}},
            },
        })

        self.assertIn("VG-7KW-AC", html)
        self.assertIn("会话记忆", html)
        self.assertIn("完成", html)

    def test_run_agent_stream_polls_async_api_and_returns_final_payload(self) -> None:
        start_payload = {"success": True, "data": {"run_id": "run_demo", "status": "running", "session_id": "session_demo"}}
        status_payload = {
            "success": True,
            "data": {
                "run_id": "run_demo",
                "status": "completed",
                "session_id": "session_demo",
                "node_statuses_compact": {"final": {"title": "结果汇总", "status": "completed", "output": {"keys": ["action"]}}},
                "customer_reply": "您好，已为您生成充电桩处理建议。",
                "debug_log_path": "data/run_logs/run_demo.json",
            },
        }

        with patch("gradio_app.call_charger_diagnosis_start_api", return_value=start_payload) as start_call:
            with patch("gradio_app.call_charger_diagnosis_run_status", return_value=status_payload):
                outputs = list(run_agent_stream("充电桩异常", "", "hybrid", 5, "session_old"))

        start_call.assert_called_once_with("充电桩异常", "", "hybrid", 5, "session_old")
        customer_reply, node_html, run_meta_text, session_state = outputs[-1]
        self.assertEqual(len(outputs[-1]), 4)
        self.assertIn("充电桩处理建议", customer_reply)
        self.assertIn("结束", node_html)
        self.assertIn("运行摘要", run_meta_text)
        self.assertIn("data/run_logs/run_demo.json", run_meta_text)
        self.assertNotIn('"tool_history"', "".join(outputs[-1]))
        self.assertNotIn('"retrieval"', "".join(outputs[-1]))
        self.assertEqual(session_state, "session_demo")

    @unittest.skipIf(TestClient is None, "当前环境未安装 fastapi，安装 requirements.txt 后会执行该测试。")
    def test_run_agent_stream_uses_real_fastapi_async_contract_end_to_end(self) -> None:
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
            client.__enter__()
            self.addCleanup(client.__exit__, None, None, None)

            def post(url: str, json: dict[str, Any], timeout: int) -> InProcessResponse:
                path = url.split("127.0.0.1:8800", 1)[-1]
                return InProcessResponse(client.post(path, json=json))

            def get(url: str, timeout: int) -> InProcessResponse:
                path = url.split("127.0.0.1:8800", 1)[-1]
                return InProcessResponse(client.get(path))

            with patch("api.workflow", real_workflow):
                with patch("api.run_store", AsyncRunManager(timeout_seconds=5, log_dir=Path(temp_dir) / "run_logs")):
                    with patch("gradio_app.requests.post", side_effect=post):
                        with patch("gradio_app.requests.get", side_effect=get):
                            outputs = list(run_agent_stream(
                                "VoltGate VG-11KW-Pro 无法启动充电，屏幕显示 C-RCD-04，漏保频繁跳闸，东莞。",
                                "",
                                "hybrid",
                                5,
                                "ui_main_chain",
                            ))

        final_output = outputs[-1]
        self.assertEqual(len(final_output), 4)
        self.assertIn("停止充电", final_output[0])
        self.assertIn("结束", final_output[1])
        self.assertIn("运行摘要", final_output[2])
        self.assertIn("run_logs", final_output[2])
        self.assertNotIn('"retrieval"', "".join(final_output))
        self.assertNotIn("memory_context_read", "".join(final_output))
        self.assertEqual(final_output[3], "ui_main_chain")

    def test_run_agent_stream_waiting_state_uses_placeholder_cards(self) -> None:
        start_payload = {"success": True, "data": {"run_id": "run_demo", "status": "running", "session_id": "session_demo"}}
        running_payload = {
            "success": True,
            "data": {
                "run_id": "run_demo",
                "status": "running",
                "session_id": "session_demo",
                "node_statuses": {"triage": {"title": "安全分诊", "status": "running"}},
            },
        }

        with patch("gradio_app.call_charger_diagnosis_start_api", return_value=start_payload):
            with patch("gradio_app.call_charger_diagnosis_run_status", return_value=running_payload):
                first_output = next(run_agent_stream("充电桩异常", "", "hybrid", 5, "session_old"))

        self.assertEqual(len(first_output), 4)
        self.assertIn("正在运行", first_output[0])
        self.assertIn("运行中", first_output[1])
        self.assertIn("运行摘要", first_output[2])

    def test_run_agent_stream_failed_state_has_fixed_outputs_and_raw_status(self) -> None:
        start_payload = {"success": True, "data": {"run_id": "run_demo", "status": "running", "session_id": "session_demo"}}
        failed_payload = {
            "success": True,
            "data": {
                "run_id": "run_demo",
                "status": "failed",
                "session_id": "session_demo",
                "node_statuses": {"retrieval": {"title": "知识库检索", "status": "failed", "output": {"error": "boom"}}},
                "result": {},
                "error": "boom",
                "trace": [],
                "tool_history": [],
            },
        }

        with patch("gradio_app.call_charger_diagnosis_start_api", return_value=start_payload):
            with patch("gradio_app.call_charger_diagnosis_run_status", return_value=failed_payload):
                outputs = list(run_agent_stream("充电桩异常", "", "hybrid", 5, "session_old"))

        self.assertEqual(len(outputs[-1]), 4)
        self.assertIn("运行失败", outputs[-1][0])
        self.assertIn("boom", outputs[-1][0])
        self.assertIn("失败", outputs[-1][1])
        self.assertIn("boom", outputs[-1][2])
        self.assertNotIn('"status": "failed"', outputs[-1][2])
        self.assertEqual(outputs[-1][3], "session_demo")

    def test_run_agent_stream_frontend_timeout_has_fixed_outputs_and_raw_status(self) -> None:
        start_payload = {"success": True, "data": {"run_id": "run_demo", "status": "running", "session_id": "session_demo"}}
        running_payload = {
            "success": True,
            "data": {
                "run_id": "run_demo",
                "status": "running",
                "session_id": "session_demo",
                "node_statuses": {"triage": {"title": "安全分诊", "status": "running"}},
                "result": {},
                "error": "",
                "trace": [],
                "tool_history": [],
            },
        }

        with patch("gradio_app.AGENT_POLL_LIMIT", 1):
            with patch("gradio_app.AGENT_POLL_INTERVAL_SECONDS", 0):
                with patch("gradio_app.call_charger_diagnosis_start_api", return_value=start_payload):
                    with patch("gradio_app.call_charger_diagnosis_run_status", return_value=running_payload):
                        outputs = list(run_agent_stream("充电桩异常", "", "hybrid", 5, "session_old"))

        self.assertEqual(len(outputs[-1]), 4)
        self.assertIn("运行超时", outputs[-1][0])
        self.assertIn("超时", outputs[-1][1])
        self.assertIn("frontend polling timeout", outputs[-1][2])
        self.assertEqual(outputs[-1][3], "session_demo")

    def test_run_agent_stream_api_unavailable_has_fixed_outputs_and_raw_status(self) -> None:
        with patch("gradio_app.call_charger_diagnosis_start_api", side_effect=RuntimeError("connection refused")):
            outputs = list(run_agent_stream("充电桩异常", "", "hybrid", 5, "session_old"))

        self.assertEqual(len(outputs[-1]), 4)
        self.assertIn("FastAPI 未连接", outputs[-1][0])
        self.assertIn("未连接", outputs[-1][1])
        self.assertIn("connection refused", outputs[-1][2])
        self.assertNotIn('"status": "api_unavailable"', outputs[-1][2])
        self.assertEqual(outputs[-1][3], "session_old")

    def test_delete_kb_calls_api_and_refreshes_choices(self) -> None:
        delete_response = Mock()
        delete_response.json.return_value = {"success": True, "data": {"database_id": "kb_old"}}
        delete_response.raise_for_status.return_value = None

        list_response = Mock()
        list_response.json.return_value = {
            "success": True,
            "data": {
                "items": [
                    {
                        "database_id": "kb_new",
                        "display_name": "充电桩知识库",
                        "label": "充电桩知识库 | 1 chunks | pypdf",
                    }
                ]
            },
        }
        list_response.raise_for_status.return_value = None

        status_response = Mock()
        status_response.json.return_value = {"success": True, "data": {"loaded": False}}
        status_response.raise_for_status.return_value = None

        with patch("gradio_app.requests.delete", return_value=delete_response) as delete_call:
            with patch("gradio_app.requests.get", side_effect=[list_response, status_response]):
                with patch("gradio_app.gr", None):
                    message, choices, agent_choices, rag_choices, status_text = delete_kb("kb_old")

        delete_call.assert_called_once()
        self.assertIn("kb_old", message)
        self.assertEqual(choices[0][1], "kb_new")
        self.assertEqual(agent_choices[0][1], "kb_new")
        self.assertEqual(rag_choices[0][1], "kb_new")
        self.assertIn("充电桩知识库", choices[0][0])
        self.assertIn("当前加载知识库", status_text)
        self.assertIn("未加载", status_text)

    def test_kb_status_loaded_badge_uses_actual_loaded_state(self) -> None:
        loaded_text = format_kb_status({
            "database_id": "kb_loaded",
            "chunk_count": 214,
            "metadata": {"display_name": "充电桩知识库", "file_names": ["a.pdf"]},
        })
        unloaded_text = format_kb_status({
            "loaded": False,
            "database_id": "kb_list_only",
            "chunk_count": 214,
            "metadata": {"display_name": "充电桩知识库", "file_names": ["a.pdf"]},
        })

        self.assertIn("已加载", loaded_text)
        self.assertIn("kb_loaded", loaded_text)
        self.assertIn("未加载", unloaded_text)
        self.assertNotIn("当前加载知识库</span><span class='as-kv-value'>kb_list_only", unloaded_text)

    def test_load_kb_marks_selected_database_as_loaded_even_without_loaded_field(self) -> None:
        load_response = Mock()
        load_response.json.return_value = {
            "success": True,
            "data": {
                "database_id": "kb_loaded",
                "chunk_count": 214,
                "metadata": {"display_name": "充电桩知识库", "file_names": ["a.pdf"]},
            },
        }
        load_response.raise_for_status.return_value = None

        with patch("gradio_app.requests.post", return_value=load_response):
            message, status_text = load_kb("kb_loaded")

        self.assertIn("kb_loaded", message)
        self.assertIn("已加载", status_text)
        self.assertIn("kb_loaded", status_text)

    def test_kb_choices_show_display_name_but_keep_ascii_database_id(self) -> None:
        list_response = Mock()
        list_response.json.return_value = {
            "success": True,
            "data": {
                "items": [
                    {
                        "database_id": "kb_abc123",
                        "display_name": "充电桩知识库",
                        "label": "充电桩知识库 | 60 chunks | pypdf | VoltGate",
                    }
                ]
            },
        }
        list_response.raise_for_status.return_value = None

        with patch("gradio_app.requests.get", return_value=list_response):
            choices = get_kb_choices()

        self.assertEqual(choices[0][1], "kb_abc123")
        self.assertIn("充电桩知识库", choices[0][0])
        self.assertEqual(first_kb_value(choices), "kb_abc123")

    def test_lightweight_gradio_callbacks_do_not_wait_for_queue(self) -> None:
        demo = create_demo()

        dependency_by_api = {
            dependency.get("api_name"): dependency
            for dependency in demo.config.get("dependencies", [])
            if dependency.get("api_name")
        }

        for api_name in [
            "refresh_kb_choices",
            "load_kb",
            "delete_kb",
            "search_rag_formatted",
            "call_kb_status_formatted",
        ]:
            self.assertFalse(dependency_by_api[api_name]["queue"])

        self.assertTrue(dependency_by_api["build_kb"]["queue"])
        self.assertTrue(dependency_by_api["run_agent_stream"]["queue"])
        self.assertEqual(len(dependency_by_api["run_agent_stream"]["outputs"]), 4)
        load_dependencies = [
            dependency
            for dependency in demo.config.get("dependencies", [])
            if any(target[1] == "load" for target in dependency.get("targets", []))
        ]
        self.assertTrue(load_dependencies)
        self.assertTrue(any(dependency.get("api_name", "").startswith("refresh_kb_choices") for dependency in load_dependencies))
        self.assertFalse(any(dependency.get("api_name", "").startswith("refresh_agent_kb_choices") for dependency in load_dependencies))
        self.assertFalse(any(dependency.get("api_name", "").startswith("call_kb_status_formatted") for dependency in load_dependencies))
        self.assertTrue(any(len(dependency.get("outputs", [])) == 4 for dependency in load_dependencies))
        self.assertTrue(all(not dependency["queue"] for dependency in load_dependencies))
        self.assertFalse(
            any(
                target[1] == "select"
                for dependency in demo.config.get("dependencies", [])
                for target in dependency.get("targets", [])
            )
        )

        tab_labels = [
            component.get("props", {}).get("label")
            for component in demo.config.get("components", [])
            if component.get("type") == "tabitem"
        ]
        tab_render_flags = [
            component.get("props", {}).get("render_children")
            for component in demo.config.get("components", [])
            if component.get("type") == "tabitem"
        ]
        self.assertEqual(tab_labels, ["充电桩安全诊断 Agent", "知识库管理", "知识检索功能", "系统状态"])
        self.assertFalse(any(tab_render_flags))

        component_labels = {
            component.get("props", {}).get("label")
            for component in demo.config.get("components", [])
            if component.get("props", {}).get("label")
        }
        self.assertNotIn("业务结果", component_labels)
        self.assertNotIn("检索结果", component_labels)
        self.assertNotIn("补充业务信息", component_labels)
        self.assertNotIn("调试与依据", component_labels)
        self.assertNotIn("测试详情", component_labels)
        self.assertNotIn("记忆与工单", component_labels)
        self.assertNotIn("手机号查询（预留）", component_labels)
        self.assertNotIn("设备序列号查询（预留）", component_labels)
        self.assertNotIn("安装地址查询（预留）", component_labels)
        self.assertNotIn("分片预览 / 测试检索", component_labels)
        self.assertNotIn("测试检索", component_labels)
        self.assertNotIn("引用来源", component_labels)
        self.assertNotIn("工具调用记录", component_labels)
        self.assertNotIn("Agent 执行轨迹", component_labels)
        self.assertNotIn("原始 JSON", component_labels)
        component_types = {component.get("type") for component in demo.config.get("components", [])}
        self.assertNotIn("examples", component_types)

        kb_dropdowns = [
            component
            for component in demo.config.get("components", [])
            if component.get("type") == "dropdown"
            and component.get("props", {}).get("label") in {"知识库选择", "已有知识库"}
        ]
        self.assertEqual(len(kb_dropdowns), 3)
        self.assertTrue(all(component.get("props", {}).get("allow_custom_value") for component in kb_dropdowns))

        memory_summary_cards = [
            component
            for component in demo.config.get("components", [])
            if component.get("type") == "html"
            and "Session 记忆" in str(component.get("props", {}).get("value", ""))
        ]
        self.assertEqual(len(memory_summary_cards), 0)


if __name__ == "__main__":
    unittest.main()
