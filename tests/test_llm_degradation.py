"""测试 LLM 不可用时全链路降级行为 — workflow.run(llm=None) 仍返回完整结果。"""

import tempfile
import unittest
from pathlib import Path
from typing import Any

from backend.graph_workflow import ChargerDiagnosisWorkflow
from backend.memory import MemoryManager


class LLMDegradationTest(unittest.TestCase):
    """验证 llm=None 时 workflow 每个节点平稳降级，不崩溃，返回 legal 输出。"""

    def setUp(self) -> None:
        self._temp_memory_dir = tempfile.TemporaryDirectory()
        self.memory_manager = MemoryManager(memory_dir=Path(self._temp_memory_dir.name))

    def tearDown(self) -> None:
        self._temp_memory_dir.cleanup()

    def _workflow(self, **kwargs: Any) -> ChargerDiagnosisWorkflow:
        kwargs.setdefault("memory_manager", self.memory_manager)
        kwargs.setdefault("llm", None)  # 显式 None
        return ChargerDiagnosisWorkflow(**kwargs)

    # ------------------------------------------------------------------
    # 基础降级：结构完整性
    # ------------------------------------------------------------------

    def test_llm_none_returns_all_required_keys(self) -> None:
        """llm=None 时结果仍包含全部顶层 key。"""
        workflow = self._workflow()
        result = workflow.run("VoltGate VG-11KW-Pro 无法启动，屏幕 VG-E01")

        expected_keys = {
            "input_safety", "triage", "case", "memory_context",
            "retrieval", "safety", "diagnosis", "warranty",
            "dispatch", "action", "audit", "governance",
            "tool_history", "trace",
        }
        self.assertEqual(set(result), expected_keys)

    def test_llm_none_triage_falls_back(self) -> None:
        """llm=None 时 triage 有合法 intent fallback。"""
        workflow = self._workflow()
        result = workflow.run("充电桩不能充电")

        triage = result["triage"]
        self.assertIn(triage["intent"], {
            "fault_diagnosis", "warranty_consultation", "safety_emergency",
            "dispatch_request", "general_question", "unknown",
        })

    def test_llm_none_case_structured_fallback_works(self) -> None:
        """llm=None 时 case 仍然通过结构化 fallback 提取字段。"""
        workflow = self._workflow()
        result = workflow.run("VoltGate VG-11KW-Pro SN123456 屏幕显示 VG-E01")

        self.assertEqual(result["case"]["brand"], "VoltGate")
        self.assertEqual(result["case"]["charger_model"], "VG-11KW-Pro")
        self.assertIn("VG-E01", result["case"]["fault_codes"])

    def test_llm_none_retrieval_still_works(self) -> None:
        """llm=None 时 RAG 检索仍然执行（不依赖 LLM）。"""
        workflow = self._workflow()
        result = workflow.run("C-RCD-04 漏保跳闸")

        retrieval = result["retrieval"]
        self.assertIn("query", retrieval)
        self.assertIn("results", retrieval)
        self.assertIn("sources", retrieval)

    def test_llm_none_safety_guard_still_fires(self) -> None:
        """llm=None 时安全护栏依旧触发（不依赖 LLM）。"""
        workflow = self._workflow()
        result = workflow.run("充电桩冒烟有烧焦味")

        self.assertEqual(result["safety"]["risk_level"], "p0_emergency")
        self.assertTrue(result["safety"]["need_onsite"])

    # ------------------------------------------------------------------
    # 降级标记
    # ------------------------------------------------------------------

    def test_llm_none_marks_llm_unavailable_in_audit(self) -> None:
        """llm=None 时 audit 包含 LLM 不可用警告。"""
        workflow = self._workflow()
        result = workflow.run("VG-11KW-Pro 屏幕不亮")

        audit_warnings = result["audit"].get("warnings", [])
        self.assertTrue(
            any("LLM" in w or "llm" in w.lower() or "LLM" in str(w) for w in audit_warnings)
            or "未启用 LLM" in str(audit_warnings),
            f"audit warnings should mention LLM unavailable: {audit_warnings}"
        )

    def test_llm_none_trace_has_llm_warning(self) -> None:
        """llm=None 时 trace 包含 LLM 不可用状态记录。"""
        workflow = self._workflow()
        result = workflow.run("设备异常")

        trace_items = result.get("trace", [])
        llm_status_items = [
            item for item in trace_items
            if item.get("node") == "llm" and item.get("status") == "warning"
        ]
        self.assertTrue(
            llm_status_items or any("llm_unavailable" in str(item) for item in trace_items),
            f"trace should record LLM unavailable: {trace_items}"
        )

    # ------------------------------------------------------------------
    # 输出安全性 — 降级不应编造
    # ------------------------------------------------------------------

    def test_llm_none_diagnosis_does_not_fabricate_causes(self) -> None:
        """llm=None 时 diagnosis 不会编造具体原因。"""
        workflow = self._workflow()
        result = workflow.run("VG-CloudMini 离线")

        diagnosis = result["diagnosis"]
        # 没有 RAG 结果时 likely_issue_areas 应被清空
        if not result["retrieval"]["results"]:
            self.assertEqual(diagnosis["likely_issue_areas"], [])

    def test_llm_none_action_has_customer_reply(self) -> None:
        """llm=None 时 action 仍有客户回复（兜底模板）。"""
        workflow = self._workflow()
        result = workflow.run("充电桩故障码 E07")

        action = result["action"]
        self.assertTrue(action.get("customer_reply"), "llm=None 时至少应有兜底回复")

    def test_llm_none_memory_context_reads_without_llm(self) -> None:
        """llm=None 时 memory_context 读取不依赖 LLM。"""
        workflow = self._workflow()
        result = workflow.run("查询设备信息")

        mem = result["memory_context"]
        self.assertIn("isolation", mem)
        self.assertFalse(mem["isolation"]["used_as_diagnostic_evidence"])

    # ------------------------------------------------------------------
    # 端到端：多轮对话降级
    # ------------------------------------------------------------------

    def test_llm_none_multi_turn_preserves_session(self) -> None:
        """llm=None 时多轮对话 session 保持连贯。"""
        workflow = self._workflow()
        sid = "session_degradation_test"

        # Turn 1
        r1 = workflow.run("VoltGate VG-11KW-Pro SN123456 C-RCD-04", session_id=sid)
        self.assertEqual(r1["memory_context"]["session"]["session_id"], sid)

        # Turn 2 — 追问
        r2 = workflow.run("风险等级是多少？", session_id=sid)
        # 不应崩溃
        self.assertIn("triage", r2)
        self.assertIsNotNone(r2.get("case"))

    # ------------------------------------------------------------------
    # 工具链完整性
    # ------------------------------------------------------------------

    def test_llm_none_warranty_tool_still_runs(self) -> None:
        """llm=None 时保修工具仍然运行。"""
        workflow = self._workflow()
        result = workflow.run("充电桩过了保修期怎么办")

        warranty = result["warranty"]
        self.assertIn("status", warranty)

    def test_llm_none_memory_write_still_runs(self) -> None:
        """llm=None 时记忆写入仍然执行。"""
        workflow = self._workflow()
        result = workflow.run("VoltGate VG-11KW-Pro")

        tool_names = [item["tool_name"] for item in result["tool_history"]]
        self.assertIn("memory_workflow_write", tool_names)


if __name__ == "__main__":
    unittest.main()
