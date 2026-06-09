"""测试 RAG 空结果时 diagnosis 具体原因被 enforce_diagnosis_grounding 清空。"""

import unittest

from backend.rules.output_rules import enforce_diagnosis_grounding


class RAGEmptyResultTest(unittest.TestCase):
    """验证知识库无结果时诊断依据强制清空，防止 LLM 编造。"""

    def test_empty_retrieval_clears_specific_causes(self) -> None:
        """retrieval results 为空时清空 likely_issue_areas / fault_code_interpretation / evidence_sources。"""
        diagnosis = {
            "summary": "可能是主板故障或漏保模块损坏。",
            "evidence_status": "grounded",
            "likely_issue_areas": ["主板故障", "漏保模块损坏"],
            "fault_code_interpretation": [{"code": "C-RCD-04", "meaning": "漏保自检失败"}],
            "evidence_sources": ["充电桩手册 第3页"],
            "safe_remote_checks": ["拍照"],
            "priority": "p1_high",
            "suggested_next_step": "上门维修",
        }
        case = {"brand": "VoltGate", "charger_model": "VG-11KW-Pro", "issue_description": "不能充电"}
        retrieval = {"results": [], "sources": []}

        result = enforce_diagnosis_grounding(diagnosis, case, retrieval)

        self.assertEqual(result["evidence_status"], "insufficient")
        self.assertEqual(result["likely_issue_areas"], [])
        self.assertEqual(result["fault_code_interpretation"], [])
        self.assertEqual(result["evidence_sources"], [])
        self.assertIn("依据不足", result["summary"])
        self.assertIn("不能自动判断具体原因", result["summary"])
        self.assertNotIn("主板", result["summary"])

    def test_empty_retrieval_sets_insufficient_status(self) -> None:
        """retrieval results 为空时 evidence_status 强制设为 insufficient。"""
        diagnosis = {"evidence_status": "partial"}
        case = {}
        retrieval = {}

        result = enforce_diagnosis_grounding(diagnosis, case, retrieval)
        self.assertEqual(result["evidence_status"], "insufficient")

    def test_non_empty_retrieval_preserves_diagnosis(self) -> None:
        """retrieval results 非空时诊断结果原样保留。"""
        diagnosis = {
            "summary": "漏保自检失败，需要安全核验。",
            "evidence_status": "grounded",
            "likely_issue_areas": ["漏保故障"],
            "fault_code_interpretation": [{"code": "C-RCD-04", "meaning": "漏保自检"}],
            "evidence_sources": ["充电桩手册 第3页"],
        }
        case = {}
        retrieval = {"results": [{"file_name": "手册.pdf", "text": "C-RCD-04"}]}

        result = enforce_diagnosis_grounding(diagnosis, case, retrieval)
        self.assertEqual(result["summary"], "漏保自检失败，需要安全核验。")
        self.assertEqual(result["evidence_status"], "grounded")
        self.assertEqual(result["likely_issue_areas"], ["漏保故障"])

    def test_empty_retrieval_truncates_long_llm_fabrication(self) -> None:
        """retrieval 为空时 LLM 编造的长篇解释被替换为安全提示。"""
        diagnosis = {
            "summary": "根据充电桩工作原理，C-RCD-04 故障码代表漏保模块检测到剩余电流超过 30mA，可能是主板电容老化导致漏电流增大，建议更换主板电容或升级漏保模块为 100mA 型号。",
        }
        case = {"brand": "VoltGate", "charger_model": "VG-11KW-Pro", "issue_description": "屏幕 VG-E01"}
        retrieval = {}

        result = enforce_diagnosis_grounding(diagnosis, case, retrieval)

        # 编造的具体原因不应出现在 summary 中
        self.assertNotIn("30mA", result["summary"])
        self.assertNotIn("电容老化", result["summary"])
        self.assertIn("依据不足", result["summary"])


if __name__ == "__main__":
    unittest.main()
