from __future__ import annotations

import json
import unittest
from typing import Any

from backend.agents.action_agent import ChargerActionAgent
from backend.agents.audit_agent import ChargerAuditAgent
from backend.agents.case_extract_agent import ChargerCaseExtractAgent
from backend.agents.diagnosis_agent import ChargerDiagnosisAgent
from backend.agents.intent_agent import ChargerTriageAgent
from backend.rules import case_rules, output_rules, safety_rules
from backend.schemas import (
    ChargerActionResult,
    ChargerAuditResult,
    ChargerCase,
    ChargerDiagnosisResult,
    DispatchDraft,
    TriageResult,
    WarrantyResult,
)


class FakeLLM:
    def __init__(self, content: dict[str, Any] | str) -> None:
        self.content = json.dumps(content, ensure_ascii=False) if isinstance(content, dict) else content
        self.calls: list[Any] = []

    def __call__(self, prompt_value: Any) -> str:
        self.calls.append(prompt_value)
        return self.content


class ChargerLLMAgentsTest(unittest.TestCase):
    def test_triage_agent_uses_charger_llm_semantics_when_available(self) -> None:
        llm = FakeLLM({
            "intent": "service_dispatch",
            "confidence": "high",
            "reason": "客户要求充电桩上门处理。",
        })
        agent = ChargerTriageAgent(llm=llm)

        triage = agent.triage("VG-WallBox2 充到一半停止，想安排上门")

        self.assertEqual(triage["intent"], "service_dispatch")
        self.assertEqual(triage["confidence"], "high")
        self.assertEqual(set(triage), set(TriageResult().to_dict()))
        self.assertTrue(llm.calls)

    def test_triage_fallback_stays_minimal_without_rule_semantics(self) -> None:
        agent = ChargerTriageAgent(llm=None)

        high_risk = agent.triage("充电桩枪线破皮，还有烧焦味")
        unclear = agent.triage("想问一下充电桩怎么预约充电")

        self.assertEqual(high_risk["intent"], "unknown")
        self.assertEqual(high_risk["confidence"], "low")
        self.assertIn("LLM 不可用", high_risk["reason"])
        self.assertEqual(unclear["intent"], "unknown")

    def test_case_extract_agent_uses_charger_schema(self) -> None:
        llm = FakeLLM({
            "brand": "VoltGate",
            "charger_model": "VG-11KW-Pro",
            "charger_series": "VG",
            "serial_number": "SNVG123456",
            "charger_type": "交流家用充电桩",
            "installation_type": "地下车库壁挂",
            "rated_power_kw": "11",
            "connector_type": "国标枪",
            "power_supply_phase": "三相",
            "breaker_or_rcd_info": "漏保频繁跳闸",
            "grounding_status": "待核验",
            "vehicle_brand_model": "某新能源 SUV",
            "issue_type": "rcd_or_grounding",
            "issue_description": "无法启动充电，屏幕显示 C-RCD-04",
            "fault_codes": ["C-RCD-04"],
            "observed_symptoms": ["无法启动充电"],
            "safety_signals": ["漏保频繁跳闸"],
            "environment_factors": ["地下车库"],
            "installation_or_recent_changes": ["昨天重启过"],
            "customer_actions": ["重启一次"],
            "customer_requests": ["上门"],
            "purchase_or_install_time": "2个月",
            "warranty_or_order_evidence": "购买平台截图",
            "city": "东莞",
            "contact_name": "",
            "contact_phone": "13800138000",
            "contact_address": "东莞市测试小区地下车库",
        })
        agent = ChargerCaseExtractAgent(llm=llm)

        case = agent.extract("VoltGate VG-11KW-Pro SNVG123456 C-RCD-04，漏保频繁跳闸，电话13800138000")

        self.assertEqual(case["brand"], "VoltGate")
        self.assertEqual(case["charger_model"], "VG-11KW-Pro")
        self.assertIn("C-RCD-04", case["fault_codes"])
        self.assertIn("漏保频繁跳闸", case["safety_signals"])
        self.assertEqual(case["contact_phone"], "13800138000")
        self.assertEqual(case["purchase_or_install_time"], "2个月")
        self.assertEqual(set(case), set(ChargerCase().to_dict()))
        self.assertTrue(llm.calls)

    def test_case_extract_agent_fallback_is_minimal_and_rules_do_structure(self) -> None:
        agent = ChargerCaseExtractAgent(llm=None)
        raw_text = (
            "VoltGate VG-7KW-AC SN:ABC123456 APP离线，屏幕显示 C-GND-01，"
            "枪线破皮，电话 13800138000，地址在广州天河。"
        )

        case = agent.extract(raw_text)

        self.assertEqual(case["brand"], "")
        self.assertEqual(case["charger_model"], "")
        self.assertEqual(case["serial_number"], "")
        self.assertEqual(case["fault_codes"], [])
        self.assertEqual(case["safety_signals"], [])
        self.assertEqual(case["contact_phone"], "")
        self.assertEqual(case["raw_text"], raw_text)

        normalized = case_rules.normalize_charger_case(case, raw_text)
        self.assertEqual(normalized["brand"], "VoltGate")
        self.assertEqual(normalized["charger_model"], "VG-7KW-AC")
        self.assertEqual(normalized["serial_number"], "ABC123456")
        self.assertIn("C-GND-01", normalized["fault_codes"])
        self.assertEqual(normalized["contact_phone"], "13800138000")
        self.assertEqual(normalized["safety_signals"], [])
        safety = safety_rules.evaluate_charger_safety(normalized, raw_text)
        self.assertIn("枪线破皮", safety["matched_safety_signals"])
        self.assertEqual(safety["risk_level"], "p1_high")
        self.assertEqual(normalized["purchase_or_install_time"], "")
        self.assertEqual(normalized["contact_address"], "")
        self.assertIn("安装地址", normalized["missing_info"])
        self.assertNotIn("model_or_sku", normalized)

    def test_diagnosis_agent_uses_llm_and_rag_for_charger_issue(self) -> None:
        llm = FakeLLM({
            "summary": "VG-11KW-Pro 显示 C-RCD-04，知识库提示与漏保自检失败相关，需要结合现场照片核验。",
            "evidence_status": "grounded",
            "likely_issue_areas": ["漏保模块或相关检测链路"],
            "fault_code_interpretation": ["C-RCD-04：漏保自检失败"],
            "safe_remote_checks": ["拍摄屏幕报错和安装环境照片", "确认是否存在进水、异味或跳闸"],
            "onsite_reasons": ["漏保相关问题需专业人员核验"],
            "priority": "p2_medium",
            "suggested_next_step": "按知识库采集证据后转人工判断是否派工。",
            "evidence_sources": ["充电桩手册.pdf 第2页"],
            "risk_flags": [],
        })
        agent = ChargerDiagnosisAgent(llm=llm)

        diagnosis = agent.diagnose(
            case={
                "brand": "VoltGate",
                "charger_model": "VG-11KW-Pro",
                "issue_description": "无法启动充电",
                "fault_codes": ["C-RCD-04"],
                "raw_text": "VG-11KW-Pro C-RCD-04 无法启动充电",
            },
            retrieval={
                "sources": ["充电桩手册.pdf 第2页"],
                "results": [{"text": "C-RCD-04 漏保自检失败。请客户拍摄屏幕、App 报错和安装环境照片。"}],
            },
            tools={"safety": {"risk_level": "p3_low", "matched_safety_signals": []}},
        )

        self.assertEqual(diagnosis["evidence_status"], "grounded")
        self.assertIn("C-RCD-04", diagnosis["fault_code_interpretation"][0])
        self.assertEqual(diagnosis["priority"], "p2_medium")
        self.assertEqual(set(diagnosis), set(ChargerDiagnosisResult().to_dict()))
        self.assertTrue(llm.calls)

    def test_diagnosis_agent_does_not_apply_grounding_guard_itself(self) -> None:
        llm = FakeLLM({
            "summary": "可以确定是主板坏了。",
            "evidence_status": "grounded",
            "likely_issue_areas": ["主板"],
            "fault_code_interpretation": ["未知故障码"],
            "safe_remote_checks": ["开盖检查"],
            "onsite_reasons": [],
            "priority": "normal",
            "suggested_next_step": "直接换主板。",
            "evidence_sources": [],
            "risk_flags": [],
        })

        case = {"charger_model": "VG-7KW-AC", "issue_description": "不能充电", "raw_text": "不能充电"}
        retrieval = {"sources": [], "results": []}
        diagnosis = ChargerDiagnosisAgent(llm=llm).diagnose(
            case=case,
            retrieval=retrieval,
            tools={"safety": {"risk_level": "p3_low", "matched_safety_signals": []}},
        )

        self.assertEqual(diagnosis["likely_issue_areas"], ["主板"])
        self.assertEqual(diagnosis["fault_code_interpretation"], ["未知故障码"])
        self.assertEqual(diagnosis["evidence_status"], "grounded")
        self.assertIn("开盖检查", diagnosis["safe_remote_checks"])
        self.assertTrue(llm.calls)

        guarded = output_rules.enforce_diagnosis_grounding(diagnosis, case, retrieval)
        self.assertEqual(guarded["likely_issue_areas"], [])
        self.assertEqual(guarded["fault_code_interpretation"], [])
        self.assertEqual(guarded["evidence_status"], "insufficient")
        self.assertIn("知识库依据不足", guarded["summary"])

    def test_high_risk_guard_overrides_unsafe_llm_diagnosis_and_reply(self) -> None:
        diagnosis_llm = FakeLLM({
            "summary": "只是轻微异常，可以继续充电观察。",
            "evidence_status": "partial",
            "likely_issue_areas": ["偶发问题"],
            "fault_code_interpretation": [],
            "safe_remote_checks": ["继续充电观察"],
            "onsite_reasons": [],
            "priority": "normal",
            "suggested_next_step": "无需升级。",
            "evidence_sources": [],
            "risk_flags": [],
        })
        safety = {
            "risk_level": "p1_high",
            "need_human": True,
            "need_onsite": True,
            "need_electrician": True,
            "reason": "命中高风险充电桩安全信号：枪线破皮。",
            "matched_safety_signals": ["枪线破皮"],
            "required_customer_actions": ["立即停止充电并暂停使用充电桩。", "远离充电桩和枪线。"],
            "forbidden_actions": [],
        }
        raw_diagnosis = ChargerDiagnosisAgent(llm=diagnosis_llm).diagnose(
            case={
                "charger_model": "VG-7KW-AC",
                "issue_description": "枪线破皮",
                "safety_signals": ["枪线破皮"],
                "raw_text": "充电桩枪线破皮",
            },
            retrieval={"sources": [], "results": []},
            tools={"safety": safety},
        )

        self.assertEqual(raw_diagnosis["priority"], "normal")
        self.assertIn("继续充电观察", raw_diagnosis["safe_remote_checks"])
        self.assertTrue(diagnosis_llm.calls)

        diagnosis = safety_rules.enforce_diagnosis(raw_diagnosis, safety)
        self.assertEqual(diagnosis["priority"], "p1_high")
        self.assertIn("枪线破皮", diagnosis["risk_flags"])

        action_llm = FakeLLM({"customer_reply": "可以继续充电观察。", "internal_advice": "无需处理。"})
        raw_action = ChargerActionAgent(llm=action_llm).generate(
            case={"charger_model": "VG-7KW-AC", "safety_signals": ["枪线破皮"], "raw_text": "充电桩枪线破皮"},
            diagnosis=diagnosis,
            warranty={"status": "unknown"},
            safety=safety,
            dispatch={},
        )

        self.assertIn("继续充电观察", raw_action["customer_reply"])
        self.assertTrue(action_llm.calls)

        action = output_rules.enforce_reply(
            raw_action,
            case={"charger_model": "VG-7KW-AC", "safety_signals": ["枪线破皮"], "raw_text": "充电桩枪线破皮"},
            safety=safety,
            warranty={"status": "unknown"},
            retrieval={"sources": [], "results": []},
            dispatch={},
        )
        self.assertIn("停止充电", action["customer_reply"])
        self.assertIn("远离", action["customer_reply"])
        self.assertIn("不要自行拆修", action["customer_reply"])

    def test_high_risk_reply_expands_rcd_specific_forbidden_actions_without_duplicates(self) -> None:
        case = {
            "charger_model": "VG-11KW-Pro",
            "fault_codes": ["C-RCD-04"],
            "raw_text": "VG-11KW-Pro 显示 C-RCD-04，漏保跳闸，我想重新合上漏保再试",
        }
        safety = safety_rules.evaluate_charger_safety(case, raw_text=case["raw_text"])

        action = output_rules.enforce_reply(
            {"customer_reply": "可以合上漏保再试，继续充电观察。", "internal_advice": ""},
            case=case,
            safety=safety,
            warranty={"status": "unknown"},
            retrieval={"sources": [], "results": []},
            dispatch={},
        )

        reply = action["customer_reply"]
        self.assertIn("停止充电", reply)
        self.assertIn("远离", reply)
        self.assertIn("切断", reply)
        self.assertIn("不要自行拆修", reply)
        self.assertEqual(reply.count("不要自行拆修"), 1)
        self.assertIn("不要重新合上漏保或空开", reply)
        self.assertIn("不要反复复位", reply)
        self.assertIn("不要继续充电观察", reply)
        self.assertTrue(output_rules.has_required_safety_reply(reply))
        self.assertNotIn("保证免费", reply)

    def test_high_risk_reply_expands_water_heat_and_burnt_smell_forbidden_actions(self) -> None:
        case = {
            "charger_model": "VG-7KW-AC",
            "raw_text": "雨水倒灌后地面积水，枪头发热，还有焦糊味",
        }
        safety = safety_rules.evaluate_charger_safety(case, raw_text=case["raw_text"])

        action = output_rules.enforce_reply(
            {"customer_reply": "先摸一下枪头看看温度。", "internal_advice": ""},
            case=case,
            safety=safety,
            warranty={"status": "unknown"},
            retrieval={"sources": [], "results": []},
            dispatch={},
        )

        reply = action["customer_reply"]
        self.assertIn("远离积水", reply)
        self.assertIn("不要触摸枪头、枪线", reply)
        self.assertIn("发热", reply)
        self.assertIn("切断", reply)
        self.assertIn("上门电工", reply)

    def test_warranty_and_dangerous_action_guards(self) -> None:
        llm = FakeLLM({
            "customer_reply": "您好，这个情况我们保证免费并一定换新，您可以开盖检修后继续充电观察。",
            "internal_advice": "话术需要复核。",
        })
        case = {"purchase_or_install_time": "6个月", "raw_text": "买了半年是不是免费维修", "customer_requests": ["换新"]}
        warranty = {"status": "possibly_in_warranty", "reason": "可能仍在保障期，需凭证核验。"}
        raw_action = ChargerActionAgent(llm=llm).generate(
            case=case,
            diagnosis={"summary": "客户咨询保修资格", "safe_remote_checks": []},
            warranty=warranty,
            safety={"risk_level": "p3_low"},
            dispatch={},
            triage={"intent": "warranty_consultation"},
        )

        self.assertIn("保证免费", raw_action["customer_reply"])
        self.assertIn("开盖检修", raw_action["customer_reply"])
        self.assertTrue(llm.calls)

        action = output_rules.enforce_reply(
            raw_action,
            case=case,
            safety={"risk_level": "p3_low"},
            warranty=warranty,
            retrieval={"sources": [], "results": []},
            dispatch={},
        )
        self.assertNotIn("保证免费", action["customer_reply"])
        self.assertNotIn("一定换新", action["customer_reply"])
        self.assertNotIn("开盖检修", action["customer_reply"])
        self.assertNotIn("继续充电观察", action["customer_reply"])
        self.assertIn("凭证", action["customer_reply"])
        self.assertEqual(set(action), set(ChargerActionResult().to_dict()))

    def test_audit_stacks_local_warnings_over_llm(self) -> None:
        case = {"purchase_or_install_time": "", "raw_text": "充电桩枪线破皮"}
        diagnosis = {"priority": "p1_high"}
        retrieval = {"sources": [], "results": []}
        action = {"customer_reply": "您可以继续充电观察。", "internal_advice": ""}
        safety = {"risk_level": "p1_high", "matched_safety_signals": ["枪线破皮"]}
        warranty = {"status": "unknown"}
        llm_audit = ChargerAuditAgent(
            llm=FakeLLM({"passed": True, "warnings": [], "final_note": "可回复", "risk_level": "p3_low"})
        ).audit(
            case=case,
            diagnosis=diagnosis,
            retrieval=retrieval,
            action=action,
            safety=safety,
            warranty=warranty,
        )

        self.assertTrue(llm_audit["passed"])
        self.assertEqual(llm_audit["warnings"], [])

        audit = output_rules.merge_with_local_audit(
            llm_audit,
            case=case,
            diagnosis=diagnosis,
            action=action,
            safety=safety,
            warranty=warranty,
            retrieval=retrieval,
        )

        self.assertFalse(audit["passed"])
        combined = " ".join(audit["warnings"])
        self.assertIn("高风险充电桩安全问题", combined)
        self.assertIn("继续充电观察", combined)
        self.assertEqual(audit["risk_level"], "p1_high")
        self.assertEqual(set(audit), set(ChargerAuditResult().to_dict()))

    def test_schema_defaults_are_charger_specific(self) -> None:
        self.assertIn("charger_model", ChargerCase().to_dict())
        self.assertIn("safety_level", DispatchDraft().to_dict())
        self.assertNotIn("model_or_sku", ChargerCase().to_dict())
        warranty = WarrantyResult().to_dict()
        self.assertIsNone(warranty["policy_months"])
        self.assertEqual(warranty["policy_sources"], [])


if __name__ == "__main__":
    unittest.main()
