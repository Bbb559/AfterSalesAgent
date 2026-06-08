"""memory_answer v2 parse + resolver 层验证脚本。

覆盖：
  - MemoryQueryResult 字段校验 / 清洗 / normalize / fallback_reason
  - MemoryFieldResolution 输出结构
  - _resolve_memory_fields() 来源路由 & 置信度
  - _build_memory_reply_v2() 基于 resolver 输出的回复模板
  - 真实链路测试：同一 session 输入后召回
  - eval_mem_001~010 用例结构校验

不依赖 LLM（不调用 API）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.schemas import MemoryFieldResolution, MemoryQueryResult

EVAL_FILE = Path(__file__).with_name("memory_answer_eval.json")


def load_eval_cases() -> list[dict[str, Any]]:
    with open(EVAL_FILE, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"eval 文件顶层应为数组，实际为 {type(data).__name__}")
    return data


def compare_fields(actual: list[str], expected: list[str]) -> bool:
    return set(actual) == set(expected)


# ============================================================================
# 测试 1: MemoryQueryResult 校验 & 清洗
# ============================================================================

def test_memory_result_validation() -> tuple[int, int]:
    passed = 0
    failed = 0

    print("=" * 60)
    print("TEST: MemoryQueryResult 校验 & 清洗")
    print("=" * 60)

    # --- validate_fields & clean_fields ---
    result = MemoryQueryResult(
        is_memory_query=True,
        target_fields=["brand", "user_name", "charger_model", "phone_number", "city"],
        query_scope="recent",
        answer_style="precise",
    )

    illegal = result.validate_fields()
    expected_illegal = ["user_name", "phone_number"]
    if set(illegal) == set(expected_illegal):
        print(f"  PASS validate_fields: 非法字段 {illegal}")
        passed += 1
    else:
        print(f"  FAIL validate_fields: 期望 {expected_illegal}，实际 {illegal}")
        failed += 1

    result.clean_fields()
    expected_clean = ["brand", "charger_model", "city"]
    if compare_fields(result.target_fields, expected_clean):
        print(f"  PASS clean_fields: 清洗后 {result.target_fields}")
        passed += 1
    else:
        print(f"  FAIL clean_fields: 期望 {expected_clean}，实际 {result.target_fields}")
        failed += 1

    # --- normalize_scope ---
    result.query_scope = "all_history"
    if not result.validate_query_scope():
        result.normalize_scope()
    if result.query_scope == "recent":
        print(f"  PASS normalize_scope: 'all_history' → 'recent'")
        passed += 1
    else:
        print(f"  FAIL normalize_scope: 期望 'recent'，实际 '{result.query_scope}'")
        failed += 1

    # --- normalize_answer_style ---
    result.answer_style = "detailed"
    if not result.validate_answer_style():
        result.normalize_answer_style()
    if result.answer_style == "precise":
        print(f"  PASS normalize_answer_style: 'detailed' → 'precise'")
        passed += 1
    else:
        print(f"  FAIL normalize_answer_style: 期望 'precise'，实际 '{result.answer_style}'")
        failed += 1

    # --- 合法值不变 ---
    result.query_scope = "session"
    result.answer_style = "summary"
    if result.validate_query_scope():
        result.normalize_scope()
    if result.validate_answer_style():
        result.normalize_answer_style()
    if result.query_scope == "session" and result.answer_style == "summary":
        print(f"  PASS 合法值不变: scope={result.query_scope}, style={result.answer_style}")
        passed += 1
    else:
        print(f"  FAIL 合法值被修改: scope={result.query_scope}, style={result.answer_style}")
        failed += 1

    # --- fallback_reason 字段 ---
    fb_result = MemoryQueryResult(is_memory_query=True, fallback_reason="parse_failed")
    if fb_result.fallback_reason == "parse_failed" and fb_result.is_memory_query:
        print(f"  PASS fallback_reason: '{fb_result.fallback_reason}' + is_memory_query=True")
        passed += 1
    else:
        print(f"  FAIL fallback_reason")
        failed += 1

    return passed, failed


# ============================================================================
# 测试 2: eval_010 validation 用例
# ============================================================================

def test_eval_cases_validation_only() -> tuple[int, int]:
    cases = load_eval_cases()
    validation_cases = [c for c in cases if c.get("type") == "validation"]

    passed = 0
    failed = 0

    print()
    print("=" * 60)
    print("TEST: eval validation 类型用例")
    print("=" * 60)

    for case in validation_cases:
        case_id = case["id"]
        mock_output = case.get("mock_llm_output", {})
        expected_clean = case.get("expected_after_clean", {})
        expected_illegal = case.get("expected_illegal_fields", [])

        result = MemoryQueryResult(
            is_memory_query=mock_output.get("is_memory_query", False),
            target_fields=mock_output.get("target_fields", []),
            query_scope=mock_output.get("query_scope", "recent"),
            entities=mock_output.get("entities", []),
            answer_style=mock_output.get("answer_style", "precise"),
        )

        illegal = result.validate_fields()
        result.clean_fields()
        result.normalize_scope()
        result.normalize_answer_style()

        all_ok = True

        if set(illegal) != set(expected_illegal):
            print(f"  FAIL {case_id}: 非法字段 期望 {expected_illegal}，实际 {illegal}")
            all_ok = False

        if not compare_fields(result.target_fields, expected_clean.get("target_fields", [])):
            print(f"  FAIL {case_id}: target_fields 期望 {expected_clean['target_fields']}，实际 {result.target_fields}")
            all_ok = False

        if result.query_scope != expected_clean.get("query_scope", "recent"):
            print(f"  FAIL {case_id}: query_scope 期望 '{expected_clean['query_scope']}'，实际 '{result.query_scope}'")
            all_ok = False

        if result.answer_style != expected_clean.get("answer_style", "precise"):
            print(f"  FAIL {case_id}: answer_style 期望 '{expected_clean['answer_style']}'，实际 '{result.answer_style}'")
            all_ok = False

        if result.is_memory_query != expected_clean.get("is_memory_query", True):
            print(f"  FAIL {case_id}: is_memory_query 期望 {expected_clean['is_memory_query']}，实际 {result.is_memory_query}")
            all_ok = False

        if all_ok:
            print(f"  PASS {case_id}: {case.get('scenario', '')}")
            passed += 1
        else:
            failed += 1

    return passed, failed


# ============================================================================
# 测试 3: _resolve_memory_fields() 来源路由 & 置信度
# ============================================================================

def test_resolver_routing() -> tuple[int, int]:
    from backend.graph_workflow import ChargerDiagnosisWorkflow

    passed = 0
    failed = 0

    print()
    print("=" * 60)
    print("TEST: _resolve_memory_fields 来源路由 & 置信度")
    print("=" * 60)

    # 模拟 memory_context（结构与 MemoryManager.recall_context() 对齐）
    mock_ctx: dict[str, Any] = {
        "last_case": {
            "brand": "星星充电",
            "charger_model": "AC-22KW",
            "city": "杭州",
            "rated_power_kw": "22",
            "fault_codes": ["C-RCD-04"],
        },
        "recent_safety": {
            "risk_level": "p2_medium",
            "need_onsite": False,
            "need_electrician": True,
        },
        "recent_ticket": {
            "priority": "high",
            "title": "漏保跳闸-杭州",
        },
        "session": {
            "last_diagnosis": {
                "summary": "疑似漏保故障",
                "suggested_next_step": "安排电工上门",
            },
            "recent_user_messages": ["充电桩无法启动，屏幕显示 C-RCD-04"],
        },
        "last_customer_reply": "好的，我已记录。请确认漏保是否复位？",
        "missing_info": ["序列号"],
        "session_search": {"available": False, "matches": [], "query": ""},
    }

    wf = ChargerDiagnosisWorkflow()
    _resolve = wf._resolve_memory_fields

    # --- Case 1: 单字段 from last_case ---
    parsed = MemoryQueryResult(is_memory_query=True, target_fields=["brand"])
    res = _resolve(parsed, mock_ctx, "", {})
    if res.resolved_values == {"brand": "星星充电"} and res.resolver_sources["brand"] == "last_case.brand" and res.confidence == "high":
        print(f"  PASS last_case.brand → high")
        passed += 1
    else:
        print(f"  FAIL last_case.brand: {res.resolved_values}, {res.resolver_sources}, {res.confidence}")
        failed += 1

    # --- Case 2: SafetyResult ---
    parsed2 = MemoryQueryResult(is_memory_query=True, target_fields=["risk_level"])
    res2 = _resolve(parsed2, mock_ctx, "", {})
    if res2.resolved_values == {"risk_level": "p2_medium"} and res2.confidence == "high":
        print(f"  PASS recent_safety.risk_level → high")
        passed += 1
    else:
        print(f"  FAIL recent_safety.risk_level: {res2.resolved_values}")
        failed += 1

    # --- Case 3: diagnosis_summary from session.last_diagnosis.summary ---
    parsed3 = MemoryQueryResult(is_memory_query=True, target_fields=["diagnosis_summary"])
    res3 = _resolve(parsed3, mock_ctx, "", {})
    if res3.resolved_values.get("diagnosis_summary") == "疑似漏保故障":
        print(f"  PASS session.last_diagnosis.summary → diagnosis_summary 映射")
        passed += 1
    else:
        print(f"  FAIL diagnosis_summary 映射: {res3.resolved_values}")
        failed += 1

    # --- Case 4: ticket_priority from recent_ticket.priority ---
    parsed4 = MemoryQueryResult(is_memory_query=True, target_fields=["ticket_priority"])
    res4 = _resolve(parsed4, mock_ctx, "", {})
    if res4.resolved_values.get("ticket_priority") == "high":
        print(f"  PASS recent_ticket.priority → ticket_priority 映射")
        passed += 1
    else:
        print(f"  FAIL ticket_priority 映射: {res4.resolved_values}")
        failed += 1

    # --- Case 5: last_customer_reply from 顶层 ---
    parsed5 = MemoryQueryResult(is_memory_query=True, target_fields=["last_customer_reply"])
    res5 = _resolve(parsed5, mock_ctx, "", {})
    if "好的，我已记录" in str(res5.resolved_values.get("last_customer_reply", "")):
        print(f"  PASS last_customer_reply 从顶层读取")
        passed += 1
    else:
        print(f"  FAIL last_customer_reply: {res5.resolved_values}")
        failed += 1

    # --- Case 6: last_user_message from session.recent_user_messages ---
    parsed6 = MemoryQueryResult(is_memory_query=True, target_fields=["last_user_message"])
    res6 = _resolve(parsed6, mock_ctx, "", {})
    if "C-RCD-04" in str(res6.resolved_values.get("last_user_message", "")):
        print(f"  PASS last_user_message 从 session.recent_user_messages 读取")
        passed += 1
    else:
        print(f"  FAIL last_user_message: {res6.resolved_values}")
        failed += 1

    # --- Case 7: missing_info from 顶层 ---
    parsed7 = MemoryQueryResult(is_memory_query=True, target_fields=["missing_info"])
    res7 = _resolve(parsed7, mock_ctx, "", {})
    if "序列号" in str(res7.resolved_values.get("missing_info", [])):
        print(f"  PASS missing_info 从顶层读取")
        passed += 1
    else:
        print(f"  FAIL missing_info: {res7.resolved_values}")
        failed += 1

    # --- Case 8: 字段不存在 → missing ---
    parsed8 = MemoryQueryResult(is_memory_query=True, target_fields=["serial_number"])
    res8 = _resolve(parsed8, mock_ctx, "", {})
    if "serial_number" in res8.missing_fields and res8.confidence == "low":
        print(f"  PASS serial_number 缺失 → confidence=low")
        passed += 1
    else:
        print(f"  FAIL serial_number: missing={res8.missing_fields}, conf={res8.confidence}")
        failed += 1

    # --- Case 9: 多字段部分命中 → medium ---
    parsed9 = MemoryQueryResult(is_memory_query=True, target_fields=["brand", "serial_number", "city"])
    res9 = _resolve(parsed9, mock_ctx, "", {})
    if (res9.confidence == "medium"
            and "brand" in res9.resolved_values
            and "city" in res9.resolved_values
            and "serial_number" in res9.missing_fields):
        print(f"  PASS 部分命中 → confidence=medium")
        passed += 1
    else:
        print(f"  FAIL 部分命中: resolved={res9.resolved_values}, missing={res9.missing_fields}, conf={res9.confidence}")
        failed += 1

    # --- Case 10: fallback 全扫描 ---
    parsed10 = MemoryQueryResult(is_memory_query=True, target_fields=[], fallback_reason="parse_failed")
    res10 = _resolve(parsed10, mock_ctx, "", {})
    if len(res10.resolved_values) > 0:
        print(f"  PASS fallback 全扫描: resolved={len(res10.resolved_values)}, missing={len(res10.missing_fields)}, conf={res10.confidence}")
        passed += 1
    else:
        print(f"  FAIL fallback 全扫描: resolved={res10.resolved_values}")
        failed += 1

    return passed, failed


# ============================================================================
# 测试 3B: FTS5 fallback 行为
# ============================================================================

def test_fts5_fallback() -> tuple[int, int]:
    """测试 FTS5 fallback 的各种场景（不调用真实 LLM）。"""
    from backend.graph_workflow import ChargerDiagnosisWorkflow

    passed = 0
    failed = 0

    print()
    print("=" * 60)
    print("TEST: FTS5 fallback")
    print("=" * 60)

    wf = ChargerDiagnosisWorkflow()

    # --- Case 1: FTS5 不覆盖 high 置信度字段 ---
    mock_ctx: dict[str, Any] = {
        "last_case": {"brand": "华为"},
        "recent_safety": {},
        "recent_ticket": {},
        "session": {"last_diagnosis": {}, "recent_user_messages": ["我家华为 7kW 家充桩不能充电"]},
        "last_customer_reply": "",
        "missing_info": [],
        "session_search": {
            "available": True,
            "query": "7kW",
            "matches": [
                {"role": "user", "content": "我家华为 7kW 家充桩不能充电，屏幕不亮"},
            ],
        },
    }
    parsed = MemoryQueryResult(
        is_memory_query=True,
        target_fields=["brand", "rated_power_kw"],
        entities=["7kW"],
    )
    res = wf._resolve_memory_fields(parsed, mock_ctx, "刚才我说的是什么功率？", {})
    # brand 应从结构化来源命中（high），不应被 FTS5 覆盖
    if res.resolver_sources.get("brand") == "last_case.brand":
        print(f"  PASS FTS5 不覆盖 high 置信度字段: brand 来源=last_case.brand")
        passed += 1
    else:
        print(f"  FAIL brand 来源应为 last_case.brand，实际={res.resolver_sources.get('brand')}")
        failed += 1
    # rated_power_kw 结构化缺失 → FTS5 尝试提取
    # 如果 LLM 可用且提取成功 → 来源应为 fts5.message[0]
    # 如果 LLM 不可用 → 保留在 missing_fields
    if "rated_power_kw" in res.resolved_values:
        source = res.resolver_sources.get("rated_power_kw", "")
        if source.startswith("fts5.message"):
            print(f"  PASS rated_power_kw FTS5 提取成功，来源={source}")
            passed += 1
        else:
            print(f"  FAIL rated_power_kw FTS5 提取后来源异常: {source}")
            failed += 1
    else:
        print(f"  PASS rated_power_kw 结构化缺失 + FTS5 未提取（LLM 不可用），保持 missing")
        passed += 1

    # --- Case 2: FTS5 不可用时跳过 ---
    mock_ctx_no_fts5: dict[str, Any] = {
        "last_case": {},
        "recent_safety": {},
        "recent_ticket": {},
        "session": {"last_diagnosis": {}, "recent_user_messages": []},
        "last_customer_reply": "",
        "missing_info": [],
        "session_search": {"available": False, "matches": [], "query": ""},
    }
    parsed2 = MemoryQueryResult(is_memory_query=True, target_fields=["brand", "city"])
    res2 = wf._resolve_memory_fields(parsed2, mock_ctx_no_fts5, "刚才我说了什么？", {})
    if res2.confidence == "low" and len(res2.missing_fields) == 2:
        print(f"  PASS FTS5 不可用时跳过，confidence=low, missing={res2.missing_fields}")
        passed += 1
    else:
        print(f"  FAIL FTS5 不可用: conf={res2.confidence}, missing={res2.missing_fields}")
        failed += 1

    # --- Case 3: FTS5 可用但无匹配 ---
    mock_ctx_no_match: dict[str, Any] = {
        "last_case": {},
        "recent_safety": {},
        "recent_ticket": {},
        "session": {"last_diagnosis": {}, "recent_user_messages": []},
        "last_customer_reply": "",
        "missing_info": [],
        "session_search": {"available": True, "matches": [], "query": "test"},
    }
    parsed3 = MemoryQueryResult(is_memory_query=True, target_fields=["serial_number"])
    res3 = wf._resolve_memory_fields(parsed3, mock_ctx_no_match, "序列号是什么？", {})
    if res3.confidence == "low" and "serial_number" in res3.missing_fields:
        print(f"  PASS FTS5 无匹配时保持 missing, confidence=low")
        passed += 1
    else:
        print(f"  FAIL FTS5 无匹配: conf={res3.confidence}, missing={res3.missing_fields}")
        failed += 1

    # --- Case 4: _fts5_extract_fields 直接调用（LLM=None 强制测试优雅降级）---
    mock_ctx_fts5: dict[str, Any] = {
        "last_case": {},
        "session": {"last_diagnosis": {}, "recent_user_messages": []},
        "recent_safety": {},
        "recent_ticket": {},
        "last_customer_reply": "",
        "missing_info": [],
        "session_search": {
            "available": True,
            "query": "华为 7kW",
            "matches": [
                {"role": "user", "content": "我家华为 7kW 家充桩不能充电"},
                {"role": "assistant", "content": "好的，已记录。"},
            ],
        },
    }
    # 用 llm=None 强制创建 workflow，确保 FTS5 优雅降级路径可测试
    wf_no_llm = ChargerDiagnosisWorkflow(llm=None)
    fts5_result = wf_no_llm._fts5_extract_fields(
        missing_fields=["brand", "rated_power_kw"],
        user_input="刚才是什么品牌和功率？",
        entities=["华为", "7kW"],
        memory_context=mock_ctx_fts5,
        state={},
    )
    if fts5_result is None:
        print(f"  PASS _fts5_extract_fields LLM=None 时返回 None（优雅降级）")
        passed += 1
    else:
        print(f"  FAIL _fts5_extract_fields LLM=None 时应返回 None，实际={fts5_result}")
        failed += 1

    # --- Case 5: trace 日志包含 FTS5 debug 字段 ---
    state_with_trace: dict[str, Any] = {"trace": []}
    wf._fts5_extract_fields(
        missing_fields=["brand"],
        user_input="test",
        entities=[],
        memory_context=mock_ctx_fts5,
        state=state_with_trace,
    )
    trace_entries = state_with_trace.get("trace", [])
    fts5_traces = [t for t in trace_entries if "fts5" in str(t.get("node", "")).lower()]
    if len(fts5_traces) >= 2:  # 至少 "FTS5 fallback 开始" + "LLM 不可用"
        print(f"  PASS trace 日志包含 FTS5 debug 字段 ({len(fts5_traces)} 条)")
        passed += 1
    else:
        print(f"  FAIL trace 缺少 FTS5 日志: {len(fts5_traces)} 条")
        failed += 1

    return passed, failed


# ============================================================================
# 测试 4: _build_memory_reply_v2 基于 resolver 输出
# ============================================================================

def test_reply_from_resolution() -> tuple[int, int]:
    from backend.graph_workflow import ChargerDiagnosisWorkflow

    passed = 0
    failed = 0

    print()
    print("=" * 60)
    print("TEST: _build_memory_reply_v2 基于 resolver 输出")
    print("=" * 60)

    # --- precise 模式: 全部命中 ---
    parsed = MemoryQueryResult(is_memory_query=True, target_fields=["brand", "charger_model"], answer_style="precise")
    resolution = MemoryFieldResolution(
        resolved_values={"brand": "华为", "charger_model": "AC-7KW"},
        missing_fields=[],
        confidence="high",
        resolver_sources={"brand": "last_case.brand", "charger_model": "last_case.charger_model"},
    )
    reply = ChargerDiagnosisWorkflow._build_memory_reply_v2(parsed, resolution)
    if "华为" in reply and "AC-7KW" in reply and "high" in reply:
        print(f"  PASS precise 全部命中 + confidence=high")
        passed += 1
    else:
        print(f"  FAIL precise 全部命中: {reply[:120]}")
        failed += 1

    # --- precise 模式: 部分命中 ---
    parsed2 = MemoryQueryResult(is_memory_query=True, target_fields=["brand", "serial_number"], answer_style="precise")
    resolution2 = MemoryFieldResolution(
        resolved_values={"brand": "华为"},
        missing_fields=["serial_number"],
        confidence="medium",
        resolver_sources={"brand": "last_case.brand"},
    )
    reply2 = ChargerDiagnosisWorkflow._build_memory_reply_v2(parsed2, resolution2)
    if "华为" in reply2 and "未找到" in reply2 and "序列号" in reply2 and "medium" in reply2:
        print(f"  PASS precise 部分命中 + 缺失提示")
        passed += 1
    else:
        print(f"  FAIL precise 部分命中: {reply2[:120]}")
        failed += 1

    # --- summary 模式 ---
    parsed3 = MemoryQueryResult(is_memory_query=True, target_fields=["brand", "city"], answer_style="summary")
    resolution3 = MemoryFieldResolution(
        resolved_values={"brand": "华为", "city": "杭州"},
        missing_fields=[],
        confidence="high",
        resolver_sources={"brand": "last_case.brand", "city": "last_case.city"},
    )
    reply3 = ChargerDiagnosisWorkflow._build_memory_reply_v2(parsed3, resolution3)
    if "当前会话已记录" in reply3 and "华为" in reply3 and "杭州" in reply3:
        print(f"  PASS summary 格式")
        passed += 1
    else:
        print(f"  FAIL summary 格式: {reply3[:120]}")
        failed += 1

    # --- parse fallback ---
    parsed4 = MemoryQueryResult(is_memory_query=True, target_fields=[], fallback_reason="parse_failed")
    resolution4 = MemoryFieldResolution()
    reply4 = ChargerDiagnosisWorkflow._build_memory_reply_v2(parsed4, resolution4)
    if "查询解析遇到了问题" in reply4 or "更具体" in reply4:
        print(f"  PASS parse_failed 回退提示")
        passed += 1
    else:
        print(f"  FAIL parse_failed 回退: {reply4[:120]}")
        failed += 1

    # --- is_memory_query=False ---
    parsed5 = MemoryQueryResult(is_memory_query=False)
    resolution5 = MemoryFieldResolution()
    reply5 = ChargerDiagnosisWorkflow._build_memory_reply_v2(parsed5, resolution5)
    if "需要查询什么" in reply5:
        print(f"  PASS is_memory_query=False 提示")
        passed += 1
    else:
        print(f"  FAIL is_memory_query=False: {reply5[:120]}")
        failed += 1

    return passed, failed


# ============================================================================
# 测试 5: 真实链路测试（同一 session 输入后召回）
# ============================================================================

def test_real_session_recall() -> tuple[int, int]:
    """模拟真实场景：用户先报告故障，再回忆刚才说的信息。

    预期：V2 resolver 能从 last_case 中取出 brand=华为、rated_power_kw=7。
    如果 last_case 未写入，debug log 应明确显示缺失来源。
    """
    from backend.graph_workflow import ChargerDiagnosisWorkflow

    passed = 0
    failed = 0

    print()
    print("=" * 60)
    print("TEST: 真实链路 — 同一 session 品牌+功率召回")
    print("=" * 60)

    # 模拟经过一轮诊断后 memory_context 的状态
    # 第一轮用户输入："我家华为 7kW 家充桩不能充电"
    # 经过 CaseExtractAgent 后，last_case 应有品牌和功率
    mock_ctx: dict[str, Any] = {
        "last_case": {
            "brand": "华为",
            "charger_model": "",
            "rated_power_kw": "7",
            "issue_description": "家充桩不能充电",
            "fault_codes": [],
        },
        "recent_safety": {},
        "recent_ticket": {},
        "session": {
            "recent_user_messages": [
                "我家华为 7kW 家充桩不能充电",
                "刚才我说的是什么品牌和功率？",
            ],
            "last_diagnosis": {},
            "last_dispatch": {},
        },
        "last_customer_reply": "好的，我已记录。请继续描述故障现象。",
        "missing_info": [],
    }

    wf = ChargerDiagnosisWorkflow()

    # 模拟第二轮：用户问 "刚才我说的是什么品牌和功率？"
    # 步骤 1: parse（用硬编码模拟 LLM 输出）
    parsed = MemoryQueryResult(
        is_memory_query=True,
        target_fields=["brand", "rated_power_kw"],
        query_scope="recent",
        answer_style="precise",
    )

    # 步骤 2: resolve
    resolution = wf._resolve_memory_fields(parsed, mock_ctx, "", {})
    print(f"  resolver 输出: resolved={resolution.resolved_values}, missing={resolution.missing_fields}, conf={resolution.confidence}")
    print(f"  resolver sources: {resolution.resolver_sources}")

    # 步骤 3: build reply
    reply = wf._build_memory_reply_v2(parsed, resolution)
    print(f"  reply: {reply}")

    # 验证
    all_ok = True
    if "华为" not in str(resolution.resolved_values.get("brand", "")):
        print(f"  FAIL: brand 未从 last_case 解析到！来源缺失，debug log 应显示。")
        print(f"       resolver_sources={resolution.resolver_sources}")
        all_ok = False
    if "7" not in str(resolution.resolved_values.get("rated_power_kw", "")):
        print(f"  FAIL: rated_power_kw 未从 last_case 解析到！来源缺失，debug log 应显示。")
        print(f"       resolver_sources={resolution.resolver_sources}")
        all_ok = False
    if resolution.confidence != "high":
        print(f"  FAIL: 两个字段都存在，confidence 应为 high，实际={resolution.confidence}")
        all_ok = False
    if "华为" not in reply or "7" not in reply:
        print(f"  FAIL: 回复中缺少品牌或功率值")
        all_ok = False

    if all_ok:
        print(f"  PASS 真实链路：品牌和功率均从 last_case 召回，confidence=high")
        passed += 1
    else:
        failed += 1

    # --- 场景 B: last_case 为空时的行为 ---
    print()
    print("  --- 场景 B: last_case 为空（未写入）---")
    empty_ctx: dict[str, Any] = {
        "last_case": {},
        "recent_safety": {},
        "recent_ticket": {},
        "session": {
            "recent_user_messages": [
                "我家华为 7kW 家充桩不能充电",
                "刚才我说的是什么品牌和功率？",
            ],
            "last_diagnosis": {},
            "last_dispatch": {},
        },
        "last_customer_reply": "",
        "missing_info": [],
    }

    parsed_b = MemoryQueryResult(
        is_memory_query=True,
        target_fields=["brand", "rated_power_kw"],
        query_scope="recent",
        answer_style="precise",
    )
    resolution_b = wf._resolve_memory_fields(parsed_b, empty_ctx, "", {})
    print(f"  resolver 输出: resolved={resolution_b.resolved_values}, missing={resolution_b.missing_fields}, conf={resolution_b.confidence}")
    reply_b = wf._build_memory_reply_v2(parsed_b, resolution_b)
    print(f"  reply: {reply_b}")

    if resolution_b.confidence == "low" and "brand" in resolution_b.missing_fields and "rated_power_kw" in resolution_b.missing_fields:
        print(f"  PASS 场景B: last_case 为空时 confidence=low，明确提示缺失字段")
        passed += 1
    else:
        print(f"  FAIL 场景B: conf={resolution_b.confidence}, missing={resolution_b.missing_fields}")
        failed += 1

    return passed, failed


# ============================================================================
# 测试 6: eval 文件结构校验
# ============================================================================

def test_parse_only_expected_structure() -> tuple[int, int]:
    from backend.schemas import (
        MEMORY_ANSWER_STYLE_VALUES,
        MEMORY_QUERY_SCOPE_VALUES,
        MEMORY_QUERY_TARGET_FIELDS,
    )

    cases = load_eval_cases()
    parse_cases = [c for c in cases if c.get("type") == "parse_only"]

    passed = 0
    failed = 0

    print()
    print("=" * 60)
    print("TEST: eval 文件 parse_only 用例结构校验")
    print("=" * 60)

    allowed_fields = set(MEMORY_QUERY_TARGET_FIELDS)

    for case in parse_cases:
        case_id = case["id"]
        expected = case.get("expected", {})
        all_ok = True

        for field in expected.get("target_fields", []):
            if field not in allowed_fields:
                print(f"  FAIL {case_id}: target_field '{field}' 不在允许列表中")
                all_ok = False

        scope = expected.get("query_scope", "")
        if scope not in MEMORY_QUERY_SCOPE_VALUES:
            print(f"  FAIL {case_id}: query_scope '{scope}' 不在允许值中")
            all_ok = False

        style = expected.get("answer_style", "")
        if style not in MEMORY_ANSWER_STYLE_VALUES:
            print(f"  FAIL {case_id}: answer_style '{style}' 不在允许值中")
            all_ok = False

        if all_ok:
            print(f"  PASS {case_id}: {case.get('scenario', '')}")
            passed += 1
        else:
            failed += 1

    return passed, failed


# ============================================================================
# 测试 7: FTS5 阶段 B 修复验证
# ============================================================================

def test_fts5_fixes() -> tuple[int, int]:
    """验证 FTS5 阶段 B 的 4 个修复：
    1. session_search 原有 matches 为空，但重新用 fts5_query 搜索能命中
    2. 全部字段由 FTS5 补齐时 confidence 仍然是 medium
    3. LLM 返回空字符串时不写入 resolved_values
    4. resolver_sources 能标明 FTS5 来源（使用 source_index）
    """
    import tempfile
    import os
    from unittest.mock import patch

    from backend.graph_workflow import ChargerDiagnosisWorkflow
    from backend.memory import MemoryManager, SessionMemory
    from backend.schemas import MemoryQueryResult, MemoryFieldResolution

    passed = 0
    failed = 0

    print()
    print("=" * 60)
    print("TEST: FTS5 阶段 B 修复验证")
    print("=" * 60)

    # =========================================================================
    # Case 1: session_search 原有 matches 为空，但重新用 fts5_query 搜索能命中
    # =========================================================================
    print()
    print("  --- Case 1: 空 matches → 重搜索命中 ---")
    tmpdir = tempfile.mkdtemp(prefix="fts5_test_")
    try:
        mm = MemoryManager(tmpdir)
        session = mm.create_session()
        session.add_message("user", "我家华为 7kW 家充桩不能充电，屏幕不亮")
        session.add_message("assistant", "好的，已记录。请问漏保是否跳闸？")
        mm.session_search.index_session(session)

        # 构造 memory_context：session_search.matches 为空（模拟旧搜索无结果）
        mock_ctx: dict[str, Any] = {
            "last_case": {},
            "recent_safety": {},
            "recent_ticket": {},
            "session": {"last_diagnosis": {}, "recent_user_messages": []},
            "last_customer_reply": "",
            "missing_info": [],
            "session_search": {"available": True, "matches": [], "query": "old_query"},
        }

        wf = ChargerDiagnosisWorkflow(llm=None, memory_manager=mm)
        state: dict[str, Any] = {
            "trace": [],
            "session_id": session.session_id,
            "memory_manager": mm,
        }
        parsed = MemoryQueryResult(
            is_memory_query=True,
            target_fields=["brand", "rated_power_kw"],
            entities=["华为", "7kW"],
        )

        # 调用 resolver（LLM=None 时 FTS5 抽取会跳过，但候选证据准备阶段会执行）
        resolution = wf._resolve_memory_fields(parsed, mock_ctx, "刚才是什么品牌和功率？", state)

        # 验证 trace 中包含重搜索来源标记
        fts5_traces = [
            t for t in state.get("trace", [])
            if "fts5" in str(t.get("node", "")).lower()
        ]
        re_search_traces = [
            t for t in fts5_traces
            if t.get("input", {}).get("fts5_search_source") == "re_search"
        ]
        if len(re_search_traces) >= 1:
            print(f"  PASS 空 matches → 重搜索触发（trace 中 fts5_search_source=re_search，共 {len(fts5_traces)} 条 FTS5 trace）")
            passed += 1
        else:
            print(f"  FAIL 未触发重搜索: fts5 traces={len(fts5_traces)}, re_search traces={len(re_search_traces)}")
            failed += 1

        # 额外验证：即使 LLM=None，候选证据也应从重搜索结果中构建（非空）
        candidate_traces = [
            t for t in fts5_traces
            if "候选证据" in str(t.get("title", ""))
        ]
        if candidate_traces:
            cand = candidate_traces[0].get("input", {}).get("fts5_matches_count", 0)
            print(f"  PASS 重搜索获取到 {cand} 条候选匹配")
            passed += 1
        else:
            # LLM=None 时在候选证据准备之前就跳过了，但 FTS5 fallback 开始的 trace 应该有
            start_traces = [t for t in fts5_traces if "开始" in str(t.get("title", ""))]
            if start_traces:
                matches_count = start_traces[0].get("input", {}).get("fts5_matches_count", 0)
                if matches_count > 0:
                    print(f"  PASS 重搜索命中 {matches_count} 条匹配")
                    passed += 1
                else:
                    print(f"  FAIL 重搜索命中 0 条: {start_traces[0].get('input', {})}")
                    failed += 1
            else:
                print(f"  FAIL 无 FTS5 开始 trace")
                failed += 1

    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    # =========================================================================
    # Case 2: 全部字段由 FTS5 补齐时 confidence 仍然是 medium
    # =========================================================================
    print()
    print("  --- Case 2: 全部字段由 FTS5 补齐 → confidence=medium ---")
    wf = ChargerDiagnosisWorkflow()

    # 构造 context：所有结构化来源为空，只有 FTS5 能提供值
    empty_ctx: dict[str, Any] = {
        "last_case": {},
        "recent_safety": {},
        "recent_ticket": {},
        "session": {"last_diagnosis": {}, "recent_user_messages": []},
        "last_customer_reply": "",
        "missing_info": [],
        "session_search": {
            "available": True,
            "query": "华为 7kW",
            "matches": [
                {"role": "user", "content": "我家华为 7kW 家充桩不能充电"},
            ],
        },
    }

    parsed = MemoryQueryResult(
        is_memory_query=True,
        target_fields=["brand", "rated_power_kw"],
        entities=["华为", "7kW"],
    )

    # 用 mock 控制 _fts5_extract_fields 返回模拟的 FTS5 抽取结果
    mock_fts5_result = {
        "extracted_values": {"brand": "华为", "rated_power_kw": "7kW"},
        "extracted_sources": {"brand": 0, "rated_power_kw": 0},
        "missing_fields": [],
    }

    with patch.object(wf, "_fts5_extract_fields", return_value=mock_fts5_result):
        resolution = wf._resolve_memory_fields(parsed, empty_ctx, "刚才是什么品牌和功率？", {})

    if resolution.confidence == "medium":
        print(f"  PASS 全部字段由 FTS5 补齐 → confidence={resolution.confidence}")
        passed += 1
    else:
        print(f"  FAIL 全部由 FTS5 补齐但 confidence={resolution.confidence}，期望 medium")
        failed += 1

    # 同时验证 resolved_values 和 resolver_sources
    if resolution.resolved_values.get("brand") == "华为" and resolution.resolved_values.get("rated_power_kw") == "7kW":
        print(f"  PASS FTS5 补齐的值正确: {resolution.resolved_values}")
        passed += 1
    else:
        print(f"  FAIL FTS5 补齐的值: {resolution.resolved_values}")
        failed += 1

    # =========================================================================
    # Case 2b: 全部来自结构化来源 → confidence=high（对照组）
    # =========================================================================
    print()
    print("  --- Case 2b: 全部来自结构化来源 → confidence=high（对照组）---")
    rich_ctx: dict[str, Any] = {
        "last_case": {"brand": "华为", "rated_power_kw": "22"},
        "recent_safety": {},
        "recent_ticket": {},
        "session": {"last_diagnosis": {}, "recent_user_messages": []},
        "last_customer_reply": "",
        "missing_info": [],
        "session_search": {"available": False, "matches": [], "query": ""},
    }
    parsed_rich = MemoryQueryResult(
        is_memory_query=True,
        target_fields=["brand", "rated_power_kw"],
    )
    resolution_rich = wf._resolve_memory_fields(parsed_rich, rich_ctx, "", {})
    if resolution_rich.confidence == "high":
        print(f"  PASS 结构化来源全命中 → confidence=high")
        passed += 1
    else:
        print(f"  FAIL 结构化来源全命中 → confidence={resolution_rich.confidence}，期望 high")
        failed += 1

    # =========================================================================
    # Case 3: LLM 返回空字符串时不写入 resolved_values
    # =========================================================================
    print()
    print("  --- Case 3: LLM 返回空字符串 → 不写入 resolved_values ---")
    mock_empty_result = {
        "extracted_values": {"brand": "", "rated_power_kw": "7kW", "city": None},
        "extracted_sources": {"brand": 0, "rated_power_kw": 0, "city": 0},
        "missing_fields": [],
    }

    with patch.object(wf, "_fts5_extract_fields", return_value=mock_empty_result):
        resolution3 = wf._resolve_memory_fields(parsed, empty_ctx, "test", {})

    # brand 和 city 应该被过滤掉（空字符串和 None），只有 rated_power_kw 保留
    if "brand" not in resolution3.resolved_values and "city" not in resolution3.resolved_values:
        print(f"  PASS 空值被正确过滤: resolved={resolution3.resolved_values}")
        passed += 1
    else:
        print(f"  FAIL 空值未被过滤: resolved={resolution3.resolved_values}")
        failed += 1

    if "rated_power_kw" in resolution3.resolved_values:
        print(f"  PASS 非空值 rated_power_kw 正常写入")
        passed += 1
    else:
        print(f"  FAIL rated_power_kw 被错误过滤")
        failed += 1

    # =========================================================================
    # Case 4: resolver_sources 能标明 FTS5 来源（使用 source_index）
    # =========================================================================
    print()
    print("  --- Case 4: resolver_sources 标明 FTS5 来源 ---")
    mock_multi_source = {
        "extracted_values": {"brand": "华为", "city": "杭州"},
        "extracted_sources": {"brand": 0, "city": 2},  # city 来自第 3 条消息
        "missing_fields": [],
    }

    with patch.object(wf, "_fts5_extract_fields", return_value=mock_multi_source):
        parsed4 = MemoryQueryResult(
            is_memory_query=True,
            target_fields=["brand", "city"],
        )
        resolution4 = wf._resolve_memory_fields(parsed4, empty_ctx, "test", {})

    sources = resolution4.resolver_sources
    brand_src = sources.get("brand", "")
    city_src = sources.get("city", "")

    if brand_src == "fts5.message[0]":
        print(f"  PASS brand 来源={brand_src}")
        passed += 1
    else:
        print(f"  FAIL brand 来源={brand_src}，期望 fts5.message[0]")
        failed += 1

    if city_src == "fts5.message[2]":
        print(f"  PASS city 来源={city_src}")
        passed += 1
    else:
        print(f"  FAIL city 来源={city_src}，期望 fts5.message[2]")
        failed += 1

    return passed, failed


# ============================================================================
# 测试 8: Answer LLM 回答生成
# ============================================================================

def test_answer_llm() -> tuple[int, int]:
    """测试 Answer LLM 回答生成全链路（mock RunnableLambda，真实调用 _build_memory_answer_llm）。

    验证 _format_fields_for_answer / _format_missing_for_answer /
    _clean_answer_output / _validate_answer_fields 在真实调用链中全部生效。
    """
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    from backend.graph_workflow import ChargerDiagnosisWorkflow
    from backend.schemas import MemoryQueryResult, MemoryFieldResolution

    def _make_mock_llm(content: str):
        """创建满足 LangChain Runnable 接口的 mock LLM。"""
        def _invoke(messages, config=None):
            return AIMessage(content=content)
        return RunnableLambda(_invoke)

    passed = 0
    failed = 0

    print()
    print("=" * 60)
    print("TEST: Answer LLM 回答生成（全链路，mock LLM.invoke）")
    print("=" * 60)

    wf = ChargerDiagnosisWorkflow()

    # =========================================================================
    # Case 1: 全部命中（confidence=high）→ 回答自然且包含字段值
    # =========================================================================
    print()
    print("  --- Case 1: 全部命中，mock LLM 返回自然回答 ---")
    parsed1 = MemoryQueryResult(
        is_memory_query=True,
        target_fields=["brand", "rated_power_kw"],
        answer_style="precise",
    )
    resolution1 = MemoryFieldResolution(
        resolved_values={"brand": "华为", "rated_power_kw": "7kW"},
        missing_fields=[],
        confidence="high",
        resolver_sources={"brand": "last_case.brand", "rated_power_kw": "last_case.rated_power_kw"},
    )
    state1: dict[str, Any] = {"trace": []}
    wf.llm = _make_mock_llm("根据之前记录的信息，您的充电桩品牌是华为，额定功率是7kW。")

    result1 = wf._build_memory_answer_llm(parsed1, resolution1, "刚才是什么品牌和功率？", state1)

    ok1 = (
        "华为" in result1
        and "7kW" in result1
        and "[confidence" not in result1
        and "resolver_sources" not in result1.lower()
    )
    if ok1:
        print(f"  PASS 回答包含字段值且无技术标记: {result1}")
        passed += 1
    else:
        print(f"  FAIL 回答: {result1}")
        failed += 1

    # 验证 trace 包含 answer_llm_used=True
    answer_traces = [t for t in state1["trace"] if "answer_llm" in str(t.get("node", "")).lower()]
    used_flag = any(t.get("input", {}).get("answer_llm_used") for t in answer_traces)
    if used_flag:
        print(f"  PASS trace 中 answer_llm_used=True")
        passed += 1
    else:
        print(f"  FAIL trace 缺少 answer_llm_used=True: {answer_traces}")
        failed += 1

    # =========================================================================
    # Case 2: 部分字段缺失（confidence=medium）
    #          → 真实 _format_fields_for_answer / _format_missing_for_answer 生效
    # =========================================================================
    print()
    print("  --- Case 2: 部分缺失，mock LLM 同时提示找到和缺失 ---")
    parsed2 = MemoryQueryResult(
        is_memory_query=True,
        target_fields=["brand", "serial_number"],
        answer_style="precise",
    )
    resolution2 = MemoryFieldResolution(
        resolved_values={"brand": "华为"},
        missing_fields=["serial_number"],
        confidence="medium",
        resolver_sources={"brand": "last_case.brand"},
    )
    state2: dict[str, Any] = {"trace": []}
    wf.llm = _make_mock_llm(
        "从当前会话记录中看，之前记录的充电桩品牌是华为。但序列号信息在当前会话记忆中没有找到。"
    )

    result2 = wf._build_memory_answer_llm(parsed2, resolution2, "品牌和序列号是什么？", state2)

    ok2 = "华为" in result2 and ("序列号" in result2 or "没有找到" in result2)
    if ok2:
        print(f"  PASS 既回答了找到的品牌，也提示了缺失的序列号: {result2[:120]}")
        passed += 1
    else:
        print(f"  FAIL 缺失提示不足: {result2}")
        failed += 1

    # 验证 _format_fields_for_answer 和 _format_missing_for_answer 都已生效
    fmt_resolved1 = ChargerDiagnosisWorkflow._format_fields_for_answer(
        resolution2.resolved_values, resolution2.resolver_sources, resolution2.confidence
    )
    fmt_missing1 = ChargerDiagnosisWorkflow._format_missing_for_answer(resolution2.missing_fields)
    if "品牌" in fmt_resolved1 and "序列号" in fmt_missing1:
        print(f"  PASS _format_fields/_format_missing 标签正确")
        passed += 1
    else:
        print(f"  FAIL format: resolved={fmt_resolved1[:60]}, missing={fmt_missing1}")
        failed += 1

    # =========================================================================
    # Case 3: LLM 返回带 [confidence: high] 的回答 → _clean_answer_output 清洗
    # =========================================================================
    print()
    print("  --- Case 3: LLM 返回含 [confidence: high] → 清洗后不泄露 ---")
    parsed3 = MemoryQueryResult(
        is_memory_query=True,
        target_fields=["brand"],
        answer_style="precise",
    )
    resolution3 = MemoryFieldResolution(
        resolved_values={"brand": "华为"},
        missing_fields=[],
        confidence="high",
        resolver_sources={"brand": "last_case.brand"},
    )
    state3: dict[str, Any] = {"trace": []}
    wf.llm = _make_mock_llm("之前记录的充电桩品牌是华为。[confidence: high]（来源：last_case.brand）")

    result3 = wf._build_memory_answer_llm(parsed3, resolution3, "品牌是什么？", state3)

    if "[confidence" not in result3 and "来源" not in result3 and "华为" in result3:
        print(f"  PASS 技术标记已清洗（无 [confidence]、无来源标注）: {result3}")
        passed += 1
    else:
        print(f"  FAIL 技术标记仍存在: {result3}")
        failed += 1

    # =========================================================================
    # Case 4: LLM 不可用 → 回退 _build_memory_reply_v2
    # =========================================================================
    print()
    print("  --- Case 4: LLM 不可用 → 回退 _build_memory_reply_v2 ---")
    wf_no_llm = ChargerDiagnosisWorkflow(llm=None)

    parsed4 = MemoryQueryResult(
        is_memory_query=True,
        target_fields=["brand", "charger_model"],
        answer_style="precise",
    )
    resolution4 = MemoryFieldResolution(
        resolved_values={"brand": "华为", "charger_model": "AC-7KW"},
        missing_fields=[],
        confidence="high",
        resolver_sources={"brand": "last_case.brand", "charger_model": "last_case.charger_model"},
    )
    state4: dict[str, Any] = {"trace": []}
    result4 = wf_no_llm._build_memory_answer_llm(
        parsed4, resolution4, "型号和品牌是什么？", state4,
    )

    if "华为" in result4 and "AC-7KW" in result4:
        print(f"  PASS LLM=None 回退 fallback 模板: {result4[:120]}")
        passed += 1
    else:
        print(f"  FAIL LLM=None 回退失败: {result4}")
        failed += 1

    # 验证 trace 包含 answer_fallback_reason
    answer_traces4 = [
        t for t in state4.get("trace", [])
        if "answer_llm" in str(t.get("node", "")).lower()
    ]
    fallback_trace = [
        t for t in answer_traces4
        if t.get("input", {}).get("answer_fallback_reason", "")
    ]
    if fallback_trace:
        print(f"  PASS trace 含 answer_fallback_reason: {fallback_trace[0]['input']['answer_fallback_reason']}")
        passed += 1
    else:
        print(f"  FAIL trace 缺少 answer_fallback_reason")
        failed += 1

    # =========================================================================
    # Case 5: LLM 编造缺失字段 → _validate_answer_fields 捕获 → fallback
    # =========================================================================
    print()
    print("  --- Case 5: LLM 编造缺失字段 → 检测到并 fallback ---")
    parsed5 = MemoryQueryResult(
        is_memory_query=True,
        target_fields=["brand", "serial_number"],
        answer_style="precise",
    )
    resolution5 = MemoryFieldResolution(
        resolved_values={"brand": "华为"},
        missing_fields=["serial_number"],
        confidence="medium",
        resolver_sources={"brand": "last_case.brand"},
    )
    state5: dict[str, Any] = {"trace": []}
    wf.llm = _make_mock_llm("您之前记录的充电桩品牌是华为，序列号是SN2024-001。")

    result5 = wf._build_memory_answer_llm(
        parsed5, resolution5, "品牌和序列号是什么？", state5,
    )

    # 编造检测应触发 fallback → 回到 _build_memory_reply_v2 模板
    if "[confidence" in result5:
        print(f"  PASS 编造被检测到，已回退到 fallback 模板: {result5[:120]}")
        passed += 1
    else:
        print(f"  FAIL 编造未被捕获或未回退: {result5}")
        failed += 1

    # 验证 trace 包含 answer_validation_failed
    all_traces5 = state5.get("trace", [])
    validation_traces = [
        t for t in all_traces5
        if "编造" in str(t.get("title", ""))
    ]
    fallback_traces5 = [
        t for t in all_traces5
        if t.get("input", {}).get("answer_fallback_reason") == "answer_validation_failed"
    ]
    if validation_traces and fallback_traces5:
        print(f"  PASS trace 含编造校验警告 + answer_fallback_reason=answer_validation_failed")
        passed += 1
    else:
        print(f"  FAIL trace 缺失编造检测: validation={len(validation_traces)}, fallback={len(fallback_traces5)}")
        failed += 1

    # 恢复 wf.llm 避免影响后续测试
    wf.llm = None

    return passed, failed


# ============================================================================
# 主入口
# ============================================================================

def main() -> int:
    total_passed = 0
    total_failed = 0

    for test_fn in [
        test_memory_result_validation,
        test_eval_cases_validation_only,
        test_resolver_routing,
        test_fts5_fallback,
        test_reply_from_resolution,
        test_real_session_recall,
        test_parse_only_expected_structure,
        test_fts5_fixes,
        test_answer_llm,
    ]:
        p, f = test_fn()
        total_passed += p
        total_failed += f

    print()
    print("=" * 60)
    print(f"总计: {total_passed} 通过, {total_failed} 失败, {total_passed + total_failed} 条")
    print("=" * 60)

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
