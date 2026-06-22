"""RAG 检索 Top-K 命中率评估脚本。

加载知识库内测试问题，调用 KnowledgeBaseManager.retrieve()，
判断 top-1 / top-3 / top-5 结果是否命中预期关键词，
输出 Hit@1、Hit@3、Hit@5 百分率。

不依赖 LLM（纯检索评估）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def load_cases(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _check_hit(results: list[dict[str, Any]], keywords: list[str]) -> bool:
    """判断 top-K 结果中是否有任一 chunk 的 text 包含任一关键词。"""
    if not keywords:
        return False
    combined = " ".join(r.get("text", "") for r in results)
    for kw in keywords:
        if kw in combined:
            return True
    return False


def run_rag_eval() -> tuple[int, int]:
    """运行 RAG 检索命中率评估。

    Returns:
        (total_passed, total_failed) — 为兼容统一的 main() 入口。
        但 RAG eval 本质不是 pass/fail，而是统计命中率；
        这里用 "passed = hit@5 命中数, failed = hit@5 未命中数" 近似。
    """
    cases_path = Path(__file__).with_name("rag_eval_questions.json")
    cases = load_cases(cases_path)
    print(f"\n加载 {len(cases)} 条 RAG 评估用例: {cases_path}")

    # ── 初始化知识库管理器并加载充电桩知识库 ──
    from backend.rag.kb_manager import KnowledgeBaseManager

    manager = KnowledgeBaseManager()
    kb_id = "kb_258935aa3681"
    print(f"\n加载知识库: {kb_id}")
    load_result = manager.load_knowledge_base(kb_id)
    if not load_result.get("success"):
        print(f"  ⚠ 知识库加载失败: {load_result.get('error')}")
        print(f"  请确认索引文件存在: data/indexes/{kb_id}/")
        return 0, len(cases)

    print(f"  已加载 {load_result['chunk_count']} 个 chunk")

    # ── 逐条检索并统计命中 ──
    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0
    total = len(cases)

    print(f"\n{'=' * 70}")
    print(f"{'ID':<12} {'问题':<40} {'Hit@1':<8} {'Hit@3':<8} {'Hit@5':<8}")
    print(f"{'=' * 70}")

    for case in cases:
        qid = case["id"]
        question = case["question"]
        expected = case.get("expected", {})
        keywords = expected.get("hit_chunk_keywords", [])
        description = expected.get("description", "")

        # 检索
        results, trace = manager.retrieve(question)

        hit1 = _check_hit(results[:1], keywords)
        hit3 = _check_hit(results[:3], keywords)
        hit5 = _check_hit(results[:5], keywords)

        if hit1:
            hit_at_1 += 1
        if hit3:
            hit_at_3 += 1
        if hit5:
            hit_at_5 += 1

        hit1_str = "HIT" if hit1 else "MISS"
        hit3_str = "HIT" if hit3 else "MISS"
        hit5_str = "HIT" if hit5 else "MISS"

        # 截断问题文本以便显示
        short_q = question[:38] + "…" if len(question) > 40 else question
        print(f"{qid:<12} {short_q:<40} {hit1_str:<8} {hit3_str:<8} {hit5_str:<8}")

        if not hit5:
            print(f"  ⚠ 未命中 | 预期关键词: {keywords} | 描述: {description}")
            if results:
                top_file = results[0].get("file_name", "?")
                top_text = results[0].get("text", "")[:80]
                print(f"         Top-1 来源: {top_file} | 内容: {top_text}…")
            else:
                print(f"         检索无结果")

    # ── 汇总报告 ──
    print(f"{'=' * 70}")
    print()
    print("=" * 60)
    print("  RAG 检索 Top-K 命中率评估报告")
    print("=" * 60)
    print(f"  测试问题数:        {total}")
    print(f"  Hit@1 (top-1):     {hit_at_1}/{total} = {hit_at_1 / total:.1%}")
    print(f"  Hit@3 (top-3):     {hit_at_3}/{total} = {hit_at_3 / total:.1%}")
    print(f"  Hit@5 (top-5):     {hit_at_5}/{total} = {hit_at_5 / total:.1%}")
    print(f"  知识库:            {kb_id} ({load_result['chunk_count']} chunks)")
    print(f"  检索模式:          hybrid (FAISS + BM25 + RRF)")
    print("=" * 60)
    print()
    print("  说明:")
    print("  - Hit@K = 预期关键词在 top-K 结果的 text 中至少命中一个")
    print("  - 命中判定采用宽松策略（substring 匹配），适合 MVP 阶段")
    print("  - 检索参数: vector_top_k=10, bm25_top_k=10, final_top_k=5")
    print("  - 此评估不依赖 LLM，纯确定性检索")
    print()

    # 兼容返回: passed = hit@5, failed = 未命中
    passed = hit_at_5
    failed = total - hit_at_5
    return passed, failed


def main() -> int:
    total_passed = 0
    total_failed = 0

    p, f = run_rag_eval()
    total_passed += p
    total_failed += f

    print("=" * 60)
    print(f"总计 RAG 评估: {total_passed} 命中, {total_failed} 未命中, {total_passed + total_failed} 条")
    print("=" * 60)

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
