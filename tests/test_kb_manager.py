from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from backend.rag.kb_manager import KnowledgeBaseManager
from backend.rag.rag_service import RAGService


class FakeVectorStore:
    def __init__(self) -> None:
        self.chunks: list[dict[str, Any]] = []

    def build(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = list(chunks)

    def save(self, index_path: Path, chunks_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text("fake-index", encoding="utf-8")
        chunks_path.write_text(json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8")

    def load(self, index_path: Path, chunks_path: Path) -> None:
        if not index_path.exists():
            raise FileNotFoundError(index_path)
        self.chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        results = []
        for index, chunk in enumerate(self.chunks[:top_k], start=1):
            item = dict(chunk)
            item["chunk_id"] = f"{chunk.get('chunk_id', 'chunk')}_vector_{query}_{index}"
            item["score"] = 1.0
            item["source"] = "fake_vector"
            item["matched_query"] = query
            results.append(item)
        return results


class FakeBM25Retriever:
    def __init__(self) -> None:
        self.chunks: list[dict[str, Any]] = []

    def build(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = list(chunks)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        results = []
        for index, chunk in enumerate(self.chunks[:top_k], start=1):
            item = dict(chunk)
            item["chunk_id"] = f"{chunk.get('chunk_id', 'chunk')}_bm25_{query}_{index}"
            item["score"] = 0.5
            item["source"] = "fake_bm25"
            item["matched_query"] = query
            results.append(item)
        return results


class KnowledgeBaseManagerTest(unittest.TestCase):
    def make_manager(self, root: Path, query_rewriter: Any | None = None) -> KnowledgeBaseManager:
        return KnowledgeBaseManager(
            index_dir=root / "indexes",
            chunks_dir=root / "chunks",
            parsed_json_dir=root / "parsed_json",
            markdown_dir=root / "markdown",
            vector_store_factory=FakeVectorStore,
            bm25_factory=FakeBM25Retriever,
            query_rewriter=query_rewriter or (lambda question, **_: [question]),
        )

    def test_list_empty_knowledge_bases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self.make_manager(Path(temp_dir))

            self.assertEqual(manager.list_knowledge_bases(), [])

    def test_list_reads_existing_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self.make_manager(Path(temp_dir))
            paths = manager.get_kb_paths("kb_demo")
            paths.index_dir.mkdir(parents=True)
            paths.index_path.write_text("fake-index", encoding="utf-8")
            paths.chunks_index_path.write_text(
                json.dumps([{"chunk_id": "c1", "file_name": "manual.pdf", "text": "E03"}], ensure_ascii=False),
                encoding="utf-8",
            )
            paths.metadata_path.write_text(
                json.dumps({"display_name": "演示知识库", "parser": "pypdf", "chunk_count": 1}, ensure_ascii=False),
                encoding="utf-8",
            )

            items = manager.list_knowledge_bases()

            self.assertEqual(items[0]["database_id"], "kb_demo")
            self.assertIn("演示知识库", items[0]["label"])

    def test_load_missing_knowledge_base_returns_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self.make_manager(Path(temp_dir))

            result = manager.load_knowledge_base("missing")

            self.assertFalse(result["success"])
            self.assertIn("知识库不存在", result["error"])

    def test_retrieve_after_loading_existing_knowledge_base(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self.make_manager(Path(temp_dir))
            paths = manager.get_kb_paths("kb_demo")
            chunks = [{"chunk_id": "c1", "file_name": "manual.pdf", "page": 1, "text": "E03 表示进水异常。"}]
            FakeVectorStore().build(chunks)
            fake_store = FakeVectorStore()
            fake_store.build(chunks)
            fake_store.save(paths.index_path, paths.chunks_index_path)

            load_result = manager.load_knowledge_base("kb_demo")
            results, trace = manager.retrieve("E03", {"retrieval_mode": "hybrid"})

            self.assertTrue(load_result["success"])
            self.assertEqual(results[0]["file_name"], "manual.pdf")
            self.assertEqual(trace["database_id"], "kb_demo")

    def test_retrieve_uses_original_query_and_three_rewrites_by_default(self) -> None:
        def fake_rewriter(question: str, **_: Any) -> list[str]:
            return [question, "HX-900 右耳无声", "HX-900 声道平衡", "HX-900 维修政策", "多余改写"]

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self.make_manager(Path(temp_dir), query_rewriter=fake_rewriter)
            chunks = [{"chunk_id": "c1", "file_name": "manual.pdf", "page": 1, "text": "充电桩售后安全说明"}]
            manager.vector_store = FakeVectorStore()
            manager.vector_store.build(chunks)
            manager.bm25_retriever = FakeBM25Retriever()
            manager.bm25_retriever.build(chunks)
            manager.current_database_id = "kb_demo"

            results, trace = manager.retrieve("HX-900 右耳没有声音", {"retrieval_mode": "hybrid", "final_top_k": 8})

            expected_queries = ["HX-900 右耳没有声音", "HX-900 右耳无声", "HX-900 声道平衡", "HX-900 维修政策"]
            self.assertEqual(trace["queries"], expected_queries)
            self.assertEqual(trace["query_rewrite"]["rewritten_queries"], expected_queries[1:])
            self.assertEqual([item["query"] for item in trace["vector_results"]], expected_queries)
            self.assertEqual([item["query"] for item in trace["bm25_results"]], expected_queries)
            self.assertTrue(results)

    def test_rewrite_queries_are_deduped_truncated_and_can_be_disabled(self) -> None:
        def fake_rewriter(question: str, **_: Any) -> list[str]:
            return [question, "重复", "重复", "这是一个很长的改写查询", "不会使用"]

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self.make_manager(Path(temp_dir), query_rewriter=fake_rewriter)
            chunks = [{"chunk_id": "c1", "file_name": "manual.pdf", "page": 1, "text": "售后说明"}]
            manager.vector_store = FakeVectorStore()
            manager.vector_store.build(chunks)
            manager.current_database_id = "kb_demo"

            _, trace = manager.retrieve(
                "原始问题",
                {"retrieval_mode": "vector", "query_rewrite_count": 2, "query_rewrite_max_length": 4},
            )
            _, disabled_trace = manager.retrieve("原始问题", {"retrieval_mode": "vector", "use_query_rewrite": False})

            self.assertEqual(trace["queries"], ["原始问题", "重复", "这是一个"])
            self.assertEqual(disabled_trace["queries"], ["原始问题"])
            self.assertFalse(disabled_trace["query_rewrite"]["enabled"])

    def test_query_rewrite_failure_falls_back_to_original_query(self) -> None:
        def broken_rewriter(question: str, **_: Any) -> list[str]:
            raise RuntimeError("改写失败")

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self.make_manager(Path(temp_dir), query_rewriter=broken_rewriter)
            chunks = [{"chunk_id": "c1", "file_name": "manual.pdf", "page": 1, "text": "售后说明"}]
            manager.vector_store = FakeVectorStore()
            manager.vector_store.build(chunks)
            manager.current_database_id = "kb_demo"

            _, trace = manager.retrieve("原始问题", {"retrieval_mode": "vector"})

            self.assertEqual(trace["queries"], ["原始问题"])
            self.assertIn("改写失败", trace["query_rewrite"]["error"])


class RAGServiceTest(unittest.TestCase):
    def test_retrieve_keeps_backend_retrieval_order_without_keyword_doc_type_boost(self) -> None:
        def fake_retrieval(question: str, **_: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            return [
                {"file_name": "first.pdf", "page": 1, "text": "先返回的结果", "doc_type": "general"},
                {"file_name": "second.pdf", "page": 2, "text": "后返回的结果", "doc_type": "warranty_policy"},
            ], {"query": question}

        result = RAGService(retrieval_func=fake_retrieval).retrieve("能不能免费处理")

        self.assertEqual([item["file_name"] for item in result["results"]], ["first.pdf", "second.pdf"])


if __name__ == "__main__":
    unittest.main()
