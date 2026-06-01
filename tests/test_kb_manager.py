from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from backend.rag.kb_manager import KnowledgeBaseManager


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
        for chunk in self.chunks[:top_k]:
            item = dict(chunk)
            item["score"] = 1.0
            item["source"] = "fake_vector"
            results.append(item)
        return results


class FakeBM25Retriever:
    def __init__(self) -> None:
        self.chunks: list[dict[str, Any]] = []

    def build(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = list(chunks)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        results = []
        for chunk in self.chunks[:top_k]:
            item = dict(chunk)
            item["score"] = 0.5
            item["source"] = "fake_bm25"
            results.append(item)
        return results


class KnowledgeBaseManagerTest(unittest.TestCase):
    def make_manager(self, root: Path) -> KnowledgeBaseManager:
        return KnowledgeBaseManager(
            index_dir=root / "indexes",
            chunks_dir=root / "chunks",
            parsed_json_dir=root / "parsed_json",
            markdown_dir=root / "markdown",
            vector_store_factory=FakeVectorStore,
            bm25_factory=FakeBM25Retriever,
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


if __name__ == "__main__":
    unittest.main()
