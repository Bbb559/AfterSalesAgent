from __future__ import annotations

import sys
import types
import unittest
from io import BytesIO
from unittest.mock import patch


class FakeUpload(BytesIO):
    def __init__(self, name: str, content: bytes = b"%PDF-1.4 fake") -> None:
        super().__init__(content)
        self.name = name

    @property
    def size(self) -> int:
        return len(self.getvalue())


class FakeDocument:
    def __init__(self, page_content: str, metadata: dict) -> None:
        self.page_content = page_content
        self.metadata = metadata


class FakeMinerULoader:
    calls: list[dict] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        FakeMinerULoader.calls.append(kwargs)

    def load(self):
        return [
            FakeDocument("第一页内容 E03", {"filename": "manual.pdf", "page": 1}),
            FakeDocument("第二页内容 QY-320", {"filename": "manual.pdf", "page": 2}),
        ]


class LangChainMinerUParserTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeMinerULoader.calls = []
        fake_module = types.ModuleType("langchain_mineru")
        fake_module.MinerULoader = FakeMinerULoader
        self.module_patch = patch.dict(sys.modules, {"langchain_mineru": fake_module})
        self.module_patch.start()

    def tearDown(self) -> None:
        self.module_patch.stop()

    def test_flash_loader_documents_are_converted_to_pages(self) -> None:
        from backend.rag.parsers import parse_pdfs

        pages = parse_pdfs([FakeUpload("manual.pdf")], parser_name="langchain-mineru-flash")

        self.assertEqual(len(pages), 2)
        self.assertEqual(pages[0]["parser"], "langchain-mineru-flash")
        self.assertEqual(pages[0]["page"], 1)
        self.assertIn("E03", pages[0]["text"])
        self.assertEqual(FakeMinerULoader.calls[0]["mode"], "flash")

    def test_precision_loader_uses_mineru_api_token(self) -> None:
        from backend import config
        from backend.rag.parsers import parse_pdfs

        with patch("backend.rag.parsers.MINERU_API_TOKEN", "token-from-env"):
            pages = parse_pdfs([FakeUpload("manual.pdf")], parser_name="langchain-mineru-precision")

        self.assertTrue(pages)
        self.assertEqual(FakeMinerULoader.calls[0]["mode"], "precision")
        self.assertEqual(FakeMinerULoader.calls[0]["token"], "token-from-env")


if __name__ == "__main__":
    unittest.main()
