from __future__ import annotations

import unittest
from io import BytesIO
from unittest.mock import patch


class FakeUpload(BytesIO):
    def __init__(self, name: str = "manual.pdf", content: bytes = b"%PDF fake") -> None:
        super().__init__(content)
        self.name = name
        self.size = len(content)

    def getbuffer(self):
        return memoryview(self.getvalue())


class ParserOptionsTest(unittest.TestCase):
    def test_config_only_show_pypdf_and_mineru(self) -> None:
        from backend.config import PARSER_OPTIONS

        self.assertEqual(PARSER_OPTIONS, ["pypdf", "mineru"])

    def test_parse_pdfs_routes_to_pypdf(self) -> None:
        import backend.rag.parsers as parsers

        expected = [{"file_name": "manual.pdf", "page": 1, "text": "pypdf text", "parser": "pypdf"}]
        with patch.object(parsers, "parse_with_pypdf", return_value=expected):
            self.assertEqual(parsers.parse_pdfs([FakeUpload()], "pypdf"), expected)

    def test_parse_pdfs_routes_to_mineru_md_content(self) -> None:
        import backend.rag.parsers as parsers

        with patch.object(parsers, "MINERU_API_TOKEN", "token"):
            with patch.object(parsers, "_create_mineru_batch", return_value={"batch_id": "b1", "file_urls": ["https://upload"]}):
                with patch.object(parsers, "_upload_files_to_mineru", return_value=None):
                    with patch.object(parsers, "_wait_for_mineru_results", return_value=[{"file_name": "manual.pdf", "md_content": "# 手册\nE03"}]):
                        pages = parsers.parse_pdfs([FakeUpload()], "mineru")

        self.assertEqual(pages[0]["parser"], "mineru")
        self.assertIn("E03", pages[0]["text"])

    def test_unsupported_parser_raises_clear_error(self) -> None:
        import backend.rag.parsers as parsers

        with self.assertRaisesRegex(ValueError, "不支持的 PDF 解析器"):
            parsers.parse_pdfs([FakeUpload()], "unknown-parser")


if __name__ == "__main__":
    unittest.main()

