from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class MinerUDownloadConfigTest(unittest.TestCase):
    def test_inline_comment_false_is_parsed_as_false(self) -> None:
        from backend.config import read_bool_env

        with patch.dict(os.environ, {"MINERU_DOWNLOAD_VERIFY_SSL": "false # 本地调试关闭证书校验"}):
            self.assertFalse(read_bool_env("MINERU_DOWNLOAD_VERIFY_SSL", True))

    def test_download_passes_false_to_requests_when_env_disables_verify(self) -> None:
        try:
            import backend.rag.parsers as parsers
        except ModuleNotFoundError as error:
            self.skipTest(f"当前环境缺少解析依赖：{error}")

        calls = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int):
                yield b"zip-bytes"

        def fake_get(*args, **kwargs):
            calls.append(kwargs)
            return FakeResponse()

        with patch.dict(os.environ, {"MINERU_DOWNLOAD_VERIFY_SSL": "false"}):
            with patch.object(parsers.requests, "get", side_effect=fake_get):
                with tempfile.TemporaryDirectory() as temp_dir:
                    parsers._download_file_with_retry(
                        "https://example.test/mineru.zip",
                        Path(temp_dir) / "mineru.zip",
                        1,
                    )

        self.assertFalse(calls[0]["verify"])


if __name__ == "__main__":
    unittest.main()
