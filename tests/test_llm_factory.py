from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.llm.factory import get_chat_model


class LLMFactoryTest(unittest.TestCase):
    def test_qwen_without_api_key_returns_none_for_fallback(self) -> None:
        with patch("backend.llm.factory.config.API_KEY", ""):
            self.assertIsNone(get_chat_model(provider="qwen"))

    def test_deepseek_without_api_key_returns_none_for_fallback(self) -> None:
        with patch("backend.llm.factory.config.DEEPSEEK_API_KEY", ""):
            self.assertIsNone(get_chat_model(provider="deepseek"))

    def test_unknown_provider_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported LLM provider"):
            get_chat_model(provider="unknown")

    def test_qwen_uses_timeout_and_retry_config(self) -> None:
        with patch("backend.llm.factory.config.API_KEY", "test-key"):
            with patch("backend.llm.factory.config.LLM_REQUEST_TIMEOUT", 12):
                with patch("backend.llm.factory.config.LLM_MAX_RETRIES", 0):
                    model = get_chat_model(provider="qwen")

        self.assertEqual(model.request_timeout, 12.0)
        self.assertEqual(model.max_retries, 0)


if __name__ == "__main__":
    unittest.main()
