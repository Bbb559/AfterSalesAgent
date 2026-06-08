from __future__ import annotations

from typing import Any

from backend import config


def get_chat_model(provider: str | None = None, temperature: float = 0.2) -> Any | None:
    """根据配置创建 LangChain 聊天模型。

    没有配置对应 key 时返回 None，让 workflow 可以稳定回退到规则逻辑。
    """
    provider = (provider or config.DEFAULT_LLM_PROVIDER).strip().lower()

    if provider == "qwen":
        if not config.API_KEY:
            return None
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=config.API_KEY,
            base_url=config.DASHSCOPE_BASE_URL,
            model=config.DEFAULT_CHAT_MODEL,
            temperature=temperature,
            timeout=config.LLM_REQUEST_TIMEOUT,
            max_retries=config.LLM_MAX_RETRIES,
        )

    if provider == "deepseek":
        if not config.DEEPSEEK_API_KEY:
            return None
        from langchain_deepseek import ChatDeepSeek

        return ChatDeepSeek(
            api_key=config.DEEPSEEK_API_KEY,
            model=config.DEEPSEEK_CHAT_MODEL,
            temperature=temperature,
            timeout=config.LLM_REQUEST_TIMEOUT,
            max_retries=config.LLM_MAX_RETRIES,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")
