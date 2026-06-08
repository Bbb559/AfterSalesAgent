"""统一的大模型调用入口。"""

from backend.llm.factory import get_chat_model

__all__ = ["get_chat_model"]
