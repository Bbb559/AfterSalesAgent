import os

from dotenv import load_dotenv
from openai import OpenAI

from backend.config import (
    DASHSCOPE_BASE_URL,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
)


load_dotenv()

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client

    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("缺少 API_KEY，请在项目根目录创建 .env，或设置 API_KEY 环境变量。")

    if _client is None:
        _client = OpenAI(
            api_key=api_key,
            base_url=DASHSCOPE_BASE_URL,
        )

    return _client


def get_embeddings(
    texts,
    model=DEFAULT_EMBEDDING_MODEL,
    batch_size=DEFAULT_EMBEDDING_BATCH_SIZE,
):
    if texts is None:
        return []
    
    if isinstance(texts, str):
        texts = [texts]

    cleaned_texts = [text.strip() for text in texts if text and text.strip()]
    if not cleaned_texts:
        return []

    all_embeddings = []

    client = _get_client()

    for batch in _batch(cleaned_texts, batch_size):
        response = client.embeddings.create(
            input=batch,
            model=model,
        )

        all_embeddings.extend([item.embedding for item in response.data])

    return all_embeddings


def _batch(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]
