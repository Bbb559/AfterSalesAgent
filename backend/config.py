from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def read_bool_env(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    # 兼容 .env 中写成 MINERU_DOWNLOAD_VERIFY_SSL=false # 注释 的情况。
    normalized = raw_value.split("#", 1)[0].strip().strip("\"'").lower()
    return normalized in {"1", "true", "yes", "y", "on"}


# 项目路径配置
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"

UPLOAD_DIR = DATA_DIR / "uploads"
PARSED_JSON_DIR = DATA_DIR / "parsed_json"
MARKDOWN_DIR = DATA_DIR / "markdown"
CHUNKS_DIR = DATA_DIR / "chunks"
INDEX_DIR = DATA_DIR / "indexes"


# FastAPI 服务配置
FASTAPI_HOST = os.getenv("FASTAPI_HOST", "127.0.0.1")
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", "8800"))
FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", f"http://{FASTAPI_HOST}:{FASTAPI_PORT}")


# Gradio 前端配置
GRADIO_HOST = os.getenv("GRADIO_HOST", "127.0.0.1")
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))


# 大模型与 Embedding 配置
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
DEFAULT_CHAT_MODEL = os.getenv("DEFAULT_CHAT_MODEL", "qwen-plus")
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-v3")
DEFAULT_EMBEDDING_BATCH_SIZE = int(os.getenv("DEFAULT_EMBEDDING_BATCH_SIZE", "10"))


# RAG 默认参数
DEFAULT_PARSER = os.getenv("DEFAULT_PARSER", "pypdf")
PARSER_OPTIONS = ["pypdf", "mineru", "langchain-mineru-flash", "langchain-mineru-precision"]
PARSER_LABELS = {
    "pypdf": "pypdf（推荐：文字型 PDF，数字更稳定）",
    "mineru": "MinerU API（旧接口，可能依赖 ZIP 下载）",
    "langchain-mineru-flash": "langchain-mineru flash（免 token，推荐先试）",
    "langchain-mineru-precision": "langchain-mineru precision（需要 MinerU token）",
}
PARSER_HELP = (
    "建议优先使用 pypdf。扫描件、图片型 PDF、复杂表格或 pypdf 解析乱码时再尝试 MinerU。"
    "如果旧 MinerU API 下载 ZIP 失败，可以改用 langchain-mineru flash 或 precision。"
    "MinerU 对部分期刊 PDF 可能出现正文数字缺失，构建后需要检查数字完整性。"
)

DEFAULT_SPLITTER = os.getenv("DEFAULT_SPLITTER", "recursive")
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "700"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "80"))
DEFAULT_VECTOR_TOP_K = int(os.getenv("DEFAULT_VECTOR_TOP_K", "10"))
DEFAULT_BM25_TOP_K = int(os.getenv("DEFAULT_BM25_TOP_K", "10"))
DEFAULT_FINAL_TOP_K = int(os.getenv("DEFAULT_FINAL_TOP_K", "5"))

FAISS_INDEX_FILE = INDEX_DIR / "faiss.index"
FAISS_CHUNKS_FILE = INDEX_DIR / "faiss_chunks.json"


# pypdf / MinerU 解析配置
MINERU_API_BASE_URL = os.getenv("MINERU_API_BASE_URL", "https://mineru.net").rstrip("/")
MINERU_API_TOKEN = os.getenv("MINERU_API_TOKEN", "")
MINERU_MODEL_VERSION = os.getenv("MINERU_MODEL_VERSION", "vlm")
MINERU_LANGUAGE = os.getenv("MINERU_LANGUAGE", "ch")
MINERU_IS_OCR = read_bool_env("MINERU_IS_OCR", False)
MINERU_ENABLE_TABLE = read_bool_env("MINERU_ENABLE_TABLE", True)
MINERU_ENABLE_FORMULA = read_bool_env("MINERU_ENABLE_FORMULA", True)
MINERU_PAGE_RANGES = os.getenv("MINERU_PAGE_RANGES", "")
MINERU_POLL_INTERVAL = int(os.getenv("MINERU_POLL_INTERVAL", "10"))
MINERU_TIMEOUT = int(os.getenv("MINERU_TIMEOUT", "1800"))
MINERU_DOWNLOAD_RETRY = int(os.getenv("MINERU_DOWNLOAD_RETRY", "3"))
MINERU_DOWNLOAD_TIMEOUT = int(os.getenv("MINERU_DOWNLOAD_TIMEOUT", "300"))
MINERU_DOWNLOAD_VERIFY_SSL = read_bool_env("MINERU_DOWNLOAD_VERIFY_SSL", True)
LANGCHAIN_MINERU_TIMEOUT = int(os.getenv("LANGCHAIN_MINERU_TIMEOUT", "1200"))
LANGCHAIN_MINERU_SPLIT_PAGES = read_bool_env("LANGCHAIN_MINERU_SPLIT_PAGES", True)


def get_mineru_download_verify_ssl() -> bool:
    # 下载 ZIP 前重新读取 .env，避免改完配置后只重启部分进程导致旧值残留。
    load_dotenv(BASE_DIR / ".env", override=True)
    return read_bool_env("MINERU_DOWNLOAD_VERIFY_SSL", True)


# 数据目录与日志目录
def ensure_project_dirs() -> None:
    for folder in [DATA_DIR, LOG_DIR, UPLOAD_DIR, PARSED_JSON_DIR, MARKDOWN_DIR, CHUNKS_DIR, INDEX_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


@dataclass
class ChunkConfig:
    splitter: str = DEFAULT_SPLITTER
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP


@dataclass
class RetrievalConfig:
    retrieval_mode: str = "hybrid"
    vector_top_k: int = DEFAULT_VECTOR_TOP_K
    bm25_top_k: int = DEFAULT_BM25_TOP_K
    final_top_k: int = DEFAULT_FINAL_TOP_K
    use_query_rewrite: bool = False
    use_rerank: bool = False
