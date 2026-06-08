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
MEMORY_DIR = DATA_DIR / "memory"


# FastAPI 服务配置
FASTAPI_HOST = os.getenv("FASTAPI_HOST", "127.0.0.1")
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", "8800"))
FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", f"http://{FASTAPI_HOST}:{FASTAPI_PORT}")


# Gradio 前端配置
GRADIO_HOST = os.getenv("GRADIO_HOST", "127.0.0.1")
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))


# 大模型与向量模型配置
API_KEY = os.getenv("API_KEY", "")
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "qwen")
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
DEFAULT_CHAT_MODEL = os.getenv("DEFAULT_CHAT_MODEL", "qwen-plus")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_CHAT_MODEL = os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat")
LLM_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "30"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "1"))
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-v3")
DEFAULT_EMBEDDING_BATCH_SIZE = int(os.getenv("DEFAULT_EMBEDDING_BATCH_SIZE", "10"))


# RAG 默认参数
DEFAULT_PARSER = os.getenv("DEFAULT_PARSER", "pypdf")
PARSER_OPTIONS = ["pypdf", "mineru"]
PARSER_LABELS = {
    "pypdf": "pypdf（推荐：文字型 PDF，数字更稳定）",
    "mineru": "MinerU API（复杂版面补充，可能依赖 ZIP 下载）",
}
PARSER_HELP = (
    "建议优先使用 pypdf。扫描件、图片型 PDF、复杂表格或 pypdf 解析乱码时再尝试 MinerU。"
    "MinerU 对部分期刊 PDF 可能出现正文数字缺失，构建后需要检查数字完整性。"
)

DEFAULT_SPLITTER = os.getenv("DEFAULT_SPLITTER", "recursive")
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "700"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "80"))
DEFAULT_VECTOR_TOP_K = int(os.getenv("DEFAULT_VECTOR_TOP_K", "10"))
DEFAULT_BM25_TOP_K = int(os.getenv("DEFAULT_BM25_TOP_K", "10"))
DEFAULT_FINAL_TOP_K = int(os.getenv("DEFAULT_FINAL_TOP_K", "5"))
DEFAULT_USE_QUERY_REWRITE = read_bool_env("DEFAULT_USE_QUERY_REWRITE", True)
DEFAULT_QUERY_REWRITE_COUNT = int(os.getenv("DEFAULT_QUERY_REWRITE_COUNT", "3"))
DEFAULT_QUERY_REWRITE_MAX_LENGTH = int(os.getenv("DEFAULT_QUERY_REWRITE_MAX_LENGTH", "200"))

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


def get_mineru_download_verify_ssl() -> bool:
    # 测试或启动脚本显式设置环境变量时优先使用当前进程值。
    if os.getenv("MINERU_DOWNLOAD_VERIFY_SSL") is not None:
        return read_bool_env("MINERU_DOWNLOAD_VERIFY_SSL", True)
    # 下载 ZIP 前读取 .env，避免模块导入时缓存旧值。
    load_dotenv(BASE_DIR / ".env", override=False)
    return read_bool_env("MINERU_DOWNLOAD_VERIFY_SSL", True)


# 数据目录与日志目录
def ensure_project_dirs() -> None:
    for folder in [DATA_DIR, LOG_DIR, UPLOAD_DIR, PARSED_JSON_DIR, MARKDOWN_DIR, CHUNKS_DIR, INDEX_DIR, MEMORY_DIR]:
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
    use_query_rewrite: bool = DEFAULT_USE_QUERY_REWRITE
    query_rewrite_count: int = DEFAULT_QUERY_REWRITE_COUNT
    query_rewrite_max_length: int = DEFAULT_QUERY_REWRITE_MAX_LENGTH
    use_rerank: bool = False


# ---------------------------------------------------------------------------
# memory_answer v2 feature flag
# ---------------------------------------------------------------------------
MEMORY_ANSWER_V2 = read_bool_env("MEMORY_ANSWER_V2", False)

# ---------------------------------------------------------------------------
# SQLite 长期记忆双写 feature flag
# ---------------------------------------------------------------------------
MEMORY_SQLITE_DUAL_WRITE = read_bool_env("MEMORY_SQLITE_DUAL_WRITE", True)

# ---------------------------------------------------------------------------
# SQLite 主读 feature flag（阶段 3 — 验证中，默认关闭）
# ---------------------------------------------------------------------------
# true  → recall_context() 优先从 SQLite 读取 session/case/ticket 维度，
#         失败时回退 JSON。
# false → 保持现有 JSON 路径不变。
MEMORY_READ_FROM_SQLITE = read_bool_env("MEMORY_READ_FROM_SQLITE", False)

# ---------------------------------------------------------------------------
# Session TTL 配置（天数）
# ---------------------------------------------------------------------------
# active 且 updated_at < now - N days → expired
MEMORY_SESSION_EXPIRE_AFTER_DAYS = int(os.getenv("MEMORY_SESSION_EXPIRE_AFTER_DAYS", "7"))
# expired 且 updated_at < now - N days → archived（从初始过期时间起算）
MEMORY_SESSION_ARCHIVE_AFTER_DAYS = int(os.getenv("MEMORY_SESSION_ARCHIVE_AFTER_DAYS", "30"))
# archived 且 updated_at < now - N days 的清理窗口（本阶段不执行物理删除）
MEMORY_SESSION_CLEANUP_AFTER_DAYS = int(os.getenv("MEMORY_SESSION_CLEANUP_AFTER_DAYS", "14"))

