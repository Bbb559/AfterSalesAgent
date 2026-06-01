import json
from pathlib import Path
import hashlib
import re

DATA_DIR = Path("data")
PARSED_JSON_DIR = DATA_DIR / "parsed_json"
MARKDOWN_DIR = DATA_DIR / "markdown"
CHUNKS_DIR = DATA_DIR / "chunks"
INDEX_DIR = DATA_DIR / "indexes"

def ensure_data_dirs():
    for folder in [PARSED_JSON_DIR, MARKDOWN_DIR, CHUNKS_DIR, INDEX_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

def safe_stem(file_name):
    stem = Path(file_name).stem
    safe = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff_-]+", "_", stem)
    safe = safe.strip("_") or "file"

    short_hash = hashlib.md5(file_name.encode("utf-8")).hexdigest()[:8]
    return f"{safe}_{short_hash}"

def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_text(text, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def pages_to_markdown(pages):
    parts = []
    for page in pages:
        parts.append(
            f"\n\n---\n\n# {page.get('file_name')} - 第{page.get('page')}页\n\n"
            f"{page.get('text', '')}"
        )
    return "".join(parts).strip()
