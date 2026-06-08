from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

from backend.config import (
    CHUNKS_DIR,
    DEFAULT_BM25_TOP_K,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_FINAL_TOP_K,
    DEFAULT_QUERY_REWRITE_COUNT,
    DEFAULT_QUERY_REWRITE_MAX_LENGTH,
    DEFAULT_PARSER,
    DEFAULT_SPLITTER,
    DEFAULT_USE_QUERY_REWRITE,
    DEFAULT_VECTOR_TOP_K,
    INDEX_DIR,
    MARKDOWN_DIR,
    PARSED_JSON_DIR,
    ensure_project_dirs,
)


@dataclass
class KnowledgeBasePaths:
    index_dir: Path
    index_path: Path
    chunks_index_path: Path
    metadata_path: Path
    chunks_backup_path: Path
    pages_path: Path
    markdown_path: Path


class UploadedFileAdapter(BytesIO):
    """把 FastAPI、Gradio 或路径文件统一成 pypdf/MinerU 可读取的文件对象。"""

    def __init__(self, name: str, content: bytes) -> None:
        super().__init__(content)
        self.name = name
        self.size = len(content)


class KnowledgeBaseManager:
    """负责知识库构建、加载、检索和索引文件管理。"""

    def __init__(
        self,
        index_dir: Path = INDEX_DIR,
        chunks_dir: Path = CHUNKS_DIR,
        parsed_json_dir: Path = PARSED_JSON_DIR,
        markdown_dir: Path = MARKDOWN_DIR,
        vector_store_factory: Callable[[], Any] | None = None,
        bm25_factory: Callable[[], Any] | None = None,
        parser_func: Callable[..., list[dict[str, Any]]] | None = None,
        query_rewriter: Callable[..., list[str]] | None = None,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.chunks_dir = Path(chunks_dir)
        self.parsed_json_dir = Path(parsed_json_dir)
        self.markdown_dir = Path(markdown_dir)
        self.vector_store_factory = vector_store_factory
        self.bm25_factory = bm25_factory
        self.parser_func = parser_func
        self.query_rewriter = query_rewriter
        self.vector_store: Any | None = None
        self.bm25_retriever: Any | None = None
        self.current_database_id = ""
        self.current_metadata: dict[str, Any] = {}

    def build_database_id(
        self,
        uploaded_files: list[UploadedFileAdapter],
        parser_name: str,
        splitter_name: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> str:
        raw = "|".join(
            [f"{file.name}:{file.size}" for file in uploaded_files]
            + [parser_name, splitter_name, str(chunk_size), str(chunk_overlap)]
        )
        short_hash = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
        return f"kb_{short_hash}"

    def get_kb_paths(self, database_id: str) -> KnowledgeBasePaths:
        db_index_dir = self.index_dir / database_id
        return KnowledgeBasePaths(
            index_dir=db_index_dir,
            index_path=db_index_dir / "faiss.index",
            chunks_index_path=db_index_dir / "chunks.json",
            metadata_path=db_index_dir / "metadata.json",
            chunks_backup_path=self.chunks_dir / f"{database_id}_chunks.json",
            pages_path=self.parsed_json_dir / f"{database_id}_pages.json",
            markdown_path=self.markdown_dir / f"{database_id}.md",
        )

    def list_knowledge_bases(self) -> list[dict[str, Any]]:
        if not self.index_dir.exists():
            return []

        items = []
        for kb_dir in self.index_dir.iterdir():
            if not kb_dir.is_dir():
                continue
            database_id = kb_dir.name
            paths = self.get_kb_paths(database_id)
            if not paths.index_path.exists() or not paths.chunks_index_path.exists():
                continue

            chunks = self._read_json(paths.chunks_index_path, [])
            metadata = self._read_json(paths.metadata_path, {})
            file_names = metadata.get("file_names") or sorted({
                chunk.get("file_name", "")
                for chunk in chunks
                if chunk.get("file_name")
            })
            label = self._build_label(database_id, metadata, len(chunks), file_names)
            items.append({
                "database_id": database_id,
                "display_name": metadata.get("display_name", "无"),
                "label": label,
                "chunk_count": metadata.get("chunk_count", len(chunks)),
                "file_names": file_names,
                "metadata": metadata,
                "updated_at": kb_dir.stat().st_mtime,
            })

        items.sort(key=lambda item: item["updated_at"], reverse=True)
        return items

    def build_knowledge_base(
        self,
        uploaded_files: list[Any],
        parser_name: str = DEFAULT_PARSER,
        splitter_name: str = DEFAULT_SPLITTER,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        display_name: str = "",
        doc_type: str = "",
        product_line: str = "",
        item_identifier: str = "",
        version: str = "",
    ) -> dict[str, Any]:
        try:
            ensure_project_dirs()
            self._ensure_dirs()
            normalized_files = self._normalize_uploaded_files(uploaded_files)
            if not normalized_files:
                return {"success": False, "error": "请先上传 PDF 文件。"}

            database_id = self.build_database_id(normalized_files, parser_name, splitter_name, chunk_size, chunk_overlap)
            parser = self.parser_func or self._load_default_parser()
            pages = parser(normalized_files, parser_name=parser_name)
            if not pages:
                return {"success": False, "error": "没有解析出任何文本。"}

            from backend.rag.cleaners import clean_pages
            from backend.rag.quality import analyze_digit_health
            from backend.rag.splitters import split_pages
            from backend.rag.utils import pages_to_markdown, save_json, save_text

            cleaned_pages = clean_pages(pages)
            if not cleaned_pages:
                return {"success": False, "error": "文本清洗后为空，无法继续构建知识库。"}

            for page in cleaned_pages:
                page["doc_type"] = doc_type
                page["product_line"] = product_line
                page["item_identifier"] = item_identifier
                page["version"] = version

            digit_health = analyze_digit_health(cleaned_pages)
            chunks = split_pages(
                cleaned_pages,
                splitter_name=splitter_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            if not chunks:
                return {"success": False, "error": "分块结果为空，无法构建知识库。"}

            vector_store = self._new_vector_store()
            vector_store.build(chunks)

            paths = self.get_kb_paths(database_id)
            vector_store.save(index_path=paths.index_path, chunks_path=paths.chunks_index_path)

            metadata = self._save_metadata(
                database_id=database_id,
                display_name=display_name,
                uploaded_files=normalized_files,
                parser_name=parser_name,
                splitter_name=splitter_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunk_count=len(chunks),
                doc_type=doc_type,
                product_line=product_line,
                item_identifier=item_identifier,
                version=version,
                digit_health=digit_health,
            )

            save_json(cleaned_pages, paths.pages_path)
            save_json(chunks, paths.chunks_backup_path)
            save_text(pages_to_markdown(cleaned_pages), paths.markdown_path)
            self._activate(database_id, vector_store, chunks, metadata)

            return {
                "success": True,
                "database_id": database_id,
                "chunk_count": len(chunks),
                "metadata": metadata,
                "digit_health": digit_health,
            }
        except Exception as exc:  # pragma: no cover - 构建链路防御边界
            return {"success": False, "error": f"构建知识库失败：{exc}"}

    def load_knowledge_base(self, database_id: str) -> dict[str, Any]:
        paths = self.get_kb_paths(database_id)
        if not paths.index_path.exists() or not paths.chunks_index_path.exists():
            return {"success": False, "error": f"知识库不存在或索引文件缺失：{database_id}"}

        try:
            vector_store = self._new_vector_store()
            vector_store.load(index_path=paths.index_path, chunks_path=paths.chunks_index_path)
            metadata = self._read_json(paths.metadata_path, {})
            self._activate(database_id, vector_store, vector_store.chunks, metadata)
            return {
                "success": True,
                "database_id": database_id,
                "chunk_count": len(vector_store.chunks),
                "metadata": metadata,
            }
        except Exception as exc:
            return {"success": False, "error": f"加载知识库失败：{exc}"}

    def delete_knowledge_base(self, database_id: str) -> dict[str, Any]:
        paths = self.get_kb_paths(database_id)
        if paths.index_dir.exists():
            shutil.rmtree(paths.index_dir)
        for path in [paths.chunks_backup_path, paths.pages_path, paths.markdown_path]:
            if path.exists():
                path.unlink()
        if self.current_database_id == database_id:
            self.vector_store = None
            self.bm25_retriever = None
            self.current_database_id = ""
            self.current_metadata = {}
        return {"success": True, "database_id": database_id}

    def retrieve(self, question: str, options: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        options = options or {}
        database_id = options.get("database_id")
        if database_id and database_id != self.current_database_id:
            load_result = self.load_knowledge_base(database_id)
            if not load_result.get("success"):
                return [], {
                    "mode": options.get("retrieval_mode", "unknown"),
                    "queries": [question],
                    "error": load_result.get("error", "知识库加载失败"),
                    "no_knowledge_base": True,
                }

        if self.vector_store is None:
            return [], {
                "mode": options.get("retrieval_mode", "unknown"),
                "queries": [question],
                "warning": "未加载知识库，请先构建或加载知识库。",
                "no_knowledge_base": True,
            }

        retrieval_mode = options.get("retrieval_mode", "hybrid")
        vector_top_k = int(options.get("vector_top_k", DEFAULT_VECTOR_TOP_K))
        bm25_top_k = int(options.get("bm25_top_k", DEFAULT_BM25_TOP_K))
        final_top_k = int(options.get("final_top_k", DEFAULT_FINAL_TOP_K))
        queries, rewrite_trace = self._prepare_retrieval_queries(question, options)
        trace = {
            "mode": retrieval_mode,
            "queries": queries,
            "database_id": self.current_database_id,
            "query_rewrite": rewrite_trace,
            "vector_results": [],
            "bm25_results": [],
            "before_rerank": [],
        }

        if retrieval_mode == "vector":
            result_groups = []
            for query in queries:
                results = self.vector_store.search(query, top_k=vector_top_k)
                trace["vector_results"].append({"query": query, "results": results})
                result_groups.append(results)
            results = self._reciprocal_rank_fusion(result_groups, final_top_k=final_top_k)
            trace["before_rerank"] = results
            return results, trace

        if retrieval_mode == "bm25":
            if self.bm25_retriever is None:
                return [], {**trace, "error": "BM25 索引未加载。"}
            result_groups = []
            for query in queries:
                results = self.bm25_retriever.search(query, top_k=bm25_top_k)
                trace["bm25_results"].append({"query": query, "results": results})
                result_groups.append(results)
            results = self._reciprocal_rank_fusion(result_groups, final_top_k=final_top_k)
            trace["before_rerank"] = results
            return results, trace

        result_groups = []
        for query in queries:
            vector_results = self.vector_store.search(query, top_k=vector_top_k)
            trace["vector_results"].append({"query": query, "results": vector_results})
            result_groups.append(vector_results)
            if self.bm25_retriever is not None:
                bm25_results = self.bm25_retriever.search(query, top_k=bm25_top_k)
                trace["bm25_results"].append({"query": query, "results": bm25_results})
                result_groups.append(bm25_results)

        results = self._reciprocal_rank_fusion(result_groups, final_top_k=final_top_k)
        trace["before_rerank"] = results
        return results, trace

    def status(self) -> dict[str, Any]:
        return {
            "loaded": bool(self.current_database_id),
            "database_id": self.current_database_id,
            "chunk_count": len(self.vector_store.chunks) if self.vector_store is not None else 0,
            "metadata": self.current_metadata,
        }

    def _activate(
        self,
        database_id: str,
        vector_store: Any,
        chunks: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> None:
        self.vector_store = vector_store
        self.bm25_retriever = self._new_bm25_retriever()
        self.bm25_retriever.build(chunks)
        self.current_database_id = database_id
        self.current_metadata = metadata

    def _save_metadata(
        self,
        database_id: str,
        display_name: str,
        uploaded_files: list[UploadedFileAdapter],
        parser_name: str,
        splitter_name: str,
        chunk_size: int,
        chunk_overlap: int,
        chunk_count: int,
        doc_type: str,
        product_line: str,
        item_identifier: str,
        version: str,
        digit_health: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = {
            "database_id": database_id,
            "display_name": display_name.strip() or "无",
            "file_names": [file.name for file in uploaded_files],
            "parser": parser_name,
            "splitter": splitter_name,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunk_count": chunk_count,
            "doc_type": doc_type,
            "product_line": product_line,
            "item_identifier": item_identifier,
            "version": version,
            "digit_health": digit_health,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        paths = self.get_kb_paths(database_id)
        paths.index_dir.mkdir(parents=True, exist_ok=True)
        paths.metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return metadata

    def _build_label(
        self,
        database_id: str,
        metadata: dict[str, Any],
        chunk_count: int,
        file_names: list[str],
    ) -> str:
        label = f"{metadata.get('display_name') or database_id} | {metadata.get('chunk_count', chunk_count)} chunks"
        for key in ["parser", "doc_type", "item_identifier"]:
            value = metadata.get(key)
            if value:
                label += f" | {value}"
        if file_names:
            label += f" | {'、'.join(file_names[:2])}"
        return label

    def _normalize_uploaded_files(self, uploaded_files: list[Any]) -> list[UploadedFileAdapter]:
        normalized = []
        for item in uploaded_files or []:
            if isinstance(item, UploadedFileAdapter):
                item.seek(0)
                normalized.append(item)
                continue

            if isinstance(item, (str, Path)):
                path = Path(item)
                normalized.append(UploadedFileAdapter(path.name, path.read_bytes()))
                continue

            name = getattr(item, "filename", None) or getattr(item, "name", "uploaded.pdf")
            file_obj = getattr(item, "file", item)
            if hasattr(file_obj, "seek"):
                file_obj.seek(0)
            content = file_obj.read()
            if isinstance(content, str):
                content = content.encode("utf-8")
            if hasattr(file_obj, "seek"):
                file_obj.seek(0)
            normalized.append(UploadedFileAdapter(Path(name).name, content))

        return normalized

    def _ensure_dirs(self) -> None:
        for folder in [self.index_dir, self.chunks_dir, self.parsed_json_dir, self.markdown_dir]:
            folder.mkdir(parents=True, exist_ok=True)

    def _new_vector_store(self) -> Any:
        if self.vector_store_factory is not None:
            return self.vector_store_factory()
        from backend.rag.stores import FaissVectorStore

        return FaissVectorStore()

    def _new_bm25_retriever(self) -> Any:
        if self.bm25_factory is not None:
            return self.bm25_factory()
        from backend.rag.retrievers import BM25Retriever

        return BM25Retriever()

    def _load_default_parser(self) -> Callable[..., list[dict[str, Any]]]:
        from backend.rag.parsers import parse_pdfs

        return parse_pdfs

    def _prepare_retrieval_queries(self, question: str, options: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
        original_query = str(question or "").strip()
        rewrite_count = max(0, int(options.get("query_rewrite_count", DEFAULT_QUERY_REWRITE_COUNT)))
        max_length = max(1, int(options.get("query_rewrite_max_length", DEFAULT_QUERY_REWRITE_MAX_LENGTH)))
        enabled = bool(options.get("use_query_rewrite", DEFAULT_USE_QUERY_REWRITE))
        base_trace = {
            "enabled": enabled,
            "original_query": original_query,
            "requested_rewrite_count": rewrite_count,
            "max_query_length": max_length,
            "rewritten_queries": [],
            "error": "",
        }

        if not enabled or rewrite_count <= 0:
            return [original_query[:max_length]], base_trace

        try:
            rewriter = self.query_rewriter or self._load_default_query_rewriter()
            queries = rewriter(
                original_query,
                rewrite_count=rewrite_count,
                max_query_length=max_length,
            )
            normalized = self._normalize_retrieval_queries(original_query, queries, rewrite_count, max_length)
            base_trace["rewritten_queries"] = normalized[1:]
            return normalized, base_trace
        except Exception as exc:
            base_trace["error"] = str(exc)
            return [original_query[:max_length]], base_trace

    def _normalize_retrieval_queries(
        self,
        original_query: str,
        candidates: list[str],
        rewrite_count: int,
        max_length: int,
    ) -> list[str]:
        queries = [original_query[:max_length]]
        for item in candidates or []:
            if not isinstance(item, str):
                continue
            query = item.strip()[:max_length]
            if query:
                queries.append(query)

        deduped = []
        seen = set()
        for query in queries:
            if query and query not in seen:
                seen.add(query)
                deduped.append(query)
        return deduped[: rewrite_count + 1] or [original_query[:max_length]]

    def _load_default_query_rewriter(self) -> Callable[..., list[str]]:
        from backend.rag.rerankers import rewrite_query

        return rewrite_query

    def _read_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default

    def _reciprocal_rank_fusion(
        self,
        result_groups: list[list[dict[str, Any]]],
        final_top_k: int = DEFAULT_FINAL_TOP_K,
        rrf_k: int = 60,
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for results in result_groups:
            for rank, item in enumerate(results, start=1):
                chunk_id = item.get("chunk_id") or f"{item.get('file_name', 'unknown')}_{item.get('page', 0)}_{rank}"
                if chunk_id not in merged:
                    merged[chunk_id] = dict(item)
                    merged[chunk_id]["sources"] = set()
                    merged[chunk_id]["scores"] = {}
                    merged[chunk_id]["rrf_score"] = 0.0
                source = item.get("source", "unknown")
                merged[chunk_id]["sources"].add(source)
                merged[chunk_id]["scores"][source] = item.get("score", 0.0)
                merged[chunk_id]["rrf_score"] += 1.0 / (rrf_k + rank)

        fused = []
        for item in merged.values():
            item["sources"] = sorted(item["sources"])
            fused.append(item)
        fused.sort(key=lambda item: item["rrf_score"], reverse=True)
        return fused[:final_top_k]
