import time
from pathlib import Path
from tempfile import NamedTemporaryFile

import requests
from pypdf import PdfReader

from backend.config import (
    MINERU_API_BASE_URL,
    MINERU_API_TOKEN,
    MINERU_ENABLE_FORMULA,
    MINERU_ENABLE_TABLE,
    MINERU_IS_OCR,
    MINERU_LANGUAGE,
    MINERU_MODEL_VERSION,
    MINERU_PAGE_RANGES,
    MINERU_POLL_INTERVAL,
    MINERU_TIMEOUT,
    MINERU_DOWNLOAD_RETRY, # 下载失败重试次数
    MINERU_DOWNLOAD_TIMEOUT, # 下载超时时间（秒）
    LANGCHAIN_MINERU_SPLIT_PAGES,
    LANGCHAIN_MINERU_TIMEOUT,
    get_mineru_download_verify_ssl,
)


def _save_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return Path(tmp.name)


def parse_with_pypdf(uploaded_files):
    pages = []
    for uploaded_file in uploaded_files:
        reader = PdfReader(uploaded_file)
        for page_index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({
                    "file_name": uploaded_file.name,
                    "page": page_index,
                    "text": text,
                    "parser": "pypdf",
                })
    return pages


def parse_with_mineru(uploaded_files):
    """使用 MinerU 官方 API 解析本地 PDF 文件.

    流程：
    1. 创建批量上传任务和签名上传 URL;
    2. 将每个 PDF 上传到其 URL;
    3. 轮询批量结果，直到所有文件都完成解析;
    4. 将 Markdown 输出转换为项目页面记录.
    """
    if not MINERU_API_TOKEN:
        raise ValueError("请先在 .env 中配置 MINERU_API_TOKEN。")

    if not uploaded_files:
        return []

    temp_paths = []
    try:
        for uploaded_file in uploaded_files:
            temp_path = _save_uploaded_file(uploaded_file)
            temp_paths.append((uploaded_file, temp_path))

        batch_info = _create_mineru_batch(temp_paths)
        _upload_files_to_mineru(temp_paths, batch_info)
        results = _wait_for_mineru_results(batch_info["batch_id"])

        return _mineru_results_to_pages(results)

    finally:
        for uploaded_file, temp_path in temp_paths:
            temp_path.unlink(missing_ok=True)
            uploaded_file.seek(0)


def parse_with_langchain_mineru(uploaded_files, mode="flash"):
    """用 langchain-mineru 直接加载 Document，绕过旧接口的 ZIP 下载分支。"""
    try:
        from langchain_mineru import MinerULoader
    except ModuleNotFoundError as error:
        raise RuntimeError("缺少 langchain-mineru，请先执行 pip install langchain-mineru。") from error

    if mode == "precision" and not MINERU_API_TOKEN:
        raise ValueError("langchain-mineru precision 模式需要在 .env 中配置 MINERU_API_TOKEN。")

    pages = []
    temp_paths = []
    try:
        for uploaded_file in uploaded_files:
            temp_path = _save_uploaded_file(uploaded_file)
            temp_paths.append((uploaded_file, temp_path))

            loader_kwargs = {
                "source": str(temp_path),
                "mode": mode,
                "language": MINERU_LANGUAGE,
                "split_pages": LANGCHAIN_MINERU_SPLIT_PAGES,
                "timeout": LANGCHAIN_MINERU_TIMEOUT,
                "ocr": MINERU_IS_OCR,
                "formula": MINERU_ENABLE_FORMULA,
                "table": MINERU_ENABLE_TABLE,
            }
            if MINERU_PAGE_RANGES:
                loader_kwargs["pages"] = MINERU_PAGE_RANGES
            if mode == "precision":
                loader_kwargs["token"] = MINERU_API_TOKEN

            loader = MinerULoader(**loader_kwargs)
            docs = loader.load()
            pages.extend(_langchain_docs_to_pages(docs, uploaded_file.name, mode))

        return pages
    finally:
        for uploaded_file, temp_path in temp_paths:
            temp_path.unlink(missing_ok=True)
            uploaded_file.seek(0)


def _langchain_docs_to_pages(docs, fallback_file_name, mode):
    pages = []
    for index, doc in enumerate(docs, start=1):
        text = (getattr(doc, "page_content", "") or "").strip()
        if not text:
            continue

        metadata = getattr(doc, "metadata", {}) or {}
        page_no = metadata.get("page") or metadata.get("page_number") or index
        file_name = metadata.get("filename") or Path(str(metadata.get("source", fallback_file_name))).name

        pages.append({
            "file_name": file_name or fallback_file_name,
            "page": page_no,
            "text": text,
            "parser": f"langchain-mineru-{mode}",
        })
    return pages

# 生成 MinerU API 请求头
def _mineru_headers():
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINERU_API_TOKEN}",
    }


def _create_mineru_batch(temp_paths):
    url = f"{MINERU_API_BASE_URL}/api/v4/file-urls/batch"
    files = [
        {
            "name": uploaded_file.name,
            "is_ocr": MINERU_IS_OCR,
            "data_id": uploaded_file.name,
        }
        for uploaded_file, _ in temp_paths
    ]

    payload = {
        "enable_formula": MINERU_ENABLE_FORMULA,
        "enable_table": MINERU_ENABLE_TABLE,
        "language": MINERU_LANGUAGE,
        "model_version": MINERU_MODEL_VERSION,
        "files": files,
    }

    if MINERU_PAGE_RANGES:
        payload["page_ranges"] = MINERU_PAGE_RANGES

    response = requests.post(
        url,
        headers=_mineru_headers(),
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json() # 转化为python字典

    if payload.get("code") != 0:
        raise RuntimeError(f"MinerU 创建批量任务失败：{payload}")

    data = payload.get("data") or {}
    batch_id = data.get("batch_id")
    file_urls = data.get("file_urls") or []

    if not batch_id or not file_urls:
        raise RuntimeError(f"MinerU 返回缺少 batch_id 或 file_urls：{payload}")

    return {
        "batch_id": batch_id,
        "file_urls": file_urls,
    }

# 将文件上传到 MinerU 返回的预签名 URL。每个 URL 只能上传对应的文件，且通常有较短的过期时间。
def _upload_files_to_mineru(temp_paths, batch_info):
    file_urls = batch_info["file_urls"]

    if len(file_urls) != len(temp_paths):
        raise RuntimeError(
            f"MinerU 上传链接数量与文件数量不一致：{len(file_urls)} != {len(temp_paths)}"
        )

    for (_, temp_path), upload_url in zip(temp_paths, file_urls):
        # MinerU 返回的是 OSS 预签名上传 URL。不要额外传 Content-Type，
        # 否则可能导致签名校验不一致并返回 403 Forbidden。
        with temp_path.open("rb") as file_obj:
            response = requests.put(
                upload_url,
                data=file_obj,
                timeout=300,
            )

        try:
            response.raise_for_status()
        except requests.HTTPError as error:
            if response.status_code == 403:
                raise RuntimeError(
                    "MinerU OSS 上传被拒绝：请检查 MINERU_API_TOKEN 是否有效、"
                    "上传链接是否过期，并确认代码没有给 OSS 上传请求额外添加 Content-Type。"
                ) from error
            raise


def _wait_for_mineru_results(batch_id):
    url = f"{MINERU_API_BASE_URL}/api/v4/extract-results/batch/{batch_id}"
    deadline = time.time() + MINERU_TIMEOUT

    while time.time() < deadline:
        response = requests.get(
            url,
            headers=_mineru_headers(),
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()

        if payload.get("code") != 0:
            raise RuntimeError(f"MinerU 查询解析结果失败：{payload}")

        data = payload.get("data") or {}
        extract_results = data.get("extract_result") or []

        if extract_results and _all_mineru_files_finished(extract_results):
            return extract_results

        time.sleep(MINERU_POLL_INTERVAL)

    raise TimeoutError(f"MinerU 解析超时，batch_id={batch_id}")


def _all_mineru_files_finished(results):
    running_states = {"pending", "running", "processing", "converting", "waiting"}

    for item in results:
        state = str(item.get("state", "")).lower()
        if state in running_states:
            return False
        if item.get("err_msg"):
            raise RuntimeError(f"MinerU 文件解析失败：{item}")
        if not item.get("full_zip_url") and not item.get("md_content"):
            return False

    return True

# 将 MinerU 的批量解析结果转换为页面记录列表
def _mineru_results_to_pages(results):
    pages = []

    for item in results:
        file_name = item.get("file_name") or item.get("data_id") or "unknown.pdf"
        markdown = (item.get("md_content") or "").strip()

        if not markdown and item.get("full_zip_url"):
            markdown = _download_mineru_markdown(item["full_zip_url"])

        if markdown:
            pages.extend(_markdown_to_pages(markdown, file_name))

    return pages


def _download_mineru_markdown(zip_url):
    import zipfile
    from tempfile import TemporaryDirectory

    last_error = None

    for attempt in range(1, MINERU_DOWNLOAD_RETRY + 1):
        try:
            with TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)
                zip_path = temp_dir / "mineru_result.zip"

                _download_file_with_retry(
                    url=zip_url,
                    output_path=zip_path,
                    attempt=attempt,
                )

                with zipfile.ZipFile(zip_path) as zip_file:
                    zip_file.extractall(temp_dir)

                markdown_files = sorted(temp_dir.rglob("*.md"))
                if not markdown_files:
                    return ""

                return markdown_files[0].read_text(
                    encoding="utf-8",
                    errors="ignore",
                ).strip()

        except (
            requests.exceptions.SSLError,
            requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout,
            requests.exceptions.ChunkedEncodingError,
            zipfile.BadZipFile,
        ) as error:
            last_error = error

            if attempt < MINERU_DOWNLOAD_RETRY:
                time.sleep(2 * attempt)
                continue

    verify_ssl = get_mineru_download_verify_ssl()
    last_error_text = f"{type(last_error).__name__}: {last_error}" if last_error else "unknown"
    if verify_ssl:
        suggestion = "当前仍在验证 SSL 证书，可以临时在 .env 中设置 MINERU_DOWNLOAD_VERIFY_SSL=false 后重启 FastAPI。"
    else:
        suggestion = (
            "当前 MINERU_DOWNLOAD_VERIFY_SSL=false 已生效。仍然失败通常说明不是证书校验问题，"
            "而是 CDN 连接中断、TLS 传输 EOF、代理/网络拦截或 ZIP 内容下载不完整。"
            "建议稍后重试、切换网络，或先改用 pypdf 构建知识库。"
        )

    raise RuntimeError(
        "MinerU 解析已完成，但下载结果 ZIP 失败。"
        f"{suggestion}"
        f"最后一次下载异常：{last_error_text}"
    ) from last_error


def _download_file_with_retry(url, output_path, attempt):
    verify_ssl = get_mineru_download_verify_ssl()
    with requests.get(
        url,
        stream=True,
        timeout=MINERU_DOWNLOAD_TIMEOUT,
        verify=verify_ssl,
    ) as response:
        response.raise_for_status()

        with output_path.open("wb") as file_obj:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_obj.write(chunk)


def _markdown_to_pages(markdown, file_name):
    # MinerU 的 markdown 默认是整文输出。这里按页眉标记尽力切页；
    # 若结果没有页标记，就作为第 1 页进入后续清洗和分块。
    page_markers = ["\n\n--- page ", "\n\n# Page ", "\n\n## Page "]

    for marker in page_markers:
        if marker in markdown:
            return _split_markdown_by_marker(markdown, marker, file_name)

    return [{
        "file_name": file_name,
        "page": 1,
        "text": markdown,
        "parser": "mineru",
    }]


def _split_markdown_by_marker(markdown, marker, file_name):
    parts = markdown.split(marker)
    pages = []

    first = parts[0].strip()
    if first:
        pages.append({
            "file_name": file_name,
            "page": 1,
            "text": first,
            "parser": "mineru",
        })

    for index, part in enumerate(parts[1:], start=2):
        text = part.strip()
        if text:
            pages.append({
                "file_name": file_name,
                "page": index,
                "text": text,
                "parser": "mineru",
            })

    return pages


def parse_pdfs(uploaded_files, parser_name="pypdf"):
    if parser_name == "langchain-mineru-flash":
        return parse_with_langchain_mineru(uploaded_files, mode="flash")
    if parser_name == "langchain-mineru-precision":
        return parse_with_langchain_mineru(uploaded_files, mode="precision")
    if parser_name == "mineru":
        return parse_with_mineru(uploaded_files)
    return parse_with_pypdf(uploaded_files)

