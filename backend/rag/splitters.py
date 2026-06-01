from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE

from backend.rag.utils import safe_stem


def simple_split_pages(
    pages,
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
):
    chunks = []

    for page in pages:
        text = page.get("text", "")
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(_build_chunk(page, chunk_text, chunk_index))
                chunk_index += 1

            start += max(1, chunk_size - chunk_overlap)

    return chunks


def recursive_split_pages(
    pages,
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ";", "；", ",", "，", " ", ""],
    )

    chunks = []

    for page in pages:
        texts = splitter.split_text(page.get("text", ""))

        for chunk_index, chunk_text in enumerate(texts):
            chunk_text = chunk_text.strip()
            if chunk_text:
                chunks.append(_build_chunk(page, chunk_text, chunk_index))

    return chunks


def split_pages(
    pages,
    splitter_name="recursive",
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
):
    if splitter_name == "simple":
        return simple_split_pages(
            pages,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    return recursive_split_pages(
        pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def chunks_to_texts(chunks):
    return [chunk["text"] for chunk in chunks]


def _build_chunk(page, chunk_text, chunk_index):
    file_name = page.get("file_name", "unknown")
    file_stem = safe_stem(file_name)
    page_no = page.get("page", 1)
    parser = page.get("parser", "unknown")

    chunk_id = f"{file_stem}_p{page_no}_c{chunk_index}"

    return {
        "chunk_id": chunk_id,
        "file_name": file_name,
        "file_stem": file_stem,
        "page": page_no,
        "chunk_index": chunk_index,
        "text": chunk_text,
        "parser": parser,
        "doc_type": page.get("doc_type", ""),
        "product_line": page.get("product_line", ""),
        "product_model": page.get("product_model", ""),
        "version": page.get("version", ""),
    }
