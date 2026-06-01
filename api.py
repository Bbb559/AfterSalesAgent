from __future__ import annotations

from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel, Field

from backend.graph_workflow import AfterSalesGraphWorkflow
from backend.rag.rag_service import RAGService


app = FastAPI(title="AfterSalesAgentV2 API", version="0.1.0")
rag_service = RAGService()
workflow = AfterSalesGraphWorkflow(rag_service=rag_service)


class AfterSalesRunRequest(BaseModel):
    user_input: str = Field(..., min_length=1, description="客户售后问题")
    retrieval_options: dict[str, Any] = Field(default_factory=dict, description="RAG 检索参数")
    database_id: str | None = Field(default=None, description="预留知识库编号，第一版可以为空")


class ApiResponse(BaseModel):
    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class LoadKnowledgeBaseRequest(BaseModel):
    database_id: str = Field(..., min_length=1, description="知识库编号")


class RagSearchRequest(BaseModel):
    question: str = Field(..., min_length=1, description="检索问题")
    retrieval_options: dict[str, Any] = Field(default_factory=dict, description="RAG 检索参数")
    database_id: str | None = Field(default=None, description="知识库编号")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "AfterSalesAgentV2 API"}


@app.get("/api/kb/list", response_model=ApiResponse)
def list_knowledge_bases() -> ApiResponse:
    return ApiResponse(success=True, data={"items": rag_service.list_knowledge_bases()})


@app.get("/api/kb/status", response_model=ApiResponse)
def knowledge_base_status() -> ApiResponse:
    return ApiResponse(success=True, data=rag_service.status())


@app.post("/api/kb/build", response_model=ApiResponse)
def build_knowledge_base(
    files: list[UploadFile] = File(...),
    parser_name: str = Form("pypdf"),
    splitter_name: str = Form("recursive"),
    chunk_size: int = Form(700),
    chunk_overlap: int = Form(80),
    display_name: str = Form(""),
    doc_type: str = Form(""),
    product_line: str = Form(""),
    product_model: str = Form(""),
    version: str = Form(""),
) -> ApiResponse:
    result = rag_service.build_knowledge_base(
        uploaded_files=files,
        parser_name=parser_name,
        splitter_name=splitter_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        display_name=display_name,
        doc_type=doc_type,
        product_line=product_line,
        product_model=product_model,
        version=version,
    )
    return ApiResponse(success=bool(result.get("success")), data=result if result.get("success") else {}, error=result.get("error", ""))


@app.post("/api/kb/load", response_model=ApiResponse)
def load_knowledge_base(payload: LoadKnowledgeBaseRequest) -> ApiResponse:
    result = rag_service.load_knowledge_base(payload.database_id)
    return ApiResponse(success=bool(result.get("success")), data=result if result.get("success") else {}, error=result.get("error", ""))


@app.delete("/api/kb/{database_id}", response_model=ApiResponse)
def delete_knowledge_base(database_id: str) -> ApiResponse:
    result = rag_service.delete_knowledge_base(database_id)
    return ApiResponse(success=bool(result.get("success")), data=result if result.get("success") else {}, error=result.get("error", ""))


@app.post("/api/rag/search", response_model=ApiResponse)
def search_rag(payload: RagSearchRequest) -> ApiResponse:
    options = dict(payload.retrieval_options)
    if payload.database_id:
        options["database_id"] = payload.database_id
    result = rag_service.retrieve(payload.question, options)
    return ApiResponse(success=not bool(result.get("error")), data=result, error=result.get("error", ""))


@app.post("/api/aftersales/run", response_model=ApiResponse)
def run_after_sales_agent(payload: AfterSalesRunRequest) -> ApiResponse:
    try:
        options = dict(payload.retrieval_options)
        if payload.database_id:
            options["database_id"] = payload.database_id
        result = workflow.run(payload.user_input, options)
        return ApiResponse(success=True, data=result)
    except Exception as exc:  # pragma: no cover - API 最外层防御边界
        return ApiResponse(success=False, error=f"售后 Agent 运行失败：{exc}")
