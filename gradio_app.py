from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests

from backend.config import FASTAPI_BASE_URL, GRADIO_HOST, GRADIO_PORT

try:
    import gradio as gr
except ModuleNotFoundError:  # pragma: no cover - 未安装 Gradio 时仍允许测试格式化函数
    gr = None


API_BASE_URL = os.getenv("FASTAPI_BASE_URL", FASTAPI_BASE_URL).rstrip("/")


def call_after_sales_api(
    user_input: str,
    database_id: str = "",
    retrieval_mode: str = "hybrid",
    final_top_k: int = 5,
) -> dict[str, Any]:
    response = requests.post(
        f"{API_BASE_URL}/api/aftersales/run",
        json={
            "user_input": user_input,
            "database_id": database_id or None,
            "retrieval_options": {
                "retrieval_mode": retrieval_mode,
                "final_top_k": final_top_k,
            },
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def format_agent_response(payload: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not payload.get("success"):
        return f"接口调用失败：{payload.get('error', '未知错误')}", {}, {}, {}

    data = payload.get("data", {})
    action = data.get("action", {})
    customer_reply = action.get("customer_reply") or "暂未生成客户回复，请检查 ActionAgent 输出。"
    debug_info = {
        "保修判断": data.get("warranty", {}),
        "是否需要人工升级": data.get("escalation", {}),
        "工单草稿": action.get("ticket", {}),
        "审核结果": data.get("audit", {}),
        "知识库引用来源": data.get("retrieval", {}).get("sources", []),
        "知识库检索结果": data.get("retrieval", {}).get("results", []),
        "检索过程": data.get("retrieval", {}).get("trace", {}),
    }
    return customer_reply, debug_info, {"tool_history": data.get("tool_history", [])}, {"trace": data.get("trace", [])}


def run_agent(
    user_input: str,
    database_id: str,
    retrieval_mode: str,
    final_top_k: int,
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not user_input.strip():
        return "请先输入客户售后问题。", {}, {}, {}
    try:
        return format_agent_response(call_after_sales_api(user_input, database_id, retrieval_mode, final_top_k))
    except Exception as exc:
        return f"无法连接 FastAPI 服务，请先启动 api.py。错误：{exc}", {}, {}, {}


def get_kb_items() -> list[dict[str, Any]]:
    response = requests.get(f"{API_BASE_URL}/api/kb/list", timeout=30)
    response.raise_for_status()
    payload = response.json()
    return payload.get("data", {}).get("items", []) if payload.get("success") else []


def get_kb_choices() -> list[str]:
    return [item["database_id"] for item in get_kb_items()]


def refresh_kb_choices() -> tuple[Any, dict[str, Any]]:
    choices = get_kb_choices()
    status = call_kb_status()
    if gr is None:
        return choices, status
    return gr.update(choices=choices, value=choices[0] if choices else None), status


def call_kb_status() -> dict[str, Any]:
    response = requests.get(f"{API_BASE_URL}/api/kb/status", timeout=30)
    response.raise_for_status()
    return response.json().get("data", {})


def build_kb(
    files: list[str] | str | None,
    display_name: str,
    doc_type: str,
    product_line: str,
    product_model: str,
    version: str,
    parser_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[str, Any, dict[str, Any]]:
    if not files:
        return "请先上传 PDF 文件。", gr.update() if gr else {}, {}
    file_paths = files if isinstance(files, list) else [files]
    opened_files = []
    try:
        multipart_files = []
        for file_path in file_paths:
            path = Path(file_path)
            file_obj = path.open("rb")
            opened_files.append(file_obj)
            multipart_files.append(("files", (path.name, file_obj, "application/pdf")))

        response = requests.post(
            f"{API_BASE_URL}/api/kb/build",
            files=multipart_files,
            data={
                "display_name": display_name,
                "doc_type": doc_type,
                "product_line": product_line,
                "product_model": product_model,
                "version": version,
                "parser_name": parser_name,
                "splitter_name": "recursive",
                "chunk_size": int(chunk_size),
                "chunk_overlap": int(chunk_overlap),
            },
            timeout=1800,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success"):
            return f"知识库构建失败：{payload.get('error', '未知错误')}", gr.update() if gr else {}, {}
        data = payload.get("data", {})
        choices = get_kb_choices()
        message = f"知识库构建完成：{data.get('database_id')}，共 {data.get('chunk_count')} 个文本块。"
        return message, gr.update(choices=choices, value=data.get("database_id")) if gr else choices, data
    except Exception as exc:
        return f"知识库构建失败：{exc}", gr.update() if gr else {}, {}
    finally:
        for file_obj in opened_files:
            file_obj.close()


def load_kb(database_id: str) -> tuple[str, dict[str, Any]]:
    if not database_id:
        return "请先选择知识库。", {}
    response = requests.post(f"{API_BASE_URL}/api/kb/load", json={"database_id": database_id}, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        return f"知识库加载失败：{payload.get('error', '未知错误')}", {}
    return f"知识库已加载：{database_id}", payload.get("data", {})


def search_rag(question: str, database_id: str, retrieval_mode: str, final_top_k: int) -> dict[str, Any]:
    if not question.strip():
        return {"error": "请先输入检索问题。"}
    response = requests.post(
        f"{API_BASE_URL}/api/rag/search",
        json={
            "question": question,
            "database_id": database_id or None,
            "retrieval_options": {
                "retrieval_mode": retrieval_mode,
                "final_top_k": final_top_k,
            },
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def create_demo() -> Any:
    if gr is None:
        raise RuntimeError("当前环境未安装 gradio，请先执行 pip install -r requirements.txt。")

    examples = [
        "QY-320 显示 E03，出水变慢，买了半年",
        "买了一年半还能免费维修吗？",
        "机器漏水把插座打湿了",
        "QY-320 不出水，已经重启过，地址在广州",
    ]

    initial_choices: list[str] = []
    try:
        initial_choices = get_kb_choices()
    except Exception:
        initial_choices = []

    with gr.Blocks(title="AfterSalesAgentV2 售后智能体工作台") as demo:
        gr.Markdown("# AfterSalesAgentV2 售后智能体工作台")
        gr.Markdown("先构建或加载知识库，再运行售后 Agent。未加载知识库时，系统仍会返回规则和工具结果。")

        with gr.Tab("知识库管理"):
            kb_files = gr.File(label="上传 PDF", file_count="multiple", file_types=[".pdf"], type="filepath")
            with gr.Row():
                display_name = gr.Textbox(label="知识库名称", value="售后知识库")
                doc_type = gr.Textbox(label="文档类型", value="产品说明书")
                product_line = gr.Textbox(label="产品线", value="净水设备")
                product_model = gr.Textbox(label="产品型号", value="QY-320")
                version = gr.Textbox(label="版本号", value="2025")
            with gr.Row():
                parser_name = gr.Radio(
                    ["pypdf", "mineru", "langchain-mineru-flash", "langchain-mineru-precision"],
                    label="PDF 解析器",
                    value="pypdf",
                )
                chunk_size = gr.Slider(300, 1500, value=700, step=50, label="chunk_size")
                chunk_overlap = gr.Slider(0, 300, value=80, step=10, label="chunk_overlap")
            build_button = gr.Button("构建知识库", variant="primary")
            kb_message = gr.Textbox(label="知识库操作结果")
            kb_dropdown = gr.Dropdown(label="已有知识库", choices=initial_choices, value=initial_choices[0] if initial_choices else None)
            with gr.Row():
                refresh_button = gr.Button("刷新知识库列表")
                load_button = gr.Button("加载知识库")
            kb_status = gr.JSON(label="知识库状态")

        with gr.Tab("售后 Agent"):
            user_input = gr.Textbox(label="客户售后问题", lines=4, value=examples[0])
            gr.Examples(examples=examples, inputs=user_input)
            with gr.Row():
                retrieval_mode = gr.Radio(["hybrid", "vector", "bm25"], label="检索方式", value="hybrid")
                final_top_k = gr.Slider(1, 10, value=5, step=1, label="最终上下文 TopK")
            run_button = gr.Button("运行售后 Agent", variant="primary")
            customer_reply = gr.Textbox(label="给客户的回复", lines=8)
            debug_info = gr.JSON(label="业务结果")
            tool_history = gr.JSON(label="工具调用记录")
            trace = gr.JSON(label="Agent 执行轨迹")

        with gr.Tab("RAG 调试"):
            rag_question = gr.Textbox(label="检索问题", value="QY-320 E03 出水变慢")
            rag_button = gr.Button("检索知识库")
            rag_result = gr.JSON(label="检索结果")

        with gr.Tab("系统状态"):
            status_button = gr.Button("刷新系统状态")
            system_status = gr.JSON(label="系统状态")

        build_button.click(
            fn=build_kb,
            inputs=[kb_files, display_name, doc_type, product_line, product_model, version, parser_name, chunk_size, chunk_overlap],
            outputs=[kb_message, kb_dropdown, kb_status],
        )
        refresh_button.click(fn=refresh_kb_choices, outputs=[kb_dropdown, kb_status])
        load_button.click(fn=load_kb, inputs=kb_dropdown, outputs=[kb_message, kb_status])
        run_button.click(
            fn=run_agent,
            inputs=[user_input, kb_dropdown, retrieval_mode, final_top_k],
            outputs=[customer_reply, debug_info, tool_history, trace],
        )
        rag_button.click(fn=search_rag, inputs=[rag_question, kb_dropdown, retrieval_mode, final_top_k], outputs=rag_result)
        status_button.click(fn=call_kb_status, outputs=system_status)

    return demo


if __name__ == "__main__":
    create_demo().launch(server_name=GRADIO_HOST, server_port=GRADIO_PORT)
