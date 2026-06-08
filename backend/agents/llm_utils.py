from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import JsonOutputParser


_JSON_PARSER = JsonOutputParser()


def invoke_json(llm: Any | None, prompt: Any, variables: dict[str, Any]) -> dict[str, Any]:
    """运行 LangChain 提示词、模型和解析器组成的 JSON 链。
    调用失败时返回空字典，让代理只应用必要的硬性兜底。
    """
    if llm is None:
        return {}

    try:
        chain = prompt | llm | _JSON_PARSER
        parsed = chain.invoke(variables)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}
