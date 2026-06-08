from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

"""基础工具类和工具注册器，提供工具执行的统一接口和错误处理机制。"""

@dataclass
class ToolResult:
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    tool_name: str = ""
    execution_time: float = 0.0


class BaseTool:
    name = ""
    description = ""
    required_params: list[str] = []

    def validate(self, params: dict[str, Any]) -> str:
        for param in self.required_params:
            if params.get(param) in (None, ""):
                return f"Missing required parameter: {param}"
        return ""

    def execute(self, **kwargs: Any) -> ToolResult:
        start = time.time()
        error = self.validate(kwargs)
        if error:
            return ToolResult(False, error=error, tool_name=self.name)

        try:
            data = self.run(**kwargs)
            return ToolResult(
                True,
                data=data,
                tool_name=self.name,
                execution_time=round(time.time() - start, 3),
            )
        except Exception as exc:  # pragma: no cover - 工具最外层防御边界
            return ToolResult(
                False,
                error=str(exc),
                tool_name=self.name,
                execution_time=round(time.time() - start, 3),
            )

    def run(self, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError


class ToolRegistry:
    def __init__(self) -> None:
        self.tools: dict[str, BaseTool] = {}
        self.history: list[dict[str, Any]] = []

    def register(self, tool: BaseTool) -> None:
        self.tools[tool.name] = tool

    def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        tool = self.tools.get(tool_name)
        if not tool:
            return ToolResult(False, error=f"Tool not found: {tool_name}", tool_name=tool_name)

        result = tool.execute(**kwargs)
        self.history.append(
            {
                "tool_name": tool_name,
                "success": result.success,
                "data": result.data,
                "error": result.error,
                "execution_time": result.execution_time,
            }
        )
        return result
