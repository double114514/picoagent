"""Tool registry with decorator-based definition and schema generation."""
from __future__ import annotations

import inspect
from typing import Any, Callable, get_type_hints

_PY_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean",
               list: "array", dict: "object"}


class ToolError(Exception):
    pass


class Tool:
    def __init__(self, func: Callable, name: str | None = None,
                 description: str | None = None):
        self.func = func
        self.name = name or func.__name__
        self.description = description or (func.__doc__ or "").strip()
        if not self.description:
            raise ToolError(f"Tool '{self.name}' needs a description or docstring")
        self.schema = self._build_schema()

    def _build_schema(self) -> dict[str, Any]:
        sig = inspect.signature(self.func)
        hints = get_type_hints(self.func)
        props, required = {}, []
        for pname, param in sig.parameters.items():
            ptype = _PY_TO_JSON.get(hints.get(pname, str), "string")
            props[pname] = {
                "type": ptype,
                "description": "",
            }
            if param.default is inspect.Parameter.empty:
                required.append(pname)
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {"type": "object", "properties": props,
                           "required": required},
        }

    def run(self, **kwargs) -> str:
        try:
            result = self.func(**kwargs)
        except ToolError:
            raise
        except Exception as exc:  # surface tool failures to the agent
            raise ToolError(f"{self.name} failed: {exc}") from exc
        return result if isinstance(result, str) else repr(result)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Tool {self.name}>"


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, func: Callable | None = None, *,
                 name: str | None = None, description: str | None = None):
        """Use as @registry.register or registry.register(func)."""
        def wrap(f: Callable) -> Tool:
            tool = Tool(f, name=name, description=description)
            self._tools[tool.name] = tool
            return tool
        return wrap(func) if func else wrap

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolError(f"Unknown tool: {name}")
        return self._tools[name]

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema for t in self._tools.values()]

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def merge(self, other: "ToolRegistry") -> "ToolRegistry":
        """Copy all tools from another registry into this one."""
        self._tools.update(other._tools)
        return self

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
