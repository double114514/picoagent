"""Built-in toolkits. Each factory returns a populated ToolRegistry.

    from picoagent.contrib import default_tools
    agent = Agent(llm=..., tools=default_tools())
"""
from ..tools import ToolRegistry
from .calculator import calculator_tools
from .file_io import file_tools
from .python_repl import repl_tools
from .web_search import search_tools


def default_tools(workspace: str | None = None) -> ToolRegistry:
    """calculator + file_io + python_repl + web_search in one registry."""
    reg = ToolRegistry()
    for source in (calculator_tools(), file_tools(workspace),
                   repl_tools(), search_tools()):
        for tool in source.all():
            reg._tools[tool.name] = tool
    return reg


__all__ = [
    "calculator_tools",
    "default_tools",
    "file_tools",
    "repl_tools",
    "search_tools",
]
