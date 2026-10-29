"""File read/write/list tools, sandboxed to a workspace directory."""
from __future__ import annotations

import os
from pathlib import Path

from ..tools import ToolError, ToolRegistry

MAX_READ_CHARS = 20_000


def file_tools(workspace: str | None = None) -> ToolRegistry:
    root = Path(workspace or os.getcwd()).resolve()
    reg = ToolRegistry()

    def _resolve(path: str) -> Path:
        p = (root / path).resolve()
        if not str(p).startswith(str(root)):
            raise ToolError(f"Path escapes workspace: {path}")
        return p

    @reg.register
    def read_file(path: str) -> str:
        """Read a UTF-8 text file inside the workspace. Returns at most
        20000 characters."""
        p = _resolve(path)
        if not p.is_file():
            raise ToolError(f"Not a file: {path}")
        return p.read_text(encoding="utf-8", errors="replace")[:MAX_READ_CHARS]

    @reg.register
    def write_file(path: str, content: str) -> str:
        """Write (overwrite) a UTF-8 text file inside the workspace."""
        p = _resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"wrote {p} ({len(content)} chars)"

    @reg.register
    def list_dir(path: str = ".") -> str:
        """List files and directories at a path inside the workspace."""
        p = _resolve(path)
        if not p.is_dir():
            raise ToolError(f"Not a directory: {path}")
        entries = sorted(e.name + ("/" if e.is_dir() else "") for e in p.iterdir())
        return "\n".join(entries[:200]) or "(empty)"

    return reg
