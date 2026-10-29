"""Python REPL tool: runs code in a subprocess with timeout."""
from __future__ import annotations

import subprocess
import sys

from ..tools import ToolError, ToolRegistry

DEFAULT_TIMEOUT = 15
MAX_OUTPUT = 5_000


def repl_tools(timeout: int = DEFAULT_TIMEOUT) -> ToolRegistry:
    reg = ToolRegistry()

    @reg.register
    def python_repl(code: str) -> str:
        """Run Python code in a fresh subprocess and return stdout+stderr.
        Use print() to see values. Times out after 15 seconds."""
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", code],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise ToolError(f"Execution timed out after {timeout}s")
        out = (proc.stdout or "") + (proc.stderr or "")
        return out[:MAX_OUTPUT] or f"(no output, exit code {proc.returncode})"

    return reg
