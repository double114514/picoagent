"""Safe arithmetic calculator tool (AST-based, no eval)."""
from __future__ import annotations

import ast
import operator

from ..tools import ToolError, ToolRegistry

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def safe_eval(expression: str) -> float:
    """Evaluate an arithmetic expression; rejects names/calls/attributes."""
    tree = ast.parse(expression, mode="eval")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.operand))
        raise ToolError(f"Unsupported expression: {ast.dump(node)}")

    return _eval(tree)


def calculator_tools() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.register
    def calculator(expression: str) -> str:
        """Evaluate a basic arithmetic expression like '2 + 3 * 4' or '2 ** 10'.
        Supports + - * / // % ** and parentheses. No variables or functions."""
        result = safe_eval(expression)
        return str(int(result)) if float(result).is_integer() else str(result)

    return reg
