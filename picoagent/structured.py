"""Structured output: constrain the agent's final answer to a JSON schema.

Supports two paths:
- OpenAI ``response_format`` with ``json_schema`` (strict mode) when the
  provider supports it.
- Prompt-based fallback: append the schema to the system prompt and parse the
  final answer as JSON, with retry on invalid output.

A plain dict JSON Schema is accepted, or a Pydantic model class (optional
dependency) via ``.model_json_schema()``.
"""
from __future__ import annotations

import json
from typing import Any

from .types import AgentResult


def to_json_schema(schema: Any) -> dict[str, Any]:
    """Normalize a schema input to a plain JSON Schema dict.

    Accepts a dict, or a Pydantic model class (has model_json_schema).
    """
    if isinstance(schema, dict):
        return schema
    if hasattr(schema, "model_json_schema"):
        return schema.model_json_schema()
    raise TypeError(
        "schema must be a dict JSON Schema or a Pydantic model class")


def schema_prompt(schema: dict[str, Any]) -> str:
    return (
        "\n\nYour FINAL answer MUST be a single JSON object matching this "
        "JSON Schema (reply with ONLY the JSON, no prose, no markdown "
        "fences):\n" + json.dumps(schema, ensure_ascii=False, indent=2))


def parse_structured(text: str) -> Any:
    """Extract and parse the first JSON value from model output."""
    cleaned = text.strip()
    if "```" in cleaned:
        for part in cleaned.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part and part[0] in "[{":
                cleaned = part
                break
    # find first balanced JSON value
    start = min((i for i in (cleaned.find("{"), cleaned.find("["))
                 if i != -1), default=-1)
    if start == -1:
        raise ValueError(f"No JSON found in output: {text[:200]}")
    return json.loads(cleaned[start:]) if _is_whole(cleaned[start:]) \
        else _extract_balanced(cleaned[start:])


def _is_whole(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except ValueError:
        return False


def _extract_balanced(s: str) -> Any:
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return json.loads(s[:i + 1])
    raise ValueError("Unbalanced JSON in output")


class StructuredResult:
    """AgentResult + parsed structured output."""

    def __init__(self, result: AgentResult, data: Any):
        self.result = result
        self.data = data

    def __getattr__(self, name):  # delegate to underlying result
        return getattr(self.result, name)
