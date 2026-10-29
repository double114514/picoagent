"""Core data types for picoagent."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    role: Role
    content: str
    name: str | None = None          # tool name when role == TOOL
    tool_call_id: str | None = None
    tool_calls: list["ToolCall"] | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        return d


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:8]}")

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class ToolResult:
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class Step:
    """One iteration of the agent loop."""
    index: int
    thought: str
    action: ToolCall | None = None
    observation: str | None = None
    is_final: bool = False


@dataclass
class AgentResult:
    answer: str
    steps: list[Step]
    messages: list[Message]
    usage: dict[str, int] = field(default_factory=dict)
    parsed: Any = None  # populated when an output_schema is configured
