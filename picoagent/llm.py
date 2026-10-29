"""LLM provider abstraction.

Supports any OpenAI-compatible chat-completions endpoint via HTTP,
plus a MockLLM for offline testing. No hard dependency on openai SDK.
"""
from __future__ import annotations

import asyncio
import json
import os
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterator

from .types import Message, Role, ToolCall


class LLMResponse:
    def __init__(self, content: str, tool_calls: list[ToolCall] | None = None,
                 usage: dict[str, int] | None = None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.usage = usage or {}


class BaseLLM(ABC):
    @abstractmethod
    def chat(self, messages: list[Message],
             tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        ...

    def chat_stream(self, messages: list[Message],
                    tools: list[dict[str, Any]] | None = None,
                    on_token: Callable[[str], None] | None = None
                    ) -> LLMResponse:
        """Streaming variant. Default falls back to non-streaming."""
        resp = self.chat(messages, tools)
        if on_token and resp.content:
            on_token(resp.content)
        return resp

    async def achat(self, messages: list[Message],
                    tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        """Async wrapper (runs sync HTTP in a thread by default)."""
        return await asyncio.to_thread(self.chat, messages, tools)


class OpenAICompatibleLLM(BaseLLM):
    """Works with OpenAI, DeepSeek, Qwen, Moonshot, vLLM, Ollama, etc."""

    def __init__(self, model: str, api_key: str | None = None,
                 base_url: str = "https://api.openai.com/v1",
                 temperature: float = 0.0, timeout: int = 60):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def _build_payload(self, messages: list[Message],
                       tools: list[dict[str, Any]] | None,
                       stream: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [m.to_dict() for m in messages],
            "stream": stream,
        }
        if tools:
            payload["tools"] = [
                {"type": "function", "function": t} for t in tools
            ]
        return payload

    def _post(self, payload: dict[str, Any]):
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        return urllib.request.urlopen(req, timeout=self.timeout)

    @staticmethod
    def _parse_choice(choice: dict[str, Any]) -> LLMResponse:
        msg = choice["message"]
        tool_calls = [
            ToolCall(id=tc["id"], name=tc["function"]["name"],
                     arguments=json.loads(tc["function"]["arguments"] or "{}"))
            for tc in msg.get("tool_calls") or []
        ]
        return LLMResponse(content=msg.get("content") or "",
                           tool_calls=tool_calls)

    def chat(self, messages: list[Message],
             tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        with self._post(self._build_payload(messages, tools)) as resp:
            data = json.loads(resp.read().decode())
        result = self._parse_choice(data["choices"][0])
        result.usage = data.get("usage") or {}
        return result

    def chat_stream(self, messages: list[Message],
                    tools: list[dict[str, Any]] | None = None,
                    on_token: Callable[[str], None] | None = None
                    ) -> LLMResponse:
        """SSE streaming: emits content deltas via on_token, returns full response."""
        payload = self._build_payload(messages, tools, stream=True)
        chunks: list[str] = []
        tc_acc: dict[int, dict[str, Any]] = {}
        with self._post(payload) as resp:
            for raw in iter(resp.readline, b""):
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                for piece in self._handle_sse_chunk(data, tc_acc):
                    chunks.append(piece)
                    if on_token:
                        on_token(piece)
        tool_calls = [
            ToolCall(id=v.get("id") or f"call_{i}",
                     name=v.get("name") or "",
                     arguments=json.loads(v.get("arguments") or "{}"))
            for i, v in sorted(tc_acc.items())
        ]
        return LLMResponse(content="".join(chunks), tool_calls=tool_calls)

    @staticmethod
    def _handle_sse_chunk(data: str,
                          tc_acc: dict[int, dict[str, Any]]) -> Iterator[str]:
        """Parse one SSE data payload; accumulate tool-call deltas, yield content."""
        delta = json.loads(data)["choices"][0].get("delta") or {}
        if delta.get("content"):
            yield delta["content"]
        for tc in delta.get("tool_calls") or []:
            slot = tc_acc.setdefault(tc.get("index", 0),
                                     {"id": "", "name": "", "arguments": ""})
            if tc.get("id"):
                slot["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["name"] += fn["name"]
            if fn.get("arguments"):
                slot["arguments"] += fn["arguments"]


class MockLLM(BaseLLM):
    """Scripted LLM for tests: returns queued responses in order."""

    def __init__(self, responses: list[LLMResponse]):
        self._queue = list(responses)
        self.calls: list[list[Message]] = []

    def chat(self, messages: list[Message],
             tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        self.calls.append(messages)
        if not self._queue:
            raise RuntimeError("MockLLM queue exhausted")
        return self._queue.pop(0)


def make_system_message(content: str) -> Message:
    return Message(role=Role.SYSTEM, content=content)
