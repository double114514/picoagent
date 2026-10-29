"""ReAct-style agent loop with tool calling, streaming, async, and event hooks."""
from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .llm import BaseLLM, LLMResponse
from .memory import Memory
from .tools import Tool, ToolError, ToolRegistry
from .types import AgentResult, Message, Role, Step


@dataclass
class AgentConfig:
    max_steps: int = 10
    system_prompt: str = (
        "You are a helpful assistant. Use the provided tools when needed. "
        "Answer concisely when the task is complete."
    )
    max_memory_messages: int = 50
    verbose: bool = False
    stream: bool = False  # use LLM streaming; emits "token" events
    output_schema: dict | None = None  # constrain final answer to JSON Schema
    max_structured_retries: int = 2  # extra turns to fix invalid JSON output
    fresh: bool = True  # False: keep existing memory (for persistent sessions)


# Event hook signatures: (event_name, payload dict) -> None
Hook = Callable[[str, dict[str, Any]], None]
# Events: llm_call / tool_call / step / finish / token

_PARSE_FAILED = object()  # sentinel for structured-parse failure


class Agent:
    """ReAct-style agent: think -> act (tool call) -> observe, loop until done."""

    def __init__(self, llm: BaseLLM, tools: ToolRegistry | None = None,
                 config: AgentConfig | None = None,
                 memory: Memory | None = None):
        self.llm = llm
        # NOTE: ToolRegistry/Memory define __len__, so `x or default` would
        # wrongly discard empty-but-valid instances. Use explicit None checks.
        self.tools = tools if tools is not None else ToolRegistry()
        self.config = config if config is not None else AgentConfig()
        self.memory = memory if memory is not None else \
            Memory(self.config.max_memory_messages)
        self._hooks: list[Hook] = []

    def on(self, hook: Hook) -> None:
        """Register an event hook: llm_call/tool_call/step/finish/token."""
        self._hooks.append(hook)

    def _emit(self, event: str, **payload) -> None:
        for h in self._hooks:
            h(event, payload)
        if self.config.verbose:
            print(f"[{event}] {payload}")

    def _chat(self) -> LLMResponse:
        messages, schemas = self.memory.messages(), self.tools.schemas() or None
        if self.config.stream:
            return self.llm.chat_stream(
                messages, schemas,
                on_token=lambda tok: self._emit("token", token=tok))
        return self.llm.chat(messages, schemas)

    def _run_tool(self, tool: Tool, arguments: dict[str, Any]) -> str:
        result = tool.func(**arguments)
        return result if isinstance(result, str) else repr(result)

    async def _arun_tool(self, tool: Tool, arguments: dict[str, Any]) -> str:
        result = tool.func(**arguments)
        if inspect.isawaitable(result):
            result = await result
        return result if isinstance(result, str) else repr(result)

    @staticmethod
    def _accumulate(total: dict[str, int], usage: dict[str, int]) -> None:
        for k, v in usage.items():
            if isinstance(v, int):
                total[k] = total.get(k, 0) + v

    def _reset(self, task: str) -> None:
        from .structured import schema_prompt
        system = self.config.system_prompt
        if self.config.output_schema is not None:
            system += schema_prompt(self.config.output_schema)
        if self.config.fresh:
            self.memory.clear()
        has_system = any(m.role == Role.SYSTEM for m in self.memory.messages())
        if not has_system:
            self.memory.add(Message(role=Role.SYSTEM, content=system))
        self.memory.add_user(task)

    def _finish(self, answer: str, steps: list[Step],
                usage: dict[str, int], truncated: bool = False,
                parsed=None) -> AgentResult:
        self._emit("finish", answer=answer, truncated=truncated)
        return AgentResult(answer=answer, steps=steps,
                           messages=self.memory.messages(), usage=usage,
                           parsed=parsed)

    def _try_parse_structured(self, content: str):
        """Return parsed JSON or None if no schema / parse failed."""
        if self.config.output_schema is None:
            return None
        from .structured import parse_structured
        try:
            return parse_structured(content)
        except ValueError:
            return _PARSE_FAILED

    def run(self, task: str) -> AgentResult:
        self._reset(task)
        steps: list[Step] = []
        total_usage: dict[str, int] = {}
        struct_retries = self.config.max_structured_retries

        for i in range(self.config.max_steps):
            self._emit("llm_call", step=i)
            resp = self._chat()
            self._accumulate(total_usage, resp.usage)

            if not resp.tool_calls:
                parsed = self._try_parse_structured(resp.content)
                if parsed is _PARSE_FAILED and struct_retries > 0:
                    struct_retries -= 1
                    self.memory.add_assistant(resp.content)
                    self.memory.add_user(
                        "Your answer was not valid JSON matching the schema. "
                        "Reply with ONLY the corrected JSON object.")
                    continue
                if parsed is _PARSE_FAILED:
                    parsed = None
                self.memory.add_assistant(resp.content)
                steps.append(Step(index=i, thought=resp.content, is_final=True))
                return self._finish(resp.content, steps, total_usage,
                                    parsed=parsed)

            self.memory.add_assistant(resp.content, tool_calls=resp.tool_calls)
            for tc in resp.tool_calls:
                self._emit("tool_call", step=i, tool=tc.name,
                           arguments=tc.arguments)
                try:
                    observation = self._run_tool(self.tools.get(tc.name),
                                                 tc.arguments)
                except ToolError as exc:
                    observation = f"Error: {exc}"
                except Exception as exc:  # noqa: BLE001 - surface to the agent
                    observation = f"Error: {tc.name} failed: {exc}"
                self.memory.add_tool_result(tc.id, tc.name, observation)
                steps.append(Step(index=i, thought=resp.content, action=tc,
                                  observation=observation))
                self._emit("step", step=steps[-1])

        return self._finish("Reached max_steps without a final answer.",
                            steps, total_usage, truncated=True)

    async def arun(self, task: str) -> AgentResult:
        """Async variant: async tools are awaited, sync tools run in threads."""
        self._reset(task)
        steps: list[Step] = []
        total_usage: dict[str, int] = {}
        struct_retries = self.config.max_structured_retries

        for i in range(self.config.max_steps):
            self._emit("llm_call", step=i)
            resp = await self.llm.achat(self.memory.messages(),
                                        self.tools.schemas() or None)
            self._accumulate(total_usage, resp.usage)

            if not resp.tool_calls:
                parsed = self._try_parse_structured(resp.content)
                if parsed is _PARSE_FAILED and struct_retries > 0:
                    struct_retries -= 1
                    self.memory.add_assistant(resp.content)
                    self.memory.add_user(
                        "Your answer was not valid JSON matching the schema. "
                        "Reply with ONLY the corrected JSON object.")
                    continue
                if parsed is _PARSE_FAILED:
                    parsed = None
                self.memory.add_assistant(resp.content)
                steps.append(Step(index=i, thought=resp.content, is_final=True))
                return self._finish(resp.content, steps, total_usage,
                                    parsed=parsed)

            self.memory.add_assistant(resp.content, tool_calls=resp.tool_calls)
            for tc in resp.tool_calls:
                self._emit("tool_call", step=i, tool=tc.name,
                           arguments=tc.arguments)
                try:
                    observation = await self._arun_tool(self.tools.get(tc.name),
                                                        tc.arguments)
                except ToolError as exc:
                    observation = f"Error: {exc}"
                except Exception as exc:  # noqa: BLE001
                    observation = f"Error: {tc.name} failed: {exc}"
                self.memory.add_tool_result(tc.id, tc.name, observation)
                steps.append(Step(index=i, thought=resp.content, action=tc,
                                  observation=observation))
                self._emit("step", step=steps[-1])

        return self._finish("Reached max_steps without a final answer.",
                            steps, total_usage, truncated=True)


def run_async(agent: Agent, task: str) -> AgentResult:
    """Convenience wrapper: asyncio.run(agent.arun(task))."""
    return asyncio.run(agent.arun(task))
