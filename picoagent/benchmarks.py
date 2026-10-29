"""Benchmarks: measure agent success rate and token efficiency on tasks.

    python -m picoagent.benchmarks            # offline self-check with MockLLM

A benchmark is a list of tasks; each task has a prompt, an expected substring
(or callable checker), and optional scripted mock responses for offline runs.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .agent import Agent, AgentConfig
from .llm import BaseLLM, LLMResponse, MockLLM
from .tools import ToolRegistry
from .types import ToolCall

Checker = Callable[[str], bool]


@dataclass
class BenchTask:
    name: str
    prompt: str
    expect: str | Checker  # substring or predicate on the final answer
    mock: list[LLMResponse] | None = None  # scripted responses for offline run


@dataclass
class BenchResult:
    name: str
    passed: bool
    answer: str
    steps: int
    usage: dict[str, int] = field(default_factory=dict)


def _check(expect: str | Checker, answer: str) -> bool:
    if callable(expect):
        return bool(expect(answer))
    return expect.lower() in answer.lower()


def run_benchmark(llm_factory: Callable[[BenchTask], BaseLLM],
                  tasks: list[BenchTask],
                  tools: ToolRegistry | None = None,
                  config: AgentConfig | None = None) -> list[BenchResult]:
    """Run tasks sequentially; llm_factory builds an LLM per task (so mocks
    stay isolated)."""
    results = []
    for task in tasks:
        agent = Agent(llm=llm_factory(task), tools=tools or ToolRegistry(),
                      config=config or AgentConfig())
        result = agent.run(task.prompt)
        results.append(BenchResult(
            name=task.name,
            passed=_check(task.expect, result.answer),
            answer=result.answer,
            steps=len(result.steps),
            usage=result.usage,
        ))
    return results


def summarize(results: list[BenchResult]) -> dict[str, Any]:
    passed = sum(r.passed for r in results)
    total_tokens = sum(r.usage.get("total_tokens", 0) for r in results)
    return {
        "total": len(results),
        "passed": passed,
        "success_rate": round(passed / len(results), 3) if results else 0.0,
        "total_tokens": total_tokens,
        "avg_steps": round(sum(r.steps for r in results) / len(results), 2)
        if results else 0.0,
    }


def offline_demo() -> dict[str, Any]:
    """Self-contained smoke benchmark using MockLLM (no API key needed)."""
    from .contrib.calculator import calculator_tools

    tasks = [
        BenchTask(
            name="arith",
            prompt="What is 6 * 7?",
            expect="42",
            mock=[LLMResponse("", [ToolCall(name="calculator",
                                            arguments={"expression": "6 * 7"})]),
                  LLMResponse("6 * 7 = 42.")],
        ),
        BenchTask(
            name="direct",
            prompt="Say 'pong'.",
            expect="pong",
            mock=[LLMResponse("pong")],
        ),
    ]
    results = run_benchmark(lambda t: MockLLM(list(t.mock or [])), tasks,
                            tools=calculator_tools())
    return summarize(results)


if __name__ == "__main__":
    print(offline_demo())
