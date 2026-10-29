"""Quickstart: a calculator agent.

Run offline (mock):   python examples/quickstart.py
Run with real LLM:    OPENAI_API_KEY=... python examples/quickstart.py --live
"""
import sys

from picoagent import (
    Agent,
    AgentConfig,
    LLMResponse,
    MockLLM,
    OpenAICompatibleLLM,
    ToolCall,
    ToolRegistry,
)


def build_tools() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.register
    def calculator(expression: str) -> str:
        """Evaluate a basic arithmetic expression like '2 + 3 * 4'."""
        allowed = set("0123456789+-*/(). ")
        if not set(expression) <= allowed:
            raise ValueError("unsafe expression")
        return str(eval(expression))

    return reg


def main() -> None:
    if "--live" in sys.argv:
        llm = OpenAICompatibleLLM(model="gpt-4o-mini")
    else:
        llm = MockLLM([
            LLMResponse("", [ToolCall(name="calculator",
                                      arguments={"expression": "12 * (3 + 4)"})]),
            LLMResponse("12 * (3 + 4) = 84."),
        ])

    agent = Agent(llm, build_tools(), AgentConfig(verbose=True))
    result = agent.run("What is 12 * (3 + 4)?")
    print("\nFinal answer:", result.answer)
    for step in result.steps:
        print(f"  step {step.index}: thought={step.thought!r} "
              f"action={step.action.name if step.action else None} "
              f"obs={step.observation!r}")


if __name__ == "__main__":
    main()
