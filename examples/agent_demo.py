"""Full-feature demo: contrib tools + trace + async.

Run offline (mock):   python examples/agent_demo.py
Run with real LLM:    OPENAI_API_KEY=... python examples/agent_demo.py --live
"""
import sys

from picoagent import (
    Agent,
    AgentConfig,
    LLMResponse,
    MockLLM,
    OpenAICompatibleLLM,
    ToolCall,
    TraceRecorder,
    run_async,
)
from picoagent.contrib import default_tools


def main() -> None:
    if "--live" in sys.argv:
        llm = OpenAICompatibleLLM(model="gpt-4o-mini")
    else:
        llm = MockLLM([
            LLMResponse("", [ToolCall(name="calculator",
                                      arguments={"expression": "2 ** 10"})]),
            LLMResponse("", [ToolCall(name="list_dir", arguments={})]),
            LLMResponse("2 ** 10 = 1024, and the directory is listed."),
        ])

    agent = Agent(llm, default_tools(), AgentConfig(verbose=True))
    with TraceRecorder("demo_trace.jsonl") as tracer:
        agent.on(tracer)
        result = run_async(agent, "Compute 2**10 and list the workspace.")

    print("\nFinal answer:", result.answer)
    print("Steps:", len(result.steps), "| trace saved to demo_trace.jsonl")


if __name__ == "__main__":
    main()
