"""End-to-end agent loop tests using MockLLM."""
from picoagent import Agent, AgentConfig, LLMResponse, MockLLM, ToolCall, ToolRegistry


def make_tools() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.register
    def add(a: int, b: int) -> str:
        """Add two numbers."""
        return str(a + b)

    return reg


def test_single_tool_call_then_answer():
    llm = MockLLM([
        LLMResponse("", [ToolCall(name="add", arguments={"a": 2, "b": 3})]),
        LLMResponse("The answer is 5."),
    ])
    agent = Agent(llm, make_tools(), AgentConfig())
    result = agent.run("What is 2+3?")

    assert result.answer == "The answer is 5."
    assert len(result.steps) == 2
    assert result.steps[0].observation == "5"
    assert result.steps[1].is_final


def test_unknown_tool_returns_error_observation():
    llm = MockLLM([
        LLMResponse("", [ToolCall(name="nope", arguments={})]),
        LLMResponse("I could not find that tool."),
    ])
    agent = Agent(llm, make_tools(), AgentConfig())
    result = agent.run("try nope")

    assert "Error" in result.steps[0].observation
    assert result.answer == "I could not find that tool."


def test_max_steps_truncation():
    llm = MockLLM([
        LLMResponse("", [ToolCall(name="add", arguments={"a": 1, "b": 1})])
        for _ in range(3)
    ])
    agent = Agent(llm, make_tools(), AgentConfig(max_steps=3))
    result = agent.run("loop forever")

    assert "max_steps" in result.answer
    assert len(result.steps) == 3


def test_hooks_receive_events():
    events = []
    llm = MockLLM([LLMResponse("done")])
    agent = Agent(llm, make_tools(), AgentConfig())
    agent.on(lambda event, payload: events.append(event))
    agent.run("hi")

    assert "llm_call" in events
    assert "finish" in events


def test_usage_is_aggregated():
    llm = MockLLM([
        LLMResponse("", [ToolCall(name="add", arguments={"a": 1, "b": 2})],
                    usage={"total_tokens": 10}),
        LLMResponse("3", usage={"total_tokens": 5}),
    ])
    agent = Agent(llm, make_tools(), AgentConfig())
    result = agent.run("1+2?")

    assert result.usage["total_tokens"] == 15
