"""Tests for async arun, streaming fallback, and async tools."""
import asyncio

from picoagent import (Agent, AgentConfig, LLMResponse, MockLLM, ToolCall,
                       ToolRegistry, run_async)


def test_arun_basic():
    llm = MockLLM([LLMResponse(content="done")])
    agent = Agent(llm=llm)
    result = run_async(agent, "hi")
    assert result.answer == "done"
    assert result.steps[0].is_final


def test_arun_with_async_tool():
    reg = ToolRegistry()

    @reg.register
    async def fetch(url: str) -> str:
        """Fetch a URL (mocked)."""
        await asyncio.sleep(0)
        return f"content of {url}"

    llm = MockLLM([
        LLMResponse(content="", tool_calls=[
            ToolCall(id="1", name="fetch", arguments={"url": "x.io"})]),
        LLMResponse(content="got it"),
    ])
    agent = Agent(llm=llm, tools=reg)
    result = run_async(agent, "fetch x.io")
    assert result.answer == "got it"
    assert result.steps[0].observation == "content of x.io"


def test_stream_fallback_emits_token():
    """BaseLLM.chat_stream default falls back to chat + single token emit."""
    tokens = []
    llm = MockLLM([LLMResponse(content="hello world")])
    agent = Agent(llm=llm, config=AgentConfig(stream=True))
    agent.on(lambda event, payload: tokens.append(payload["token"])
             if event == "token" else None)
    result = agent.run("hi")
    assert result.answer == "hello world"
    assert tokens == ["hello world"]


def test_sse_chunk_parsing():
    from picoagent.llm import OpenAICompatibleLLM

    acc: dict = {}
    chunks = [
        '{"choices":[{"delta":{"content":"He"}}]}',
        '{"choices":[{"delta":{"content":"llo"}}]}',
        '{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"calc","arguments":"{\\"a\\":"}}]}}]}',
        '{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}',
    ]
    emitted = []
    for c in chunks:
        emitted.extend(OpenAICompatibleLLM._handle_sse_chunk(c, acc))
    assert emitted == ["He", "llo"]
    assert acc[0]["id"] == "c1"
    assert acc[0]["name"] == "calc"
    assert acc[0]["arguments"] == '{"a":1}'
