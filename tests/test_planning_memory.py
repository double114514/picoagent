"""Tests for PlannerExecutor and SummarizingMemory."""
import pytest

from picoagent import (LLMResponse, MockLLM, PlannerExecutor, Role,
                       SummarizingMemory, parse_plan)


def test_parse_plan_plain():
    assert parse_plan('["a", "b"]') == ["a", "b"]


def test_parse_plan_with_fences():
    text = 'Here is the plan:\n```json\n["step1", "step2"]\n```'
    assert parse_plan(text) == ["step1", "step2"]


def test_parse_plan_invalid():
    with pytest.raises(ValueError):
        parse_plan("no json here")


def test_planner_executor_end_to_end():
    llm = MockLLM([
        LLMResponse(content='["research topic", "summarize findings"]'),
        LLMResponse(content="research done"),
        LLMResponse(content="summary done"),
        LLMResponse(content="Final synthesized answer."),
    ])
    pe = PlannerExecutor(llm=llm)
    result = pe.run("write a report")
    assert result.plan == ["research topic", "summarize findings"]
    assert result.step_answers == ["research done", "summary done"]
    assert result.answer == "Final synthesized answer."


def test_planner_retries_on_bad_output():
    llm = MockLLM([
        LLMResponse(content="not json"),
        LLMResponse(content='["only step"]'),
        LLMResponse(content="ok"),
        LLMResponse(content="final"),
    ])
    pe = PlannerExecutor(llm=llm)
    result = pe.run("task")
    assert result.plan == ["only step"]


def test_summarizing_memory_keeps_summary():
    llm = MockLLM([LLMResponse(content=f"- summary round {i}") for i in range(10)])
    m = SummarizingMemory(llm=llm, max_messages=3)
    from picoagent import Message
    m.add(Message(role=Role.SYSTEM, content="sys"))
    for i in range(6):
        m.add_user(f"msg{i}")
    msgs = m.messages()
    assert len(msgs) == 3
    assert m.summary.startswith("- summary round")
    assert any("Summary of earlier conversation" in (x.content or "")
               for x in msgs)
