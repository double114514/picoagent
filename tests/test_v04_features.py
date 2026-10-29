"""Tests for multi-agent, Reflexion, and prompt templates."""
import pytest

from picoagent import (AgentConfig, FewShotPrompt, LLMResponse, MockLLM,
                       PromptTemplate, ReflexionAgent, Team, ToolCall,
                       ToolRegistry)
from picoagent.contrib.calculator import calculator_tools


# ---------- multi-agent ----------

def test_team_delegation():
    """Coordinator delegates to a math specialist via delegate_math tool."""
    llm = MockLLM([
        # coordinator first: calls delegate_math
        LLMResponse("", [ToolCall(name="delegate_math",
                                  arguments={"task": "compute 3*7"})]),
        # specialist agent: calculator tool call, then its answer
        LLMResponse("", [ToolCall(name="calculator",
                                  arguments={"expression": "3 * 7"})]),
        LLMResponse("3 * 7 = 21"),
        # coordinator final answer
        LLMResponse("The answer is 21."),
    ])
    team = Team(llm)
    team.add_specialist("math", calculator_tools(), "Solves arithmetic")
    result = team.run("What is 3 times 7?")
    assert result.answer == "The answer is 21."
    assert "delegate_math" in team.coordinator_tools
    assert result.steps[0].observation == "3 * 7 = 21"


# ---------- reflexion ----------

def test_reflexion_approved_first_round():
    llm = MockLLM([
        LLMResponse(content="draft answer"),   # ReAct draft
        LLMResponse(content="APPROVED"),       # critique
    ])
    ra = ReflexionAgent(llm=llm)
    result = ra.run("task")
    assert result.approved is True
    assert result.answer == "draft answer"
    assert result.rounds == 1


def test_reflexion_revises_until_approved():
    llm = MockLLM([
        LLMResponse(content="draft v1"),
        LLMResponse(content="Missing detail X."),
        LLMResponse(content="draft v2 with X"),
        LLMResponse(content="APPROVED"),
    ])
    ra = ReflexionAgent(llm=llm, max_rounds=3)
    result = ra.run("task")
    assert result.approved is True
    assert result.answer == "draft v2 with X"
    assert len(result.drafts) == 2


def test_reflexion_max_rounds_without_approval():
    llm = MockLLM([
        LLMResponse(content="draft v1"),
        LLMResponse(content="still bad"),
        LLMResponse(content="draft v2"),
        LLMResponse(content="still bad"),
        LLMResponse(content="draft v3"),
    ])
    ra = ReflexionAgent(llm=llm, max_rounds=2)
    result = ra.run("task")
    assert result.approved is False
    assert result.answer == "draft v3"
    assert result.rounds == 2


# ---------- prompt templates ----------

def test_prompt_template_render():
    t = PromptTemplate("把{text}翻译成{lang}")
    assert t.render(text="hello", lang="法语") == "把hello翻译成法语"
    assert set(t.variables) == {"text", "lang"}


def test_prompt_template_missing_var():
    t = PromptTemplate("hi {name}")
    with pytest.raises(KeyError, match="name"):
        t.render()


def test_few_shot_prompt():
    fs = FewShotPrompt("判断情感", input_label="文本", output_label="情感")
    fs.add("太棒了", "positive").add("糟透了", "negative")
    prompt = fs.render("还行吧")
    assert "文本: 太棒了" in prompt
    assert "情感: negative" in prompt
    assert prompt.endswith("情感:")
    assert prompt.count("文本:") == 3
