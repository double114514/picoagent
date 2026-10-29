"""Tests for structured output, access control, persistent memory, benchmarks."""
import pytest

from picoagent import (
    AccessPolicy,
    Agent,
    AgentConfig,
    LLMResponse,
    MockLLM,
    SQLiteMemory,
    ToolError,
    guarded_registry,
    parse_structured,
    to_json_schema,
)
from picoagent.benchmarks import offline_demo
from picoagent.contrib.calculator import calculator_tools

# ---------- structured output ----------

def test_parse_structured_plain():
    assert parse_structured('{"a": 1}') == {"a": 1}


def test_parse_structured_with_prose():
    assert parse_structured('Here you go: {"x": [1, 2]} done') == {"x": [1, 2]}


def test_parse_structured_invalid():
    with pytest.raises(ValueError):
        parse_structured("no json at all")


def test_agent_structured_output():
    schema = {"type": "object", "properties": {"total": {"type": "integer"}}}
    llm = MockLLM([LLMResponse(content='{"total": 84}')])
    agent = Agent(llm=llm, config=AgentConfig(output_schema=schema))
    result = agent.run("compute")
    assert result.parsed == {"total": 84}


def test_agent_structured_retry_then_success():
    schema = {"type": "object"}
    llm = MockLLM([
        LLMResponse(content="oops not json"),
        LLMResponse(content='{"ok": true}'),
    ])
    agent = Agent(llm=llm, config=AgentConfig(output_schema=schema))
    result = agent.run("compute")
    assert result.parsed == {"ok": True}
    assert len(llm.calls) == 2  # retried once


def test_to_json_schema_dict_passthrough():
    s = {"type": "object"}
    assert to_json_schema(s) is s


# ---------- access control ----------

def test_access_deny():
    reg = calculator_tools()
    guarded = guarded_registry(reg, AccessPolicy(reg, deny={"calculator"}))
    with pytest.raises(ToolError, match="denied"):
        guarded.get("calculator").run(expression="1+1")


def test_access_allow_list():
    reg = calculator_tools()
    guarded = guarded_registry(reg, AccessPolicy(reg, allow={"other"}))
    with pytest.raises(ToolError, match="allow-list"):
        guarded.get("calculator").run(expression="1+1")


def test_access_rate_limit():
    reg = calculator_tools()
    guarded = guarded_registry(reg, AccessPolicy(reg, rate_limit_per_minute=2))
    tool = guarded.get("calculator")
    tool.run(expression="1+1")
    tool.run(expression="2+2")
    with pytest.raises(ToolError, match="rate limit"):
        tool.run(expression="3+3")


def test_access_confirm_reject():
    reg = calculator_tools()
    guarded = guarded_registry(
        reg, AccessPolicy(reg, confirm=lambda name, args: False))
    with pytest.raises(ToolError, match="rejected"):
        guarded.get("calculator").run(expression="1+1")


# ---------- persistent memory ----------

def test_sqlite_memory_persists(tmp_path):
    db = tmp_path / "chat.db"
    m = SQLiteMemory(db, session_id="s1")
    m.add_user("hello")
    m.close()

    m2 = SQLiteMemory(db, session_id="s1")
    assert len(m2) == 1
    assert m2.messages()[0].content == "hello"


def test_sqlite_sessions_isolated(tmp_path):
    db = tmp_path / "chat.db"
    SQLiteMemory(db, session_id="a").add_user("aaa")
    SQLiteMemory(db, session_id="b").add_user("bbb")
    assert set(SQLiteMemory.list_sessions(db)) == {"a", "b"}
    ma = SQLiteMemory(db, session_id="a")
    assert len(ma) == 1 and ma.messages()[0].content == "aaa"


def test_agent_resume_with_persistent_memory(tmp_path):
    db = tmp_path / "chat.db"
    mem = SQLiteMemory(db, session_id="s1")
    llm = MockLLM([LLMResponse(content="first answer"),
                   LLMResponse(content="second answer")])
    agent = Agent(llm=llm, memory=mem, config=AgentConfig(fresh=False))
    agent.run("turn 1")
    agent.run("turn 2")
    contents = [m.content for m in mem.messages()]
    assert contents == [contents[0], "turn 1", "first answer",
                        "turn 2", "second answer"]
    mem.close()


# ---------- benchmarks ----------

def test_offline_demo_benchmark():
    summary = offline_demo()
    assert summary["total"] == 2
    assert summary["passed"] == 2
    assert summary["success_rate"] == 1.0
