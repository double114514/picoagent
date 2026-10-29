"""Tests for contrib tools and TraceRecorder."""
import pytest

from picoagent import ToolError, TraceRecorder, read_trace
from picoagent.contrib import calculator_tools, file_tools, repl_tools
from picoagent.contrib.calculator import safe_eval


def test_calculator_basic():
    calc = calculator_tools().get("calculator")
    assert calc.run(expression="2 + 3 * 4") == "14"
    assert calc.run(expression="2 ** 10") == "1024"
    assert calc.run(expression="7 / 2") == "3.5"


def test_calculator_rejects_unsafe():
    with pytest.raises(ToolError):
        safe_eval("__import__('os').system('ls')")
    with pytest.raises(ToolError):
        safe_eval("open('x')")


def test_file_tools_sandbox(tmp_path):
    reg = file_tools(str(tmp_path))
    reg.get("write_file").run(path="a/b.txt", content="hello")
    assert reg.get("read_file").run(path="a/b.txt") == "hello"
    assert "b.txt" in reg.get("list_dir").run(path="a")
    with pytest.raises(ToolError):
        reg.get("read_file").run(path="../escape.txt")


def test_python_repl():
    repl = repl_tools().get("python_repl")
    assert repl.run(code="print(1 + 1)") == "2\n"
    assert "ZeroDivisionError" in repl.run(code="1/0")


def test_trace_recorder(tmp_path):
    from picoagent import Agent, LLMResponse, MockLLM
    trace_file = tmp_path / "run.jsonl"
    with TraceRecorder(trace_file) as tracer:
        agent = Agent(llm=MockLLM([LLMResponse(content="hi")]))
        agent.on(tracer)
        agent.run("hello")
    events = read_trace(trace_file)
    names = [e["event"] for e in events]
    assert names == ["llm_call", "finish"]
    assert events[-1]["payload"]["answer"] == "hi"
