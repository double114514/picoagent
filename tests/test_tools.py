"""Tool registry and schema tests."""
import pytest

from picoagent import ToolError, ToolRegistry


def test_register_and_schema():
    reg = ToolRegistry()

    @reg.register
    def greet(name: str, loud: bool = False) -> str:
        """Greet someone."""
        return f"Hello, {name}!"

    schema = reg.get("greet").schema
    assert schema["name"] == "greet"
    assert schema["description"] == "Greet someone."
    assert schema["parameters"]["properties"]["name"]["type"] == "string"
    assert schema["parameters"]["required"] == ["name"]


def test_run_and_type_coercion():
    reg = ToolRegistry()

    @reg.register
    def mul(a: int, b: int) -> int:
        """Multiply."""
        return a * b

    assert reg.get("mul").run(a=3, b=4) == "12"  # non-str -> repr


def test_unknown_tool_raises():
    reg = ToolRegistry()
    with pytest.raises(ToolError):
        reg.get("missing")


def test_tool_exception_wrapped():
    reg = ToolRegistry()

    @reg.register
    def boom() -> str:
        """Always fails."""
        raise ValueError("bad")

    with pytest.raises(ToolError, match="boom failed"):
        reg.get("boom").run()


def test_missing_description_raises():
    reg = ToolRegistry()
    with pytest.raises(ToolError):
        @reg.register
        def nodoc(x: int) -> int:
            return x
