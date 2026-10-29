"""picoagent: a minimal, engineering-grade ReAct agent framework."""
from .access import AccessPolicy, guarded_registry
from .agent import Agent, AgentConfig, run_async
from .llm import BaseLLM, LLMResponse, MockLLM, OpenAICompatibleLLM
from .memory import Memory, SummarizingMemory
from .multiagent import Team, agent_as_tool
from .persistent import SQLiteMemory
from .planning import PlannerExecutor, PlanResult, parse_plan
from .prompts import FewShotPrompt, PromptTemplate
from .reflexion import ReflexionAgent, ReflexionResult
from .structured import parse_structured, to_json_schema
from .tools import Tool, ToolError, ToolRegistry
from .trace import TraceRecorder, read_trace
from .types import AgentResult, Message, Role, Step, ToolCall

__version__ = "1.0.0"

__all__ = [
    "Agent", "AgentConfig", "run_async", "AccessPolicy", "guarded_registry",
    "BaseLLM", "LLMResponse", "MockLLM", "OpenAICompatibleLLM", "Memory",
    "SummarizingMemory", "SQLiteMemory", "Team", "agent_as_tool",
    "PlannerExecutor", "PlanResult", "parse_plan", "ReflexionAgent",
    "ReflexionResult", "PromptTemplate", "FewShotPrompt", "parse_structured",
    "to_json_schema", "Tool", "ToolError", "ToolRegistry", "TraceRecorder",
    "read_trace", "AgentResult", "Message", "Role", "Step", "ToolCall",
    "__version__",
]

