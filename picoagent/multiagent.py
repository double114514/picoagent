"""Multi-agent collaboration: expose an Agent as a tool for another Agent.

The parent agent can delegate subtasks to specialist sub-agents via normal
tool calls. Each sub-agent runs its own ReAct loop with its own tools.
"""
from __future__ import annotations

from .agent import Agent, AgentConfig
from .llm import BaseLLM
from .tools import ToolRegistry


def agent_as_tool(agent: Agent, name: str, description: str,
                  registry: ToolRegistry) -> None:
    """Register `agent` as a tool named `name` on `registry`.

    The parent agent calls it like any tool: {"task": "..."}.
    """
    def delegate(task: str) -> str:
        return agent.run(task).answer

    delegate.__name__ = name
    registry.register(delegate, name=name, description=description)


class Team:
    """A coordinator agent plus named specialist sub-agents.

        team = Team(llm)
        team.add_specialist("math", math_tools, "Solves arithmetic")
        result = team.run("算 3*7 然后写进文件")
    """

    def __init__(self, llm: BaseLLM,
                 coordinator_tools: ToolRegistry | None = None,
                 config: AgentConfig | None = None):
        self.llm = llm
        self.config = config if config is not None else AgentConfig()
        # ToolRegistry defines __len__; explicit None check, not `or`.
        self.coordinator_tools = coordinator_tools \
            if coordinator_tools is not None else ToolRegistry()
        self.specialists: dict[str, Agent] = {}

    def add_specialist(self, name: str, tools: ToolRegistry,
                       description: str,
                       config: AgentConfig | None = None) -> Agent:
        specialist = Agent(
            llm=self.llm, tools=tools,
            config=config or AgentConfig(
                max_steps=self.config.max_steps,
                system_prompt=(
                    f"You are the '{name}' specialist. {description} "
                    "Complete the delegated subtask and report the result "
                    "concisely.")))
        self.specialists[name] = specialist
        agent_as_tool(specialist, name=f"delegate_{name}",
                      description=(f"Delegate a subtask to the {name} "
                                   f"specialist. {description}"),
                      registry=self.coordinator_tools)
        return specialist

    def coordinator(self) -> Agent:
        return Agent(llm=self.llm, tools=self.coordinator_tools,
                     config=self.config)

    def run(self, task: str):
        return self.coordinator().run(task)
