"""Reflexion strategy: draft answer -> self-critique -> revise, N rounds.

Reference: Shinn et al., "Reflexion: Language Agents with Verbal
Reinforcement Learning" (2023). This is the text-only variant: the model
critiques its own draft and improves it, optionally using tools each round.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .agent import Agent, AgentConfig
from .llm import BaseLLM
from .tools import ToolRegistry
from .types import Message, Role

CRITIQUE_PROMPT = (
    "You are a strict reviewer. Critique the draft answer below against the "
    "original task. Point out factual errors, missing parts, and unclear "
    "reasoning. If the draft is fully correct and complete, reply with "
    "exactly: APPROVED"
)

REVISE_PROMPT = (
    "You are a reviser. Given the task, the previous draft, and the critique, "
    "write an improved final answer. Address every point in the critique."
)


@dataclass
class ReflexionResult:
    task: str
    drafts: list[str] = field(default_factory=list)
    critiques: list[str] = field(default_factory=list)
    answer: str = ""
    rounds: int = 0
    approved: bool = False


class ReflexionAgent:
    """Runs a ReAct agent for the draft, then critique/revise rounds.

        ra = ReflexionAgent(llm, tools, max_rounds=2)
        result = ra.run("解释 Python GIL")
    """

    def __init__(self, llm: BaseLLM, tools: ToolRegistry | None = None,
                 config: AgentConfig | None = None, max_rounds: int = 2):
        self.llm = llm
        self.tools = tools if tools is not None else ToolRegistry()
        self.config = config if config is not None else AgentConfig()
        self.max_rounds = max_rounds

    def _critique(self, task: str, draft: str) -> str:
        return self.llm.chat([
            Message(role=Role.SYSTEM, content=CRITIQUE_PROMPT),
            Message(role=Role.USER,
                    content=f"Task: {task}\n\nDraft answer:\n{draft}"),
        ]).content

    def _revise(self, task: str, draft: str, critique: str) -> str:
        return self.llm.chat([
            Message(role=Role.SYSTEM, content=REVISE_PROMPT),
            Message(role=Role.USER,
                    content=f"Task: {task}\n\nPrevious draft:\n{draft}\n\n"
                            f"Critique:\n{critique}"),
        ]).content

    def run(self, task: str, verbose: bool = False) -> ReflexionResult:
        result = ReflexionResult(task=task)

        # Round 0: full ReAct loop with tools for the initial draft
        agent = Agent(llm=self.llm, tools=self.tools,
                      config=AgentConfig(max_steps=self.config.max_steps,
                                         system_prompt=self.config.system_prompt))
        draft = agent.run(task).answer
        result.drafts.append(draft)

        for r in range(self.max_rounds):
            critique = self._critique(task, draft)
            result.critiques.append(critique)
            if verbose:
                print(f"[critique {r + 1}] {critique}")
            if critique.strip() == "APPROVED":
                result.approved = True
                break
            draft = self._revise(task, draft, critique)
            result.drafts.append(draft)
            if verbose:
                print(f"[revise {r + 1}] {draft}")

        result.answer = result.drafts[-1]
        result.rounds = len(result.critiques)
        return result
