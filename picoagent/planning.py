"""Plan-and-Execute strategy: plan once, execute each step with a ReAct agent."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .agent import Agent, AgentConfig
from .llm import BaseLLM
from .tools import ToolRegistry
from .types import Message, Role

PLAN_PROMPT = (
    "You are a planner. Break the user's task into concrete ordered steps. "
    "Reply with ONLY a JSON array of step strings, e.g. "
    '["Step 1 ...", "Step 2 ..."]. No prose, no markdown fences.'
)

SYNTH_PROMPT = (
    "You are a synthesizer. Given the task and the results of each executed "
    "step, write the final answer for the user. Be concise and complete."
)


@dataclass
class PlanResult:
    task: str
    plan: list[str]
    step_answers: list[str] = field(default_factory=list)
    answer: str = ""


def parse_plan(text: str) -> list[str]:
    """Extract a JSON list of steps from LLM output, tolerating fences/prose."""
    cleaned = text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("["):
                cleaned = part
                break
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"Planner output is not a JSON array: {text[:200]}")
    steps = json.loads(cleaned[start:end + 1])
    if not isinstance(steps, list) or not all(isinstance(s, str) for s in steps):
        raise ValueError("Plan must be a JSON array of strings")
    return steps


class PlannerExecutor:
    """Plan-and-Execute: LLM drafts a plan, a ReAct agent executes each step,
    then the LLM synthesizes the final answer."""

    def __init__(self, llm: BaseLLM, tools: ToolRegistry | None = None,
                 config: AgentConfig | None = None,
                 max_plan_retries: int = 2):
        self.llm = llm
        # ToolRegistry defines __len__; avoid `or` which drops empty registries.
        self.tools = tools if tools is not None else ToolRegistry()
        self.config = config if config is not None else AgentConfig()
        self.max_plan_retries = max_plan_retries

    def _plan(self, task: str) -> list[str]:
        messages = [Message(role=Role.SYSTEM, content=PLAN_PROMPT),
                    Message(role=Role.USER, content=task)]
        last_error: Exception | None = None
        for _ in range(self.max_plan_retries + 1):
            resp = self.llm.chat(messages)
            try:
                return parse_plan(resp.content)
            except ValueError as exc:
                last_error = exc
                messages.append(Message(role=Role.ASSISTANT, content=resp.content))
                messages.append(Message(
                    role=Role.USER,
                    content="Invalid format. Reply with ONLY a JSON array of strings."))
        raise ValueError(f"Planner failed to produce a valid plan: {last_error}")

    def run(self, task: str, verbose: bool = False) -> PlanResult:
        plan = self._plan(task)
        if verbose:
            print(f"[plan] {plan}")

        result = PlanResult(task=task, plan=plan)
        context: list[str] = []
        for i, step in enumerate(plan):
            step_task = (
                f"Overall task: {task}\n"
                f"Current step ({i + 1}/{len(plan)}): {step}\n"
                + (f"Results so far: {context}\n" if context else "")
                + "Complete this step and report the result concisely."
            )
            agent = Agent(llm=self.llm, tools=self.tools,
                          config=AgentConfig(
                              max_steps=self.config.max_steps,
                              system_prompt=self.config.system_prompt))
            step_result = agent.run(step_task)
            result.step_answers.append(step_result.answer)
            context.append(f"Step {i + 1}: {step_result.answer}")
            if verbose:
                print(f"[step {i + 1}] {step_result.answer}")

        synth_messages = [
            Message(role=Role.SYSTEM, content=SYNTH_PROMPT),
            Message(role=Role.USER,
                    content=f"Task: {task}\n\nStep results:\n"
                            + "\n".join(context)),
        ]
        result.answer = self.llm.chat(synth_messages).content
        return result
