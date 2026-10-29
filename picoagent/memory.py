"""Conversation memory with sliding-window and summarization hooks."""
from __future__ import annotations

from .types import Message, Role


class Memory:
    """Stores messages; trims oldest non-system turns when over limit."""

    def __init__(self, max_messages: int = 50):
        self.max_messages = max_messages
        self._messages: list[Message] = []

    def add(self, message: Message) -> None:
        self._messages.append(message)
        self._trim()

    def add_user(self, content: str) -> None:
        self.add(Message(role=Role.USER, content=content))

    def add_assistant(self, content: str, tool_calls=None) -> None:
        self.add(Message(role=Role.ASSISTANT, content=content,
                         tool_calls=tool_calls))

    def add_tool_result(self, tool_call_id: str, name: str, content: str) -> None:
        self.add(Message(role=Role.TOOL, content=content,
                         name=name, tool_call_id=tool_call_id))

    def _trim(self) -> None:
        if len(self._messages) <= self.max_messages:
            return
        system = [m for m in self._messages if m.role == Role.SYSTEM]
        rest = [m for m in self._messages if m.role != Role.SYSTEM]
        keep = self.max_messages - len(system)
        self._messages = system + rest[-keep:]

    def messages(self) -> list[Message]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)


SUMMARY_PROMPT = (
    "Summarize the following conversation fragment in 3-5 bullet points, "
    "keeping facts, decisions, and tool results that matter for continuing "
    "the task. Be compact."
)


class SummarizingMemory(Memory):
    """When trimming, summarizes dropped messages via the LLM instead of
    silently discarding them. The summary is kept as a system message."""

    def __init__(self, llm, max_messages: int = 20):
        super().__init__(max_messages=max_messages)
        self._llm = llm
        self._summary = ""

    def _trim(self) -> None:
        if len(self._messages) <= self.max_messages:
            return
        marker = "[Summary of earlier conversation]"
        base_system = [m for m in self._messages
                       if m.role == Role.SYSTEM
                       and not (m.content or "").startswith(marker)]
        rest = [m for m in self._messages if m.role != Role.SYSTEM]
        # reserve one slot for the (single, reusable) summary message
        keep = self.max_messages - len(base_system) - 1
        if keep <= 0:
            dropped, kept = rest, []
        else:
            dropped, kept = rest[:-keep], rest[-keep:]

        summary_msg = None
        if dropped:
            fragment = "\n".join(
                f"{m.role.value}: {m.content[:500]}" for m in dropped if m.content
            )
            prior = f"Previous summary:\n{self._summary}\n\n" if self._summary else ""
            resp = self._llm.chat([
                Message(role=Role.SYSTEM, content=SUMMARY_PROMPT),
                Message(role=Role.USER, content=prior + "New fragment:\n" + fragment),
            ])
            self._summary = resp.content
            summary_msg = Message(role=Role.SYSTEM,
                                  content=f"{marker}\n{self._summary}")
        self._messages = base_system + ([summary_msg] if summary_msg else []) + kept

    @property
    def summary(self) -> str:
        return self._summary
