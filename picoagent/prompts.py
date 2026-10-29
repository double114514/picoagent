"""Prompt templates: zero-dependency {placeholder} rendering + few-shot
example management."""
from __future__ import annotations

import string
from dataclasses import dataclass, field
from typing import Any


class PromptTemplate:
    """A str.format-style template with strict variable checking.

        t = PromptTemplate("把{text}翻译成{lang}")
        t.render(text="hello", lang="法语")
    """

    def __init__(self, template: str):
        self.template = template
        self.variables = [
            fname for _, fname, _, _ in string.Formatter().parse(template)
            if fname
        ]

    def render(self, **kwargs: Any) -> str:
        missing = [v for v in self.variables if v not in kwargs]
        if missing:
            raise KeyError(f"Missing template variables: {missing}")
        return self.template.format(**kwargs)

    def __repr__(self) -> str:  # pragma: no cover
        return f"PromptTemplate(vars={self.variables})"


@dataclass
class Example:
    """One few-shot example: input/output pair (extensible)."""
    input: str
    output: str
    extra: dict[str, Any] = field(default_factory=dict)


class FewShotPrompt:
    """Instruction + N examples + query, rendered as one prompt string.

        fs = FewShotPrompt(
            instruction="判断情感，输出 positive/negative",
            input_label="文本", output_label="情感")
        fs.add("太棒了", "positive").add("糟透了", "negative")
        prompt = fs.render("还行吧")
    """

    def __init__(self, instruction: str,
                 input_label: str = "Input",
                 output_label: str = "Output"):
        self.instruction = instruction
        self.input_label = input_label
        self.output_label = output_label
        self.examples: list[Example] = []

    def add(self, input_text: str, output_text: str,
            **extra: Any) -> FewShotPrompt:
        self.examples.append(Example(input=input_text, output=output_text,
                                     extra=extra))
        return self

    def render(self, query: str) -> str:
        parts = [self.instruction, ""]
        for ex in self.examples:
            parts.append(f"{self.input_label}: {ex.input}")
            parts.append(f"{self.output_label}: {ex.output}")
            parts.append("")
        parts.append(f"{self.input_label}: {query}")
        parts.append(f"{self.output_label}:")
        return "\n".join(parts)
