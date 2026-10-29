# picoagent 文档

工程级的极简 Python Agent 框架：手写 ReAct 循环、工具调用、记忆管理、规划策略、结构化输出与多 Agent 协作，零重型依赖。

- [快速开始](quickstart.md)
- [核心概念](concepts.md)
- [API 参考](api.md)

## 安装

```bash
pip install -e ".[dev]"
```

## 30 秒上手

```python
from picoagent import Agent, OpenAICompatibleLLM
from picoagent.contrib import calculator_tools

agent = Agent(llm=OpenAICompatibleLLM(model="gpt-4o-mini"),
              tools=calculator_tools())
print(agent.run("12 * (3 + 4) = ?").answer)
```
