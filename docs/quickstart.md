# 快速开始

## 环境准备

```bash
pip install -e ".[dev]"
export OPENAI_API_KEY=sk-...   # 或任何 OpenAI 兼容端点的 key
```

## 第一个 Agent

```python
from picoagent import Agent, AgentConfig, OpenAICompatibleLLM, ToolRegistry

tools = ToolRegistry()

@tools.register
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression like '2 + 3 * 4'."""
    from picoagent.contrib.calculator import safe_eval
    return str(safe_eval(expression))

agent = Agent(llm=OpenAICompatibleLLM(model="gpt-4o-mini"), tools=tools,
              config=AgentConfig(verbose=True))
print(agent.run("What is 12 * (3 + 4)?").answer)
```

## 异步

```python
from picoagent import run_async
result = run_async(agent, "再算一次 2**10")
```

## 离线开发

没有 API Key 也能开发调试——用 `MockLLM` 脚本化模型响应：

```python
from picoagent import MockLLM, LLMResponse, ToolCall

llm = MockLLM([
    LLMResponse("", [ToolCall(name="calculator",
                              arguments={"expression": "1+1"})]),
    LLMResponse("结果是 2"),
])
```

更多示例见 `examples/` 目录。
