# picoagent

[![CI](https://github.com/double114514/picoagent/actions/workflows/ci.yml/badge.svg)](https://github.com/double114514/picoagent/actions/workflows/ci.yml)
[![Docs](https://github.com/double114514/picoagent/actions/workflows/docs.yml/badge.svg)](https://double114514.github.io/picoagent/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

一个**工程级的极简 Python Agent 框架**：手写 ReAct 循环、工具调用、记忆管理、规划策略、结构化输出与事件钩子。**零重型依赖**——仅用 Python 标准库实现 LLM 通信（`urllib`），适合学习 Agent 原理，也可直接作为 AI 应用开发的项目底座。

📖 在线文档：<https://double114514.github.io/picoagent/>

## 特性

- **ReAct Agent 循环**：思考 → 工具调用 → 观察 → 再思考，直到给出最终答案或达到步数上限
- **同步 + 异步**：`agent.run()` 与 `await agent.arun()` 双通道，工具支持普通函数和 `async def`
- **流式输出**：SSE 流式解析（`chat_stream`），通过 `token` 事件逐 token 推送
- **结构化输出**：`output_schema` 约束最终答案为指定 JSON Schema，非法输出自动纠偏重试，结果在 `result.parsed`
- **声明式工具注册**：`@registry.register` 装饰器即可定义工具，自动从类型注解生成 JSON Schema
- **LLM 抽象层**：`BaseLLM` 接口，兼容任意 OpenAI 协议端点（OpenAI / DeepSeek / Qwen / vLLM / Ollama），内置 `MockLLM` 支持完全离线测试
- **记忆管理**：滑动窗口裁剪保留 system prompt；`SummarizingMemory` 超窗时用 LLM 自动总结旧消息；`SQLiteMemory` 跨会话持久化（`fresh=False` 续聊）
- **规划策略**：除 ReAct 外提供 `PlannerExecutor`（Plan-and-Execute：先规划 → 逐步执行 → 汇总合成）
- **内置工具包**（`picoagent/contrib/`）：calculator（AST 安全求值）、file_io（工作目录沙箱）、python_repl（子进程 + 超时）、web_search（DuckDuckGo）
- **工具访问控制**：`AccessPolicy` 支持白/黑名单、每分钟速率限制、人工确认钩子
- **可观测性**：`TraceRecorder` 将全部事件落为 JSONL trace，可回放分析
- **评测基准**：`picoagent.benchmarks` 定义任务集、统计成功率与 token 效率，内置离线自检
- **多 Agent 协作**：`Team` 把专家 Agent 注册为协调者的委派工具，子任务通过普通工具调用分发
- **Reflexion 策略**：起草 → 自我批评 → 修订循环，直到 APPROVED 或轮数上限
- **Prompt 模板**：`PromptTemplate`（严格变量检查）+ `FewShotPrompt`（few-shot 示例管理），零依赖
- **Token 用量聚合**：自动累计每次运行的 token 消耗
- **工程化**：pytest 测试套件（49 个用例）、Ruff lint、GitHub Actions CI（4 个 Python 版本矩阵 + 基准冒烟）、mkdocs-material 文档站（push 即部署 GitHub Pages）、MIT 协议

## 快速开始

```bash
pip install -e ".[dev]"
pytest -q                        # 运行测试（无需 API Key）
python examples/quickstart.py    # 离线 mock：ReAct 计算器
python examples/agent_demo.py    # 离线 mock：contrib 工具 + 异步 + trace
python -m picoagent.benchmarks   # 离线基准自检
```

使用真实大模型：

```bash
export OPENAI_API_KEY=sk-...
python examples/agent_demo.py --live
```

### 最小示例

```python
from picoagent import Agent, AgentConfig, OpenAICompatibleLLM, ToolRegistry

tools = ToolRegistry()

@tools.register
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression like '2 + 3 * 4'."""
    return str(eval(expression))

agent = Agent(
    llm=OpenAICompatibleLLM(model="gpt-4o-mini"),
    tools=tools,
    config=AgentConfig(max_steps=10, verbose=True),
)
result = agent.run("What is 12 * (3 + 4)?")
print(result.answer)   # 12 * (3 + 4) = 84.
print(result.usage)    # {'total_tokens': ...}
```

### 异步 + 追踪

```python
from picoagent import Agent, TraceRecorder, run_async
from picoagent.contrib import default_tools

agent = Agent(llm=..., tools=default_tools())
with TraceRecorder("run.jsonl") as tracer:
    agent.on(tracer)
    result = run_async(agent, "帮我算 2**10 并列出当前目录")
```

### 结构化输出

```python
schema = {"type": "object",
          "properties": {"total": {"type": "integer"}},
          "required": ["total"]}
agent = Agent(llm=..., config=AgentConfig(output_schema=schema))
result = agent.run("算一下 12 * (3 + 4)")
print(result.parsed)   # {'total': 84}
```

### 工具访问控制

```python
from picoagent import AccessPolicy, guarded_registry
from picoagent.contrib import default_tools

policy = AccessPolicy(default_tools(), deny={"python_repl"},
                      rate_limit_per_minute=10,
                      confirm=lambda name, args: input(f"允许 {name}? y/n ") == "y")
agent = Agent(llm=..., tools=guarded_registry(default_tools(), policy))
```

### 持久化记忆（跨会话续聊）

```python
from picoagent import Agent, AgentConfig, SQLiteMemory

memory = SQLiteMemory("chat.db", session_id="user-42")
agent = Agent(llm=..., memory=memory, config=AgentConfig(fresh=False))
agent.run("第一轮……")   # 进程重启后历史仍在
```

### 多 Agent 协作

```python
from picoagent import Team
from picoagent.contrib import calculator_tools, file_tools

team = Team(llm)
team.add_specialist("math", calculator_tools(), "负责数学计算")
team.add_specialist("files", file_tools("."), "负责读写文件")
result = team.run("算 3*7 并把结果写进 result.txt")
```

### Reflexion 自我反思

```python
from picoagent import ReflexionAgent

ra = ReflexionAgent(llm, max_rounds=2)
result = ra.run("解释 Python GIL 对多线程性能的影响")
print(result.answer, result.approved)
```

## 架构

```
picoagent/
├── types.py      # Message / ToolCall / Step / AgentResult 等核心数据类型
├── llm.py        # BaseLLM 抽象、OpenAICompatibleLLM(SSE)、MockLLM
├── tools.py      # Tool、ToolRegistry（装饰器注册 + Schema 自动生成）
├── memory.py     # Memory（滑动窗口）、SummarizingMemory（自动总结）
├── persistent.py # SQLiteMemory（跨会话持久化）
├── agent.py      # Agent.run()/arun() 主循环 + 事件钩子 + 用量聚合
├── planning.py   # PlannerExecutor（Plan-and-Execute 策略）
├── reflexion.py  # ReflexionAgent（自我反思修订策略）
├── multiagent.py # Team / agent_as_tool（多 Agent 委派协作）
├── prompts.py    # PromptTemplate / FewShotPrompt
├── structured.py # 结构化输出：schema 注入、JSON 提取、纠偏重试
├── access.py     # AccessPolicy（白/黑名单、限流、确认钩子）
├── trace.py      # TraceRecorder（JSONL 事件追踪）
├── benchmarks.py # 评测基准：任务定义、成功率/token 统计、离线自检
└── contrib/      # calculator / file_io / python_repl / web_search 工具包
```

文档站：`mkdocs.yml` + `docs/`，push 到 main 后由 GitHub Actions 自动部署到 GitHub Pages。

核心循环（`agent.py`）：每轮把记忆消息发给 LLM → 若无 tool_calls 则返回最终答案 → 否则执行所有工具、把观察写回记忆、进入下一轮，直到 `max_steps`。事件（`llm_call` / `tool_call` / `step` / `finish` / `token`）通过 `agent.on(hook)` 订阅。

## 路线图（交接给下一个开发者）

已完成 ✅：

1. ✅ 流式输出（SSE + `token` 钩子）
2. ✅ 异步支持（`arun()` + async 工具）
3. ✅ 内置工具包（calculator / file_io / python_repl / web_search，含安全沙箱）
4. ✅ 记忆增强（`SummarizingMemory` 自动总结）
5. ✅ 规划策略（`PlannerExecutor`）
6. ✅ 可观测性（`TraceRecorder` JSONL 追踪 + 回放）
7. ✅ 结构化输出（`output_schema` + 纠偏重试 + `result.parsed`）
8. ✅ 工具访问控制（`AccessPolicy`：白/黑名单、限流、确认钩子）
9. ✅ 持久化记忆（`SQLiteMemory` 跨会话）
10. ✅ 评测基准（`picoagent.benchmarks`，含离线自检）
11. ✅ 多 Agent 协作（`Team` + `agent_as_tool` 委派）
12. ✅ Reflexion 自我反思策略
13. ✅ Prompt 模板系统（`PromptTemplate` / `FewShotPrompt`）
14. ✅ 文档站（mkdocs-material + GitHub Pages 自动部署）

待办（按优先级）：

1. **真实基准任务集**：接入 HotpotQA / GAIA 子集，评估真实模型成功率与 token 效率
2. **OpenTelemetry**：导出 OTEL trace/spans 对接 Jaeger/DataDog
3. **Tree-of-Thoughts**：多分支搜索策略（BFS/DFS + 评估器）
4. **流式工具结果**：长任务工具支持进度事件
5. **Anthropic / Gemini 原生协议适配器**（当前走 OpenAI 兼容层）

## 设计决策记录

- **不依赖 openai SDK**：用 `urllib` 直连 OpenAI 兼容端点，保持零依赖、方便读者看清协议细节
- **工具错误不中断循环**：异常转为 observation 文本喂回模型，让模型自我修正
- **MockLLM 脚本化**：测试队列式返回预设响应，使全部测试可离线、确定性运行
- **calculator 用 AST 而非 eval**：白名单 AST 求值，拒绝导入/调用/属性访问，避免代码注入
- **python_repl 用子进程 + `-I` 隔离**：`-I` 忽略环境与 site-packages，配合超时保证安全
- **SummarizingMemory 复用单条 summary**：避免 system 消息无限增长，稳定内存占用
- **避免 `or` 默认值陷阱**：`ToolRegistry`/`Memory` 定义了 `__len__`，空实例布尔值为 False，构造器一律用显式 `is not None` 判断
- **结构化输出用 prompt + 纠偏重试**：不依赖 provider 的 json mode，任何 OpenAI 兼容端点都能用

## License

MIT
