# API 参考

## `Agent`

```python
Agent(llm, tools=None, config=None, memory=None)
agent.run(task) -> AgentResult        # 同步
await agent.arun(task) -> AgentResult # 异步
agent.on(hook)                        # 订阅事件
```

## `AgentConfig`

| 字段 | 默认 | 说明 |
|---|---|---|
| `max_steps` | 10 | ReAct 最大轮数 |
| `system_prompt` | 内置 | 系统提示词 |
| `max_memory_messages` | 50 | 记忆窗口 |
| `verbose` | False | 打印事件到 stdout |
| `stream` | False | SSE 流式 + `token` 事件 |
| `output_schema` | None | 结构化输出 JSON Schema |
| `max_structured_retries` | 2 | 结构解析失败重试次数 |
| `fresh` | True | False 时保留已有记忆（持久化场景） |

## `AgentResult`

`answer` / `steps` / `messages` / `usage` / `parsed`

## LLM

- `OpenAICompatibleLLM(model, api_key=None, base_url=..., temperature=0.0)`
- `MockLLM(responses)`：离线脚本化测试

## 工具

- `ToolRegistry`：`@reg.register` / `reg.get(name)` / `reg.schemas()` / `reg.merge(other)`
- `picoagent.contrib`：`calculator_tools()` / `file_tools(workspace)` / `repl_tools()` / `search_tools()` / `default_tools()`
- `AccessPolicy(registry, allow=None, deny=None, rate_limit_per_minute=None, confirm=None)` + `guarded_registry(reg, policy)`

## 策略

- `PlannerExecutor(llm, tools, config).run(task) -> PlanResult`
- `ReflexionAgent(llm, tools, config, max_rounds=2).run(task) -> ReflexionResult`
- `Team(llm).add_specialist(name, tools, description).run(task)`

## 其他

- `TraceRecorder(path)`：JSONL 事件追踪，`read_trace(path)` 回放
- `PromptTemplate(template).render(**vars)` / `FewShotPrompt(instruction).add(x, y).render(query)`
- `picoagent.benchmarks`：`BenchTask` / `run_benchmark` / `summarize`
