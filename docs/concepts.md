# 核心概念

## ReAct 循环

`Agent.run()` 每轮把记忆消息发给 LLM：

1. LLM 返回纯文本 → 视为最终答案，结束
2. LLM 返回 tool_calls → 执行全部工具，观察写回记忆，进入下一轮
3. 达到 `max_steps` → 返回截断提示

## 事件钩子

`agent.on(hook)` 订阅事件：`llm_call` / `tool_call` / `step` / `finish` / `token`。hook 签名为 `(event: str, payload: dict)`。

## 记忆

| 类 | 行为 |
|---|---|
| `Memory` | 滑动窗口，超窗丢弃最旧非 system 消息 |
| `SummarizingMemory` | 超窗时调用 LLM 把被丢弃消息总结为一条 system 消息 |
| `SQLiteMemory` | 持久化到 SQLite，`AgentConfig(fresh=False)` 跨会话续聊 |

## 规划策略

- **ReAct**：默认，边想边做
- **PlannerExecutor**：先一次性出计划，逐步用子 Agent 执行，最后汇总
- **ReflexionAgent**：起草 → 自我批评 → 修订，直到 APPROVED 或达到轮数上限

## 多 Agent 协作

`Team` 把专家 Agent 注册为协调者的 `delegate_<name>` 工具，协调者通过普通工具调用委派子任务。

## 安全

- calculator：AST 白名单求值，拒绝任意代码
- file_io：路径解析限制在工作目录内
- python_repl：子进程 + `-I` 隔离 + 超时
- AccessPolicy：白/黑名单、限流、人工确认
