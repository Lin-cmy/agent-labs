# Agent Labs —— 从零手写 Agent 系列实验

不依赖 Agent 框架，亲手实现 AI Agent 的每一层机制：循环、工具、协议、记忆、容错。

**核心理念**：先用最原始的方式把 Agent 手写一遍，再用框架/协议做对比——每一层"框架替你做了什么"都亲眼见过。

## 文件清单

| 文件 | 是什么 | 对应概念 |
|---|---|---|
| `agent_loop.py` | 手写最小 Agent Loop（2 工具 + 5 步上限 + JSONL trace） | ReAct 循环：observe → think → act → observe |
| `tool_calling.py` | 三工具多功能 Agent（天气/计算/文件搜索） | 工具 schema、并行调用、跨工具切换、不可信参数安全 |
| `mcp_server.py` | MCP 工具服务器（FastMCP，stdio 传输） | MCP server 侧：标准工具 schema 暴露 |
| `mcp_client.py` | MCP 客户端（跨进程调用工具） | MCP client 侧：传输层 / 会话层分层 |
| `agent_mcp.py` | Agent 通过 MCP 协议调用跨进程工具 | 适配器接入、工具来源可插拔 |
| `agent_memory.py` | 多轮对话记忆 + tenacity 指数退避重试 | 记忆管理、容错策略 |
| `trace.jsonl` / `tool_calling_trace.jsonl` | 各练习的决策轨迹日志 | 可观测性（每步的工具选择、参数、观察、成本） |

## 运行前提

- Python 3.12（conda 环境 `cook-qa`，依赖 `langchain-openai`、`langchain-core<1.0`、`mcp<2`、`langchain-mcp-adapters<0.2`、`tenacity`）
- 环境变量 `DEEPSEEK_API_KEY`（DeepSeek 官方 API key）
- 所有脚本从本目录运行（`mcp_client.py` / `agent_mcp.py` 通过相对路径启动 `mcp_server.py` 子进程）

## 快速开始

```bash
python agent_loop.py        # 手写循环，5 条固定任务 + trace
python tool_calling.py      # 三工具 Agent，5 条任务
python mcp_client.py        # MCP 跨进程调用演示
python agent_mcp.py         # Agent + MCP 工具
python agent_memory.py      # 交互式多轮对话（输入"退出"结束）
```

## 学习主线（面试叙事）

1. **Day 1-2 手写循环**：不依赖框架实现 ReAct——while 循环 + 模型输出解析 + 工具执行 + 观察喂回。理解"Agent = Harness + Model"中的 Harness 是什么；
2. **Day 3-4 工具系统**：工具 schema 从函数签名自动生成；模型生成的工具参数是**不可信输入**（eval 白名单 + 空 `__builtins__` 双重防护）；观察模型并行调用、跨工具切换、失败收手的决策行为；
3. **Day 5 MCP 协议**：手写 MCP server 与 client，体验"工具从进程私有函数变成标准服务"——传输层（stdio）与会话层（JSON-RPC）分层设计；
4. **Day 5 加分**：loop 骨架不变，工具来源从本进程函数换成 MCP 服务器——工具可插拔的实证；
5. **Day 6 记忆与容错**：messages 列表跨轮次保留实现多轮记忆（代词指代验证）；tenacity 指数退避重试（1s→2s→4s，3 次后 reraise）。

**与框架的对照**（面试对比"手写 vs 框架"的依据）：

| 手写实现 | 框架对应物 |
|---|---|
| `while` 循环 | `AgentExecutor` |
| `messages` 管理 | `agent_scratchpad` |
| `execute_tool` 分发器 | 框架自动执行 |
| 手写 `trace.jsonl` | LangSmith / verbose 输出 |

## 技术栈

Python · LangChain（`bind_tools` / LCEL）· DeepSeek API（OpenAI 兼容）· MCP SDK（FastMCP）· langchain-mcp-adapters · tenacity

## 目录说明

所有脚本平铺在根目录（未按 Day 分子目录）——`mcp_client.py` 和 `agent_mcp.py` 通过 `args=["mcp_server.py"]` 以相对路径启动子进程，移动文件会破坏这层耦合；且各脚本的相对路径均以本目录为工作目录。
