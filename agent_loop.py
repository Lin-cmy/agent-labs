"""
Day 1-2 练习：手写最小 Agent Loop
学习目标：理解 observe -> think -> act -> observe 循环

验收标准（AgentGuide）：
  1. 只接 2 个工具：search_notes(query)、write_summary(text)
  2. 限制最大 5 步
  3. 每步写入 JSONL trace：step, thought_summary, tool, args, observation, cost_estimate
  4. 能处理 5 条固定任务
  5. 出错时不崩溃，返回可读失败原因
"""
import json
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool


# ============ 0. 配置 ============
# TODO: 初始化 llm = ChatOpenAI(...)
#   - model="deepseek-chat"
#   - api_key=os.getenv("DEEPSEEK_API_KEY")
#   - base_url="https://api.deepseek.com"
#   - temperature 建议 0~0.3（让工具调用输出更稳定）

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.1
)

MAX_STEPS = 5          # 步数上限
TRACE_PATH = "trace.jsonl"

# 本地笔记库（供 search_notes 检索，可自行补充到 8~10 条）
NOTES = [
    "笔记1：RAG 的核心思想是检索增强生成——先查资料，再回答问题。",
    "笔记2：Agent Loop 是观察-思考-行动-观察的循环。",
    "笔记3：ReAct 是 Reasoning + Acting 的组合。",
]

STOP_PAIRS = {"帮我", "查询", "一下", "什么", "怎么", "哪些", "有没有", "请帮", "查找", "搜索", "关于", "内容", "笔记", "搜一"}


# TOOLS 列表必须放在工具定义【之后】创建：bind_tools 需要的是工具对象，不是名字字符串

# ============ 1. 工具函数（只写函数本体） ============
# 思路：工具函数就是普通 Python 函数，模型只能"请求调用"，真正执行的是这里。
# TODO: search_notes(query) -> str
#   - 在 NOTES 里做关键词匹配（query 中的关键词命中笔记即算匹配）
#   - 返回最多 3 条，格式：编号 + 内容
#   - 没有命中返回"未找到相关笔记"

def _search_notes_impl(query: str) -> str:
    key_words = get_key_words(query)

    hits = []
    for i, note in enumerate(NOTES):
        score = len(key_words & get_key_words(note))
        if score > 0:
            hits.append((score, note))

    if not hits:
        return "未找到相关笔记"

    hits.sort(key=lambda x: -x[0])
    return "\n".join(f"- {note}" for _, note in hits[:3])

def get_key_words(text: str) -> set:
    pairs = {text[i:i+2] for i in range(len(text) - 1)}
    return pairs - STOP_PAIRS


# TODO: write_summary(text) -> str
#   - 把 text 写入 summary.md（覆盖写即可）
#   - 返回"已保存到 summary.md"

def _write_summary_impl(text: str) -> str:
    with open("summary.md", "w", encoding="utf-8") as f:
        f.write(text)
    return "已保存到 summary.md"

# ============ 2. 工具 schema（让模型知道工具的存在） ============
# 思路：模型需要三样信息——工具叫什么、干什么、要什么参数。
# 用 @tool 装饰器包一层函数即可自动生成 schema（函数文档字符串会变成工具描述）。
# 装饰时注意：新函数本体只需 return 调用上面的实现（或直接把 @tool 加在上面函数上）。
# 提示：@tool 装饰后的对象有 .name / .description / .args 属性。
# TODO: 用 @tool 包装 search_notes / write_summary，放进 TOOLS 列表

@tool
def search_notes(query: str) -> str:
    """在本地笔记库中搜索与查询主题相关的笔记内容。当用户询问某个主题、需要查找笔记时使用此工具。"""
    return _search_notes_impl(query)

@tool
def write_summary(text: str) -> str:
    """将文本信息保存到summary.md中，当模型需要保存输出内容时使用此工具。"""
    return _write_summary_impl(text)

TOOL_MAP = {
    "search_notes": search_notes,
    "write_summary": write_summary
}

# bind_tools 需要【工具对象】的列表（方括号 + 装饰后的对象，不是花括号 + 字符串）
TOOLS = [search_notes, write_summary]

# ============ 3. 工具执行器 ============
# TODO: execute_tool(name: str, args: dict) -> str
# 思路：按 name 分发到对应工具函数。
# 要求：整个分发过程包在 try/except 里——
#   出错时返回 f"工具执行失败: {e}"（把失败作为观察喂回模型，而不是崩溃）。
#   同时返回错误文本时要让模型知道"这次调用失败了"，它才会换策略。

def execute_tool(name: str, args: dict) -> str:
    """执行工具并返回观察结果。任何错误都转为可读文本返回，绝不抛出。"""
    try:
        tool_obj = TOOL_MAP.get(name, "")
        if not tool_obj:
            return f"未知工具: {name}，可用工具: {list(TOOL_MAP.keys())}"
        result = tool_obj.invoke(args)
        return str(result)
    except Exception as e:
        return f"工具执行失败: {e}"

# ============ 4. Agent Loop 核心 ============
# TODO: run_agent(query: str) -> str
# 思路（对照下面的骨架补全）：
#   messages = [SystemMessage(系统提示词), HumanMessage(query)]
#   循环 step in range(MAX_STEPS)：
#     a. response = llm.invoke(messages)          # 注意：这里要调用 bind_tools 之后的模型
#     b. 如果 response.tool_calls 非空：
#          - 对每个 tool_call：execute_tool 执行，构造 ToolMessage 追加进 messages
#          - 写一行 trace.jsonl
#     c. 否则（没有工具调用）：返回 response.content（最终答案）
#   循环耗尽：返回"达到最大步数上限，任务未完成"
#
# 系统提示词要点：
#   - 角色：你是一个笔记助手
#   - 可用工具：search_notes（查笔记）、write_summary（写总结）
#   - 流程要求：先查笔记再回答；需要总结时先写文件再告知
#   - 查不到时的行为：如实告知未找到，不要编造

# bind_tools 的思路：llm.bind_tools(TOOLS) 返回一个新的可调用对象，
# 模型的输出里才会带 tool_calls 字段（AIMessage.tool_calls）。

def run_agent(query: str) -> str:
    messages = [
        SystemMessage("""- 角色：你是一个笔记助手
- 可用工具：search_notes（查笔记）、write_summary（写总结）
- 流程要求：先查笔记再回答；需要总结时先写文件再告知
- 查不到时的行为：如实告知未找到，不要编造"""),
        HumanMessage(query)]

    llm_with_tools = llm.bind_tools(TOOLS)

    for step in range(MAX_STEPS):
        response = llm_with_tools.invoke(messages)
        if response.tool_calls:
            messages.append(response)

            for tool_call in response.tool_calls:
                observation = execute_tool(tool_call["name"], tool_call["args"])
                messages.append(ToolMessage(
                    content=observation,
                    tool_call_id=tool_call["id"]
                ))
                write_trace(step, tool_call["name"], tool_call["args"], observation, response.usage_metadata["total_tokens"])
        else:
            return response.content

    return "达到最大步数上限，任务未完成"

# ============ 5. trace 记录函数 ============
# TODO: write_trace(step, tool_name, args, observation, cost) -> None
# 思路：把字段组成 dict，json.dumps(..., ensure_ascii=False) 转成一行，
# 以追加模式写入 TRACE_PATH。cost 可以从 response.usage_metadata 取 total_tokens。
# thought_summary 字段先填空字符串占位，跑完后回看 trace 再补写体会。
def write_trace(step: int, tool_name: str, args: dict, observation: str, cost: int) -> None:
    """把每一步的决策轨迹追加写入 trace.jsonl（一行一个 JSON 对象）"""
    record = {
        "step": step,
        "thought_summary": "",
        "tool": tool_name,
        "args": args,
        "observation": observation[:200],
        "cost_estimate": cost
    }
    with open(TRACE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ============ 6. 测试区 ============
# TODO: TASKS 列表放 5 条固定任务，覆盖三种情况：
#   1) 正常多步（需要先查后写/答）
#   2) 搜索无结果（考验模型面对"未找到"怎么收场）
#   3) 可直接回答（可能一步结束）
# 逐条调用 run_agent，打印分隔线和结果。
# 验收自查：
#   - 跑完 5 条任务不崩溃
#   - trace.jsonl 每步一行、字段齐全
#   - 出错任务返回的是可读的失败原因
if __name__ == "__main__":
    TASKS = [
        "什么是 RAG？",
        "RAG 和 Agent Loop 的区别和联系是什么？",
        "重庆大学的推免政策是什么？",
        "你好。",
        "Agent Loop 是对 ReAct 思想的实现吗？"
    ]

    for task in TASKS:
        print("=" * 50)
        print(f"正在回答 {task}")
        answer = run_agent(task)
        print(f"\n最终回答: {answer}")
        print("=" * 50)
