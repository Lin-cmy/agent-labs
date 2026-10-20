import os
import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

MAX_STEPS = 5
TRACE_PATH = "tool_calling_trace.jsonl"

WEATHER_DB = {
    "北京": "晴，25℃，东南风2级",
    "广州": "多云转雷阵雨，31℃",
    "上海": "小雨，22℃"
}

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.1
)

@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。输入城市名（如"北京"），返回该城市的天气情况。"""
    if city in WEATHER_DB:
        return f"{city} 天气: {WEATHER_DB[city]}"
    else:
        return f"未知城市: {city}"

@tool
def calculate(expression: str) -> str:
    """计算四则运算表达式。输入如 '(23+7)*3' 的算式，返回计算结果。"""
    if not re.fullmatch(r"[\d+\-*/().\s]+", expression):
        return f"非法表达式: {expression}（仅支持数字和 + - * / 括号）"
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算失败: {e}"

@tool
def search_files(keyword: str) -> str:
    """搜索当前目录下文件名包含关键词的文件。输入关键词（如"py"或"agent"），返回匹配的文件列表。"""
    matches = [f for f in os.listdir(".") if keyword in f]
    if not matches:
        return f"未找到文件名包含 '{keyword}' 的文件"
    return "\n".join(f"- {m}" for m in sorted(matches)[:10])

TOOLS = [get_weather, calculate, search_files]

TOOL_MAP = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_files": search_files
}

def execute_tool(name: str, args: dict) -> str:
    """执行工具并返回观察结果。任何错误都转为可读文本返回，绝不抛出。"""
    try:
        tool_obj = TOOL_MAP.get(name)
        if tool_obj is None:
            return f"未知工具: {name}，可用工具: {list(TOOL_MAP.keys())}"
        result = tool_obj.invoke(args)
        return result
    except Exception as e:
        return f"执行工具时出错：{e}"

def run_agent(query: str) -> str:
    messages = [
        SystemMessage("""- 角色：你是一个生活助手，可以查天气、算算术、搜索文件
- 可用工具：get_weather（查天气）、calculate（计算）、search_files（搜文件）
- 使用原则：
  1. 涉及天气/计算/文件搜索的问题，必须先调用对应工具，拿到结果后再回答
  2. 工具返回"未知城市"或报错时，如实告知用户，绝不要编造数据
  3. 一次要查多个城市或多个算式时，可以同时发起多个工具调用
  4. 与工具无关的问题（如问候），直接回答，不要强行调用工具"""),
        HumanMessage(query)
    ]

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

def write_trace(step: int, tool_name: str, args: dict, observation: str, cost: int) -> None:
    record = {
        "step": step,
        "thought_summary": "",
        "tool": tool_name,
        "args": args,
        "observation": observation,
        "cost_estimate": cost
    }

    with open(TRACE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    TASKS = [
        "帮我算一下 (23+7)*3 等于多少",
        "查一下北京和广州的天气",
        "看看这个目录下有哪些 .py 文件",
        "先算 123*456 是多少，再查一下上海的天气",
        "查一下火星的天气"
    ]

    for task in TASKS:
        print("=" * 50)
        print(f"正在回答问题 {task}...")
        answer = run_agent(task)
        print(f"\n最终回答: {answer}")
        print("=" * 50)