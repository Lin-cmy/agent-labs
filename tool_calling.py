"""
Day 3-4 练习：三工具多功能 Agent

- 工具：get_weather（mock 天气）、calculate（AST 结构白名单的四则计算）、search_files（文件搜索）
- 复用 Day 1-2 的手写循环骨架，只替换工具集（验证"loop 骨架与工具解耦"）
- 演示：工具 schema（签名+docstring 自动生成）、并行工具调用、跨工具切换、失败处理（未知城市）
- 安全点：calculate 的参数是模型生成的不可信输入，采用 AST 结构白名单 + 空 __builtins__
  + 长度/指数/位宽三重预算（拒绝 9**9**9 这类算力炸弹；NFKC 归一化全角输入避免误杀）

运行：python tool_calling.py（需要 DEEPSEEK_API_KEY）
trace 输出：tool_calling_trace.jsonl
"""
import os
import json
import ast
import math
import unicodedata

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

MAX_STEPS = 5
TRACE_PATH = "tool_calling_trace.jsonl"

MAX_EXPR_LEN = 200          # 表达式长度上限，挡住超长输入
MAX_ABS_EXPONENT = 64       # 幂指数上限，挡住 9**9**9 这类算力炸弹
MAX_RESULT_BITS = 4096      # 结果位宽上限

# 只允许"表达式 + 四则运算 + 数字常量"这几种语法节点。
# 出现任何未枚举节点（Call / Attribute / Name / Lambda ...）即拒绝。
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
)

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

class UnsafeExpression(ValueError):
    """表达式不满足白名单约束时抛出，由 calculate 转成可读文本。"""


def _validate_expr(node: ast.AST) -> None:
    """AST 结构白名单校验。

    为什么不用正则字符白名单：字符合法不等于结构合法——它会放行 9**9**9
    （** 只是两个合法字符），还会误杀 1e5、全角数字这类合法输入。
    节点枚举把"能做什么运算"显式写死，审计时看这一张表就够了。
    """
    for sub in ast.walk(node):
        if not isinstance(sub, _ALLOWED_NODES):
            raise UnsafeExpression(f"不允许的语法: {type(sub).__name__}")
        if isinstance(sub, ast.Constant) and not isinstance(sub.value, (int, float)):
            raise UnsafeExpression("只允许数字常量")
        if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Pow):
            exp = sub.right
            if not (isinstance(exp, ast.Constant) and isinstance(exp.value, int)):
                raise UnsafeExpression("幂指数必须是整数字面量")
            if abs(exp.value) > MAX_ABS_EXPONENT:
                raise UnsafeExpression(f"幂指数超过上限 {MAX_ABS_EXPONENT}")


@tool
def calculate(expression: str) -> str:
    """计算四则运算表达式。输入如 '(23+7)*3' 的算式，返回计算结果。"""
    if not isinstance(expression, str):
        return f"非法表达式: {expression!r}（需要字符串）"
    if len(expression) > MAX_EXPR_LEN:
        return f"非法表达式: 长度 {len(expression)} 超过上限 {MAX_EXPR_LEN}"
    # 模型偶尔输出全角数字（１＋１），归一化后是合法意图，不该被判非法
    normalized = unicodedata.normalize("NFKC", expression).strip()
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as e:
        return f"非法表达式: {expression}（{e.msg}）"
    try:
        _validate_expr(tree)
    except UnsafeExpression as e:
        return f"非法表达式: {expression}（{e}）"
    # 走到这里 tree 只含数字常量与四则运算符，eval 已无执行面；
    # 空 __builtins__ 仍保留，作为纵深防御的一层而不是唯一的防线。
    try:
        result = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, {})
    except ZeroDivisionError:
        return f"计算失败: {expression}（除数为 0）"
    except (ValueError, OverflowError, MemoryError) as e:
        return f"计算失败: {expression}（{type(e).__name__}）"
    if isinstance(result, float):
        if math.isinf(result) or math.isnan(result):
            return f"计算失败: {expression}（结果不是有限数）"
        result = round(result, 10)
    if isinstance(result, int) and result.bit_length() > MAX_RESULT_BITS:
        return f"计算失败: {expression}（结果规模超过 {MAX_RESULT_BITS} 位）"
    return f"{expression} = {result}"

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
                # 必须在循环体内逐条落盘：并行调用多个工具时，写在循环外
                # 只会留下最后一条，事后无法复盘"模型到底请求了哪些调用"。
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
        "查一下火星的天气",
        "一千一百减九十九等于多少"
    ]

    for task in TASKS:
        print("=" * 50)
        print(f"正在回答问题 {task}...")
        answer = run_agent(task)
        print(f"\n最终回答: {answer}")
        print("=" * 50)