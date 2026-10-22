import json
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.1
)

MAX_STEPS = 5          # 步数上限
TRACE_PATH = "trace.jsonl"

NOTES = [
    "笔记1：RAG 的核心思想是检索增强生成——先查资料，再回答问题。",
    "笔记2：Agent Loop 是观察-思考-行动-观察的循环。",
    "笔记3：ReAct 是 Reasoning + Acting 的组合。",
]

STOP_PAIRS = {"帮我", "查询", "一下", "什么", "怎么", "哪些", "有没有", "请帮", "查找", "搜索", "关于", "内容", "笔记", "搜一"}

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

def _write_summary_impl(text: str) -> str:
    with open("summary.md", "w", encoding="utf-8") as f:
        f.write(text)
    return "已保存到 summary.md"

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

TOOLS = [search_notes, write_summary]

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

# def run_agent(query: str) -> str:
#     messages = [
#         SystemMessage("""- 角色：你是一个笔记助手
# - 可用工具：search_notes（查笔记）、write_summary（写总结）
# - 流程要求：先查笔记再回答；需要总结时先写文件再告知
# - 查不到时的行为：如实告知未找到，不要编造"""),
#         HumanMessage(query)]

#     llm_with_tools = llm.bind_tools(TOOLS)

#     for step in range(MAX_STEPS):
#         response = llm_with_tools.invoke(messages)
#         if response.tool_calls:
#             messages.append(response)

#             for tool_call in response.tool_calls:
#                 observation = execute_tool(tool_call["name"], tool_call["args"])
#                 messages.append(ToolMessage(
#                     content=observation,
#                     tool_call_id=tool_call["id"]
#                 ))
#                 write_trace(step, tool_call["name"], tool_call["args"], observation, response.usage_metadata["total_tokens"])
#         else:
#             return response.content

#     return "达到最大步数上限，任务未完成"

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

class ChatAgent:
    def __init__(self):
        self.messages = [
            SystemMessage("""- 角色：你是一个笔记助手
- 可用工具：search_notes（查笔记）、write_summary（写总结）
- 流程要求：先查笔记再回答；需要总结时先写文件再告知
- 查不到时的行为：如实告知未找到，不要编造
- 对话历史就在你的上下文里，不需要用工具查询对话内容""")
        ]
        self.llm_with_tools = llm.bind_tools(TOOLS)

    def chat(self, user_input: str) -> str:
        self.messages.append(HumanMessage(user_input))

        for step in range(MAX_STEPS):
            response = self.llm_with_tools.invoke(self.messages)

            if response.tool_calls:
                self.messages.append(response)

                for tool_call in response.tool_calls:
                    observation = execute_tool(tool_call["name"], tool_call["args"])
                    self.messages.append(ToolMessage(
                        content=observation,
                        tool_call_id=tool_call["id"]
                    ))
                write_trace(step, tool_call["name"], tool_call["args"], observation, response.usage_metadata["total_tokens"])
            else:
                self.messages.append(AIMessage(response.content))
                return response.content
        return "达到最大步数上限，任务未完成"


if __name__ == "__main__":
    agent = ChatAgent()
    while True:
        user = input("你: ").strip()
        if user == "退出":
            break
        print(f"助手: {agent.chat(user)}")