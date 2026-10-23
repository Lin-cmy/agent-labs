"""
Day 5 练习：MCP 工具服务器（FastMCP，stdio 传输）

- 把 Day 1-2 的笔记工具注册为 MCP 工具：@mcp.tool() 装饰器自动生成标准 schema
  （name / description / inputSchema / outputSchema）
- stdio 模式下静默等待 JSON-RPC 请求：单独运行时"挂起"是正常现象，不是卡死
- 依赖：pip install "mcp<2"（mcp 2.x 已把 FastMCP 改名 MCPServer，本文件基于 1.x API）

运行：由 mcp_client.py / agent_mcp.py 以子进程方式启动；也可单独运行验证存活
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("notes-server")

NOTES = [
    "笔记1：RAG 的核心思想是检索增强生成——先查资料，再回答问题。",
    "笔记2：Agent Loop 是观察-思考-行动-观察的循环。",
    "笔记3：ReAct 是 Reasoning + Acting 的组合。",
]

STOP_PAIRS = {"帮我", "查询", "一下", "什么", "怎么", "哪些", "有没有", "请帮", "查找", "搜索", "关于", "内容", "笔记", "搜一"}
SAVE_PATH = "summary.md"

@mcp.tool()
def search_notes(query: str) -> str:
    """在本地笔记库中搜索与查询主题相关的笔记内容。"""
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

@mcp.tool()
def write_summary(text: str) -> str:
    """将文本保存到 summary.md"""
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    return f"已保存到 {SAVE_PATH}"

if __name__ == "__main__":
    mcp.run()