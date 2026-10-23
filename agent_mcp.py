"""
Day 5 练习：Agent 通过 MCP 协议调用跨进程工具

- 与 agent_loop.py 的对比实验：loop 骨架不变，工具来源从"本进程私有函数"换成"MCP 服务器"
- langchain-mcp-adapters（<0.2）把 MCP 工具转成 LangChain 工具，直接喂给 bind_tools
- 注意：适配器 0.1.x 不支持 async with，用"实例化 + await get_tools()"模式

运行：python agent_mcp.py（需要 DEEPSEEK_API_KEY 和同目录的 mcp_server.py）
"""
import sys, asyncio, os
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

MAX_STEP = 5

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.1
)

async def run_agent(query: str) -> str:
    # 0.1.x 适配器不支持 async with，用"实例化 + await get_tools()"模式
    client = MultiServerMCPClient({
        "notes": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["mcp_server.py"]
        }
    })
    mcp_tools = await client.get_tools()
    print("MCP 工具: ", [t.name for t in mcp_tools])

    llm_with_tools = llm.bind_tools(mcp_tools)

    messages = [
        SystemMessage("""- 角色：你是一个笔记助手
- 可用工具：search_notes（查笔记）、write_summary（写总结）
- 流程要求：先查笔记再回答；需要总结时先写文件再告知
- 查不到时的行为：如实告知未找到，不要编造"""),
        HumanMessage(query)
    ]

    for step in range(MAX_STEP):
        response = await llm_with_tools.ainvoke(messages)
        if response.tool_calls:
            messages.append(response)
            for call in response.tool_calls:
                tool = {t.name: t for t in mcp_tools}[call["name"]]
                result = await tool.ainvoke(call["args"])
                messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        else:
            return response.content
    return "达到最大步数上限"

if __name__ == "__main__":
    answer = asyncio.run(run_agent("帮我查一下 RAG 相关的笔记"))
    print(answer)