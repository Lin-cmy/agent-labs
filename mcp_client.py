"""
Day 5 练习：MCP 客户端 —— 跨进程调用工具

- 通过 stdio 管道启动 mcp_server.py 子进程，用 JSON-RPC 协议发现并调用其工具
- 三层结构：StdioServerParameters（配置：怎么启动 server）→ stdio_client（传输层：字节怎么搬运）
  → ClientSession（会话层：协议怎么说话）
- 演示"MCP 解决了什么问题"：工具不再是本进程私有函数，而是任何 MCP 客户端都能调用的标准服务

运行：python mcp_client.py（依赖同目录的 mcp_server.py）
"""
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server_params = StdioServerParameters(command="python", args=["mcp_server.py"])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Server 提供的工具: ", [t.name for t in tools.tools])

            result = await session.call_tool("search_notes", {"query": "RAG"})
            print("调用结果:", result.content[0].text)

asyncio.run(main())