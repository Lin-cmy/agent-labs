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