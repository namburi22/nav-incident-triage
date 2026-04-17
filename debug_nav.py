import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["nav_mcp_server_v2.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            
            get_nav = next(t for t in tools if t.name == "get_fund_nav")
            
            # See exactly what MCP returns
            result = await get_nav.ainvoke({"fund_id": "FUND001"})
            
            print(f"Type: {type(result)}")
            print(f"Value: {result}")

asyncio.run(main())