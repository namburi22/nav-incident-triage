import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

async def main():
    # Connect to your MCP server
    server_params = StdioServerParameters(
        command="python",
        args=["nav_mcp_server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize connection
            await session.initialize()

            # Discover tools dynamically — agent doesn't know these upfront
            print("Discovering tools from MCP server...")
            tools = await load_mcp_tools(session)
            print(f"Found {len(tools)} tools:")
            for tool in tools:
                print(f"  - {tool.name}")

            # Create agent with discovered tools
            agent = create_react_agent(llm, tools)

            print("\n=== Asking agent about FUND001 ===\n")
            response = await agent.ainvoke({
                "messages": [("user", "FUND001 NAV has failed. Check the feeds, history and tell me what happened and what to do.")]
            })

            print("Agent response:")
            print(response["messages"][-1].content)

asyncio.run(main())