import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- State ---
class IncidentState(TypedDict):
    fund_id: str
    question: str
    nav_report: str
    feed_report: str
    consumer_report: str
    final_summary: str
    tools: list

# --- Async Nodes ---
async def fund_agent(state: IncidentState) -> IncidentState:
    print("\n[FUND AGENT] Investigating NAV...")
    
    tools = state["tools"]
    get_nav = next(t for t in tools if t.name == "get_fund_nav")
    get_history = next(t for t in tools if t.name == "get_incident_history")
    
    nav = await get_nav.ainvoke({"fund_id": state["fund_id"]})
    history = await get_history.ainvoke({"fund_id": state["fund_id"]})
    
    prompt = f"""
    You are a Fund NAV specialist.
    Fund: {state['fund_id']}
    NAV Data: {nav}
    Incident History: {history}
    
    Provide a brief fund status report focusing on:
    - Current NAV health
    - Historical patterns
    - Risk assessment
    """
    response = llm.invoke(prompt)
    state["nav_report"] = response.content
    print("[FUND AGENT] Done")
    return state

async def feed_agent(state: IncidentState) -> IncidentState:
    print("\n[FEED AGENT] Investigating feeds...")
    
    tools = state["tools"]
    get_feeds = next(t for t in tools if t.name == "get_feeds_for_fund")
    check_feed = next(t for t in tools if t.name == "check_feed_status")
    
    raw_feeds = await get_feeds.ainvoke({"fund_id": state["fund_id"]})
    
    # MCP wraps responses — extract string feed_ids
    feed_ids = []
    for item in raw_feeds:
        if isinstance(item, str):
            feed_ids.append(item)
        elif isinstance(item, dict):
            # extract text value from MCP wrapper
            feed_ids.append(item.get("text", item.get("feed_id", str(item))))
    
    print(f"  Feeds found: {feed_ids}")
    
    feed_statuses = []
    for feed_id in feed_ids:
        status = await check_feed.ainvoke({"feed_id": feed_id})
        # handle MCP wrapper on status too
        if isinstance(status, list):
            status = status[0] if status else {}
        if isinstance(status, dict) and "text" in status:
            import json
            try:
                status = json.loads(status["text"])
            except:
                pass
        feed_statuses.append({"feed_id": feed_id, "status": status})
        print(f"  Feed {feed_id}: {status}")
    
    prompt = f"""
    You are a Data Feed specialist.
    Fund: {state['fund_id']}
    Feed Statuses: {feed_statuses}
    
    Provide a brief feed health report:
    - Which feeds are failing
    - Severity of each issue
    - Likely impact on NAV calculation
    """
    response = llm.invoke(prompt)
    state["feed_report"] = response.content
    print("[FEED AGENT] Done")
    return state

async def consumer_agent(state: IncidentState) -> IncidentState:
    print("\n[CONSUMER AGENT] Checking downstream impact...")
    
    tools = state["tools"]
    get_consumers = next(t for t in tools if t.name == "get_impacted_consumers")
    
    consumers = await get_consumers.ainvoke({"fund_id": state["fund_id"]})
    
    prompt = f"""
    You are a Downstream Impact specialist.
    Fund: {state['fund_id']}
    Impacted Consumers: {consumers}
    
    Provide a brief impact report:
    - Which systems are affected
    - Business severity per system
    - Urgency of resolution
    """
    response = llm.invoke(prompt)
    state["consumer_report"] = response.content
    print("[CONSUMER AGENT] Done")
    return state

async def supervisor(state: IncidentState) -> IncidentState:
    print("\n[SUPERVISOR] Synthesizing all reports...")
    
    prompt = f"""
    You are an Incident Management Supervisor.
    Original Question: {state['question']}
    
    Fund Report: {state['nav_report']}
    Feed Report: {state['feed_report']}
    Consumer Report: {state['consumer_report']}
    
    Synthesize into a final executive incident summary:
    1. Situation — what happened
    2. Impact — who is affected and severity
    3. Root cause — why it happened
    4. Action plan — immediate steps with owner
    5. ETA — realistic resolution timeline
    """
    response = llm.invoke(prompt)
    state["final_summary"] = response.content
    print("[SUPERVISOR] Done")
    return state

async def should_investigate(state: IncidentState) -> str:
    print("\n[DECISION] Evaluating severity...")
    
    tools = state["tools"]
    get_nav = next(t for t in tools if t.name == "get_fund_nav")
    nav = await get_nav.ainvoke({"fund_id": state["fund_id"]})
    # MCP sometimes wraps response differently — handle both
    if isinstance(nav, list):
        nav = nav[0] if nav else {}
    if isinstance(nav, dict):
        nav_status = nav.get("status", "UNKNOWN")
    else:
        nav_status = str(nav)

    print(f"[DECISION] NAV status: {nav_status}")

    prompt = f"""
    Fund: {state['fund_id']}
    NAV Status: {nav_status}
    
    Should we investigate further?
    FAILED or PENDING = investigate
    SUCCESS = skip
    
    Respond with exactly one word: 'investigate' or 'skip'
    """
    response = llm.invoke(prompt)
    decision = response.content.strip().lower()
    print(f"[DECISION] → {decision}")
    return decision

# --- Build Graph ---
def build_graph():
    graph = StateGraph(IncidentState)
    
    graph.add_node("fund_agent", fund_agent)
    graph.add_node("feed_agent", feed_agent)
    graph.add_node("consumer_agent", consumer_agent)
    graph.add_node("supervisor", supervisor)
    
    graph.set_entry_point("fund_agent")
    
    graph.add_conditional_edges(
        "fund_agent",
        should_investigate,
        {
            "investigate": "feed_agent",
            "skip": END
        }
    )
    
    graph.add_edge("feed_agent", "consumer_agent")
    graph.add_edge("consumer_agent", "supervisor")
    graph.add_edge("supervisor", END)
    
    return graph.compile()

# --- Main ---
async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["nav_mcp_server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            print("Connecting to MCP server...")
            tools = await load_mcp_tools(session)
            print(f"Discovered {len(tools)} tools from MCP server")
            
            app = build_graph()
            
            # FUND001 — should investigate
            print("\n=== Final System: FUND001 ===\n")
            result = await app.ainvoke({
                "fund_id": "FUND001",
                "question": "FUND001 NAV has failed. What is the full situation and action plan?",
                "nav_report": "",
                "feed_report": "",
                "consumer_report": "",
                "final_summary": "",
                "tools": tools
            })
            
            print("\n" + "="*60)
            print("EXECUTIVE INCIDENT SUMMARY")
            print("="*60)
            print(result["final_summary"])
            
            # FUND002 — should skip
            print("\n=== Final System: FUND002 ===\n")
            result = await app.ainvoke({
                "fund_id": "FUND002",
                "question": "Check FUND002 status",
                "nav_report": "",
                "feed_report": "",
                "consumer_report": "",
                "final_summary": "",
                "tools": tools
            })
            
            if result["final_summary"]:
                print(result["final_summary"])
            else:
                print("NAV healthy — no investigation needed")

asyncio.run(main())