import asyncio
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- Load RAG knowledge base ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./nav_knowledge_base",
    embedding_function=embeddings,
    collection_name="nav_incidents"
)

# --- State ---
class IncidentState(TypedDict):
    fund_id: str
    question: str
    nav_report: str
    feed_report: str
    consumer_report: str
    knowledge_report: str
    final_summary: str
    tools: list

def parse_mcp_response(result):
    """Parse MCP wrapped response into clean dict or list."""
    # MCP returns list of content blocks
    if isinstance(result, list):
        if not result:
            return {}
        item = result[0]
        if isinstance(item, dict) and "text" in item:
            try:
                return json.loads(item["text"])
            except:
                return item["text"]
        return item
    return result

# --- RAG helper ---
def query_knowledge_base(query: str, k: int = 3) -> str:
    results = vectorstore.similarity_search(query, k=k)
    if not results:
        return "No relevant historical incidents found."
    context = ""
    for i, doc in enumerate(results, 1):
        context += f"\n--- Result {i} ---\n"
        context += doc.page_content.strip()
        context += "\n"
    return context

# --- Agents ---
async def fund_agent(state: IncidentState) -> IncidentState:
    print("\n[FUND AGENT] Investigating NAV...")
    tools = state["tools"]
    get_nav = next(t for t in tools if t.name == "get_fund_nav")
    get_history = next(t for t in tools if t.name == "get_incident_history")
    
    raw = await get_nav.ainvoke({"fund_id": state["fund_id"]})
    nav = parse_mcp_response(raw)
    history = await get_history.ainvoke({"fund_id": state["fund_id"]})
    
    # handle MCP wrapper
    if isinstance(nav, list):
        nav = nav[0] if nav else {}

    prompt = f"""
    You are a Fund NAV specialist.
    Fund: {state['fund_id']}
    NAV Data: {nav}
    Incident History: {history}
    
    Provide a brief fund status report:
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
    
    feed_ids = []
    for item in raw_feeds:
        if isinstance(item, str):
            feed_ids.append(item)
        elif isinstance(item, dict):
            feed_ids.append(item.get("text", str(item)))
    
    feed_statuses = []
    for feed_id in feed_ids:
        status = await check_feed.ainvoke({"feed_id": feed_id})
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

async def knowledge_agent(state: IncidentState) -> IncidentState:
    print("\n[KNOWLEDGE AGENT] Querying historical knowledge base...")
    
    # Build smart query from current incident context
    query = f"{state['fund_id']} feed failure NAV incident resolution playbook"
    
    context = query_knowledge_base(query, k=4)
    
    prompt = f"""
    You are a Knowledge Management specialist with access to historical incident database.
    
    Current Incident: {state['question']}
    Fund: {state['fund_id']}
    
    Historical Context Retrieved:
    {context}
    
    Based on historical incidents and playbooks provide:
    - Similar past incidents and what caused them
    - Proven resolution steps from playbooks
    - Average resolution time based on history
    - Any recurring patterns to watch for
    """
    response = llm.invoke(prompt)
    state["knowledge_report"] = response.content
    print("[KNOWLEDGE AGENT] Done")
    return state

async def supervisor(state: IncidentState) -> IncidentState:
    print("\n[SUPERVISOR] Synthesizing all reports...")
    
    prompt = f"""
    You are an Incident Management Supervisor.
    Original Question: {state['question']}
    
    Fund Report: {state['nav_report']}
    Feed Report: {state['feed_report']}
    Consumer Report: {state['consumer_report']}
    Knowledge Report: {state['knowledge_report']}
    
    Synthesize into a final executive incident summary:
    1. Situation — what happened
    2. Impact — who is affected and severity
    3. Root cause — why it happened
    4. Action plan — specific steps with owners based on historical playbooks
    5. ETA — based on historical resolution times
    """
    response = llm.invoke(prompt)
    state["final_summary"] = response.content
    print("[SUPERVISOR] Done")
    return state

# --- Decision ---
async def should_investigate(state: IncidentState) -> str:
    print("\n[DECISION] Evaluating severity...")
    tools = state["tools"]
    get_nav = next(t for t in tools if t.name == "get_fund_nav")
    raw = await get_nav.ainvoke({"fund_id": state["fund_id"]})
    
    nav_status = "UNKNOWN"
    try:
        if isinstance(raw, list) and len(raw) > 0:
            item = raw[0]
            if isinstance(item, dict) and "text" in item:
                parsed = json.loads(item["text"])
                nav_status = parsed.get("status", "UNKNOWN")
            elif hasattr(item, "text"):
                parsed = json.loads(item.text)
                nav_status = parsed.get("status", "UNKNOWN")
    except Exception as e:
        print(f"[DECISION] Parse error: {e}")

    print(f"[DECISION] NAV status: {nav_status}")

    if nav_status in ["FAILED", "PENDING"]:
        print("[DECISION] → investigate")
        return "investigate"
    else:
        print("[DECISION] → skip")
        return "skip"

# --- Build Graph ---
def build_graph():
    graph = StateGraph(IncidentState)
    
    graph.add_node("fund_agent", fund_agent)
    graph.add_node("feed_agent", feed_agent)
    graph.add_node("consumer_agent", consumer_agent)
    graph.add_node("knowledge_agent", knowledge_agent)
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
    graph.add_edge("consumer_agent", "knowledge_agent")
    graph.add_edge("knowledge_agent", "supervisor")
    graph.add_edge("supervisor", END)
    
    return graph.compile()

# --- Main ---
async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["nav_mcp_server_v2.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            print("Connecting to MCP server...")
            tools = await load_mcp_tools(session)
            print(f"Discovered {len(tools)} tools from MCP server")
            
            app = build_graph()
            
            # FUND001 — FAILED
            print("\n=== Full System: FUND001 ===\n")
            result = await app.ainvoke({
                "fund_id": "FUND001",
                "question": "FUND001 NAV has failed. What is the full situation and action plan?",
                "nav_report": "",
                "feed_report": "",
                "consumer_report": "",
                "knowledge_report": "",
                "final_summary": "",
                "tools": tools
            })
            
            print("\n" + "="*60)
            print("EXECUTIVE INCIDENT SUMMARY — FUND001")
            print("="*60)
            print(result["final_summary"])

            print("\n=== Full System: FUND002 ===\n")
            result = await app.ainvoke({
                    "fund_id": "FUND002",
                    "question": "Check FUND002 status",
                    "nav_report": "",
                    "feed_report": "",
                    "consumer_report": "",
                    "knowledge_report": "",
                    "final_summary": "",
                    "tools": tools
            })

            if result["final_summary"]:
             print(result["final_summary"])
            else:
             print("NAV healthy — no investigation needed")

asyncio.run(main())