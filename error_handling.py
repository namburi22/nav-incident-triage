import asyncio
import json
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional

load_dotenv()

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-4o", temperature=0)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./nav_knowledge_base",
    embedding_function=embeddings,
    collection_name="nav_incidents"
)

# --- State with error tracking ---
class IncidentState(TypedDict):
    fund_id: str
    question: str
    nav_report: str
    feed_report: str
    consumer_report: str
    knowledge_report: str
    final_summary: str
    errors: List[str]      # track errors per agent
    tools: list

# --- MCP response parser ---
def parse_mcp_response(result):
    if isinstance(result, list):
        if not result:
            return {}
        item = result[0]
        if isinstance(item, dict) and "text" in item:
            try:
                return json.loads(item["text"])
            except:
                return item["text"]
        elif hasattr(item, "text"):
            try:
                return json.loads(item.text)
            except:
                return item.text
        return item
    return result

# --- Tool call with retry ---
async def safe_tool_call(tool, args: dict, retries: int = 3, fallback=None):
    """Call a tool with retry logic and fallback."""
    for attempt in range(retries):
        try:
            result = await tool.ainvoke(args)
            return result
        except Exception as e:
            logger.warning(f"Tool {tool.name} failed attempt {attempt+1}: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(1)  # wait before retry
            else:
                logger.error(f"Tool {tool.name} failed after {retries} attempts")
                return fallback
    return fallback

# --- RAG helper with error handling ---
def query_knowledge_base(query: str, k: int = 3) -> str:
    try:
        results = vectorstore.similarity_search(query, k=k)
        if not results:
            return "No relevant historical incidents found."
        context = ""
        for i, doc in enumerate(results, 1):
            context += f"\n--- Result {i} ---\n"
            context += doc.page_content.strip()
            context += "\n"
        return context
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        return "Knowledge base temporarily unavailable."

# --- Agents with error handling ---
async def fund_agent(state: IncidentState) -> IncidentState:
    print("\n[FUND AGENT] Investigating NAV...")
    errors = state.get("errors", [])
    
    try:
        tools = state["tools"]
        get_nav = next(t for t in tools if t.name == "get_fund_nav")
        get_history = next(t for t in tools if t.name == "get_incident_history")
        
        # Safe tool calls with fallback
        raw = await safe_tool_call(
            get_nav,
            {"fund_id": state["fund_id"]},
            fallback={"status": "UNKNOWN", "nav": 0.0}
        )
        nav = parse_mcp_response(raw)
        
        history = await safe_tool_call(
            get_history,
            {"fund_id": state["fund_id"]},
            fallback=[]
        )

        prompt = f"""
        You are a Fund NAV specialist.
        Fund: {state['fund_id']}
        NAV Data: {nav}
        Incident History: {history}
        
        Provide a brief fund status report:
        - Current NAV health
        - Historical patterns
        - Risk assessment
        
        Note: If data is unavailable say so clearly.
        """
        response = llm.invoke(prompt)
        state["nav_report"] = response.content
        
    except Exception as e:
        error_msg = f"Fund agent failed: {str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)
        state["nav_report"] = "Fund data unavailable due to system error."
    
    state["errors"] = errors
    print("[FUND AGENT] Done")
    return state

async def feed_agent(state: IncidentState) -> IncidentState:
    print("\n[FEED AGENT] Investigating feeds...")
    errors = state.get("errors", [])
    
    try:
        tools = state["tools"]
        get_feeds = next(t for t in tools if t.name == "get_feeds_for_fund")
        check_feed = next(t for t in tools if t.name == "check_feed_status")
        
        raw_feeds = await safe_tool_call(
            get_feeds,
            {"fund_id": state["fund_id"]},
            fallback=[]
        )
        
        feed_ids = []
        if raw_feeds:
            for item in raw_feeds:
                if isinstance(item, str):
                    feed_ids.append(item)
                elif isinstance(item, dict):
                    feed_ids.append(item.get("text", str(item)))
        
        if not feed_ids:
            state["feed_report"] = "No feed data available."
            state["errors"] = errors
            return state
        
        feed_statuses = []
        for feed_id in feed_ids:
            status = await safe_tool_call(
                check_feed,
                {"feed_id": feed_id},
                fallback={"status": "UNKNOWN"}
            )
            if isinstance(status, list):
                status = status[0] if status else {}
            if isinstance(status, dict) and "text" in status:
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
        
        Note: If data is unavailable say so clearly.
        """
        response = llm.invoke(prompt)
        state["feed_report"] = response.content
        
    except Exception as e:
        error_msg = f"Feed agent failed: {str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)
        state["feed_report"] = "Feed data unavailable due to system error."
    
    state["errors"] = errors
    print("[FEED AGENT] Done")
    return state

async def consumer_agent(state: IncidentState) -> IncidentState:
    print("\n[CONSUMER AGENT] Checking downstream impact...")
    errors = state.get("errors", [])
    
    try:
        tools = state["tools"]
        get_consumers = next(t for t in tools if t.name == "get_impacted_consumers")
        
        consumers = await safe_tool_call(
            get_consumers,
            {"fund_id": state["fund_id"]},
            fallback=[]
        )

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
        
    except Exception as e:
        error_msg = f"Consumer agent failed: {str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)
        state["consumer_report"] = "Consumer data unavailable due to system error."
    
    state["errors"] = errors
    print("[CONSUMER AGENT] Done")
    return state

async def knowledge_agent(state: IncidentState) -> IncidentState:
    print("\n[KNOWLEDGE AGENT] Querying knowledge base...")
    errors = state.get("errors", [])
    
    try:
        query = f"{state['fund_id']} feed failure NAV incident resolution playbook"
        context = query_knowledge_base(query, k=4)

        prompt = f"""
        You are a Knowledge Management specialist.
        Current Incident: {state['question']}
        Fund: {state['fund_id']}
        
        Historical Context:
        {context}
        
        Provide:
        - Similar past incidents and causes
        - Proven resolution steps
        - Average resolution time
        - Recurring patterns
        """
        response = llm.invoke(prompt)
        state["knowledge_report"] = response.content
        
    except Exception as e:
        error_msg = f"Knowledge agent failed: {str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)
        state["knowledge_report"] = "Historical knowledge unavailable due to system error."
    
    state["errors"] = errors
    print("[KNOWLEDGE AGENT] Done")
    return state

async def supervisor(state: IncidentState) -> IncidentState:
    print("\n[SUPERVISOR] Synthesizing all reports...")
    errors = state.get("errors", [])
    
    try:
        # Include any errors in summary so supervisor is aware
        error_context = ""
        if errors:
            error_context = f"\nNote: The following errors occurred during investigation: {errors}"

        prompt = f"""
        You are an Incident Management Supervisor.
        Original Question: {state['question']}
        
        Fund Report: {state['nav_report']}
        Feed Report: {state['feed_report']}
        Consumer Report: {state['consumer_report']}
        Knowledge Report: {state['knowledge_report']}
        {error_context}
        
        Synthesize into a final executive incident summary:
        1. Situation — what happened
        2. Impact — who is affected and severity
        3. Root cause — why it happened
        4. Action plan — specific steps with owners
        5. ETA — based on historical resolution times
        6. Data gaps — note any missing data due to errors
        """
        response = llm.invoke(prompt)
        state["final_summary"] = response.content
        
    except Exception as e:
        error_msg = f"Supervisor failed: {str(e)}"
        logger.error(error_msg)
        state["final_summary"] = "Unable to generate summary due to system error. Please investigate manually."
    
    print("[SUPERVISOR] Done")
    return state

# --- Decision with error handling ---
async def should_investigate(state: IncidentState) -> str:
    print("\n[DECISION] Evaluating severity...")
    
    try:
        tools = state["tools"]
        get_nav = next(t for t in tools if t.name == "get_fund_nav")
        raw = await safe_tool_call(
            get_nav,
            {"fund_id": state["fund_id"]},
            fallback={"status": "UNKNOWN"}
        )
        
        nav_status = "UNKNOWN"
        if isinstance(raw, list) and len(raw) > 0:
            item = raw[0]
            if isinstance(item, dict) and "text" in item:
                parsed = json.loads(item["text"])
                nav_status = parsed.get("status", "UNKNOWN")
            elif hasattr(item, "text"):
                parsed = json.loads(item.text)
                nav_status = parsed.get("status", "UNKNOWN")
        elif isinstance(raw, dict):
            nav_status = raw.get("status", "UNKNOWN")

        print(f"[DECISION] NAV status: {nav_status}")

        # If unknown — default to investigate to be safe
        if nav_status in ["FAILED", "PENDING", "UNKNOWN"]:
            print("[DECISION] → investigate")
            return "investigate"
        else:
            print("[DECISION] → skip")
            return "skip"
            
    except Exception as e:
        logger.error(f"Decision function failed: {e}")
        print("[DECISION] → investigate (defaulting due to error)")
        return "investigate"

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
            
            print("\n=== Error Handling Test: FUND001 ===\n")
            result = await app.ainvoke({
                "fund_id": "FUND001",
                "question": "FUND001 NAV has failed. What is the situation?",
                "nav_report": "",
                "feed_report": "",
                "consumer_report": "",
                "knowledge_report": "",
                "final_summary": "",
                "errors": [],
                "tools": tools
            })
            
            print("\n" + "="*60)
            print("EXECUTIVE SUMMARY")
            print("="*60)
            print(result["final_summary"])
            
            if result["errors"]:
                print("\n⚠️  Errors during investigation:")
                for err in result["errors"]:
                    print(f"  - {err}")
            else:
                print("\n✅ No errors during investigation")
            
            print("\n=== Error Handling Test: UNKNOWN FUND ===\n")
            
            result = await app.ainvoke({
                "fund_id": "FUND999",  # doesn't exist
                "question": "Check FUND999 status",
                "nav_report": "",
                "feed_report": "",
                "consumer_report": "",
                "knowledge_report": "",
                "final_summary": "",
                "errors": [],
                "tools": tools
            })

            print("\n" + "="*60)
            print("EXECUTIVE SUMMARY — UNKNOWN FUND")
            print("="*60)
            print(result["final_summary"])

            if result["errors"]:
                print("\n⚠️  Errors during investigation:")
            for err in result["errors"]:
                 print(f"  - {err}")
            else:
                 print("\n✅ No errors during investigation")

asyncio.run(main())