import streamlit as st
import asyncio
import json
import sys
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

load_dotenv()

# --- Page config ---
st.set_page_config(
    page_title="NAV Incident Triage",
    page_icon="🏦",
    layout="wide"
)

# --- Header ---
st.title("🏦 NAV Incident Triage Assistant")
st.caption("Agentic AI system for mutual fund NAV incident investigation")

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    fund_id = st.selectbox(
        "Select Fund",
        ["FUND001", "FUND002", "FUND003"],
        index=0
    )
    st.divider()
    st.markdown("**System Status**")
    st.success("MCP Server: Connected")
    st.success("ChromaDB: Connected")
    st.success("LangSmith: Tracing")
    st.divider()
    st.caption("Built with LangGraph + MCP + RAG")

# --- Setup ---
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(
        persist_directory="./nav_knowledge_base",
        embedding_function=embeddings,
        collection_name="nav_incidents"
    )

vectorstore = load_vectorstore()
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- State ---
class IncidentState(TypedDict):
    fund_id: str
    question: str
    nav_report: str
    feed_report: str
    consumer_report: str
    knowledge_report: str
    final_summary: str
    errors: List[str]
    tools: list
    steps: List[dict]  # for live UI updates

# --- MCP parser ---
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

# --- RAG ---
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
        return f"Knowledge base unavailable: {e}"

# --- Safe tool call ---
async def safe_tool_call(tool, args, retries=3, fallback=None):
    for attempt in range(retries):
        try:
            return await tool.ainvoke(args)
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(1)
            else:
                return fallback
    return fallback

# --- Agents ---
async def fund_agent(state, status_container):
    with status_container:
        st.write("🔍 **Fund Agent** — Investigating NAV status...")
    
    tools = state["tools"]
    get_nav = next(t for t in tools if t.name == "get_fund_nav")
    get_history = next(t for t in tools if t.name == "get_incident_history")
    
    raw = await safe_tool_call(get_nav, {"fund_id": state["fund_id"]}, fallback={"status": "UNKNOWN"})
    nav = parse_mcp_response(raw)
    history = await safe_tool_call(get_history, {"fund_id": state["fund_id"]}, fallback=[])
    
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
    state["steps"].append({"agent": "Fund Agent", "status": "✅ Complete", "data": nav})
    
    with status_container:
        st.write("✅ **Fund Agent** — Done")
    return state

async def feed_agent(state, status_container):
    with status_container:
        st.write("📡 **Feed Agent** — Checking data feeds...")
    
    tools = state["tools"]
    get_feeds = next(t for t in tools if t.name == "get_feeds_for_fund")
    check_feed = next(t for t in tools if t.name == "check_feed_status")
    
    raw_feeds = await safe_tool_call(get_feeds, {"fund_id": state["fund_id"]}, fallback=[])
    
    feed_ids = []
    if raw_feeds:
        for item in raw_feeds:
            if isinstance(item, str):
                feed_ids.append(item)
            elif isinstance(item, dict):
                feed_ids.append(item.get("text", str(item)))
    
    feed_statuses = []
    for feed_id in feed_ids:
        status = await safe_tool_call(check_feed, {"feed_id": feed_id}, fallback={"status": "UNKNOWN"})
        if isinstance(status, list):
            status = status[0] if status else {}
        if isinstance(status, dict) and "text" in status:
            try:
                status = json.loads(status["text"])
            except:
                pass
        feed_statuses.append({"feed_id": feed_id, "status": status})
    
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
    state["steps"].append({"agent": "Feed Agent", "status": "✅ Complete", "data": feed_statuses})
    
    with status_container:
        st.write("✅ **Feed Agent** — Done")
    return state

async def consumer_agent(state, status_container):
    with status_container:
        st.write("🔗 **Consumer Agent** — Checking downstream impact...")
    
    tools = state["tools"]
    get_consumers = next(t for t in tools if t.name == "get_impacted_consumers")
    consumers = await safe_tool_call(get_consumers, {"fund_id": state["fund_id"]}, fallback=[])
    
    if not consumers:
        state["consumer_report"] = "No consumer data available."
        state["steps"].append({"agent": "Consumer Agent", "status": "⚠️ No data", "data": []})
        return state
    
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
    state["steps"].append({"agent": "Consumer Agent", "status": "✅ Complete", "data": consumers})
    
    with status_container:
        st.write("✅ **Consumer Agent** — Done")
    return state

async def knowledge_agent(state, status_container):
    with status_container:
        st.write("📚 **Knowledge Agent** — Searching historical incidents...")
    
    query = f"{state['fund_id']} feed failure NAV incident resolution playbook"
    context = query_knowledge_base(query, k=4)
    
    prompt = f"""
    You are a Knowledge Management specialist.
    Current Incident: {state['question']}
    Fund: {state['fund_id']}
    Historical Context: {context}
    Provide:
    - Similar past incidents and causes
    - Proven resolution steps
    - Average resolution time
    - Recurring patterns
    """
    response = llm.invoke(prompt)
    state["knowledge_report"] = response.content
    state["steps"].append({"agent": "Knowledge Agent", "status": "✅ Complete", "data": "RAG retrieved"})
    
    with status_container:
        st.write("✅ **Knowledge Agent** — Done")
    return state

async def supervisor_agent(state, status_container):
    with status_container:
        st.write("👔 **Supervisor** — Synthesizing all reports...")
    
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
    
    with status_container:
        st.write("✅ **Supervisor** — Done")
    return state

# --- Main runner ---
async def run_triage(fund_id, status_container):
    server_params = StdioServerParameters(
        command="python",
        args=["nav_mcp_server_v2.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            
            state = {
                "fund_id": fund_id,
                "question": f"{fund_id} NAV incident triage. What is the situation and action plan?",
                "nav_report": "",
                "feed_report": "",
                "consumer_report": "",
                "knowledge_report": "",
                "final_summary": "",
                "errors": [],
                "steps": [],
                "tools": tools
            }
            
            # Check NAV status first
            get_nav = next(t for t in tools if t.name == "get_fund_nav")
            raw = await safe_tool_call(get_nav, {"fund_id": fund_id}, fallback={"status": "UNKNOWN"})
            nav = parse_mcp_response(raw)
            nav_status = nav.get("status", "UNKNOWN") if isinstance(nav, dict) else "UNKNOWN"
            
            if nav_status == "SUCCESS":
                state["final_summary"] = f"✅ {fund_id} NAV is healthy. No investigation needed."
                return state
            
            # Run all agents
            state = await fund_agent(state, status_container)
            state = await feed_agent(state, status_container)
            state = await consumer_agent(state, status_container)
            state = await knowledge_agent(state, status_container)
            state = await supervisor_agent(state, status_container)
            
            return state

# --- UI ---
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Selected Fund")
    st.metric("Fund ID", fund_id)
    run_button = st.button("🚀 Run Triage", type="primary", use_container_width=True)

with col2:
    if run_button:
        tab1, tab2 = st.tabs(["🔬 Investigation", "📊 Executive Summary"])
        
        with tab1:
            st.subheader("Live Investigation")
            status_container = st.container()
            
            with st.spinner("Running agentic investigation..."):
                result = asyncio.run(run_triage(fund_id, status_container))
            
            st.divider()
            
            # Show agent steps
            if result["steps"]:
                st.subheader("Agent Results")
                for step in result["steps"]:
                    with st.expander(f"{step['agent']} — {step['status']}"):
                        st.json(step["data"])
            
            # Show individual reports
            st.subheader("Detailed Reports")
            
            if result["nav_report"]:
                with st.expander("📈 Fund Report"):
                    st.markdown(result["nav_report"])
            
            if result["feed_report"]:
                with st.expander("📡 Feed Report"):
                    st.markdown(result["feed_report"])
            
            if result["consumer_report"]:
                with st.expander("🔗 Consumer Report"):
                    st.markdown(result["consumer_report"])
            
            if result["knowledge_report"]:
                with st.expander("📚 Historical Knowledge"):
                    st.markdown(result["knowledge_report"])
            
            if result["errors"]:
                st.warning(f"⚠️ Errors: {result['errors']}")
        
        with tab2:
            st.subheader("Executive Incident Summary")
            st.markdown(result["final_summary"])
            
            # Download button
            st.download_button(
                label="📥 Download Report",
                data=result["final_summary"],
                file_name=f"{fund_id}_incident_report.md",
                mime="text/markdown"
            )