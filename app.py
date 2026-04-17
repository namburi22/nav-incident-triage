import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import asyncio
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import tool
from typing import TypedDict, List

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
    st.success("Agents: Ready")
    st.success("RAG: Connected")
    st.success("LangSmith: Tracing")
    st.divider()
    st.caption("Built with LangGraph + MCP + RAG")

# --- Tools ---
@tool
def get_fund_nav(fund_id: str) -> dict:
    """Get NAV status for a fund."""
    funds = {
        "FUND001": {"nav": 142.35, "status": "FAILED", "last_updated": "2026-04-14 08:00"},
        "FUND002": {"nav": 98.12, "status": "SUCCESS", "last_updated": "2026-04-14 08:05"},
        "FUND003": {"nav": 0.00, "status": "PENDING", "last_updated": "2026-04-14 07:45"},
    }
    return funds.get(fund_id, {"error": f"Fund {fund_id} not found"})

@tool
def get_feeds_for_fund(fund_id: str) -> list:
    """Get feed IDs for a fund."""
    mapping = {
        "FUND001": ["FEED_PRICE_01", "FEED_CORP_ACTION"],
        "FUND002": ["FEED_PRICE_02"],
        "FUND003": ["FEED_PRICE_01"],
    }
    return mapping.get(fund_id, [])

@tool
def check_feed_status(feed_id: str) -> dict:
    """Check feed status."""
    feeds = {
        "FEED_PRICE_01": {"status": "DOWN", "last_success": "2026-04-13 22:00"},
        "FEED_PRICE_02": {"status": "UP", "last_success": "2026-04-14 08:00"},
        "FEED_CORP_ACTION": {"status": "DELAYED", "last_success": "2026-04-14 06:00"},
    }
    return feeds.get(feed_id, {"error": f"Feed {feed_id} not found"})

@tool
def get_incident_history(fund_id: str) -> list:
    """Get incident history for a fund."""
    history = {
        "FUND001": [
            {"date": "2026-03-10", "issue": "Price feed down", "resolution": "Feed restarted", "duration_mins": 45},
            {"date": "2026-02-22", "issue": "Corporate action missing", "resolution": "Manual override", "duration_mins": 120},
        ],
        "FUND003": [
            {"date": "2026-04-01", "issue": "NAV timeout", "resolution": "Reprocessed", "duration_mins": 30}
        ],
    }
    return history.get(fund_id, [])

@tool
def get_impacted_consumers(fund_id: str) -> list:
    """Get downstream consumers for a fund."""
    consumers = {
        "FUND001": ["RetailPortal", "AdvisorDashboard", "RegulatoryReporter", "SettlementEngine"],
        "FUND002": ["RetailPortal"],
        "FUND003": ["AdvisorDashboard", "SettlementEngine"],
    }
    return consumers.get(fund_id, [])

# --- RAG Knowledge Base ---
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    documents = [
        Document(page_content="Incident: FEED_PRICE_01 Down. Date: 2026-03-10. Fund: FUND001. Root Cause: Network timeout between feed provider and ingestion service. Resolution: Feed ops team restarted ingestion service. Duration: 45 minutes. Impact: FUND001 NAV delayed.", metadata={"type": "incident"}),
        Document(page_content="Incident: FEED_PRICE_01 Intermittent. Date: 2026-01-15. Fund: FUND001 FUND003. Root Cause: Feed provider database under heavy load. Resolution: Switched to backup feed. Duration: 120 minutes.", metadata={"type": "incident"}),
        Document(page_content="Incident: Corporate Action Missing. Date: 2026-02-22. Fund: FUND001. Root Cause: Vendor did not process weekend announcements. Resolution: Manual override applied. Duration: 120 minutes. Impact: RegulatoryReporter flagged discrepancy.", metadata={"type": "incident"}),
        Document(page_content="Incident: NAV Calculation Timeout. Date: 2026-04-01. Fund: FUND003. Root Cause: Large number of corporate actions processed simultaneously. Resolution: Reprocessed with increased timeout. Duration: 30 minutes.", metadata={"type": "incident"}),
        Document(page_content="Incident: SettlementEngine Failure. Date: 2026-02-10. Fund: FUND001. Root Cause: Cascading failure from FEED_PRICE_01 outage causing late NAV. Resolution: Manual NAV publication. Duration: 120 minutes.", metadata={"type": "incident"}),
        Document(page_content="Incident: RegulatoryReporter Missing NAV. Date: 2025-10-15. Fund: FUND001 FUND003. Root Cause: NAV failure not detected before reporting window. Resolution: Amended report submitted. Duration: 240 minutes.", metadata={"type": "incident"}),
        Document(page_content="Playbook: FEED_PRICE_01 Recovery. Step 1: Check feed provider status page. Step 2: Restart ingestion service via feed ops dashboard. Step 3: If not restored in 15 minutes switch to backup feed FEED_PRICE_BACKUP. Step 4: Notify NAV calculation team. Typical resolution: 30-45 minutes. Escalation: If not resolved in 60 minutes escalate to feed provider account manager.", metadata={"type": "playbook"}),
        Document(page_content="Playbook: Corporate Action Recovery. Step 1: Check vendor portal for data availability. Step 2: If delay less than 30 minutes wait for auto catch-up. Step 3: Initiate manual data pull from vendor API. Step 4: Validate corporate action data completeness. Typical resolution: 60-120 minutes.", metadata={"type": "playbook"}),
    ]
    return FAISS.from_documents(documents, embeddings)

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
    steps: List[dict]

# --- RAG helper ---
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

# --- Agents ---
async def fund_agent(state, status_container):
    with status_container:
        st.write("🔍 **Fund Agent** — Investigating NAV status...")

    try:
        nav = get_fund_nav.invoke({"fund_id": state["fund_id"]})
        history = get_incident_history.invoke({"fund_id": state["fund_id"]})

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
    except Exception as e:
        state["nav_report"] = f"Fund data unavailable: {e}"
        state["steps"].append({"agent": "Fund Agent", "status": "⚠️ Error", "data": str(e)})

    with status_container:
        st.write("✅ **Fund Agent** — Done")
    return state

async def feed_agent(state, status_container):
    with status_container:
        st.write("📡 **Feed Agent** — Checking data feeds...")

    try:
        feed_ids = get_feeds_for_fund.invoke({"fund_id": state["fund_id"]})
        feed_statuses = []
        for feed_id in feed_ids:
            status = check_feed_status.invoke({"feed_id": feed_id})
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
    except Exception as e:
        state["feed_report"] = f"Feed data unavailable: {e}"
        state["steps"].append({"agent": "Feed Agent", "status": "⚠️ Error", "data": str(e)})

    with status_container:
        st.write("✅ **Feed Agent** — Done")
    return state

async def consumer_agent(state, status_container):
    with status_container:
        st.write("🔗 **Consumer Agent** — Checking downstream impact...")

    try:
        consumers = get_impacted_consumers.invoke({"fund_id": state["fund_id"]})

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
    except Exception as e:
        state["consumer_report"] = f"Consumer data unavailable: {e}"
        state["steps"].append({"agent": "Consumer Agent", "status": "⚠️ Error", "data": str(e)})

    with status_container:
        st.write("✅ **Consumer Agent** — Done")
    return state

async def knowledge_agent(state, status_container):
    with status_container:
        st.write("📚 **Knowledge Agent** — Searching historical incidents...")

    try:
        query = f"{state['fund_id']} feed failure NAV incident resolution playbook"
        context = query_knowledge_base(query, k=4)

        prompt = f"""
        You are a Knowledge Management specialist.
        Current Incident: {state['question']}
        Fund: {state['fund_id']}
        Historical Context: {context}
        Provide:
        - Similar past incidents and causes
        - Proven resolution steps from playbooks
        - Average resolution time based on history
        - Recurring patterns to watch for
        """
        response = llm.invoke(prompt)
        state["knowledge_report"] = response.content
        state["steps"].append({"agent": "Knowledge Agent", "status": "✅ Complete", "data": "RAG retrieved"})
    except Exception as e:
        state["knowledge_report"] = f"Knowledge base unavailable: {e}"
        state["steps"].append({"agent": "Knowledge Agent", "status": "⚠️ Error", "data": str(e)})

    with status_container:
        st.write("✅ **Knowledge Agent** — Done")
    return state

async def supervisor_agent(state, status_container):
    with status_container:
        st.write("👔 **Supervisor** — Synthesizing all reports...")

    try:
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
    except Exception as e:
        state["final_summary"] = f"Unable to generate summary: {e}"

    with status_container:
        st.write("✅ **Supervisor** — Done")
    return state

# --- Main runner ---
async def run_triage(fund_id, status_container):
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
    }

    # Check NAV status first
    nav = get_fund_nav.invoke({"fund_id": fund_id})
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
        tab1, tab2 = st.tabs(["🔬 Investigation (Engineers)", "📊 Executive Summary (Directors)"])

        with tab1:
            st.subheader("Live Investigation")
            status_container = st.container()

            with st.spinner("Running agentic investigation..."):
                result = asyncio.run(run_triage(fund_id, status_container))

            st.divider()

            if result["steps"]:
                st.subheader("Agent Results")
                for step in result["steps"]:
                    with st.expander(f"{step['agent']} — {step['status']}"):
                        st.json(step["data"])

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

        with tab2:
            st.subheader("Executive Incident Summary")
            st.markdown(result["final_summary"])

            st.download_button(
                label="📥 Download Report",
                data=result["final_summary"],
                file_name=f"{fund_id}_incident_report.md",
                mime="text/markdown"
            )