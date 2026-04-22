import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import asyncio
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.callbacks import BaseCallbackHandler
from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, Text
from sqlalchemy.orm import declarative_base, Session
from typing import TypedDict, List, Optional, Dict

load_dotenv()

# --- Cost Constants ---
COSTS = {
    "gpt-4o":      {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini": {"input": 0.15,  "output": 0.60},
}

# --- Fund Registry ---
FUND_REGISTRY = {
    "Equity": {
        "description": "US and International Equity Funds",
        "funds": {
            "FUND001": {"name": "US Large Cap Growth",   "aum_b": 2.4},
            "FUND002": {"name": "International Equity",  "aum_b": 1.1},
            "FUND003": {"name": "Small Cap Value",       "aum_b": 0.8},
        }
    },
    "Fixed Income": {
        "description": "Bond and Credit Funds",
        "funds": {
            "FUND004": {"name": "Investment Grade Bond", "aum_b": 3.2},
            "FUND005": {"name": "High Yield Credit",     "aum_b": 0.9},
            "FUND006": {"name": "Municipal Bond",        "aum_b": 1.5},
        }
    },
    "Money Market": {
        "description": "Cash and Liquidity Funds",
        "funds": {
            "FUND007": {"name": "Prime Money Market",    "aum_b": 5.1},
            "FUND008": {"name": "Government MMF",        "aum_b": 4.2},
        }
    },
    "ETF": {
        "description": "Exchange Traded Funds",
        "funds": {
            "FUND009": {"name": "S&P 500 ETF",           "aum_b": 8.5},
            "FUND010": {"name": "Bond ETF",              "aum_b": 2.1},
        }
    }
}

# --- NAV / Feed / Consumer Data ---
NAV_DATA = {
    "FUND001": {"status": "FAILED",  "nav": 142.35},
    "FUND002": {"status": "SUCCESS", "nav": 98.12},
    "FUND003": {"status": "PENDING", "nav": 0.00},
    "FUND004": {"status": "SUCCESS", "nav": 45.20},
    "FUND005": {"status": "FAILED",  "nav": 0.00},
    "FUND006": {"status": "SUCCESS", "nav": 112.50},
    "FUND007": {"status": "SUCCESS", "nav": 1.00},
    "FUND008": {"status": "SUCCESS", "nav": 1.00},
    "FUND009": {"status": "PENDING", "nav": 0.00},
    "FUND010": {"status": "SUCCESS", "nav": 55.30},
}

FEED_DATA = {
    "FUND001": ["FEED_PRICE_01", "FEED_CORP_ACTION"],
    "FUND002": ["FEED_PRICE_02"],
    "FUND003": ["FEED_PRICE_01"],
    "FUND004": ["FEED_BOND_01", "FEED_CREDIT_01"],
    "FUND005": ["FEED_BOND_01", "FEED_HY_01"],
    "FUND006": ["FEED_MUNI_01"],
    "FUND007": ["FEED_MM_01"],
    "FUND008": ["FEED_MM_01", "FEED_GOVT_01"],
    "FUND009": ["FEED_PRICE_01", "FEED_ETF_01"],
    "FUND010": ["FEED_BOND_01"],
}

FEED_STATUS = {
    "FEED_PRICE_01":   {"status": "DOWN",    "hours_down": 10},
    "FEED_PRICE_02":   {"status": "UP"},
    "FEED_CORP_ACTION":{"status": "DELAYED", "delay_mins": 120},
    "FEED_BOND_01":    {"status": "DOWN",    "hours_down": 2},
    "FEED_CREDIT_01":  {"status": "UP"},
    "FEED_HY_01":      {"status": "UP"},
    "FEED_MUNI_01":    {"status": "UP"},
    "FEED_MM_01":      {"status": "UP"},
    "FEED_GOVT_01":    {"status": "UP"},
    "FEED_ETF_01":     {"status": "UP"},
}

CONSUMER_DATA = {
    "FUND001": ["RetailPortal", "AdvisorDashboard", "RegulatoryReporter", "SettlementEngine"],
    "FUND002": ["RetailPortal"],
    "FUND003": ["AdvisorDashboard", "SettlementEngine"],
    "FUND004": ["SettlementEngine", "RiskAnalytics"],
    "FUND005": ["RiskAnalytics"],
    "FUND006": ["RegulatoryReporter", "AdvisorDashboard"],
    "FUND007": ["RetailPortal", "SettlementEngine"],
    "FUND008": ["RetailPortal"],
    "FUND009": ["RetailPortal", "AdvisorDashboard", "RegulatoryReporter"],
    "FUND010": ["AdvisorDashboard"],
}

# --- Token Tracker ---
class TokenTracker(BaseCallbackHandler):
    def __init__(self, agent_name: str, model: str):
        self.agent_name = agent_name
        self.model = model
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def on_llm_end(self, response, **kwargs):
        usage = response.llm_output.get("token_usage", {})
        self.total_input_tokens += usage.get("prompt_tokens", 0)
        self.total_output_tokens += usage.get("completion_tokens", 0)

    @property
    def total_cost(self) -> float:
        rates = COSTS.get(self.model, COSTS["gpt-4o"])
        return (
            (self.total_input_tokens / 1_000_000) * rates["input"] +
            (self.total_output_tokens / 1_000_000) * rates["output"]
        )

    def summary(self) -> dict:
        return {
            "agent": self.agent_name,
            "model": self.model,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "cost_usd": round(self.total_cost, 6),
        }

# --- Memory Database ---
Base = declarative_base()
engine = create_engine("sqlite:///nav_memory.db", echo=False)

class IncidentMemory(Base):
    __tablename__ = "incident_memories"
    id = Column(Integer, primary_key=True, autoincrement=True)
    fund_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.now)
    nav_status = Column(String)
    severity = Column(String)
    feeds_down = Column(String)
    consumers_impacted = Column(String)
    resolution_taken = Column(String)
    resolution_time_mins = Column(Float, nullable=True)
    key_lesson = Column(Text)
    sre_decision = Column(String, nullable=True)

Base.metadata.create_all(engine)

def save_incident_memory(incident: dict):
    with Session(engine) as session:
        memory = IncidentMemory(
            fund_id=incident["fund_id"],
            timestamp=datetime.now(),
            nav_status=incident.get("nav_status", "UNKNOWN"),
            severity=incident.get("severity", "UNKNOWN"),
            feeds_down=json.dumps(incident.get("feeds_down", [])),
            consumers_impacted=json.dumps(incident.get("consumers_impacted", [])),
            resolution_taken=incident.get("resolution_taken", ""),
            resolution_time_mins=incident.get("resolution_time_mins"),
            key_lesson=incident.get("key_lesson", ""),
            sre_decision=incident.get("sre_decision")
        )
        session.add(memory)
        session.commit()

def get_incident_memories(fund_id: str, limit: int = 10) -> List[dict]:
    with Session(engine) as session:
        memories = (
            session.query(IncidentMemory)
            .filter(IncidentMemory.fund_id == fund_id)
            .order_by(IncidentMemory.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "timestamp": m.timestamp.strftime("%Y-%m-%d %H:%M"),
                "nav_status": m.nav_status,
                "severity": m.severity,
                "feeds_down": json.loads(m.feeds_down),
                "consumers_impacted": json.loads(m.consumers_impacted),
                "resolution_taken": m.resolution_taken,
                "resolution_time_mins": m.resolution_time_mins,
                "key_lesson": m.key_lesson,
                "sre_decision": m.sre_decision,
            }
            for m in memories
        ]

def get_all_memories() -> List[dict]:
    with Session(engine) as session:
        memories = (
            session.query(IncidentMemory)
            .order_by(IncidentMemory.timestamp.desc())
            .all()
        )
        return [
            {
                "timestamp": m.timestamp.strftime("%Y-%m-%d %H:%M"),
                "fund_id": m.fund_id,
                "nav_status": m.nav_status,
                "severity": m.severity,
                "feeds_down": json.loads(m.feeds_down),
                "resolution_taken": m.resolution_taken,
                "resolution_time_mins": m.resolution_time_mins,
                "sre_decision": m.sre_decision,
                "key_lesson": m.key_lesson,
            }
            for m in memories
        ]

# --- Page config ---
st.set_page_config(
    page_title="NAV Incident Triage",
    page_icon="🏦",
    layout="wide"
)

# --- LLMs ---
llm_powerful  = ChatOpenAI(model="gpt-4o",      temperature=0)
llm_efficient = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- Portfolio Triage Functions ---
async def run_fund_triage_portfolio(fund_id: str, fund_meta: dict) -> dict:
    """Triage a single fund for portfolio scan."""
    nav = NAV_DATA.get(fund_id, {"status": "UNKNOWN", "nav": 0})
    feed_ids = FEED_DATA.get(fund_id, [])
    feed_statuses = {fid: FEED_STATUS.get(fid, {"status": "UNKNOWN"}) for fid in feed_ids}
    feeds_down = [fid for fid, fs in feed_statuses.items() if fs.get("status") == "DOWN"]
    consumers = CONSUMER_DATA.get(fund_id, [])
    nav_status = nav.get("status", "UNKNOWN")
    high_risk = [c for c in consumers if c in ["RegulatoryReporter", "SettlementEngine"]]

    if nav_status == "FAILED" and (feeds_down or high_risk):
        severity = "CRITICAL"
    elif nav_status in ["FAILED", "PENDING"] or feeds_down:
        severity = "WARNING"
    else:
        severity = "HEALTHY"

    if severity != "HEALTHY":
        response = await llm_efficient.ainvoke(
            f"Fund {fund_id} — {fund_meta['name']}. NAV: {nav_status}. Feeds down: {feeds_down}. "
            f"Consumers: {consumers}. One sentence situation summary."
        )
        summary = response.content
    else:
        summary = f"{fund_meta['name']} NAV healthy — no action required"

    return {
        "fund_id": fund_id,
        "fund_name": fund_meta["name"],
        "aum_b": fund_meta["aum_b"],
        "nav_status": nav_status,
        "nav_value": nav.get("nav", 0),
        "severity": severity,
        "feeds_down": feeds_down,
        "consumers": consumers,
        "summary": summary,
    }

async def run_category_triage(category: str, category_meta: dict) -> dict:
    """Triage all funds in a category in parallel."""
    fund_results = await asyncio.gather(*[
        run_fund_triage_portfolio(fund_id, fund_meta)
        for fund_id, fund_meta in category_meta["funds"].items()
    ])
    fund_results = list(fund_results)

    total    = len(fund_results)
    healthy  = sum(1 for f in fund_results if f["severity"] == "HEALTHY")
    warning  = sum(1 for f in fund_results if f["severity"] == "WARNING")
    critical = sum(1 for f in fund_results if f["severity"] == "CRITICAL")
    health_score = (healthy / total) * 100
    total_aum    = sum(f["aum_b"] for f in fund_results)
    aum_at_risk  = sum(f["aum_b"] for f in fund_results if f["severity"] != "HEALTHY")

    all_feeds_down = []
    for f in fund_results:
        all_feeds_down.extend(f["feeds_down"])
    shared_feeds = list(set([f for f in all_feeds_down if all_feeds_down.count(f) > 1]))

    if critical > 0:
        risk_level = "CRITICAL"
    elif warning > 0:
        risk_level = "WARNING"
    else:
        risk_level = "HEALTHY"

    priority_score = (critical * 3 + warning * 1) * (aum_at_risk / max(total_aum, 1)) * 100

    if risk_level != "HEALTHY":
        critical_funds = [f["fund_id"] for f in fund_results if f["severity"] == "CRITICAL"]
        warning_funds  = [f["fund_id"] for f in fund_results if f["severity"] == "WARNING"]
        response = await llm_powerful.ainvoke(
            f"Category: {category}. Health: {health_score:.0f}%. "
            f"Critical: {critical_funds}. Warning: {warning_funds}. "
            f"Shared feeds down: {shared_feeds}. AUM at risk: ${aum_at_risk:.1f}B. "
            f"Two sentence situation and recommended action."
        )
        category_summary = response.content
    else:
        category_summary = f"All {total} {category} funds healthy — no action required"

    return {
        "category": category,
        "description": category_meta["description"],
        "total_funds": total,
        "healthy": healthy,
        "warning": warning,
        "critical": critical,
        "health_score": health_score,
        "total_aum_b": total_aum,
        "aum_at_risk_b": aum_at_risk,
        "priority_score": priority_score,
        "fund_results": fund_results,
        "shared_feeds_down": shared_feeds,
        "category_summary": category_summary,
        "risk_level": risk_level,
    }

async def run_portfolio_scan() -> dict:
    """Scan entire portfolio — all categories in parallel."""
    start = time.time()

    category_results = await asyncio.gather(*[
        run_category_triage(category, meta)
        for category, meta in FUND_REGISTRY.items()
    ])
    category_results = list(category_results)

    total_funds  = sum(c["total_funds"]   for c in category_results)
    healthy      = sum(c["healthy"]       for c in category_results)
    warning      = sum(c["warning"]       for c in category_results)
    critical     = sum(c["critical"]      for c in category_results)
    total_aum    = sum(c["total_aum_b"]   for c in category_results)
    aum_at_risk  = sum(c["aum_at_risk_b"] for c in category_results)
    overall_health = (healthy / total_funds) * 100

    ranked = sorted(category_results, key=lambda x: x["priority_score"], reverse=True)
    highest_priority = ranked[0]["category"] if ranked[0]["priority_score"] > 0 else "None"

    critical_cats = [c["category"] for c in category_results if c["risk_level"] == "CRITICAL"]
    warning_cats  = [c["category"] for c in category_results if c["risk_level"] == "WARNING"]

    response = await llm_powerful.ainvoke(
        f"Portfolio Health: {overall_health:.0f}%. Funds: {total_funds} total, "
        f"{healthy} healthy, {warning} warning, {critical} critical. "
        f"AUM: ${total_aum:.1f}B total, ${aum_at_risk:.1f}B at risk. "
        f"Critical categories: {critical_cats}. Warning: {warning_cats}. "
        f"Highest priority: {highest_priority}. "
        f"Executive summary for VP Engineering — 3 sentences max."
    )

    elapsed = time.time() - start

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_funds": total_funds,
        "healthy": healthy,
        "warning": warning,
        "critical": critical,
        "overall_health": overall_health,
        "total_aum_b": total_aum,
        "aum_at_risk_b": aum_at_risk,
        "categories": ranked,
        "highest_priority_category": highest_priority,
        "portfolio_summary": response.content,
        "scan_time_secs": round(elapsed, 2),
    }

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
    st.success("Memory: Connected")
    st.success("LangSmith: Tracing")
    st.divider()

    past = get_incident_memories(fund_id, limit=3)
    if past:
        st.markdown(f"**Last {len(past)} incidents for {fund_id}**")
        for p in past:
            color = "🔴" if p["severity"] == "CRITICAL" else "🟡"
            st.caption(f"{color} {p['timestamp']} — {p['nav_status']}")
    else:
        st.caption(f"No past incidents for {fund_id}")

    st.divider()
    st.caption("Built with LangGraph + MCP + RAG")

# --- Header ---
st.title("🏦 NAV Incident Triage Assistant")
st.caption("Agentic AI system for mutual fund NAV incident investigation")

# --- RAG Knowledge Base ---
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    documents = [
        Document(page_content="Incident: FEED_PRICE_01 Down. Date: 2026-03-10. Fund: FUND001. Root Cause: Network timeout. Resolution: Feed ops team restarted ingestion service. Duration: 45 minutes.", metadata={"type": "incident"}),
        Document(page_content="Incident: FEED_PRICE_01 Intermittent. Date: 2026-01-15. Fund: FUND001 FUND003. Root Cause: Feed provider database under heavy load. Resolution: Switched to backup feed. Duration: 120 minutes.", metadata={"type": "incident"}),
        Document(page_content="Incident: Corporate Action Missing. Date: 2026-02-22. Fund: FUND001. Root Cause: Vendor did not process weekend announcements. Resolution: Manual override applied. Duration: 120 minutes.", metadata={"type": "incident"}),
        Document(page_content="Incident: NAV Calculation Timeout. Date: 2026-04-01. Fund: FUND003. Root Cause: Large number of corporate actions. Resolution: Reprocessed with increased timeout. Duration: 30 minutes.", metadata={"type": "incident"}),
        Document(page_content="Incident: SettlementEngine Failure. Date: 2026-02-10. Fund: FUND001. Root Cause: Cascading failure from FEED_PRICE_01. Resolution: Manual NAV publication. Duration: 120 minutes.", metadata={"type": "incident"}),
        Document(page_content="Incident: RegulatoryReporter Missing NAV. Date: 2025-10-15. Fund: FUND001 FUND003. Root Cause: NAV failure not detected before reporting window. Resolution: Amended report submitted. Duration: 240 minutes.", metadata={"type": "incident"}),
        Document(page_content="Playbook: FEED_PRICE_01 Recovery. Step 1: Check feed provider status. Step 2: Restart ingestion service. Step 3: Switch to backup feed after 15 minutes. Step 4: Notify NAV team. Typical resolution: 30-45 minutes.", metadata={"type": "playbook"}),
        Document(page_content="Playbook: Corporate Action Recovery. Step 1: Check vendor portal. Step 2: Wait 30 minutes for auto catch-up. Step 3: Manual data pull. Step 4: Validate completeness. Typical resolution: 60-120 minutes.", metadata={"type": "playbook"}),
    ]
    return FAISS.from_documents(documents, embeddings)

vectorstore = load_vectorstore()

# --- Tools ---
@tool
def get_fund_nav(fund_id: str) -> dict:
    """Get NAV status for a fund."""
    return {
        "FUND001": {"nav": 142.35, "status": "FAILED", "last_updated": "2026-04-14 08:00"},
        "FUND002": {"nav": 98.12,  "status": "SUCCESS", "last_updated": "2026-04-14 08:05"},
        "FUND003": {"nav": 0.00,   "status": "PENDING", "last_updated": "2026-04-14 07:45"},
    }.get(fund_id, {"error": f"Fund {fund_id} not found"})

@tool
def get_feeds_for_fund(fund_id: str) -> list:
    """Get feed IDs for a fund."""
    return {
        "FUND001": ["FEED_PRICE_01", "FEED_CORP_ACTION"],
        "FUND002": ["FEED_PRICE_02"],
        "FUND003": ["FEED_PRICE_01"],
    }.get(fund_id, [])

@tool
def check_feed_status(feed_id: str) -> dict:
    """Check feed status."""
    return {
        "FEED_PRICE_01": {"status": "DOWN",    "last_success": "2026-04-13 22:00", "hours_down": 10},
        "FEED_PRICE_02": {"status": "UP",      "last_success": "2026-04-14 08:00"},
        "FEED_CORP_ACTION": {"status": "DELAYED", "last_success": "2026-04-14 06:00", "delay_mins": 120},
    }.get(feed_id, {"error": f"Feed {feed_id} not found"})

@tool
def get_incident_history(fund_id: str) -> list:
    """Get incident history for a fund."""
    return {
        "FUND001": [
            {"date": "2026-03-10", "issue": "Price feed down", "resolution": "Feed restarted", "duration_mins": 45},
            {"date": "2026-02-22", "issue": "Corporate action missing", "resolution": "Manual override", "duration_mins": 120},
        ],
        "FUND003": [
            {"date": "2026-04-01", "issue": "NAV timeout", "resolution": "Reprocessed", "duration_mins": 30}
        ],
    }.get(fund_id, [])

@tool
def get_impacted_consumers(fund_id: str) -> list:
    """Get downstream consumers for a fund."""
    return {
        "FUND001": ["RetailPortal", "AdvisorDashboard", "RegulatoryReporter", "SettlementEngine"],
        "FUND002": ["RetailPortal"],
        "FUND003": ["AdvisorDashboard", "SettlementEngine"],
    }.get(fund_id, [])

# --- State ---
class IncidentState(TypedDict):
    fund_id: str
    question: str
    nav_report: str
    feed_report: str
    consumer_report: str
    knowledge_report: str
    final_summary: str
    severity: str
    severity_reason: str
    feeds_down: List[str]
    consumers: List[str]
    errors: List[str]
    steps: List[dict]
    cost_tracker: List[dict]
    sre_decision: Optional[str]
    sre_rationale: Optional[str]

# --- RAG helper ---
def query_knowledge_base(query: str, k: int = 3) -> str:
    try:
        results = vectorstore.similarity_search(query, k=k)
        if not results:
            return "No relevant historical incidents found."
        context = ""
        for i, doc in enumerate(results, 1):
            context += f"\n--- Result {i} ---\n{doc.page_content.strip()}\n"
        return context
    except Exception as e:
        return f"Knowledge base unavailable: {e}"

# --- Fund-level Agents ---
async def fund_agent(state, status_container):
    with status_container:
        st.write("🔍 **Fund Agent** — Investigating NAV status...")
    tracker = TokenTracker("Fund Agent", "gpt-4o")
    llm_t = llm_powerful.with_config({"callbacks": [tracker]})
    try:
        nav = get_fund_nav.invoke({"fund_id": state["fund_id"]})
        history = get_incident_history.invoke({"fund_id": state["fund_id"]})
        response = llm_t.invoke(f"""
        You are a Fund NAV specialist.
        Fund: {state['fund_id']} | NAV: {nav} | History: {history}
        Provide a 3-bullet fund status report covering health, patterns, and risk.
        """)
        state["nav_report"] = response.content
        state["steps"].append({"agent": "Fund Agent", "status": "✅ Complete", "data": nav})
    except Exception as e:
        state["nav_report"] = f"Fund data unavailable: {e}"
        state["steps"].append({"agent": "Fund Agent", "status": "⚠️ Error", "data": str(e)})
    state["cost_tracker"].append(tracker.summary())
    with status_container:
        st.write("✅ **Fund Agent** — Done")
    return state

async def feed_agent(state, status_container):
    with status_container:
        st.write("📡 **Feed Agent** — Checking data feeds...")
    tracker = TokenTracker("Feed Agent", "gpt-4o")
    llm_t = llm_powerful.with_config({"callbacks": [tracker]})
    try:
        feed_ids = get_feeds_for_fund.invoke({"fund_id": state["fund_id"]})
        feed_statuses = []
        feeds_down = []
        for feed_id in feed_ids:
            status = check_feed_status.invoke({"feed_id": feed_id})
            feed_statuses.append({"feed_id": feed_id, **status})
            if status.get("status") == "DOWN":
                feeds_down.append(feed_id)
        state["feeds_down"] = feeds_down
        response = llm_t.invoke(f"""
        You are a Data Feed specialist.
        Fund: {state['fund_id']} | Feeds: {feed_statuses}
        Provide a 3-bullet feed health report covering failures, severity, and NAV impact.
        """)
        state["feed_report"] = response.content
        state["steps"].append({"agent": "Feed Agent", "status": "✅ Complete", "data": feed_statuses})
    except Exception as e:
        state["feed_report"] = f"Feed data unavailable: {e}"
        state["feeds_down"] = []
        state["steps"].append({"agent": "Feed Agent", "status": "⚠️ Error", "data": str(e)})
    state["cost_tracker"].append(tracker.summary())
    with status_container:
        st.write("✅ **Feed Agent** — Done")
    return state

async def consumer_agent(state, status_container):
    with status_container:
        st.write("🔗 **Consumer Agent** — Checking downstream impact...")
    tracker = TokenTracker("Consumer Agent", "gpt-4o-mini")
    llm_t = llm_efficient.with_config({"callbacks": [tracker]})
    try:
        consumers = get_impacted_consumers.invoke({"fund_id": state["fund_id"]})
        state["consumers"] = consumers
        if not consumers:
            state["consumer_report"] = "No consumer data available."
            state["steps"].append({"agent": "Consumer Agent", "status": "⚠️ No data", "data": []})
            state["cost_tracker"].append(tracker.summary())
            return state
        response = llm_t.invoke(f"""
        You are a Downstream Impact specialist.
        Fund: {state['fund_id']} | Consumers: {consumers}
        Flag RegulatoryReporter and SettlementEngine as HIGH RISK.
        Provide a 3-bullet impact report.
        """)
        state["consumer_report"] = response.content
        state["steps"].append({"agent": "Consumer Agent", "status": "✅ Complete", "data": consumers})
    except Exception as e:
        state["consumer_report"] = f"Consumer data unavailable: {e}"
        state["consumers"] = []
        state["steps"].append({"agent": "Consumer Agent", "status": "⚠️ Error", "data": str(e)})
    state["cost_tracker"].append(tracker.summary())
    with status_container:
        st.write("✅ **Consumer Agent** — Done")
    return state

async def knowledge_agent(state, status_container):
    with status_container:
        st.write("📚 **Knowledge Agent** — Searching historical incidents...")
    tracker = TokenTracker("Knowledge Agent", "gpt-4o")
    llm_t = llm_powerful.with_config({"callbacks": [tracker]})
    try:
        query = f"{state['fund_id']} feed failure NAV incident resolution playbook"
        context = query_knowledge_base(query, k=4)
        response = llm_t.invoke(f"""
        You are a Knowledge Management specialist.
        Incident: {state['question']} | Fund: {state['fund_id']}
        Historical Context: {context}
        Provide: similar past incidents, proven resolution steps, avg resolution time, recurring patterns.
        """)
        state["knowledge_report"] = response.content
        state["steps"].append({"agent": "Knowledge Agent", "status": "✅ Complete", "data": "RAG retrieved"})
    except Exception as e:
        state["knowledge_report"] = f"Knowledge base unavailable: {e}"
        state["steps"].append({"agent": "Knowledge Agent", "status": "⚠️ Error", "data": str(e)})
    state["cost_tracker"].append(tracker.summary())
    with status_container:
        st.write("✅ **Knowledge Agent** — Done")
    return state

async def severity_agent(state, status_container):
    with status_container:
        st.write("⚖️ **Severity Agent** — Classifying incident...")
    tracker = TokenTracker("Severity Agent", "gpt-4o-mini")
    llm_t = llm_efficient.with_config({"callbacks": [tracker]})
    try:
        response = llm_t.invoke(f"""
        Classify severity as CRITICAL or STANDARD.
        Fund: {state['fund_id']}
        NAV Report: {state['nav_report']}
        Feed Report: {state['feed_report']}
        Consumers: {state.get('consumers', [])}
        CRITICAL if: NAV FAILED + feed down + 3+ consumers
        CRITICAL if: RegulatoryReporter or SettlementEngine impacted
        STANDARD: everything else
        Respond JSON only no markdown:
        {{"severity": "CRITICAL", "reason": "one sentence"}}
        """)
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content.strip())
        state["severity"] = result.get("severity", "CRITICAL")
        state["severity_reason"] = result.get("reason", "")
    except Exception as e:
        state["severity"] = "CRITICAL"
        state["severity_reason"] = "Parse failed — defaulting to CRITICAL"
    state["cost_tracker"].append(tracker.summary())
    with status_container:
        st.write(f"✅ **Severity Agent** — {state['severity']}")
    return state

async def supervisor_agent(state, status_container):
    with status_container:
        st.write("👔 **Supervisor** — Synthesizing all reports...")
    tracker = TokenTracker("Supervisor", "gpt-4o")
    llm_t = llm_powerful.with_config({"callbacks": [tracker]})
    try:
        past = get_incident_memories(state["fund_id"], limit=3)
        memory_context = ""
        if past:
            memory_context = f"\nPast {len(past)} incidents:\n"
            for p in past:
                memory_context += f"- {p['timestamp']}: {p['nav_status']}, resolved in {p['resolution_time_mins']} min. Lesson: {p['key_lesson']}\n"
        response = llm_t.invoke(f"""
        You are an Incident Management Supervisor.
        Question: {state['question']}
        Fund Report: {state['nav_report']}
        Feed Report: {state['feed_report']}
        Consumer Report: {state['consumer_report']}
        Knowledge Report: {state['knowledge_report']}
        {memory_context}
        Synthesize into final executive summary:
        1. Situation — what happened
        2. Impact — who is affected and severity
        3. Root cause — why it happened
        4. Action plan — specific steps with owners
        5. ETA — based on historical resolution times
        """)
        state["final_summary"] = response.content
    except Exception as e:
        state["final_summary"] = f"Unable to generate summary: {e}"
    state["cost_tracker"].append(tracker.summary())
    with status_container:
        st.write("✅ **Supervisor** — Done")
    return state

# --- Main fund triage runner ---
async def run_triage(fund_id, status_container):
    state = {
        "fund_id": fund_id,
        "question": f"{fund_id} NAV incident triage. What is the situation and action plan?",
        "nav_report": "", "feed_report": "", "consumer_report": "",
        "knowledge_report": "", "final_summary": "",
        "severity": "", "severity_reason": "",
        "feeds_down": [], "consumers": [],
        "errors": [], "steps": [], "cost_tracker": [],
        "sre_decision": None, "sre_rationale": None,
    }
    nav = get_fund_nav.invoke({"fund_id": fund_id})
    nav_status = nav.get("status", "UNKNOWN") if isinstance(nav, dict) else "UNKNOWN"
    if nav_status == "SUCCESS":
        state["final_summary"] = f"✅ {fund_id} NAV is healthy. No investigation needed."
        state["severity"] = "STANDARD"
        return state
    state = await fund_agent(state, status_container)
    state = await feed_agent(state, status_container)
    state = await consumer_agent(state, status_container)
    state = await knowledge_agent(state, status_container)
    state = await severity_agent(state, status_container)
    state = await supervisor_agent(state, status_container)
    return state

# ═══════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Selected Fund")
    nav_data = get_fund_nav.invoke({"fund_id": fund_id})
    nav_status = nav_data.get("status", "UNKNOWN")
    status_color = {"FAILED": "🔴", "PENDING": "🟡", "SUCCESS": "🟢"}.get(nav_status, "⚪")
    st.metric("Fund ID", fund_id)
    st.metric("NAV Status", f"{status_color} {nav_status}")
    st.metric("NAV Value", f"${nav_data.get('nav', 0):.2f}")

    st.divider()
    run_button = st.button("🚀 Run Fund Triage", type="primary", use_container_width=True)
    portfolio_button = st.button("🗂️ Scan Portfolio", type="secondary", use_container_width=True)

with col2:

    # --- Run Fund Triage ---
    if run_button:
        if "sre_decision" in st.session_state:
            del st.session_state["sre_decision"]
        if "sre_rationale" in st.session_state:
            del st.session_state["sre_rationale"]
        status_container = st.container()
        with st.spinner("Running agentic investigation..."):
            result = asyncio.run(run_triage(fund_id, status_container))
        st.session_state["triage_result"] = result
        st.session_state["triage_fund"] = fund_id
        st.session_state["active_view"] = "fund"

    # --- Run Portfolio Scan ---
    if portfolio_button:
        with st.spinner("🗂️ Scanning all fund categories in parallel..."):
            portfolio_result = asyncio.run(run_portfolio_scan())
        st.session_state["portfolio_result"] = portfolio_result
        st.session_state["active_view"] = "portfolio"

    # ── PORTFOLIO VIEW ──────────────────────────────────────────
    if st.session_state.get("active_view") == "portfolio" and "portfolio_result" in st.session_state:
        pr = st.session_state["portfolio_result"]

        st.subheader("🗂️ Portfolio Health Dashboard")
        st.caption(f"Scanned at {pr['timestamp']} — completed in {pr['scan_time_secs']}s")

        # Top metrics
        h_color = "normal" if pr["overall_health"] >= 80 else "inverse"
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Overall Health", f"{pr['overall_health']:.0f}%")
        with m2:
            st.metric("Total AUM", f"${pr['total_aum_b']:.1f}B")
        with m3:
            st.metric("AUM at Risk", f"${pr['aum_at_risk_b']:.1f}B")
        with m4:
            st.metric("Critical Funds", pr["critical"])
        with m5:
            st.metric("Warning Funds", pr["warning"])

        st.divider()

        # Category grid
        st.subheader("Category Health")
        cols = st.columns(len(pr["categories"]))

        for i, cat in enumerate(pr["categories"]):
            with cols[i]:
                if cat["risk_level"] == "CRITICAL":
                    st.error(f"🔴 {cat['category']}")
                elif cat["risk_level"] == "WARNING":
                    st.warning(f"🟡 {cat['category']}")
                else:
                    st.success(f"🟢 {cat['category']}")

                st.metric("Health", f"{cat['health_score']:.0f}%")
                st.metric("AUM", f"${cat['total_aum_b']:.1f}B")
                st.metric("At Risk", f"${cat['aum_at_risk_b']:.1f}B")

                fund_summary = f"{cat['healthy']}✅ {cat['warning']}🟡 {cat['critical']}🔴"
                st.caption(fund_summary)

        st.divider()

        # Highest priority alert
        if pr["highest_priority_category"] != "None":
            st.error(f"🚨 HIGHEST PRIORITY: **{pr['highest_priority_category']}** — requires immediate SRE attention")

        # Executive summary
        with st.expander("📊 Executive Summary for VP Engineering"):
            st.markdown(pr["portfolio_summary"])

        st.divider()

        # Category drill-down
        st.subheader("Category Detail")
        for cat in pr["categories"]:
            risk_icon = "🔴" if cat["risk_level"] == "CRITICAL" else "🟡" if cat["risk_level"] == "WARNING" else "🟢"
            with st.expander(f"{risk_icon} {cat['category']} — {cat['health_score']:.0f}% healthy — ${cat['total_aum_b']:.1f}B AUM"):
                st.markdown(f"**Summary:** {cat['category_summary']}")

                if cat["shared_feeds_down"]:
                    st.warning(f"⚠️ Shared feed issue: **{', '.join(cat['shared_feeds_down'])}** affecting multiple funds")

                # Fund table
                for f in cat["fund_results"]:
                    fund_icon = "🔴" if f["severity"] == "CRITICAL" else "🟡" if f["severity"] == "WARNING" else "🟢"
                    fc1, fc2, fc3, fc4 = st.columns([1, 2, 1, 3])
                    with fc1:
                        st.write(f"{fund_icon} **{f['fund_id']}**")
                    with fc2:
                        st.write(f["fund_name"])
                    with fc3:
                        st.write(f"${f['aum_b']}B | {f['nav_status']}")
                    with fc4:
                        if f["feeds_down"]:
                            st.write(f"⬇️ {', '.join(f['feeds_down'])}")
                        else:
                            st.write("Feeds OK")

                # Category SRE decision
                if cat["risk_level"] == "CRITICAL":
                    st.divider()
                    st.markdown(f"### SRE Decision — {cat['category']} Category")

                    cat_key = cat["category"].replace(" ", "_")
                    cat_decision = st.radio(
                        f"Approve P1 for {cat['category']}?",
                        ["Select...", "✅ APPROVE — Escalate to P1", "❌ REJECT — Monitor standard"],
                        key=f"cat_radio_{cat_key}"
                    )
                    cat_rationale = st.text_area(
                        "Rationale:",
                        key=f"cat_rationale_{cat_key}",
                        placeholder=f"e.g. {cat['shared_feeds_down']} affecting {cat['critical']} funds..."
                    )
                    if st.button(f"Submit Decision — {cat['category']}", key=f"cat_submit_{cat_key}"):
                        if cat_decision == "Select...":
                            st.error("Please select a decision.")
                        elif not cat_rationale.strip():
                            st.error("Please provide a rationale.")
                        else:
                            sre_dec = "APPROVE" if "APPROVE" in cat_decision else "REJECT"
                            # Save memory for all critical funds in category
                            for f in cat["fund_results"]:
                                if f["severity"] == "CRITICAL":
                                    save_incident_memory({
                                        "fund_id": f["fund_id"],
                                        "nav_status": f["nav_status"],
                                        "severity": f["severity"],
                                        "feeds_down": f["feeds_down"],
                                        "consumers_impacted": f["consumers"],
                                        "resolution_taken": cat_rationale,
                                        "resolution_time_mins": 82,
                                        "key_lesson": f"Category SRE {sre_dec} for {cat['category']}: {cat_rationale}",
                                        "sre_decision": sre_dec,
                                    })
                            if sre_dec == "APPROVE":
                                st.error(f"🚨 P1 ESCALATED — {cat['category']} category")
                                st.markdown(f"**Covers:** {cat['critical']} critical funds, ${cat['aum_at_risk_b']:.1f}B AUM")
                            else:
                                st.warning(f"📊 Standard monitoring — {cat['category']} category")
                            st.success(f"✅ Decision logged for all {cat['category']} critical funds")

    # ── FUND VIEW ───────────────────────────────────────────────
    elif st.session_state.get("active_view") == "fund" and "triage_result" in st.session_state:
        result = st.session_state["triage_result"]
        current_fund = st.session_state.get("triage_fund", fund_id)

        st.caption(f"Showing results for: **{current_fund}**")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔬 Investigation",
            "🚨 SRE Approval",
            "📊 Executive Summary",
            "💰 Cost Report",
            "🧠 Memory"
        ])

        with tab1:
            st.subheader("Investigation Results")
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
            st.subheader("🚨 SRE Team Approval Gate")
            if result.get("severity") != "CRITICAL":
                st.success(f"✅ Severity is {result.get('severity', 'STANDARD')} — no SRE approval needed.")
            else:
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Fund", result.get("fund_id"))
                with col_b:
                    st.metric("Severity", "🔴 CRITICAL")
                with col_c:
                    st.metric("Feeds Down", len(result.get("feeds_down", [])))

                st.divider()
                st.markdown("**Incident Summary for SRE Review:**")
                st.markdown(f"**Reason:** {result.get('severity_reason', 'N/A')}")
                st.markdown(f"**Feeds Down:** {', '.join(result.get('feeds_down', [])) or 'None'}")
                st.markdown(f"**Impacted Systems:** {', '.join(result.get('consumers', [])) or 'None'}")
                st.markdown(f"**Historical avg resolution:** ~82 minutes")

                with st.expander("📋 Full Supervisor Report"):
                    st.markdown(result.get("final_summary", ""))

                st.divider()
                st.markdown("### SRE Decision")

                decision = st.radio(
                    "Approve P1 escalation?",
                    ["Select...", "✅ APPROVE — Escalate to P1", "❌ REJECT — Monitor standard"],
                    index=0, key="sre_radio"
                )
                rationale = st.text_area(
                    "Rationale (required):",
                    placeholder="e.g. FEED_PRICE_01 down 10 hours, SettlementEngine at risk...",
                    key="sre_rationale_input"
                )

                if st.button("Submit SRE Decision", type="primary", key="sre_submit"):
                    if decision == "Select...":
                        st.error("Please select a decision.")
                    elif not rationale.strip():
                        st.error("Please provide a rationale.")
                    else:
                        sre_decision = "APPROVE" if "APPROVE" in decision else "REJECT"
                        save_incident_memory({
                            "fund_id": result.get("fund_id"),
                            "nav_status": "FAILED",
                            "severity": result.get("severity"),
                            "feeds_down": result.get("feeds_down", []),
                            "consumers_impacted": result.get("consumers", []),
                            "resolution_taken": rationale,
                            "resolution_time_mins": 82,
                            "key_lesson": f"SRE {sre_decision}: {rationale}",
                            "sre_decision": sre_decision,
                        })
                        st.session_state["sre_decision"] = sre_decision
                        st.session_state["sre_rationale"] = rationale

                if "sre_decision" in st.session_state:
                    sre_decision = st.session_state["sre_decision"]
                    sre_rationale = st.session_state["sre_rationale"]
                    st.divider()
                    if sre_decision == "APPROVE":
                        st.error("🚨 P1 ESCALATED")
                        st.markdown(f"**Decision:** APPROVED")
                        st.markdown(f"**Rationale:** {sre_rationale}")
                        st.markdown("""
                        **Immediate Actions:**
                        - Page on-call SRE team
                        - Open P1 incident ticket
                        - Notify Fund Operations manager
                        - Start 15-minute update cycle
                        """)
                    else:
                        st.warning("📊 STANDARD MONITORING")
                        st.markdown(f"**Decision:** REJECTED")
                        st.markdown(f"**Rationale:** {sre_rationale}")
                        st.markdown("""
                        **Monitoring Plan:**
                        - Check feed status every 15 minutes
                        - Escalate to P1 if not resolved in 60 minutes
                        - Keep stakeholders informed via email
                        """)
                    st.success("✅ Decision logged to memory database")

        with tab3:
            st.subheader("Executive Incident Summary")
            severity = result.get("severity", "")
            if severity == "CRITICAL":
                st.error(f"🔴 Severity: CRITICAL — {result.get('severity_reason', '')}")
            elif severity == "STANDARD":
                st.warning("🟡 Severity: STANDARD")
            else:
                st.success("✅ NAV Healthy")
            st.markdown(result.get("final_summary", ""))
            st.download_button(
                label="📥 Download Report",
                data=result.get("final_summary", ""),
                file_name=f"{current_fund}_incident_report.md",
                mime="text/markdown"
            )

        with tab4:
            st.subheader("💰 Cost Report")
            trackers = result.get("cost_tracker", [])
            if not trackers:
                st.info("No cost data available.")
            else:
                total_cost   = sum(t["cost_usd"]     for t in trackers)
                total_tokens = sum(t["total_tokens"]  for t in trackers)
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("Total Cost", f"${total_cost:.6f}")
                with col_b:
                    st.metric("Total Tokens", f"{total_tokens:,}")
                with col_c:
                    st.metric("Cost per 1K runs", f"${total_cost * 1000:.2f}")
                with col_d:
                    mini = sum(1 for t in trackers if t["model"] == "gpt-4o-mini")
                    st.metric("Cheap model used", f"{mini}/{len(trackers)} agents")
                st.divider()
                st.markdown("**Per Agent Breakdown:**")
                for t in trackers:
                    icon = "💰" if t["model"] == "gpt-4o-mini" else "🧠"
                    with st.expander(f"{icon} {t['agent']} — ${t['cost_usd']:.6f} ({t['model']})"):
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric("Input Tokens", f"{t['input_tokens']:,}")
                        with c2:
                            st.metric("Output Tokens", f"{t['output_tokens']:,}")
                        with c3:
                            st.metric("Cost", f"${t['cost_usd']:.6f}")
                st.divider()
                st.markdown("**Routing Strategy:**")
                st.markdown("- 🧠 `gpt-4o` — Fund, Feed, Knowledge, Supervisor (complex reasoning)")
                st.markdown("- 💰 `gpt-4o-mini` — Consumer, Severity (simple classification, 10x cheaper)")
                mini_tokens = sum(t["total_tokens"] for t in trackers if t["model"] == "gpt-4o-mini")
                cost_if_all = total_cost + (mini_tokens / 1_000_000) * (
                    COSTS["gpt-4o"]["input"] - COSTS["gpt-4o-mini"]["input"]
                )
                savings = cost_if_all - total_cost
                st.info(f"💡 Smart routing saves ${savings:.6f} per run (${savings * 1000:.4f} per 1K runs)")

        with tab5:
            st.subheader("🧠 Incident Memory")
            st.caption("Persistent across sessions — agent learns from every run")
            all_memories = get_all_memories()
            if not all_memories:
                st.info("No incidents in memory yet. Run a triage and submit an SRE decision to populate memory.")
            else:
                critical = sum(1 for m in all_memories if m["severity"] == "CRITICAL")
                approved = sum(1 for m in all_memories if m["sre_decision"] == "APPROVE")
                avg_time = sum(
                    m["resolution_time_mins"] for m in all_memories if m["resolution_time_mins"]
                ) / max(len(all_memories), 1)
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Total Incidents", len(all_memories))
                with c2:
                    st.metric("Critical", critical)
                with c3:
                    st.metric("P1 Escalated", approved)
                with c4:
                    st.metric("Avg Resolution", f"{avg_time:.0f} min")
                st.divider()
                funds_in_memory = list(set(m["fund_id"] for m in all_memories))
                selected_fund = st.selectbox("Filter by fund:", ["All"] + funds_in_memory, key="memory_filter")
                filtered = all_memories if selected_fund == "All" else [
                    m for m in all_memories if m["fund_id"] == selected_fund
                ]
                for m in filtered:
                    sev_icon = "🔴" if m["severity"] == "CRITICAL" else "🟡"
                    sre_icon = "✅" if m["sre_decision"] == "APPROVE" else "❌" if m["sre_decision"] == "REJECT" else "—"
                    with st.expander(f"{sev_icon} {m['timestamp']} — {m['fund_id']} — SRE: {sre_icon}"):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(f"**NAV Status:** {m['nav_status']}")
                            st.markdown(f"**Severity:** {m['severity']}")
                            st.markdown(f"**Feeds Down:** {', '.join(m['feeds_down']) or 'None'}")
                        with c2:
                            st.markdown(f"**SRE Decision:** {m['sre_decision'] or 'N/A'}")
                            st.markdown(f"**Resolution Time:** {m['resolution_time_mins']} min")
                            st.markdown(f"**Resolution:** {m['resolution_taken']}")
                        st.markdown(f"**Key Lesson:** {m['key_lesson']}")

    else:
        st.info("Select a fund and click **🚀 Run Fund Triage** — or click **🗂️ Scan Portfolio** for a full view.")