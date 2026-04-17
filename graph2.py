from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- Real tools from agent4.py ---
@tool
def get_fund_nav(fund_id: str) -> dict:
    """Get the current NAV status for a fund."""
    funds = {
        "FUND001": {"nav": 142.35, "status": "FAILED", "last_updated": "2026-04-14 08:00"},
        "FUND002": {"nav": 98.12, "status": "SUCCESS", "last_updated": "2026-04-14 08:05"},
        "FUND003": {"nav": 0.00, "status": "PENDING", "last_updated": "2026-04-14 07:45"},
    }
    return funds.get(fund_id, {"error": f"Fund {fund_id} not found"})

@tool
def get_feeds_for_fund(fund_id: str) -> list:
    """Get all feed IDs associated with a fund."""
    fund_feeds = {
        "FUND001": ["FEED_PRICE_01", "FEED_CORP_ACTION"],
        "FUND002": ["FEED_PRICE_02"],
        "FUND003": ["FEED_PRICE_01"],
    }
    return fund_feeds.get(fund_id, [])

@tool
def check_feed_status(feed_id: str) -> dict:
    """Check the status of a data feed."""
    feeds = {
        "FEED_PRICE_01": {"status": "DOWN", "last_success": "2026-04-13 22:00"},
        "FEED_PRICE_02": {"status": "UP", "last_success": "2026-04-14 08:00"},
        "FEED_CORP_ACTION": {"status": "DELAYED", "last_success": "2026-04-14 06:00"},
    }
    return feeds.get(feed_id, {"error": f"Feed {feed_id} not found"})

@tool
def get_incident_history(fund_id: str) -> list:
    """Get past incidents for a fund."""
    history = {
        "FUND001": [
            {"date": "2026-03-10", "issue": "Price feed down", "resolution": "Feed restarted", "duration_mins": 45},
            {"date": "2026-02-22", "issue": "Corporate action missing", "resolution": "Manual override", "duration_mins": 120},
        ],
        "FUND003": [
            {"date": "2026-04-01", "issue": "NAV calculation timeout", "resolution": "Reprocessed", "duration_mins": 30},
        ],
    }
    return history.get(fund_id, [])

@tool
def get_impacted_consumers(fund_id: str) -> list:
    """Get downstream consumers impacted by a fund NAV failure."""
    consumers = {
        "FUND001": ["RetailPortal", "AdvisorDashboard", "RegulatoryReporter", "SettlementEngine"],
        "FUND002": ["RetailPortal"],
        "FUND003": ["AdvisorDashboard", "SettlementEngine"],
    }
    return consumers.get(fund_id, [])

# --- State ---
class IncidentState(TypedDict):
    fund_id: str
    nav_status: str
    nav_value: float
    feeds: List[dict]
    incidents: List[dict]
    consumers: List[str]
    summary: str

# --- Nodes now call real tools ---
def check_nav(state: IncidentState) -> IncidentState:
    print(f"[NODE 1] Checking NAV for {state['fund_id']}")
    result = get_fund_nav.invoke({"fund_id": state["fund_id"]})
    state["nav_status"] = result.get("status", "UNKNOWN")
    state["nav_value"] = result.get("nav", 0.0)
    return state

def check_feeds(state: IncidentState) -> IncidentState:
    print(f"[NODE 2] Getting feeds for {state['fund_id']}")
    feed_ids = get_feeds_for_fund.invoke({"fund_id": state["fund_id"]})
    feed_statuses = []
    for feed_id in feed_ids:
        status = check_feed_status.invoke({"feed_id": feed_id})
        feed_statuses.append({"feed_id": feed_id, **status})
        print(f"         Feed {feed_id}: {status['status']}")
    state["feeds"] = feed_statuses
    return state

def check_history(state: IncidentState) -> IncidentState:
    print(f"[NODE 3] Checking incident history for {state['fund_id']}")
    state["incidents"] = get_incident_history.invoke({"fund_id": state["fund_id"]})
    return state

def check_consumers(state: IncidentState) -> IncidentState:
    print(f"[NODE 4] Checking impacted consumers for {state['fund_id']}")
    state["consumers"] = get_impacted_consumers.invoke({"fund_id": state["fund_id"]})
    print(f"         Impacted: {state['consumers']}")
    return state

def summarize(state: IncidentState) -> IncidentState:
    print(f"[NODE 5] Generating summary")
    prompt = f"""
    Fund: {state['fund_id']}
    NAV Status: {state['nav_status']} (Value: {state['nav_value']})
    Feed Issues: {state['feeds']}
    Past Incidents: {state['incidents']}
    Impacted Consumers: {state['consumers']}
    
    Write a concise incident triage summary with:
    1. Root cause
    2. Business impact
    3. Recommended immediate action
    4. Estimated resolution time
    """
    response = llm.invoke(prompt)
    state["summary"] = response.content
    return state

# --- LLM decision ---
def should_investigate(state: IncidentState) -> str:
    print(f"[DECISION] LLM evaluating severity...")
    prompt = f"""
    Fund: {state['fund_id']}
    NAV Status: {state['nav_status']}
    
    Should we investigate further?
    FAILED or PENDING = investigate
    SUCCESS = skip
    
    Respond with exactly one word: 'investigate' or 'skip'
    """
    response = llm.invoke(prompt)
    decision = response.content.strip().lower()
    print(f"[DECISION] → {decision}")
    return decision

# --- Build graph ---
graph = StateGraph(IncidentState)

graph.add_node("check_nav", check_nav)
graph.add_node("check_feeds", check_feeds)
graph.add_node("check_history", check_history)
graph.add_node("check_consumers", check_consumers)
graph.add_node("summarize", summarize)

graph.set_entry_point("check_nav")

graph.add_conditional_edges(
    "check_nav",
    should_investigate,
    {
        "investigate": "check_feeds",
        "skip": END
    }
)

graph.add_edge("check_feeds", "check_history")
graph.add_edge("check_history", "check_consumers")
graph.add_edge("check_consumers", "summarize")
graph.add_edge("summarize", END)

app = graph.compile()

# --- Run ---
print("\n=== FUND001 ===\n")
result = app.invoke({
    "fund_id": "FUND001",
    "nav_status": "",
    "nav_value": 0.0,
    "feeds": [],
    "incidents": [],
    "consumers": [],
    "summary": ""
})
print(f"\nSUMMARY:\n{result['summary']}")

print("\n=== FUND002 ===\n")
result = app.invoke({
    "fund_id": "FUND002",
    "nav_status": "",
    "nav_value": 0.0,
    "feeds": [],
    "incidents": [],
    "consumers": [],
    "summary": ""
})
print(f"Status: {result['nav_status']} — No investigation needed")