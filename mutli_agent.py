from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, List

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- Tools ---
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
    question: str
    fund_report: str
    feed_report: str
    consumer_report: str
    final_summary: str

# --- Specialized Agents ---

def fund_agent(state: IncidentState) -> IncidentState:
    print("\n[FUND AGENT] Investigating NAV status...")
    
    nav = get_fund_nav.invoke({"fund_id": state["fund_id"]})
    history = get_incident_history.invoke({"fund_id": state["fund_id"]})
    
    prompt = f"""
    You are a Fund NAV specialist.
    Fund: {state['fund_id']}
    NAV Data: {nav}
    Incident History: {history}
    
    Provide a brief fund status report focusing on NAV health and historical patterns.
    """
    response = llm.invoke(prompt)
    state["fund_report"] = response.content
    print(f"[FUND AGENT] Done")
    return state

def feed_agent(state: IncidentState) -> IncidentState:
    print("\n[FEED AGENT] Investigating feed status...")
    
    feed_ids = get_feeds_for_fund.invoke({"fund_id": state["fund_id"]})
    feed_statuses = []
    for feed_id in feed_ids:
        status = check_feed_status.invoke({"feed_id": feed_id})
        feed_statuses.append({"feed_id": feed_id, **status})
    
    prompt = f"""
    You are a Data Feed specialist.
    Fund: {state['fund_id']}
    Feed Statuses: {feed_statuses}
    
    Provide a brief feed health report identifying any issues and their severity.
    """
    response = llm.invoke(prompt)
    state["feed_report"] = response.content
    print(f"[FEED AGENT] Done")
    return state

def consumer_agent(state: IncidentState) -> IncidentState:
    print("\n[CONSUMER AGENT] Investigating downstream impact...")
    
    consumers = get_impacted_consumers.invoke({"fund_id": state["fund_id"]})
    
    prompt = f"""
    You are a Downstream Impact specialist.
    Fund: {state['fund_id']}
    Impacted Consumers: {consumers}
    
    Provide a brief impact report on which systems are affected and business severity.
    """
    response = llm.invoke(prompt)
    state["consumer_report"] = response.content
    print(f"[CONSUMER AGENT] Done")
    return state

def supervisor(state: IncidentState) -> IncidentState:
    print("\n[SUPERVISOR] Synthesizing all reports...")
    
    prompt = f"""
    You are an Incident Management Supervisor.
    Question: {state['question']}
    
    Fund Report: {state['fund_report']}
    Feed Report: {state['feed_report']}
    Consumer Report: {state['consumer_report']}
    
    Synthesize all reports into a final executive incident summary with:
    1. Situation — what happened
    2. Impact — who is affected
    3. Root cause — why it happened
    4. Action plan — what to do now
    5. ETA — when will it be resolved
    """
    response = llm.invoke(prompt)
    state["final_summary"] = response.content
    return state

# --- Build Graph ---
graph = StateGraph(IncidentState)

graph.add_node("fund_agent", fund_agent)
graph.add_node("feed_agent", feed_agent)
graph.add_node("consumer_agent", consumer_agent)
graph.add_node("supervisor", supervisor)

# All three specialists run first then supervisor synthesizes
graph.set_entry_point("fund_agent")
graph.add_edge("fund_agent", "feed_agent")
graph.add_edge("feed_agent", "consumer_agent")
graph.add_edge("consumer_agent", "supervisor")
graph.add_edge("supervisor", END)

app = graph.compile()

# --- Run ---
print("\n=== Multi-Agent NAV Incident Triage ===\n")

result = app.invoke({
    "fund_id": "FUND001",
    "question": "FUND001 NAV has failed. What is the situation and what should we do?",
    "fund_report": "",
    "feed_report": "",
    "consumer_report": "",
    "final_summary": ""
})

print("\n" + "="*50)
print("FINAL EXECUTIVE SUMMARY")
print("="*50)
print(result["final_summary"])