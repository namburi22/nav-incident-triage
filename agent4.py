from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- Simulated domain tools ---
# In production these would hit real APIs, databases, feeds

@tool
def get_fund_nav(fund_id: str) -> dict:
    """Get the current NAV status for a fund."""
    # Simulated data
    funds = {
        "FUND001": {"nav": 142.35, "status": "FAILED", "last_updated": "2026-04-14 08:00"},
        "FUND002": {"nav": 98.12, "status": "SUCCESS", "last_updated": "2026-04-14 08:05"},
        "FUND003": {"nav": 0.00, "status": "PENDING", "last_updated": "2026-04-14 07:45"},
    }
    return funds.get(fund_id, {"error": f"Fund {fund_id} not found"})

@tool
def check_feed_status(feed_id: str) -> dict:
    """Check the status of a data feed."""
    feeds = {
        "FEED_PRICE_01": {"status": "DOWN", "last_success": "2026-04-13 22:00", "impact": ["FUND001", "FUND003"]},
        "FEED_PRICE_02": {"status": "UP", "last_success": "2026-04-14 08:00", "impact": ["FUND002"]},
        "FEED_CORP_ACTION": {"status": "DELAYED", "last_success": "2026-04-14 06:00", "impact": ["FUND001"]},
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

@tool
def estimate_resolution(issue_type: str) -> dict:
    """Estimate resolution time based on issue type."""
    estimates = {
        "feed_down": {"eta_mins": 45, "action": "Escalate to feed ops team"},
        "calculation_timeout": {"eta_mins": 20, "action": "Trigger reprocessing job"},
        "corporate_action_missing": {"eta_mins": 90, "action": "Manual data entry required"},
        "unknown": {"eta_mins": 60, "action": "Escalate to L2 support"},
    }
    return estimates.get(issue_type, estimates["unknown"])

# --- Agent setup ---
memory = MemorySaver()

agent = create_react_agent(
    llm,
    tools=[
        get_fund_nav,
        check_feed_status,
        get_incident_history,
        get_impacted_consumers,
        estimate_resolution
    ],
    checkpointer=memory
)

config = {"configurable": {"thread_id": "incident_001"}}

print("=== NAV Incident Triage Agent ===")
print("Type your question. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    response = agent.invoke(
        {"messages": [("user", user_input)]},
        config=config
    )

    last_message = response["messages"][-1]
    print(f"\nAgent: {last_message.content}\n")