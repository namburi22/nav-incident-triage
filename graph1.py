from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# State — this is what flows through every node
class IncidentState(TypedDict):
    fund_id: str
    nav_status: str
    feeds: list
    incidents: list
    summary: str

# Node 1 — check NAV
def check_nav(state: IncidentState) -> IncidentState:
    print(f"[NODE 1] Checking NAV for {state['fund_id']}")
    # Simulated
    nav_data = {
        "FUND001": "FAILED",
        "FUND002": "SUCCESS",
        "FUND003": "PENDING"
    }
    state["nav_status"] = nav_data.get(state["fund_id"], "UNKNOWN")
    return state

# Node 2 — check feeds
def check_feeds(state: IncidentState) -> IncidentState:
    print(f"[NODE 2] Checking feeds for {state['fund_id']}")
    fund_feeds = {
        "FUND001": ["FEED_PRICE_01 - DOWN", "FEED_CORP_ACTION - DELAYED"],
        "FUND002": ["FEED_PRICE_02 - UP"],
        "FUND003": ["FEED_PRICE_01 - DOWN"],
    }
    state["feeds"] = fund_feeds.get(state["fund_id"], [])
    return state

# Node 3 — get incident history
def check_history(state: IncidentState) -> IncidentState:
    print(f"[NODE 3] Checking incident history for {state['fund_id']}")
    history = {
        "FUND001": ["2026-03-10: Price feed down", "2026-02-22: Corp action missing"],
        "FUND003": ["2026-04-01: NAV timeout"],
    }
    state["incidents"] = history.get(state["fund_id"], [])
    return state

# Node 4 — summarize
def summarize(state: IncidentState) -> IncidentState:
    print(f"[NODE 4] Generating summary")
    prompt = f"""
    Fund: {state['fund_id']}
    NAV Status: {state['nav_status']}
    Feed Issues: {state['feeds']}
    Past Incidents: {state['incidents']}
    
    Write a concise incident triage summary with recommended action.
    """
    response = llm.invoke(prompt)
    state["summary"] = response.content
    return state

# Decision edge — should we investigate further or skip?
# Replace this function only — everything else stays the same

def should_investigate(state: IncidentState) -> str:
    print(f"[DECISION] LLM deciding whether to investigate...")
    
    prompt = f"""
    Fund: {state['fund_id']}
    NAV Status: {state['nav_status']}
    
    Based on the NAV status, should we investigate further?
    
    Rules:
    - FAILED means investigation is critical
    - PENDING means investigation is warranted  
    - SUCCESS means no investigation needed
    
    Respond with exactly one word: 'investigate' or 'skip'
    """
    
    response = llm.invoke(prompt)
    decision = response.content.strip().lower()
    print(f"[DECISION] LLM decided: {decision}")
    return decision

# Build the graph
graph = StateGraph(IncidentState)

# Add nodes
graph.add_node("check_nav", check_nav)
graph.add_node("check_feeds", check_feeds)
graph.add_node("check_history", check_history)
graph.add_node("summarize", summarize)

# Entry point
graph.set_entry_point("check_nav")

# Conditional edge — only investigate if NAV failed
graph.add_conditional_edges(
    "check_nav",
    should_investigate,
    {
        "investigate": "check_feeds",
        "skip": END
    }
)

# Fixed edges after investigation
graph.add_edge("check_feeds", "check_history")
graph.add_edge("check_history", "summarize")
graph.add_edge("summarize", END)

# Compile
app = graph.compile()

# Run for a failed fund
print("\n=== Running for FUND001 (FAILED) ===\n")
result = app.invoke({"fund_id": "FUND001", "nav_status": "", "feeds": [], "incidents": [], "summary": ""})
print(f"\nSUMMARY:\n{result['summary']}")

# Run for a healthy fund
print("\n=== Running for FUND002 (SUCCESS) ===\n")
result = app.invoke({"fund_id": "FUND002", "nav_status": "", "feeds": [], "incidents": [], "summary": ""})
print(f"\nStatus: {result['nav_status']} — No investigation needed")

print("\n=== Running for FUND003 (PENDING) ===\n")
result = app.invoke({"fund_id": "FUND003", "nav_status": "", "feeds": [], "incidents": [], "summary": ""})
if result['summary']:
    print(f"\nSUMMARY:\n{result['summary']}")
else:
    print(f"\nStatus: {result['nav_status']} — No investigation needed")