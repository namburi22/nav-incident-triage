import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List, Optional
from datetime import datetime

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

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
    return {
        "FUND001": ["FEED_PRICE_01", "FEED_CORP_ACTION"],
        "FUND002": ["FEED_PRICE_02"],
        "FUND003": ["FEED_PRICE_01"],
    }.get(fund_id, [])

@tool
def check_feed_status(feed_id: str) -> dict:
    """Check feed status."""
    return {
        "FEED_PRICE_01": {"status": "DOWN", "down_since": "2026-04-13 22:00", "hours_down": 10},
        "FEED_PRICE_02": {"status": "UP", "last_success": "2026-04-14 08:00"},
        "FEED_CORP_ACTION": {"status": "DELAYED", "delay_mins": 120},
    }.get(feed_id, {"status": "UNKNOWN"})

@tool
def get_impacted_consumers(fund_id: str) -> list:
    """Get downstream consumers."""
    return {
        "FUND001": ["RetailPortal", "AdvisorDashboard", "RegulatoryReporter", "SettlementEngine"],
        "FUND002": ["RetailPortal"],
        "FUND003": ["AdvisorDashboard", "SettlementEngine"],
    }.get(fund_id, [])

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

# --- Specialist Agents ---
def run_fund_agent(fund_id: str, nav_data: dict) -> str:
    """Independent Fund Agent — reasons about NAV health."""
    print("[SUPERVISOR] → Fund Agent running...")
    messages = [
        SystemMessage(content="""You are a Fund NAV specialist at a fintech firm.
        Analyze the NAV data and incident history. Be concise and factual.
        Focus on: current health, risk level, historical patterns."""),
        HumanMessage(content=f"""
        Fund: {fund_id}
        NAV Data: {nav_data}
        Incident History: {get_incident_history.invoke({"fund_id": fund_id})}

        Provide a 3-bullet fund status report.
        """)
    ]
    response = llm.invoke(messages)
    return response.content

def run_feed_agent(fund_id: str, feed_statuses: list) -> str:
    """Independent Feed Agent — reasons about feed health."""
    print("[SUPERVISOR] → Feed Agent running...")
    messages = [
        SystemMessage(content="""You are a Data Feed specialist at a fintech firm.
        Analyze feed statuses and their impact on NAV calculation.
        Focus on: what is down, how long, NAV impact severity."""),
        HumanMessage(content=f"""
        Fund: {fund_id}
        Feed Statuses: {feed_statuses}

        Provide a 3-bullet feed health report.
        """)
    ]
    response = llm.invoke(messages)
    return response.content

def run_consumer_agent(fund_id: str, consumers: list) -> str:
    """Independent Consumer Agent — reasons about downstream impact."""
    print("[SUPERVISOR] → Consumer Agent running...")
    messages = [
        SystemMessage(content="""You are a Downstream Impact specialist at a fintech firm.
        Analyze which systems are affected and the business severity.
        Focus on: financial risk, regulatory risk, client impact.
        Flag RegulatoryReporter and SettlementEngine as HIGH RISK."""),
        HumanMessage(content=f"""
        Fund: {fund_id}
        Impacted Consumers: {consumers}

        Provide a 3-bullet impact report.
        """)
    ]
    response = llm.invoke(messages)
    return response.content

# --- State ---
class IncidentState(TypedDict):
    fund_id: str
    nav_data: dict
    feed_statuses: List[dict]
    consumers: List[str]
    severity: str
    severity_reason: str
    sre_decision: Optional[str]
    sre_rationale: Optional[str]
    audit_log: List[dict]
    final_summary: str

# --- Graph Nodes ---
def investigate(state: IncidentState) -> IncidentState:
    print(f"\n[SUPERVISOR] Dispatching specialist agents for {state['fund_id']}...")

    # Gather raw data
    nav = get_fund_nav.invoke({"fund_id": state["fund_id"]})
    feed_ids = get_feeds_for_fund.invoke({"fund_id": state["fund_id"]})
    feed_statuses = [
        {"feed_id": fid, **check_feed_status.invoke({"feed_id": fid})}
        for fid in feed_ids
    ]
    consumers = get_impacted_consumers.invoke({"fund_id": state["fund_id"]})

    state["nav_data"] = nav
    state["feed_statuses"] = feed_statuses
    state["consumers"] = consumers

    # Dispatch to specialist agents
    fund_report = run_fund_agent(state["fund_id"], nav)
    feed_report = run_feed_agent(state["fund_id"], feed_statuses)
    consumer_report = run_consumer_agent(state["fund_id"], consumers)

    # Supervisor synthesizes all reports
    print("[SUPERVISOR] Synthesizing specialist reports...")
    supervisor_messages = [
        SystemMessage(content="""You are an Incident Management Supervisor.
        You receive reports from specialist agents and synthesize into one clear picture.
        Be concise — 3-4 sentences total."""),
        HumanMessage(content=f"""
        Fund Agent Report: {fund_report}
        Feed Agent Report: {feed_report}
        Consumer Agent Report: {consumer_report}

        Synthesize: What is the situation and how urgent is it?
        """)
    ]
    synthesis = llm.invoke(supervisor_messages)

    print(f"\n--- Supervisor Synthesis ---")
    print(synthesis.content)

    state["audit_log"].append({
        "timestamp": datetime.now().isoformat(),
        "event": "INVESTIGATION_COMPLETE",
        "nav_status": nav.get("status"),
        "fund_agent_report": fund_report,
        "feed_agent_report": feed_report,
        "consumer_agent_report": consumer_report,
        "supervisor_synthesis": synthesis.content,
        "feeds_checked": len(feed_statuses),
        "consumers_impacted": len(consumers)
    })

    print(f"\n[SUPERVISOR] Investigation complete")
    return state

def assess_severity(state: IncidentState) -> IncidentState:
    print(f"\n[SEVERITY] Assessing incident severity...")

    feeds_down = [f for f in state["feed_statuses"] if f.get("status") == "DOWN"]
    hours_down = max([f.get("hours_down", 0) for f in feeds_down], default=0)
    consumer_count = len(state["consumers"])
    nav_status = state["nav_data"].get("status", "UNKNOWN")

    prompt = f"""
    You are an incident severity classifier for a fintech platform.

    Incident Data:
    - Fund: {state['fund_id']}
    - NAV Status: {nav_status}
    - Feeds DOWN: {len(feeds_down)} (longest outage: {hours_down} hours)
    - Downstream Systems Impacted: {consumer_count}
    - Systems: {state['consumers']}

    Classify severity as CRITICAL or STANDARD based on:
    - CRITICAL: NAV FAILED + any feed down 4+ hours + 3+ consumers impacted
    - CRITICAL: RegulatoryReporter impacted (compliance risk)
    - CRITICAL: SettlementEngine impacted (financial transaction risk)
    - STANDARD: Everything else

    Respond in JSON only, no markdown, no explanation:
    {{"severity": "CRITICAL", "reason": "one sentence explanation"}}
    """

    response = llm.invoke(prompt)
    try:
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content.strip())
        state["severity"] = result["severity"]
        state["severity_reason"] = result["reason"]
    except:
        state["severity"] = "CRITICAL"
        state["severity_reason"] = "Could not parse severity — defaulting to CRITICAL for safety"

    state["audit_log"].append({
        "timestamp": datetime.now().isoformat(),
        "event": "SEVERITY_ASSESSED",
        "severity": state["severity"],
        "reason": state["severity_reason"]
    })

    print(f"[SEVERITY] → {state['severity']}: {state['severity_reason']}")
    return state

def request_sre_approval(state: IncidentState) -> IncidentState:
    """This node PAUSES and waits for SRE input."""
    investigation = next(
        (e for e in state["audit_log"] if e["event"] == "INVESTIGATION_COMPLETE"), {}
    )

    print(f"\n{'='*60}")
    print("🚨 SRE APPROVAL REQUIRED — P1 ESCALATION")
    print(f"{'='*60}")
    print(f"\nFund:     {state['fund_id']}")
    print(f"Severity: {state['severity']}")
    print(f"Reason:   {state['severity_reason']}")
    print(f"\nAgent Synthesis:")
    print(f"  {investigation.get('supervisor_synthesis', 'N/A')}")
    print(f"\nFeed Status:")
    for f in state["feed_statuses"]:
        hours = f.get("hours_down", "")
        hours_str = f" ({hours}h down)" if hours else ""
        print(f"  {f['feed_id']}: {f.get('status')}{hours_str}")
    print(f"\nImpacted Systems: {', '.join(state['consumers'])}")
    print(f"\nHistorical avg resolution: ~82 minutes")
    print(f"{'='*60}")

    decision = input("\n[SRE TEAM] Approve P1 escalation? (approve/reject): ").strip().lower()
    rationale = input("[SRE TEAM] Rationale: ").strip()

    state["sre_decision"] = "APPROVE" if decision == "approve" else "REJECT"
    state["sre_rationale"] = rationale

    state["audit_log"].append({
        "timestamp": datetime.now().isoformat(),
        "event": "SRE_DECISION",
        "decision": state["sre_decision"],
        "rationale": state["sre_rationale"],
        "approver": "SRE_TEAM"
    })

    print(f"\n[SRE] Decision logged: {state['sre_decision']}")
    return state

def escalate_p1(state: IncidentState) -> IncidentState:
    print(f"\n[ESCALATE] P1 ESCALATION APPROVED — initiating response...")

    prompt = f"""
    Generate a P1 incident escalation summary.

    Fund: {state['fund_id']}
    NAV Status: {state['nav_data'].get('status')}
    Feed Issues: {state['feed_statuses']}
    Impacted: {state['consumers']}
    SRE Approved by: SRE Team
    SRE Rationale: {state['sre_rationale']}

    Include:
    1. P1 Declaration — what triggered escalation
    2. Immediate actions — who does what in the next 15 minutes
    3. Communication plan — who to notify
    4. ETA — realistic based on historical data (~82 min average)
    """
    response = llm.invoke(prompt)
    state["final_summary"] = f"🚨 P1 ESCALATED\n\n{response.content}"

    state["audit_log"].append({
        "timestamp": datetime.now().isoformat(),
        "event": "P1_ESCALATED",
        "summary": "P1 incident declared and response initiated"
    })

    print("[ESCALATE] P1 escalation complete")
    return state

def monitor_standard(state: IncidentState) -> IncidentState:
    print(f"\n[MONITOR] Moving to standard monitoring...")

    # Check if this is SRE rejected or standard severity
    sre_rejected = state.get("sre_decision") == "REJECT"

    prompt = f"""
    Generate a standard incident monitoring plan.

    Fund: {state['fund_id']}
    NAV Status: {state['nav_data'].get('status')}
    Feed Issues: {state['feed_statuses']}
    {'SRE rejected P1 escalation. Rationale: ' + str(state.get('sre_rationale')) if sre_rejected else 'Severity assessed as STANDARD.'}

    Include:
    1. Monitoring cadence — how often to check
    2. Escalation trigger — when to re-evaluate P1
    3. Stakeholder communication — who to keep informed
    """
    response = llm.invoke(prompt)
    label = "⚠️ SRE REJECTED P1 — STANDARD MONITORING" if sre_rejected else "📊 STANDARD MONITORING"
    state["final_summary"] = f"{label}\n\n{response.content}"

    state["audit_log"].append({
        "timestamp": datetime.now().isoformat(),
        "event": "STANDARD_MONITORING",
        "summary": "Standard monitoring initiated"
    })

    print("[MONITOR] Standard monitoring plan generated")
    return state

def healthy_close(state: IncidentState) -> IncidentState:
    state["final_summary"] = f"✅ {state['fund_id']} NAV healthy — no action required"
    state["audit_log"].append({
        "timestamp": datetime.now().isoformat(),
        "event": "HEALTHY_CLOSE"
    })
    print(f"\n[CLOSE] {state['fund_id']} NAV healthy — closing")
    return state

# --- Routing ---
def route_after_investigation(state: IncidentState) -> str:
    if state["nav_data"].get("status") == "SUCCESS":
        return "healthy"
    return "assess"

def route_after_assessment(state: IncidentState) -> str:
    if state["severity"] == "CRITICAL":
        return "sre_approval"
    return "monitor_standard_direct"

def route_after_sre(state: IncidentState) -> str:
    if state["sre_decision"] == "APPROVE":
        return "escalate"
    return "monitor"

# --- Build Graph ---
def build_graph():
    graph = StateGraph(IncidentState)

    graph.add_node("investigate", investigate)
    graph.add_node("assess_severity", assess_severity)
    graph.add_node("request_sre_approval", request_sre_approval)
    graph.add_node("escalate_p1", escalate_p1)
    graph.add_node("monitor_standard", monitor_standard)
    graph.add_node("healthy_close", healthy_close)

    graph.set_entry_point("investigate")

    graph.add_conditional_edges(
        "investigate",
        route_after_investigation,
        {
            "healthy": "healthy_close",
            "assess": "assess_severity"
        }
    )

    graph.add_conditional_edges(
        "assess_severity",
        route_after_assessment,
        {
            "sre_approval": "request_sre_approval",
            "monitor_standard_direct": "monitor_standard"
        }
    )

    graph.add_conditional_edges(
        "request_sre_approval",
        route_after_sre,
        {
            "escalate": "escalate_p1",
            "monitor": "monitor_standard"
        }
    )

    graph.add_edge("escalate_p1", END)
    graph.add_edge("monitor_standard", END)
    graph.add_edge("healthy_close", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

# --- Run ---
def run(fund_id: str):
    app = build_graph()
    config = {"configurable": {"thread_id": f"incident_{fund_id}"}}

    initial_state = {
        "fund_id": fund_id,
        "nav_data": {},
        "feed_statuses": [],
        "consumers": [],
        "severity": "",
        "severity_reason": "",
        "sre_decision": None,
        "sre_rationale": None,
        "audit_log": [],
        "final_summary": ""
    }

    result = app.invoke(initial_state, config=config)

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(result["final_summary"])

    print(f"\n{'='*60}")
    print("AUDIT TRAIL")
    print(f"{'='*60}")
    for entry in result["audit_log"]:
        print(f"\n[{entry['timestamp']}] {entry['event']}")
        for k, v in entry.items():
            if k not in ["timestamp", "event"]:
                # Truncate long agent reports in audit display
                val = str(v)
                print(f"  {k}: {val[:120]}..." if len(val) > 120 else f"  {k}: {val}")

if __name__ == "__main__":
    print("Select fund to triage:")
    print("  FUND001 — NAV FAILED  (triggers multi-agent + SRE approval)")
    print("  FUND002 — NAV SUCCESS (healthy close)")
    print("  FUND003 — NAV PENDING (triggers multi-agent + SRE approval)")
    fund_id = input("\nEnter fund ID: ").strip() or "FUND001"
    run(fund_id)