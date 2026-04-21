import json
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List, Optional
from datetime import datetime
import time

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- Tools ---
@tool
def get_fund_nav(fund_id: str) -> dict:
    """Get NAV status for a fund."""
    return {
        "FUND001": {"nav": 142.35, "status": "FAILED", "last_updated": "2026-04-14 08:00"},
        "FUND002": {"nav": 98.12, "status": "SUCCESS", "last_updated": "2026-04-14 08:05"},
        "FUND003": {"nav": 0.00, "status": "PENDING", "last_updated": "2026-04-14 07:45"},
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

# --- Async Specialist Agents ---
async def run_fund_agent(fund_id: str, nav_data: dict) -> str:
    """Async Fund Agent."""
    messages = [
        SystemMessage(content="""You are a Fund NAV specialist at a fintech firm.
        Analyze NAV data and incident history. Be concise and factual.
        Focus on: current health, risk level, historical patterns."""),
        HumanMessage(content=f"""
        Fund: {fund_id}
        NAV Data: {nav_data}
        Incident History: {get_incident_history.invoke({"fund_id": fund_id})}
        Provide a 3-bullet fund status report.
        """)
    ]
    response = await llm.ainvoke(messages)
    return response.content

async def run_feed_agent(fund_id: str, feed_statuses: list) -> str:
    """Async Feed Agent."""
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
    response = await llm.ainvoke(messages)
    return response.content

async def run_consumer_agent(fund_id: str, consumers: list) -> str:
    """Async Consumer Agent."""
    messages = [
        SystemMessage(content="""You are a Downstream Impact specialist at a fintech firm.
        Analyze which systems are affected and business severity.
        Flag RegulatoryReporter and SettlementEngine as HIGH RISK."""),
        HumanMessage(content=f"""
        Fund: {fund_id}
        Impacted Consumers: {consumers}
        Provide a 3-bullet impact report.
        """)
    ]
    response = await llm.ainvoke(messages)
    return response.content

# --- State ---
class IncidentState(TypedDict):
    fund_id: str
    nav_data: dict
    feed_statuses: List[dict]
    consumers: List[str]
    fund_report: str
    feed_report: str
    consumer_report: str
    severity: str
    severity_reason: str
    sre_decision: Optional[str]
    sre_rationale: Optional[str]
    audit_log: List[dict]
    final_summary: str
    sequential_time: float
    parallel_time: float

# --- Sequential investigation (for comparison) ---
async def run_sequential(fund_id: str, nav_data: dict, feed_statuses: list, consumers: list):
    """Run agents one after another."""
    print("\n[SEQUENTIAL] Running agents one by one...")
    start = time.time()

    print("  → Fund Agent...")
    fund_report = await run_fund_agent(fund_id, nav_data)

    print("  → Feed Agent...")
    feed_report = await run_feed_agent(fund_id, feed_statuses)

    print("  → Consumer Agent...")
    consumer_report = await run_consumer_agent(fund_id, consumers)

    elapsed = time.time() - start
    print(f"[SEQUENTIAL] Done in {elapsed:.2f}s")
    return fund_report, feed_report, consumer_report, elapsed

# --- Parallel investigation ---
async def run_parallel(fund_id: str, nav_data: dict, feed_statuses: list, consumers: list):
    """Run agents simultaneously."""
    print("\n[PARALLEL] Running agents simultaneously...")
    start = time.time()

    # All three fire at the same time
    fund_report, feed_report, consumer_report = await asyncio.gather(
        run_fund_agent(fund_id, nav_data),
        run_feed_agent(fund_id, feed_statuses),
        run_consumer_agent(fund_id, consumers)
    )

    elapsed = time.time() - start
    print(f"[PARALLEL] Done in {elapsed:.2f}s")
    return fund_report, feed_report, consumer_report, elapsed

# --- Graph Nodes ---
async def investigate_parallel(state: IncidentState) -> IncidentState:
    print(f"\n[SUPERVISOR] Starting investigation for {state['fund_id']}...")

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

    # Run sequential first for comparison
    seq_fund, seq_feed, seq_consumer, seq_time = await run_sequential(
        state["fund_id"], nav, feed_statuses, consumers
    )
    state["sequential_time"] = seq_time

    # Now run parallel
    par_fund, par_feed, par_consumer, par_time = await run_parallel(
        state["fund_id"], nav, feed_statuses, consumers
    )
    state["parallel_time"] = par_time

    # Use parallel results
    state["fund_report"] = par_fund
    state["feed_report"] = par_feed
    state["consumer_report"] = par_consumer

    # Print timing comparison
    speedup = seq_time / par_time if par_time > 0 else 1
    print(f"\n{'='*50}")
    print(f"⚡ TIMING COMPARISON")
    print(f"  Sequential: {seq_time:.2f}s")
    print(f"  Parallel:   {par_time:.2f}s")
    print(f"  Speedup:    {speedup:.1f}x faster")
    print(f"{'='*50}")

    # Supervisor synthesizes parallel results
    print("\n[SUPERVISOR] Synthesizing reports...")
    supervisor_messages = [
        SystemMessage(content="""You are an Incident Management Supervisor.
        Synthesize specialist reports into one clear picture.
        Be concise — 3-4 sentences."""),
        HumanMessage(content=f"""
        Fund Agent: {par_fund}
        Feed Agent: {par_feed}
        Consumer Agent: {par_consumer}
        What is the situation and how urgent is it?
        """)
    ]
    synthesis = await llm.ainvoke(supervisor_messages)

    print(f"\n--- Supervisor Synthesis ---")
    print(synthesis.content)

    state["audit_log"].append({
        "timestamp": datetime.now().isoformat(),
        "event": "PARALLEL_INVESTIGATION_COMPLETE",
        "nav_status": nav.get("status"),
        "sequential_time_secs": round(seq_time, 2),
        "parallel_time_secs": round(par_time, 2),
        "speedup": round(speedup, 1),
        "supervisor_synthesis": synthesis.content,
        "feeds_checked": len(feed_statuses),
        "consumers_impacted": len(consumers)
    })

    return state

def assess_severity(state: IncidentState) -> IncidentState:
    print(f"\n[SEVERITY] Assessing...")

    feeds_down = [f for f in state["feed_statuses"] if f.get("status") == "DOWN"]
    hours_down = max([f.get("hours_down", 0) for f in feeds_down], default=0)

    prompt = f"""
    Classify incident severity as CRITICAL or STANDARD.

    - Fund: {state['fund_id']}
    - NAV Status: {state['nav_data'].get('status')}
    - Feeds DOWN: {len(feeds_down)} (longest: {hours_down} hours)
    - Consumers: {state['consumers']}

    CRITICAL if: NAV FAILED + feed down 4+ hours + 3+ consumers
    CRITICAL if: RegulatoryReporter or SettlementEngine impacted
    STANDARD: everything else

    Respond in JSON only, no markdown:
    {{"severity": "CRITICAL", "reason": "one sentence"}}
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
        state["severity_reason"] = "Parse failed — defaulting to CRITICAL"

    state["audit_log"].append({
        "timestamp": datetime.now().isoformat(),
        "event": "SEVERITY_ASSESSED",
        "severity": state["severity"],
        "reason": state["severity_reason"]
    })

    print(f"[SEVERITY] → {state['severity']}: {state['severity_reason']}")
    return state

def request_sre_approval(state: IncidentState) -> IncidentState:
    investigation = next(
        (e for e in state["audit_log"] if e["event"] == "PARALLEL_INVESTIGATION_COMPLETE"), {}
    )

    print(f"\n{'='*60}")
    print("🚨 SRE APPROVAL REQUIRED — P1 ESCALATION")
    print(f"{'='*60}")
    print(f"Fund:     {state['fund_id']}")
    print(f"Severity: {state['severity']}")
    print(f"Reason:   {state['severity_reason']}")
    print(f"\nAgent Synthesis:")
    print(f"  {investigation.get('supervisor_synthesis', 'N/A')}")
    print(f"\nFeed Status:")
    for f in state["feed_statuses"]:
        hours = f.get("hours_down", "")
        hours_str = f" ({hours}h down)" if hours else ""
        print(f"  {f['feed_id']}: {f.get('status')}{hours_str}")
    print(f"\nImpacted: {', '.join(state['consumers'])}")
    print(f"Historical avg resolution: ~82 minutes")
    print(f"\n⚡ Investigation completed in {state['parallel_time']:.2f}s (parallel) vs {state['sequential_time']:.2f}s (sequential)")
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
    print(f"\n[ESCALATE] P1 approved — generating response plan...")
    prompt = f"""
    Generate a P1 escalation summary.
    Fund: {state['fund_id']}
    NAV: {state['nav_data'].get('status')}
    Feeds: {state['feed_statuses']}
    Consumers: {state['consumers']}
    SRE Rationale: {state['sre_rationale']}

    Include: P1 Declaration, Immediate Actions (15 min), Communication Plan, ETA.
    """
    response = llm.invoke(prompt)
    state["final_summary"] = f"🚨 P1 ESCALATED\n\n{response.content}"
    state["audit_log"].append({
        "timestamp": datetime.now().isoformat(),
        "event": "P1_ESCALATED"
    })
    print("[ESCALATE] Done")
    return state

def monitor_standard(state: IncidentState) -> IncidentState:
    print(f"\n[MONITOR] Standard monitoring initiated...")
    sre_rejected = state.get("sre_decision") == "REJECT"
    prompt = f"""
    Generate a standard monitoring plan.
    Fund: {state['fund_id']}
    NAV: {state['nav_data'].get('status')}
    Feeds: {state['feed_statuses']}
    {'SRE rejected P1. Rationale: ' + str(state.get('sre_rationale')) if sre_rejected else 'Severity is STANDARD.'}

    Include: Monitoring cadence, Escalation triggers, Stakeholder updates.
    """
    response = llm.invoke(prompt)
    label = "⚠️ SRE REJECTED — STANDARD MONITORING" if sre_rejected else "📊 STANDARD MONITORING"
    state["final_summary"] = f"{label}\n\n{response.content}"
    state["audit_log"].append({
        "timestamp": datetime.now().isoformat(),
        "event": "STANDARD_MONITORING"
    })
    print("[MONITOR] Done")
    return state

def healthy_close(state: IncidentState) -> IncidentState:
    state["final_summary"] = f"✅ {state['fund_id']} NAV healthy — no action required"
    state["audit_log"].append({
        "timestamp": datetime.now().isoformat(),
        "event": "HEALTHY_CLOSE"
    })
    print(f"\n[CLOSE] {state['fund_id']} healthy — closing")
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

    graph.add_node("investigate", investigate_parallel)
    graph.add_node("assess_severity", assess_severity)
    graph.add_node("request_sre_approval", request_sre_approval)
    graph.add_node("escalate_p1", escalate_p1)
    graph.add_node("monitor_standard", monitor_standard)
    graph.add_node("healthy_close", healthy_close)

    graph.set_entry_point("investigate")

    graph.add_conditional_edges(
        "investigate",
        route_after_investigation,
        {"healthy": "healthy_close", "assess": "assess_severity"}
    )
    graph.add_conditional_edges(
        "assess_severity",
        route_after_assessment,
        {"sre_approval": "request_sre_approval", "monitor_standard_direct": "monitor_standard"}
    )
    graph.add_conditional_edges(
        "request_sre_approval",
        route_after_sre,
        {"escalate": "escalate_p1", "monitor": "monitor_standard"}
    )

    graph.add_edge("escalate_p1", END)
    graph.add_edge("monitor_standard", END)
    graph.add_edge("healthy_close", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

# --- Run ---
async def run(fund_id: str):
    app = build_graph()
    config = {"configurable": {"thread_id": f"incident_{fund_id}"}}

    initial_state = {
        "fund_id": fund_id,
        "nav_data": {},
        "feed_statuses": [],
        "consumers": [],
        "fund_report": "",
        "feed_report": "",
        "consumer_report": "",
        "severity": "",
        "severity_reason": "",
        "sre_decision": None,
        "sre_rationale": None,
        "audit_log": [],
        "final_summary": "",
        "sequential_time": 0.0,
        "parallel_time": 0.0
    }

    result = await app.ainvoke(initial_state, config=config)

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
                val = str(v)
                print(f"  {k}: {val[:120]}..." if len(val) > 120 else f"  {k}: {val}")

if __name__ == "__main__":
    print("Select fund:")
    print("  FUND001 — NAV FAILED  (parallel agents + SRE gate)")
    print("  FUND002 — NAV SUCCESS (healthy close)")
    print("  FUND003 — NAV PENDING (parallel agents + SRE gate)")
    fund_id = input("\nEnter fund ID: ").strip() or "FUND001"
    
    # Windows Python 3.10 fix — prevents event loop cleanup noise
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(run(fund_id))