import json
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.callbacks import BaseCallbackHandler
from typing import List, Dict, Any, Optional
import time

load_dotenv()

# --- Cost Constants (USD per 1M tokens) ---
COSTS = {
    "gpt-4o": {
        "input":  2.50,   # per 1M tokens
        "output": 10.00,
    },
    "gpt-4o-mini": {
        "input":  0.15,
        "output": 0.60,
    }
}

# --- Token Tracker ---
class TokenTracker(BaseCallbackHandler):
    """Tracks token usage and cost per LLM call."""

    def __init__(self, agent_name: str, model: str):
        self.agent_name = agent_name
        self.model = model
        self.calls = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def on_llm_end(self, response, **kwargs):
        usage = response.llm_output.get("token_usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        cost = self.calculate_cost(input_tokens, output_tokens)
        self.calls.append({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
        })

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        rates = COSTS.get(self.model, COSTS["gpt-4o"])
        input_cost = (input_tokens / 1_000_000) * rates["input"]
        output_cost = (output_tokens / 1_000_000) * rates["output"]
        return input_cost + output_cost

    @property
    def total_cost(self) -> float:
        return sum(c["cost_usd"] for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def summary(self) -> dict:
        return {
            "agent": self.agent_name,
            "model": self.model,
            "calls": len(self.calls),
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.total_cost, 6),
        }

# --- Smart Router ---
class ModelRouter:
    """Routes queries to the right model based on complexity."""

    def __init__(self):
        # Expensive model — for complex reasoning
        self.powerful = ChatOpenAI(model="gpt-4o", temperature=0)
        # Cheap model — for simple classification
        self.efficient = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.routing_log = []

    def route(self, task_type: str) -> tuple[ChatOpenAI, str]:
        """
        Route to the right model based on task type.

        Simple tasks  → gpt-4o-mini (10x cheaper)
        Complex tasks → gpt-4o (better reasoning)
        """
        simple_tasks = [
            "severity_classification",
            "status_check",
            "feed_count",
            "binary_decision",
        ]
        complex_tasks = [
            "synthesis",
            "action_plan",
            "pattern_recognition",
            "executive_summary",
        ]

        if task_type in simple_tasks:
            model = "gpt-4o-mini"
            self.routing_log.append({"task": task_type, "routed_to": model, "reason": "simple classification"})
            return self.efficient, model
        else:
            model = "gpt-4o"
            self.routing_log.append({"task": task_type, "routed_to": model, "reason": "complex reasoning"})
            return self.powerful, model

router = ModelRouter()

# --- Tools ---
@tool
def get_fund_nav(fund_id: str) -> dict:
    """Get NAV status for a fund."""
    return {
        "FUND001": {"nav": 142.35, "status": "FAILED", "last_updated": "2026-04-21 08:00"},
        "FUND002": {"nav": 98.12, "status": "SUCCESS", "last_updated": "2026-04-21 08:05"},
        "FUND003": {"nav": 0.00, "status": "PENDING", "last_updated": "2026-04-21 07:45"},
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
        "FEED_PRICE_01": {"status": "DOWN", "hours_down": 10},
        "FEED_PRICE_02": {"status": "UP"},
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

# --- Cost-Optimized Agents ---
async def severity_agent(fund_id: str, nav: dict, feed_statuses: list, consumers: list) -> tuple[str, dict]:
    """Simple classification — use cheap model."""
    llm, model = router.route("severity_classification")
    tracker = TokenTracker("severity_agent", model)
    llm_tracked = llm.with_config({"callbacks": [tracker]})

    feeds_down = [f for f in feed_statuses if f.get("status") == "DOWN"]

    messages = [
        SystemMessage(content="Classify incident severity. Respond JSON only."),
        HumanMessage(content=f"""
        NAV: {nav.get('status')}
        Feeds DOWN: {len(feeds_down)}
        Consumers: {consumers}

        CRITICAL if: NAV FAILED + feed down + 3+ consumers
        CRITICAL if: RegulatoryReporter or SettlementEngine impacted
        STANDARD: everything else

        {{"severity": "CRITICAL", "reason": "one sentence"}}
        """)
    ]

    response = await llm_tracked.ainvoke(messages)
    try:
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content.strip())
    except:
        result = {"severity": "CRITICAL", "reason": "Parse failed"}

    return result, tracker.summary()

async def fund_agent(fund_id: str, nav: dict) -> tuple[str, dict]:
    """Moderate complexity — use powerful model."""
    llm, model = router.route("pattern_recognition")
    tracker = TokenTracker("fund_agent", model)
    llm_tracked = llm.with_config({"callbacks": [tracker]})

    messages = [
        SystemMessage(content="You are a Fund NAV specialist. Be concise."),
        HumanMessage(content=f"""
        Fund: {fund_id}
        NAV: {nav}
        Provide a 3-bullet fund status report.
        """)
    ]
    response = await llm_tracked.ainvoke(messages)
    return response.content, tracker.summary()

async def feed_agent(fund_id: str, feed_statuses: list) -> tuple[str, dict]:
    """Moderate complexity — use powerful model."""
    llm, model = router.route("pattern_recognition")
    tracker = TokenTracker("feed_agent", model)
    llm_tracked = llm.with_config({"callbacks": [tracker]})

    messages = [
        SystemMessage(content="You are a Feed specialist. Be concise."),
        HumanMessage(content=f"""
        Fund: {fund_id}
        Feeds: {feed_statuses}
        Provide a 3-bullet feed health report.
        """)
    ]
    response = await llm_tracked.ainvoke(messages)
    return response.content, tracker.summary()

async def consumer_agent(fund_id: str, consumers: list) -> tuple[str, dict]:
    """Simple status check — use cheap model."""
    llm, model = router.route("status_check")
    tracker = TokenTracker("consumer_agent", model)
    llm_tracked = llm.with_config({"callbacks": [tracker]})

    messages = [
        SystemMessage(content="You are a Downstream Impact specialist. Flag RegulatoryReporter and SettlementEngine as HIGH RISK."),
        HumanMessage(content=f"""
        Fund: {fund_id}
        Consumers: {consumers}
        Provide a 3-bullet impact report.
        """)
    ]
    response = await llm_tracked.ainvoke(messages)
    return response.content, tracker.summary()

async def supervisor_agent(fund_report: str, feed_report: str, consumer_report: str) -> tuple[str, dict]:
    """Complex synthesis — always use powerful model."""
    llm, model = router.route("executive_summary")
    tracker = TokenTracker("supervisor_agent", model)
    llm_tracked = llm.with_config({"callbacks": [tracker]})

    messages = [
        SystemMessage(content="You are an Incident Supervisor. Synthesize concisely in 3-4 sentences."),
        HumanMessage(content=f"""
        Fund Agent: {fund_report}
        Feed Agent: {feed_report}
        Consumer Agent: {consumer_report}
        Situation, urgency, recommended action, ETA.
        """)
    ]
    response = await llm_tracked.ainvoke(messages)
    return response.content, tracker.summary()

# --- Cost Report ---
def print_cost_report(agent_summaries: list, routing_log: list):
    print(f"\n{'='*60}")
    print("COST REPORT")
    print(f"{'='*60}")

    total_cost = sum(s["cost_usd"] for s in agent_summaries)
    total_tokens = sum(s["total_tokens"] for s in agent_summaries)

    print(f"\n{'Agent':<20} {'Model':<15} {'Tokens':>8} {'Cost':>10}")
    print(f"{'─'*55}")

    for s in agent_summaries:
        print(f"{s['agent']:<20} {s['model']:<15} {s['total_tokens']:>8} ${s['cost_usd']:>9.6f}")

    print(f"{'─'*55}")
    print(f"{'TOTAL':<20} {'':15} {total_tokens:>8} ${total_cost:>9.6f}")

    # Routing decisions
    print(f"\n{'='*60}")
    print("ROUTING DECISIONS")
    print(f"{'='*60}")
    for r in routing_log:
        icon = "💰" if r["routed_to"] == "gpt-4o-mini" else "🧠"
        print(f"  {icon} {r['task']:<30} → {r['routed_to']} ({r['reason']})")

    # Savings calculation
    mini_tasks = [r for r in routing_log if r["routed_to"] == "gpt-4o-mini"]
    powerful_tasks = [r for r in routing_log if r["routed_to"] == "gpt-4o"]

    print(f"\n{'='*60}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Tasks routed to gpt-4o-mini: {len(mini_tasks)} (10x cheaper)")
    print(f"  Tasks routed to gpt-4o:      {len(powerful_tasks)} (full reasoning)")
    print(f"  Total cost this run:          ${total_cost:.6f}")
    print(f"  Projected cost per 100 runs:  ${total_cost * 100:.4f}")
    print(f"  Projected cost per 1000 runs: ${total_cost * 1000:.4f}")

    # What if everything was gpt-4o?
    mini_tokens = sum(
        s["total_tokens"] for s in agent_summaries
        if s["model"] == "gpt-4o-mini"
    )
    cost_if_all_powerful = total_cost + (mini_tokens / 1_000_000) * (
        COSTS["gpt-4o"]["input"] - COSTS["gpt-4o-mini"]["input"]
    )
    savings = cost_if_all_powerful - total_cost
    print(f"\n  Cost if all gpt-4o:           ${cost_if_all_powerful:.6f}")
    print(f"  Savings from routing:          ${savings:.6f} per run")
    print(f"  Savings per 1000 runs:         ${savings * 1000:.4f}")

# --- Main Runner ---
async def run_cost_optimized_triage(fund_id: str) -> dict:
    print(f"\n[COST AGENT] Starting cost-optimized triage for {fund_id}")

    # Gather data
    nav = get_fund_nav.invoke({"fund_id": fund_id})
    feed_ids = get_feeds_for_fund.invoke({"fund_id": fund_id})
    feed_statuses = [
        {"feed_id": fid, **check_feed_status.invoke({"feed_id": fid})}
        for fid in feed_ids
    ]
    consumers = get_impacted_consumers.invoke({"fund_id": fund_id})

    start = time.time()

    # Run agents in parallel — each routed to right model
    print("[COST AGENT] Running parallel agents with smart routing...")
    (
        (severity_result, severity_summary),
        (fund_report, fund_summary),
        (feed_report, feed_summary),
        (consumer_report, consumer_summary),
    ) = await asyncio.gather(
        severity_agent(fund_id, nav, feed_statuses, consumers),
        fund_agent(fund_id, nav),
        feed_agent(fund_id, feed_statuses),
        consumer_agent(fund_id, consumers),
    )

    # Supervisor always uses powerful model
    synthesis, supervisor_summary = await supervisor_agent(
        fund_report, feed_report, consumer_report
    )

    elapsed = time.time() - start

    # Print cost report
    all_summaries = [
        severity_summary,
        fund_summary,
        feed_summary,
        consumer_summary,
        supervisor_summary,
    ]
    print_cost_report(all_summaries, router.routing_log)

    print(f"\n{'='*60}")
    print("TRIAGE RESULT")
    print(f"{'='*60}")
    print(f"Fund:      {fund_id}")
    print(f"Severity:  {severity_result.get('severity')}")
    print(f"Reason:    {severity_result.get('reason')}")
    print(f"Time:      {elapsed:.2f}s")
    print(f"\nSynthesis:\n{synthesis}")

    return {
        "fund_id": fund_id,
        "severity": severity_result.get("severity"),
        "synthesis": synthesis,
        "agent_costs": all_summaries,
        "elapsed": elapsed,
    }

if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_cost_optimized_triage("FUND001"))