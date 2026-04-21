import json
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- Feed Registry ---
# In production this would be a database or API call
# This simulates your 160+ feed registry
FEED_REGISTRY = {
    "FEED_PRICE_01": {
        "name": "Primary Price Feed",
        "type": "price",
        "vendor": "Bloomberg",
        "funds": ["FUND001", "FUND003"],
        "criticality": "HIGH",
        "current_status": "DOWN",
        "hours_down": 10,
        "last_success": "2026-04-20 22:00",
        "sla_minutes": 30,
    },
    "FEED_PRICE_02": {
        "name": "Secondary Price Feed",
        "type": "price",
        "vendor": "Refinitiv",
        "funds": ["FUND002"],
        "criticality": "HIGH",
        "current_status": "UP",
        "last_success": "2026-04-21 08:00",
        "sla_minutes": 30,
    },
    "FEED_CORP_ACTION": {
        "name": "Corporate Actions Feed",
        "type": "corporate_action",
        "vendor": "FactSet",
        "funds": ["FUND001"],
        "criticality": "MEDIUM",
        "current_status": "DELAYED",
        "delay_mins": 120,
        "last_success": "2026-04-21 06:00",
        "sla_minutes": 60,
    },
    "FEED_FX_RATES": {
        "name": "FX Rates Feed",
        "type": "fx",
        "vendor": "Reuters",
        "funds": ["FUND001", "FUND002", "FUND003"],
        "criticality": "HIGH",
        "current_status": "UP",
        "last_success": "2026-04-21 08:10",
        "sla_minutes": 15,
    },
    "FEED_BENCHMARK": {
        "name": "Benchmark Index Feed",
        "type": "benchmark",
        "vendor": "MSCI",
        "funds": ["FUND002", "FUND003"],
        "criticality": "LOW",
        "current_status": "UP",
        "last_success": "2026-04-21 07:55",
        "sla_minutes": 120,
    },
}

# --- Consumer Registry ---
CONSUMER_REGISTRY = {
    "RetailPortal": {"criticality": "HIGH", "sla_minutes": 60},
    "AdvisorDashboard": {"criticality": "HIGH", "sla_minutes": 60},
    "RegulatoryReporter": {"criticality": "CRITICAL", "sla_minutes": 30},
    "SettlementEngine": {"criticality": "CRITICAL", "sla_minutes": 15},
    "RiskAnalytics": {"criticality": "MEDIUM", "sla_minutes": 120},
}

# --- Dynamic Tool Factory ---
class ToolFactory:
    """Generates tools dynamically from registry metadata."""

    def __init__(self, feed_registry: dict):
        self.feed_registry = feed_registry
        self.generated_tools = {}
        print(f"[TOOL FACTORY] Initialized with {len(feed_registry)} feeds in registry")

    def create_feed_tool(self, feed_id: str) -> StructuredTool:
        """Dynamically create a tool for a specific feed."""
        feed_meta = self.feed_registry[feed_id]

        # Tool input schema — generated from metadata
        class FeedStatusInput(BaseModel):
            include_history: bool = Field(
                default=False,
                description="Include historical status data"
            )

        # Tool function — closure captures feed_id and metadata
        def check_feed(include_history: bool = False) -> dict:
            result = {
                "feed_id": feed_id,
                "name": feed_meta["name"],
                "vendor": feed_meta["vendor"],
                "type": feed_meta["type"],
                "criticality": feed_meta["criticality"],
                "status": feed_meta["current_status"],
                "funds_affected": feed_meta["funds"],
                "sla_minutes": feed_meta["sla_minutes"],
                "last_success": feed_meta.get("last_success"),
            }
            if feed_meta["current_status"] == "DOWN":
                result["hours_down"] = feed_meta.get("hours_down", 0)
                result["sla_breached"] = (
                    feed_meta.get("hours_down", 0) * 60 > feed_meta["sla_minutes"]
                )
            if feed_meta["current_status"] == "DELAYED":
                result["delay_mins"] = feed_meta.get("delay_mins", 0)
                result["sla_breached"] = (
                    feed_meta.get("delay_mins", 0) > feed_meta["sla_minutes"]
                )
            return result

        # Self-describing docstring — generated from metadata
        docstring = (
            f"Check status of {feed_meta['name']} ({feed_id}). "
            f"Type: {feed_meta['type']}. "
            f"Vendor: {feed_meta['vendor']}. "
            f"Criticality: {feed_meta['criticality']}. "
            f"Affects funds: {', '.join(feed_meta['funds'])}. "
            f"SLA: {feed_meta['sla_minutes']} minutes."
        )
        check_feed.__doc__ = docstring

        # Create structured tool
        tool = StructuredTool.from_function(
            func=check_feed,
            name=f"check_{feed_id.lower()}",
            description=docstring,
            args_schema=FeedStatusInput,
        )

        self.generated_tools[feed_id] = tool
        return tool

    def generate_all_tools(self) -> Dict[str, StructuredTool]:
        """Generate tools for every feed in registry."""
        print(f"[TOOL FACTORY] Generating tools for all {len(self.feed_registry)} feeds...")
        for feed_id in self.feed_registry:
            self.create_feed_tool(feed_id)
            print(f"  ✅ Tool generated: check_{feed_id.lower()}")
        print(f"[TOOL FACTORY] {len(self.generated_tools)} tools ready")
        return self.generated_tools

    def generate_tools_for_fund(self, fund_id: str) -> Dict[str, StructuredTool]:
        """Generate tools only for feeds relevant to a specific fund."""
        relevant_feeds = [
            fid for fid, meta in self.feed_registry.items()
            if fund_id in meta.get("funds", [])
        ]
        print(f"[TOOL FACTORY] Generating {len(relevant_feeds)} tools for {fund_id}...")
        tools = {}
        for feed_id in relevant_feeds:
            tool = self.create_feed_tool(feed_id)
            tools[feed_id] = tool
            print(f"  ✅ Tool generated: check_{feed_id.lower()}")
        return tools

    def add_feed_at_runtime(self, feed_id: str, feed_meta: dict) -> StructuredTool:
        """Add a new feed to registry and generate its tool immediately."""
        self.feed_registry[feed_id] = feed_meta
        tool = self.create_feed_tool(feed_id)
        print(f"[TOOL FACTORY] Runtime tool added: {feed_id}")
        return tool

    def list_tools(self):
        """Show all generated tools with descriptions."""
        print(f"\n[TOOL FACTORY] Generated Tools ({len(self.generated_tools)}):")
        for feed_id, tool in self.generated_tools.items():
            print(f"  📡 {tool.name}")
            print(f"     {tool.description[:100]}...")

# --- Dynamic Triage Agent ---
async def run_dynamic_triage(fund_id: str, factory: ToolFactory) -> dict:
    """Agent that uses dynamically generated tools."""
    print(f"\n[DYNAMIC AGENT] Starting triage for {fund_id}")

    # Generate tools specific to this fund at runtime
    fund_tools = factory.generate_tools_for_fund(fund_id)

    if not fund_tools:
        return {"error": f"No feeds found for {fund_id}"}

    # Run all feed checks in parallel using dynamic tools
    print(f"[DYNAMIC AGENT] Running {len(fund_tools)} dynamic tools in parallel...")

    async def check_feed_async(feed_id: str, tool: StructuredTool) -> dict:
        result = tool.invoke({"include_history": False})
        return result

    feed_results = await asyncio.gather(*[
        check_feed_async(fid, tool)
        for fid, tool in fund_tools.items()
    ])

    feeds_down = [f for f in feed_results if f.get("status") == "DOWN"]
    feeds_delayed = [f for f in feed_results if f.get("status") == "DELAYED"]
    sla_breaches = [f for f in feed_results if f.get("sla_breached")]

    print(f"[DYNAMIC AGENT] Results: {len(feeds_down)} down, {len(feeds_delayed)} delayed, {len(sla_breaches)} SLA breaches")

    # Agent reasons over dynamic results
    tool_descriptions = "\n".join([
        f"- {t.name}: {t.description[:80]}"
        for t in fund_tools.values()
    ])

    messages = [
        SystemMessage(content="""You are an incident triage specialist.
        You have access to dynamically generated tools for each feed.
        Analyze all feed statuses and provide a structured incident assessment."""),
        HumanMessage(content=f"""
        Fund: {fund_id}
        Available Tools Used: 
        {tool_descriptions}

        Feed Results:
        {json.dumps(feed_results, indent=2)}

        Feeds Down: {[f['feed_id'] for f in feeds_down]}
        SLA Breaches: {[f['feed_id'] for f in sla_breaches]}

        Provide:
        1. Severity — CRITICAL or STANDARD with reason
        2. Priority order — which feed to fix first and why
        3. ETA — based on feed SLAs and criticality
        """)
    ]

    response = await llm.ainvoke(messages)

    return {
        "fund_id": fund_id,
        "tools_generated": len(fund_tools),
        "feed_results": feed_results,
        "feeds_down": [f["feed_id"] for f in feeds_down],
        "sla_breaches": [f["feed_id"] for f in sla_breaches],
        "assessment": response.content,
    }

# --- Demo ---
async def run_demo():
    print("\n" + "="*60)
    print("DYNAMIC TOOL GENERATION DEMO")
    print("="*60)

    factory = ToolFactory(FEED_REGISTRY)

    # Demo 1 — Generate all tools
    print("\n--- DEMO 1: Generate tools for entire registry ---")
    all_tools = factory.generate_all_tools()
    factory.list_tools()

    # Demo 2 — Triage FUND001 with dynamic tools
    print("\n--- DEMO 2: Triage FUND001 with dynamic tools ---")
    result1 = await run_dynamic_triage("FUND001", factory)
    print(f"\nTools generated for FUND001: {result1['tools_generated']}")
    print(f"Feeds checked: {[f['feed_id'] for f in result1['feed_results']]}")
    print(f"Feeds down: {result1['feeds_down']}")
    print(f"SLA breaches: {result1['sla_breaches']}")
    print(f"\nAssessment:\n{result1['assessment']}")

    # Demo 3 — Add new feed at runtime
    print("\n--- DEMO 3: Add new feed at runtime ---")
    print("[DEMO] Simulating new feed onboarded to platform...")

    new_feed = {
        "name": "ESG Data Feed",
        "type": "esg",
        "vendor": "Sustainalytics",
        "funds": ["FUND001", "FUND002"],
        "criticality": "MEDIUM",
        "current_status": "UP",
        "last_success": "2026-04-21 08:00",
        "sla_minutes": 240,
    }

    new_tool = factory.add_feed_at_runtime("FEED_ESG_01", new_feed)
    print(f"[DEMO] New tool created: {new_tool.name}")
    print(f"[DEMO] Description: {new_tool.description}")

    # Use the new tool immediately
    result = new_tool.invoke({"include_history": False})
    print(f"[DEMO] New tool output: {result}")

    # Demo 4 — Triage FUND001 again — now includes ESG feed
    print("\n--- DEMO 4: Triage FUND001 again — ESG feed now included ---")
    result2 = await run_dynamic_triage("FUND001", factory)
    print(f"Tools generated: {result2['tools_generated']} (was {result1['tools_generated']} before)")
    print(f"Feeds checked: {[f['feed_id'] for f in result2['feed_results']]}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✅ {len(all_tools)} tools generated from registry automatically")
    print(f"✅ Fund-specific tools generated at runtime")
    print(f"✅ New feed added mid-session — agent used it immediately")
    print(f"✅ No hardcoded tool definitions — fully dynamic")
    print(f"\nIn production: point FEED_REGISTRY at your actual feed database")
    print(f"Agent automatically picks up new feeds as they are onboarded")

if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_demo())