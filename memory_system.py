import json
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, Text
from sqlalchemy.orm import declarative_base, Session
from typing import List, Optional

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- Database Setup ---
Base = declarative_base()
engine = create_engine("sqlite:///nav_memory.db", echo=False)

class IncidentMemory(Base):
    """Stores one memory per triage run."""
    __tablename__ = "incident_memories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    fund_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.now)
    nav_status = Column(String)
    severity = Column(String)
    feeds_down = Column(String)        # JSON list
    consumers_impacted = Column(String) # JSON list
    resolution_taken = Column(String)
    resolution_time_mins = Column(Float, nullable=True)
    key_lesson = Column(Text)          # LLM-extracted insight
    sre_decision = Column(String, nullable=True)

Base.metadata.create_all(engine)
print("[MEMORY] Database ready — nav_memory.db")

# --- Memory Manager ---
class MemoryManager:

    def save_incident(self, incident: dict):
        """Save a triage run to memory."""
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
            print(f"[MEMORY] Saved incident for {incident['fund_id']}")

    def get_memories(self, fund_id: str, limit: int = 5) -> List[dict]:
        """Retrieve past incidents for a fund."""
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
                    "timestamp": m.timestamp.isoformat(),
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

    def get_feed_history(self, feed_id: str) -> List[dict]:
        """Get all incidents where a specific feed was down."""
        with Session(engine) as session:
            memories = (
                session.query(IncidentMemory)
                .filter(IncidentMemory.feeds_down.contains(feed_id))
                .order_by(IncidentMemory.timestamp.desc())
                .all()
            )
            return [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "fund_id": m.fund_id,
                    "resolution_taken": m.resolution_taken,
                    "resolution_time_mins": m.resolution_time_mins,
                    "key_lesson": m.key_lesson,
                }
                for m in memories
            ]

    def format_memories_for_agent(self, memories: List[dict]) -> str:
        """Format memories into context the agent can reason over."""
        if not memories:
            return "No previous incidents found for this fund."

        context = f"Past {len(memories)} incidents:\n"
        for i, m in enumerate(memories, 1):
            context += f"\n[Incident {i} — {m['timestamp'][:10]}]\n"
            context += f"  NAV Status: {m['nav_status']}\n"
            context += f"  Severity: {m['severity']}\n"
            context += f"  Feeds Down: {m['feeds_down']}\n"
            context += f"  Resolution: {m['resolution_taken']}\n"
            if m['resolution_time_mins']:
                context += f"  Time to Resolve: {m['resolution_time_mins']} minutes\n"
            context += f"  Key Lesson: {m['key_lesson']}\n"
        return context

memory_manager = MemoryManager()

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
        "FEED_PRICE_01": {"status": "DOWN", "down_since": "2026-04-20 22:00", "hours_down": 10},
        "FEED_PRICE_02": {"status": "UP", "last_success": "2026-04-21 08:00"},
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

# --- Memory-Aware Triage Agent ---
async def run_memory_aware_triage(fund_id: str) -> dict:
    """Triage agent that loads and uses past incident memory."""

    print(f"\n[MEMORY AGENT] Starting triage for {fund_id}")

    # Load past memories FIRST
    past_memories = memory_manager.get_memories(fund_id, limit=5)
    memory_context = memory_manager.format_memories_for_agent(past_memories)

    print(f"[MEMORY AGENT] Loaded {len(past_memories)} past incidents")
    if past_memories:
        print(f"[MEMORY AGENT] Most recent: {past_memories[0]['timestamp'][:10]}")

    # Gather current data
    nav = get_fund_nav.invoke({"fund_id": fund_id})
    feed_ids = get_feeds_for_fund.invoke({"fund_id": fund_id})
    feed_statuses = [
        {"feed_id": fid, **check_feed_status.invoke({"feed_id": fid})}
        for fid in feed_ids
    ]
    consumers = get_impacted_consumers.invoke({"fund_id": fund_id})
    feeds_down = [f["feed_id"] for f in feed_statuses if f.get("status") == "DOWN"]

    # Check feed-specific history
    feed_memories = {}
    for feed_id in feeds_down:
        feed_history = memory_manager.get_feed_history(feed_id)
        if feed_history:
            feed_memories[feed_id] = feed_history
            print(f"[MEMORY AGENT] Found {len(feed_history)} past incidents for {feed_id}")

    # Memory-aware agents run in parallel
    async def fund_agent():
        messages = [
            SystemMessage(content="""You are a Fund NAV specialist with access to incident history.
            Use past incidents to identify patterns and make better recommendations.
            Be specific — reference actual past incidents when relevant."""),
            HumanMessage(content=f"""
            Fund: {fund_id}
            Current NAV: {nav}
            Current Feed Status: {feed_statuses}

            PAST INCIDENT HISTORY:
            {memory_context}

            Provide a 3-bullet fund status report.
            Reference past incidents if you see a pattern.
            """)
        ]
        r = await llm.ainvoke(messages)
        return r.content

    async def feed_agent():
        feed_context = ""
        for feed_id, history in feed_memories.items():
            feed_context += f"\n{feed_id} past failures:\n"
            for h in history[:3]:
                feed_context += f"  - {h['timestamp'][:10]}: {h['resolution_taken']} ({h['resolution_time_mins']} min)\n"

        messages = [
            SystemMessage(content="""You are a Data Feed specialist with memory of past outages.
            Use feed history to recommend the fastest resolution path.
            If this feed failed before, tell the team exactly what fixed it."""),
            HumanMessage(content=f"""
            Fund: {fund_id}
            Current Feed Statuses: {feed_statuses}

            FEED OUTAGE HISTORY:
            {feed_context if feed_context else "No previous feed failures on record."}

            Provide a 3-bullet feed health report.
            Include specific resolution steps based on what worked before.
            """)
        ]
        r = await llm.ainvoke(messages)
        return r.content

    async def consumer_agent():
        messages = [
            SystemMessage(content="""You are a Downstream Impact specialist.
            Flag RegulatoryReporter and SettlementEngine as HIGH RISK.
            Use past incident context to assess recurring impact patterns."""),
            HumanMessage(content=f"""
            Fund: {fund_id}
            Impacted Consumers: {consumers}

            PAST CONTEXT:
            {memory_context}

            Provide a 3-bullet impact report.
            Note if any consumers were repeatedly impacted.
            """)
        ]
        r = await llm.ainvoke(messages)
        return r.content

    print("[MEMORY AGENT] Running parallel agents with memory context...")
    fund_report, feed_report, consumer_report = await asyncio.gather(
        fund_agent(), feed_agent(), consumer_agent()
    )

    # Supervisor synthesis with memory
    supervisor_messages = [
        SystemMessage(content="""You are an Incident Supervisor with full incident history.
        Synthesize current reports AND past patterns into actionable guidance.
        Be specific about what worked before and expected resolution time."""),
        HumanMessage(content=f"""
        Current Investigation:
        Fund Agent: {fund_report}
        Feed Agent: {feed_report}
        Consumer Agent: {consumer_report}

        Historical Context:
        {memory_context}

        Synthesize: situation, pattern recognition, recommended action, expected ETA.
        """)
    ]
    synthesis = await llm.ainvoke(supervisor_messages)

    # Extract key lesson for storage
    lesson_messages = [
        SystemMessage(content="Extract one concise key lesson from this incident for future reference. Max 2 sentences."),
        HumanMessage(content=f"""
        Fund: {fund_id}
        NAV: {nav.get('status')}
        Feeds Down: {feeds_down}
        Synthesis: {synthesis.content}
        """)
    ]
    lesson = await llm.ainvoke(lesson_messages)

    result = {
        "fund_id": fund_id,
        "nav_status": nav.get("status"),
        "severity": "CRITICAL" if feeds_down and nav.get("status") == "FAILED" else "STANDARD",
        "feeds_down": feeds_down,
        "consumers_impacted": consumers,
        "fund_report": fund_report,
        "feed_report": feed_report,
        "consumer_report": consumer_report,
        "synthesis": synthesis.content,
        "key_lesson": lesson.content,
        "past_incidents_loaded": len(past_memories),
    }

    return result

# --- Save incident to memory after triage ---
def save_to_memory(result: dict, resolution_taken: str, resolution_time_mins: float, sre_decision: str = None):
    """Call this after triage to persist the incident."""
    memory_manager.save_incident({
        "fund_id": result["fund_id"],
        "nav_status": result["nav_status"],
        "severity": result["severity"],
        "feeds_down": result["feeds_down"],
        "consumers_impacted": result["consumers_impacted"],
        "resolution_taken": resolution_taken,
        "resolution_time_mins": resolution_time_mins,
        "key_lesson": result["key_lesson"],
        "sre_decision": sre_decision,
    })

# --- Demo Runner ---
async def run_demo():
    print("\n" + "="*60)
    print("CROSS-SESSION MEMORY DEMO")
    print("="*60)
    print("This demo runs FUND001 twice.")
    print("Second run will use memory from the first.")
    print("="*60)

    # --- RUN 1 ---
    print("\n" + "─"*60)
    print("RUN 1 — First time seeing this incident")
    print("─"*60)

    result1 = await run_memory_aware_triage("FUND001")

    print(f"\n[RUN 1] Synthesis:")
    print(result1["synthesis"])

    # Save run 1 to memory
    save_to_memory(
        result=result1,
        resolution_taken="Restarted FEED_PRICE_01 ingestion service, switched to backup feed",
        resolution_time_mins=45,
        sre_decision="APPROVE"
    )
    print(f"\n[MEMORY] Run 1 saved to database")

    # --- RUN 2 ---
    print("\n" + "─"*60)
    print("RUN 2 — Agent now has memory of Run 1")
    print("─"*60)

    result2 = await run_memory_aware_triage("FUND001")

    print(f"\n[RUN 2] Synthesis:")
    print(result2["synthesis"])

    # Save run 2 to memory
    save_to_memory(
        result=result2,
        resolution_taken="Applied known fix — restarted ingestion service immediately",
        resolution_time_mins=20,
        sre_decision="APPROVE"
    )
    print(f"\n[MEMORY] Run 2 saved to database")

    # --- Compare ---
    print("\n" + "="*60)
    print("MEMORY IMPACT")
    print("="*60)
    print(f"Run 1 — Past incidents loaded: {result1['past_incidents_loaded']}")
    print(f"Run 2 — Past incidents loaded: {result2['past_incidents_loaded']}")
    print(f"\nRun 1 key lesson: {result1['key_lesson']}")
    print(f"\nRun 2 key lesson: {result2['key_lesson']}")
    print(f"\n✅ Agent is now learning from each run")
    print(f"   Database: nav_memory.db")
    print(f"   Run again tomorrow — it will remember today")

if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_demo())