from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Use local embeddings — no API key needed
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Synthetic incident history — realistic NAV platform scenarios
documents = [
    # FEED_PRICE_01 incidents
    Document(
        page_content="""
        Incident: FEED_PRICE_01 Down
        Date: 2026-03-10
        Fund: FUND001
        Issue: Price feed FEED_PRICE_01 went down at 07:45 AM causing NAV calculation failure.
        Root Cause: Network timeout between feed provider and ingestion service.
        Resolution: Feed ops team restarted the ingestion service. Feed restored at 08:30 AM.
        Duration: 45 minutes
        Impact: FUND001 NAV delayed. RetailPortal and AdvisorDashboard affected.
        Prevention: Added health check ping every 5 minutes on feed connection.
        """,
        metadata={"fund": "FUND001", "feed": "FEED_PRICE_01", "type": "outage", "duration_mins": 45}
    ),
    Document(
        page_content="""
        Incident: FEED_PRICE_01 Intermittent Failures
        Date: 2026-01-15
        Fund: FUND001, FUND003
        Issue: FEED_PRICE_01 sending incomplete price data intermittently.
        Root Cause: Feed provider upstream database under heavy load during market open.
        Resolution: Switched to backup price feed temporarily. Primary restored after 2 hours.
        Duration: 120 minutes
        Impact: NAV calculations for FUND001 and FUND003 had to be reprocessed.
        Prevention: Implemented backup feed failover mechanism.
        """,
        metadata={"fund": "FUND001,FUND003", "feed": "FEED_PRICE_01", "type": "intermittent", "duration_mins": 120}
    ),
    Document(
        page_content="""
        Incident: FEED_PRICE_01 Complete Outage
        Date: 2025-11-22
        Fund: FUND001, FUND003
        Issue: FEED_PRICE_01 completely unavailable from 06:00 AM to 09:00 AM.
        Root Cause: Feed provider infrastructure maintenance not communicated in advance.
        Resolution: Manual price data entry for critical funds. Automated feeds restored at 09:00 AM.
        Duration: 180 minutes
        Impact: All funds dependent on FEED_PRICE_01 had delayed NAV publication.
        Prevention: Established advance notification SLA with feed provider.
        """,
        metadata={"fund": "FUND001,FUND003", "feed": "FEED_PRICE_01", "type": "outage", "duration_mins": 180}
    ),

    # FEED_CORP_ACTION incidents
    Document(
        page_content="""
        Incident: Corporate Action Feed Missing Data
        Date: 2026-02-22
        Fund: FUND001
        Issue: FEED_CORP_ACTION missing dividend data for 3 securities in FUND001.
        Root Cause: Corporate action vendor did not process weekend announcements.
        Resolution: Manual override applied. Data backfilled from secondary source.
        Duration: 120 minutes
        Impact: FUND001 NAV temporarily incorrect. RegulatoryReporter flagged discrepancy.
        Prevention: Added weekend corporate action monitoring job.
        """,
        metadata={"fund": "FUND001", "feed": "FEED_CORP_ACTION", "type": "missing_data", "duration_mins": 120}
    ),
    Document(
        page_content="""
        Incident: Corporate Action Feed Delayed
        Date: 2026-01-08
        Fund: FUND001, FUND002
        Issue: FEED_CORP_ACTION delayed by 3 hours due to vendor system upgrade.
        Root Cause: Vendor performed unscheduled system maintenance during market hours.
        Resolution: Delayed NAV publication by 2 hours. Feed caught up automatically.
        Duration: 180 minutes
        Impact: Settlement engine delayed. Advisors notified of late NAV publication.
        Prevention: Vendor maintenance windows now restricted to off-market hours.
        """,
        metadata={"fund": "FUND001,FUND002", "feed": "FEED_CORP_ACTION", "type": "delay", "duration_mins": 180}
    ),

    # NAV calculation incidents
    Document(
        page_content="""
        Incident: NAV Calculation Timeout
        Date: 2026-04-01
        Fund: FUND003
        Issue: NAV calculation job timed out after 30 minutes.
        Root Cause: Unusually large number of corporate actions processed simultaneously.
        Resolution: Job reprocessed with increased timeout threshold.
        Duration: 30 minutes
        Impact: FUND003 NAV published 1 hour late. AdvisorDashboard showed stale data.
        Prevention: Increased timeout threshold and added parallel processing.
        """,
        metadata={"fund": "FUND003", "feed": "INTERNAL", "type": "timeout", "duration_mins": 30}
    ),
    Document(
        page_content="""
        Incident: NAV Calculation Error Due to Bad Price Data
        Date: 2025-12-10
        Fund: FUND002
        Issue: NAV calculation produced incorrect value due to stale price data.
        Root Cause: FEED_PRICE_02 cached yesterday's prices due to daylight saving time issue.
        Resolution: Cache cleared, prices refreshed, NAV recalculated.
        Duration: 60 minutes
        Impact: FUND002 NAV published incorrect value for 1 hour. Regulatory report amended.
        Prevention: Added timestamp validation on all incoming price data.
        """,
        metadata={"fund": "FUND002", "feed": "FEED_PRICE_02", "type": "bad_data", "duration_mins": 60}
    ),

    # Settlement and downstream incidents
    Document(
        page_content="""
        Incident: SettlementEngine Failure Due to Late NAV
        Date: 2026-02-10
        Fund: FUND001
        Issue: SettlementEngine could not process trades because NAV was not published by deadline.
        Root Cause: Cascading failure from FEED_PRICE_01 outage causing late NAV publication.
        Resolution: NAV published manually. Settlement processed with 2 hour delay.
        Duration: 120 minutes
        Impact: 450 trades delayed. Compliance team notified. No financial loss.
        Prevention: Added NAV publication deadline alerts at T-30 minutes.
        """,
        metadata={"fund": "FUND001", "feed": "FEED_PRICE_01", "type": "cascading", "duration_mins": 120}
    ),
    Document(
        page_content="""
        Incident: RegulatoryReporter Missing NAV Data
        Date: 2025-10-15
        Fund: FUND001, FUND003
        Issue: RegulatoryReporter submitted incomplete NAV data to regulator.
        Root Cause: NAV calculation failure not detected before regulatory reporting window.
        Resolution: Amended report submitted within 4 hours. Regulator notified.
        Duration: 240 minutes
        Impact: Regulatory compliance risk. Internal audit triggered.
        Prevention: Added pre-submission NAV validation gate in reporting pipeline.
        """,
        metadata={"fund": "FUND001,FUND003", "feed": "FEED_PRICE_01", "type": "regulatory", "duration_mins": 240}
    ),

    # Resolution playbooks
    Document(
        page_content="""
        Playbook: FEED_PRICE_01 Recovery Steps
        When FEED_PRICE_01 goes down follow these steps:
        Step 1: Check feed provider status page for outage announcement.
        Step 2: Attempt to restart ingestion service via feed ops dashboard.
        Step 3: If not restored in 15 minutes switch to backup price feed FEED_PRICE_BACKUP.
        Step 4: Notify NAV calculation team of feed switch.
        Step 5: Monitor NAV calculation for anomalies after feed restoration.
        Step 6: Send stakeholder notification if NAV publication will be delayed.
        Typical resolution time: 30-45 minutes.
        Escalation: If not resolved in 60 minutes escalate to feed provider account manager.
        """,
        metadata={"type": "playbook", "feed": "FEED_PRICE_01"}
    ),
    Document(
        page_content="""
        Playbook: Corporate Action Feed Recovery Steps
        When FEED_CORP_ACTION is delayed or missing data:
        Step 1: Check vendor portal for data availability status.
        Step 2: If delay less than 30 minutes wait for automatic catch up.
        Step 3: If delay more than 30 minutes initiate manual data pull from vendor API.
        Step 4: Validate corporate action data completeness before NAV calculation.
        Step 5: If data incomplete apply manual override with secondary source data.
        Step 6: Document all manual overrides for audit trail.
        Typical resolution time: 60-120 minutes.
        Escalation: Contact vendor support if delay exceeds 2 hours.
        """,
        metadata={"type": "playbook", "feed": "FEED_CORP_ACTION"}
    ),
]

# Create ChromaDB vector store
print("Creating ChromaDB knowledge base...")
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./nav_knowledge_base",
    collection_name="nav_incidents"
)

print(f"Created knowledge base with {len(documents)} documents")
print("\nTesting retrieval...")

# Test queries
test_queries = [
    "price feed down NAV failure",
    "corporate action missing data",
    "how to recover from feed outage",
    "settlement engine delayed"
]

for query in test_queries:
    print(f"\nQuery: '{query}'")
    results = vectorstore.similarity_search(query, k=2)
    for r in results:
        # Print first line of each result
        first_line = r.page_content.strip().split('\n')[0]
        print(f"  → {first_line}")

print("\nKnowledge base ready.")