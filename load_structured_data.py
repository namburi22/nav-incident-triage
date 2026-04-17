from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory="./nav_knowledge_base",
    embedding_function=embeddings,
    collection_name="nav_incidents"
)

# --- Fund NAV data as documents ---
fund_documents = [
    Document(
        page_content="FUND001 current NAV status failed value 142.35 last updated 2026-04-14 08:00",
        metadata={
            "type": "fund_status",
            "fund_id": "FUND001",
            "nav": 142.35,
            "status": "FAILED",
            "last_updated": "2026-04-14 08:00"
        }
    ),
    Document(
        page_content="FUND002 current NAV status success value 98.12 last updated 2026-04-14 08:05",
        metadata={
            "type": "fund_status",
            "fund_id": "FUND002",
            "nav": 98.12,
            "status": "SUCCESS",
            "last_updated": "2026-04-14 08:05"
        }
    ),
    Document(
        page_content="FUND003 current NAV status pending value 0.00 last updated 2026-04-14 07:45",
        metadata={
            "type": "fund_status",
            "fund_id": "FUND003",
            "nav": 0.00,
            "status": "PENDING",
            "last_updated": "2026-04-14 07:45"
        }
    ),
]

# --- Feed status data as documents ---
feed_documents = [
    Document(
        page_content="FEED_PRICE_01 price feed status down last success 2026-04-13 22:00",
        metadata={
            "type": "feed_status",
            "feed_id": "FEED_PRICE_01",
            "status": "DOWN",
            "last_success": "2026-04-13 22:00"
        }
    ),
    Document(
        page_content="FEED_PRICE_02 price feed status up last success 2026-04-14 08:00",
        metadata={
            "type": "feed_status",
            "feed_id": "FEED_PRICE_02",
            "status": "UP",
            "last_success": "2026-04-14 08:00"
        }
    ),
    Document(
        page_content="FEED_CORP_ACTION corporate action feed status delayed last success 2026-04-14 06:00",
        metadata={
            "type": "feed_status",
            "feed_id": "FEED_CORP_ACTION",
            "status": "DELAYED",
            "last_success": "2026-04-14 06:00"
        }
    ),
]

# --- Fund to feed mapping ---
mapping_documents = [
    Document(
        page_content="FUND001 uses feeds FEED_PRICE_01 FEED_CORP_ACTION",
        metadata={
            "type": "fund_feed_mapping",
            "fund_id": "FUND001",
            "feeds": "FEED_PRICE_01,FEED_CORP_ACTION"
        }
    ),
    Document(
        page_content="FUND002 uses feeds FEED_PRICE_02",
        metadata={
            "type": "fund_feed_mapping",
            "fund_id": "FUND002",
            "feeds": "FEED_PRICE_02"
        }
    ),
    Document(
        page_content="FUND003 uses feeds FEED_PRICE_01",
        metadata={
            "type": "fund_feed_mapping",
            "fund_id": "FUND003",
            "feeds": "FEED_PRICE_01"
        }
    ),
]

# --- Consumer mapping ---
consumer_documents = [
    Document(
        page_content="FUND001 downstream consumers RetailPortal AdvisorDashboard RegulatoryReporter SettlementEngine",
        metadata={
            "type": "consumer_mapping",
            "fund_id": "FUND001",
            "consumers": "RetailPortal,AdvisorDashboard,RegulatoryReporter,SettlementEngine"
        }
    ),
    Document(
        page_content="FUND002 downstream consumers RetailPortal",
        metadata={
            "type": "consumer_mapping",
            "fund_id": "FUND002",
            "consumers": "RetailPortal"
        }
    ),
    Document(
        page_content="FUND003 downstream consumers AdvisorDashboard SettlementEngine",
        metadata={
            "type": "consumer_mapping",
            "fund_id": "FUND003",
            "consumers": "AdvisorDashboard,SettlementEngine"
        }
    ),
]

# Add all to ChromaDB
all_docs = fund_documents + feed_documents + mapping_documents + consumer_documents
vectorstore.add_documents(all_docs)

print(f"Added {len(all_docs)} structured documents to ChromaDB")
print("\nTesting retrieval...")

# Test fund query
results = vectorstore.get(where={
    "$and": [
        {"fund_id": {"$eq": "FUND001"}},
        {"type": {"$eq": "fund_status"}}
    ]
})
print(f"\nFUND001 status: {results['metadatas'][0]}")

# Test feed query
results = vectorstore.get(where={
    "$and": [
        {"feed_id": {"$eq": "FEED_PRICE_01"}},
        {"type": {"$eq": "feed_status"}}
    ]
})
print(f"FEED_PRICE_01 status: {results['metadatas'][0]}")

# Test mapping query
results = vectorstore.get(where={
    "$and": [
        {"fund_id": {"$eq": "FUND001"}},
        {"type": {"$eq": "fund_feed_mapping"}}
    ]
})
print(f"FUND001 feeds: {results['metadatas'][0]['feeds']}")

print("\nStructured data loaded successfully.")