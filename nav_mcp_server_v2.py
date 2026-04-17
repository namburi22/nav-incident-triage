from mcp.server.fastmcp import FastMCP
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Connect to ChromaDB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./nav_knowledge_base",
    embedding_function=embeddings,
    collection_name="nav_incidents"
)

mcp = FastMCP("NAV Incident Server v2")

@mcp.tool()
def get_fund_nav(fund_id: str) -> dict:
    """Get the current NAV status for a fund."""
    results = vectorstore.get(where={
        "$and": [
            {"fund_id": {"$eq": fund_id}},
            {"type": {"$eq": "fund_status"}}
        ]
    })
    if not results["metadatas"]:
        return {"error": f"Fund {fund_id} not found"}
    return results["metadatas"][0]

@mcp.tool()
def get_feeds_for_fund(fund_id: str) -> list:
    """Get all feed IDs associated with a fund."""
    results = vectorstore.get(where={
        "$and": [
            {"fund_id": {"$eq": fund_id}},
            {"type": {"$eq": "fund_feed_mapping"}}
        ]
    })
    if not results["metadatas"]:
        return []
    feeds_str = results["metadatas"][0]["feeds"]
    return feeds_str.split(",")

@mcp.tool()
def check_feed_status(feed_id: str) -> dict:
    """Check the status of a data feed."""
    results = vectorstore.get(where={
        "$and": [
            {"feed_id": {"$eq": feed_id}},
            {"type": {"$eq": "feed_status"}}
        ]
    })
    if not results["metadatas"]:
        return {"error": f"Feed {feed_id} not found"}
    return results["metadatas"][0]

@mcp.tool()
def get_incident_history(fund_id: str) -> list:
    """Get past incidents for a fund."""
    results = vectorstore.similarity_search(
        f"{fund_id} incident history",
        k=3,
        filter={"type": "outage"}
    )
    history = []
    for doc in results:
        history.append(doc.page_content.strip())
    return history

@mcp.tool()
def get_impacted_consumers(fund_id: str) -> list:
    """Get downstream consumers impacted by a fund NAV failure."""
    results = vectorstore.get(where={
        "$and": [
            {"fund_id": {"$eq": fund_id}},
            {"type": {"$eq": "consumer_mapping"}}
        ]
    })
    if not results["metadatas"]:
        return []
    consumers_str = results["metadatas"][0]["consumers"]
    return consumers_str.split(",")

if __name__ == "__main__":
    print("Starting NAV MCP Server v2 — powered by ChromaDB...")
    mcp.run(transport="stdio")