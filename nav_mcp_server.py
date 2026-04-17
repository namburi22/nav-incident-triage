from mcp.server.fastmcp import FastMCP

# Create MCP server
mcp = FastMCP("NAV Incident Server")

# Expose tools via MCP — same domain tools, now served over protocol
@mcp.tool()
def get_fund_nav(fund_id: str) -> dict:
    """Get the current NAV status for a fund."""
    funds = {
        "FUND001": {"nav": 142.35, "status": "FAILED", "last_updated": "2026-04-14 08:00"},
        "FUND002": {"nav": 98.12, "status": "SUCCESS", "last_updated": "2026-04-14 08:05"},
        "FUND003": {"nav": 0.00, "status": "PENDING", "last_updated": "2026-04-14 07:45"},
    }
    return funds.get(fund_id, {"error": f"Fund {fund_id} not found"})

@mcp.tool()
def get_feeds_for_fund(fund_id: str) -> list:
    """Get all feed IDs associated with a fund."""
    fund_feeds = {
        "FUND001": ["FEED_PRICE_01", "FEED_CORP_ACTION"],
        "FUND002": ["FEED_PRICE_02"],
        "FUND003": ["FEED_PRICE_01"],
    }
    return fund_feeds.get(fund_id, [])

@mcp.tool()
def check_feed_status(feed_id: str) -> dict:
    """Check the status of a data feed."""
    feeds = {
        "FEED_PRICE_01": {"status": "DOWN", "last_success": "2026-04-13 22:00"},
        "FEED_PRICE_02": {"status": "UP", "last_success": "2026-04-14 08:00"},
        "FEED_CORP_ACTION": {"status": "DELAYED", "last_success": "2026-04-14 06:00"},
    }
    return feeds.get(feed_id, {"error": f"Feed {feed_id} not found"})

@mcp.tool()
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

@mcp.tool()
def get_impacted_consumers(fund_id: str) -> list:
    """Get downstream consumers impacted by a fund NAV failure."""
    consumers = {
        "FUND001": ["RetailPortal", "AdvisorDashboard", "RegulatoryReporter", "SettlementEngine"],
        "FUND002": ["RetailPortal"],
        "FUND003": ["AdvisorDashboard", "SettlementEngine"],
    }
    return consumers.get(fund_id, [])

if __name__ == "__main__":
    print("Starting NAV MCP Server...")
    mcp.run(transport="stdio")