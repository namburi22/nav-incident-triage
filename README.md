# 🏦 NAV Incident Triage Assistant

> *Agentic AI system that automates mutual fund NAV incident investigation — from 45-minute manual triage to 30-second automated analysis.*

🔗 **Live Demo**: [nav-incident-triage.streamlit.app](https://nav-incident-triage.streamlit.app)

---

## What It Does

When a NAV calculation fails, a traditional L2 engineer spends 30–45 minutes manually:
- Checking which feeds are down
- Correlating feed failures to NAV impact
- Looking up past incidents for patterns
- Identifying downstream systems at risk
- Writing up an incident summary

This system does all of that automatically — in under 30 seconds — using a multi-agent architecture that reasons the way an experienced engineer would.

---

## Architecture

```
User selects Fund → Run Triage
         ↓
LangGraph — controls the flow
         ↓
[DECISION] LLM evaluates NAV severity
         ↓
    FAILED/PENDING              SUCCESS
         ↓                         ↓
  ┌──────────────┐              → Skip (no investigation needed)
  │  Fund Agent  │  NAV status + incident history
  │  Feed Agent  │  Feed health + discovery
  │Consumer Agent│  Downstream system impact
  │Knowledge Agent│ RAG: historical incidents + playbooks
  └──────────────┘
         ↓
    SUPERVISOR
    Synthesizes all reports → Executive summary + ETA
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Orchestration | LangGraph | Controlled agent flow with LLM-driven edges |
| Tool Protocol | MCP (Model Context Protocol) | Standardised tool exposure — any agent, any tool |
| Knowledge | ChromaDB + RAG | Semantic search over historical incidents and playbooks |
| LLM | GPT-4o | Reasoning engine for all agents |
| Observability | LangSmith | Full trace of every agent decision, latency, token cost |
| UI | Streamlit | Dual-audience interface — engineers and directors |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | Local embeddings, no API cost |

---

## Agents

### Fund Agent
- Fetches current NAV status and value
- Retrieves incident history for the fund
- Produces a fund health report with risk assessment

### Feed Agent
- Discovers which feeds are associated with the fund
- Checks status of each feed individually
- Identifies severity and NAV calculation impact

### Consumer Agent
- Maps downstream systems dependent on the fund's NAV
- Assesses business severity per system
- Determines urgency of resolution

### Knowledge Agent
- Queries ChromaDB vector store semantically
- Retrieves similar past incidents and proven resolution playbooks
- Calculates average resolution time from historical data

### Supervisor
- Receives all four specialist reports
- Synthesizes into executive incident summary
- Produces structured output: Situation → Impact → Root Cause → Action Plan → ETA

---

## Key Design Decisions

**Why MCP?**
Standard protocol for tool exposure. Any agent can consume any MCP server without custom integration. Adding new data sources means updating the server — not rewriting agents.

**Why LangGraph over pure agents?**
Production systems need guaranteed execution paths, conditional routing, and audit trails. LangGraph gives control where needed and flexibility where appropriate.

**Why multi-agent over single agent?**
Each specialist has focused context and tooling. The supervisor synthesizes without domain noise. Scaling means adding nodes — not rewriting code.

**Why RAG + structured tools?**
Real-time structured data (feed status, NAV values) comes from tools. Historical knowledge (incident patterns, playbooks) comes from ChromaDB. Different retrieval strategies for different data types.

**Why ChromaDB as unified data layer?**
One source of truth for both structured data (fund/feed metadata) and unstructured knowledge (incident history, playbooks). Semantic search over historical patterns that SQL cannot replicate.

---

## Project Structure

```
agent-bootcamp/
├── app.py                      # Streamlit UI — production demo
├── nav_mcp_server_v2.py        # MCP server — ChromaDB-powered tools
├── full_system.py              # Complete system — MCP + LangGraph + Multi-agent
├── error_handling.py           # Production error handling patterns
├── final_system.py             # MCP + async agents integrated
├── multi_agent.py              # Multi-agent supervisor pattern
├── graph1.py / graph2.py       # LangGraph state machine examples
├── agent1.py → agent4.py       # Progressive agent complexity
├── create_knowledge_base.py    # RAG knowledge base setup
├── load_structured_data.py     # Structured fund/feed data loader
├── rag_agent.py                # RAG tool demonstration
├── nav_knowledge_base/         # ChromaDB persistence
├── requirements.txt
├── runtime.txt                 # Python 3.11
└── README.md
```

---

## Setup

### Prerequisites
- Python 3.11+
- OpenAI API key
- LangSmith API key (optional — for observability)

### Installation

```bash
# Clone the repo
git clone https://github.com/namburi22/nav-incident-triage.git
cd nav-incident-triage

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file:

```
OPENAI_API_KEY=your_openai_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT=nav-incident-triage
```

### Initialize Knowledge Base

```bash
# Create incident knowledge base with historical data
python create_knowledge_base.py

# Load structured fund and feed data
python load_structured_data.py
```

### Run Locally

```bash
# Terminal 1 — Start MCP server
python nav_mcp_server_v2.py

# Terminal 2 — Start Streamlit UI
streamlit run app.py
```

---

## Demo Guide

1. Open [nav-incident-triage.streamlit.app](https://nav-incident-triage.streamlit.app)
2. Select a fund from the sidebar:
   - **FUND001** — NAV FAILED (triggers full investigation)
   - **FUND002** — NAV SUCCESS (skips investigation)
   - **FUND003** — NAV PENDING (triggers investigation)
3. Click **🚀 Run Triage**
4. Watch agents work live in the **Investigation** tab
5. View clean executive output in the **Executive Summary** tab
6. Download the incident report as Markdown

---

## Learning Journey

This system was built across a structured learning plan covering:

| Day | Concept | File |
|---|---|---|
| 1 | Plain LLM call | agent1.py |
| 1 | ReAct agent + tool use | agent2.py |
| 1 | Memory across turns | agent3.py |
| 1 | Domain tools + triage agent | agent4.py |
| 2 | LangGraph state machine | graph1.py |
| 2 | Tools wired into nodes | graph2.py |
| 2 | Multi-agent + supervisor | multi_agent.py |
| 2 | MCP server + client | nav_mcp_server.py |
| 3 | RAG knowledge base | create_knowledge_base.py |
| 3 | RAG + agents connected | rag_agent.py |
| 4 | Full system — everything connected | full_system.py |
| 4 | Error handling + retry logic | error_handling.py |
| 5 | Streamlit UI deployed | app.py |

---

## Production Path

| Current (Demo) | Production |
|---|---|
| Synthetic NAV/feed data | Real feed APIs, ServiceNow tickets |
| Local ChromaDB | Managed vector store (Pinecone, Weaviate) |
| Single Streamlit instance | Kubernetes deployment |
| Static fund registry | Dynamic fund registry from database |
| Manual knowledge base | Auto-ingestion from incident tickets |

---

## What This Demonstrates

**For Engineers:**
- ReAct agent reasoning loop
- LangGraph state machine with conditional edges
- MCP tool protocol implementation
- RAG with semantic + metadata filtering
- Async multi-agent orchestration
- Error handling and retry patterns
- LangSmith observability integration

**For Directors:**
- Institutional knowledge encoded as queryable AI
- Compliance-ready audit trail per investigation
- Dual-audience output — technical and executive
- Production deployment with CI/CD via GitHub
- Scalable architecture — add agents as nodes

---

## Author

**Deepak Namburi**
Principal Software Engineer — Fidelity Investments
13+ years fintech platform engineering
Specialising in NAV lifecycle, fund accounting, high-volume distributed systems

---

## Built With

- [LangChain](https://langchain.com) — LLM framework
- [LangGraph](https://langchain-ai.github.io/langgraph) — Agent orchestration
- [MCP](https://modelcontextprotocol.io) — Tool protocol
- [ChromaDB](https://chromadb.com) — Vector store
- [LangSmith](https://smith.langchain.com) — Observability
- [Streamlit](https://streamlit.io) — UI framework
- [OpenAI GPT-4o](https://openai.com) — Reasoning engine