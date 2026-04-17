from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Load existing knowledge base
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./nav_knowledge_base",
    embedding_function=embeddings,
    collection_name="nav_incidents"
)

@tool
def search_incident_knowledge(query: str, fund_id: str = "") -> str:
    """Search historical incident knowledge base for relevant incidents and playbooks."""
    
    # Search without filter first — simpler and more reliable
    results = vectorstore.similarity_search(
        query,
        k=3
    )
    
    if not results:
        return "No relevant incidents found in knowledge base."
    
    # Format results for LLM
    context = ""
    for i, doc in enumerate(results, 1):
        context += f"\n--- Result {i} ---\n"
        context += doc.page_content.strip()
        context += "\n"
    
    return context

# Test it
print("Testing RAG tool...\n")

result = search_incident_knowledge.invoke({
    "query": "FEED_PRICE_01 is down what should I do",
    "fund_id": "FUND001"
})

print("RAG Result:")
print(result)

print("\n" + "="*50)
print("Now asking LLM to reason over retrieved context...")

prompt = f"""
You are a NAV incident specialist.
A user asked: "FEED_PRICE_01 is down, what should I do?"

Here is relevant knowledge from past incidents:
{result}

Based on this historical knowledge, provide specific actionable guidance.
"""

response = llm.invoke(prompt)
print("\nLLM Response:")
print(response.content)