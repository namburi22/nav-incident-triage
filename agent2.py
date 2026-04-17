from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Define a tool — this is something the agent can DECIDE to use
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

# Now give the LLM these tools — it becomes an agent
agent = create_react_agent(llm, tools=[multiply, add])

# Ask it something that requires tool use
response = agent.invoke({
    "messages": [("user", "What is 25 * 47, then add 100 to that result?")]
})

# Print the full message chain — watch it think
for message in response["messages"]:
    print(f"\n[{message.type.upper()}]: {message.content}")