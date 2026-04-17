from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

# Memory — this is what makes the agent remember across turns
memory = MemorySaver()

agent = create_react_agent(llm, tools=[multiply, add], checkpointer=memory)

# thread_id ties messages together — same thread = same memory
config = {"configurable": {"thread_id": "session_1"}}

print("Agent ready. Type your questions. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    response = agent.invoke(
        {"messages": [("user", user_input)]},
        config=config
    )

    # Get just the last message — the final answer
    last_message = response["messages"][-1]
    print(f"Agent: {last_message.content}\n")