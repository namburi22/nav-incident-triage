from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# The brain — GPT-4o as our reasoning engine
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# First — plain LLM call, no agency yet
# This is just question -> answer, no thinking loop
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is 25 * 47?")
]

response = llm.invoke(messages)
print("Plain LLM response:", response.content)