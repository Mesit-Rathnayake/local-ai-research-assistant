from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from tools import search_documents


llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)

llm_with_tools = llm.bind_tools(
    [search_documents]
)


question = input("Question: ")

response = llm_with_tools.invoke(
    [HumanMessage(content=question)]
)

print("\nLLM Response:")
print(response)

print("\nTool Calls:")

if response.tool_calls:
    for tool_call in response.tool_calls:
        print(tool_call)
else:
    print("No tool call.")