from typing import TypedDict

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from tools import search_documents


class State(TypedDict):
    messages: list


llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)

tools = [search_documents]

llm_with_tools = llm.bind_tools(tools)

tool_node = ToolNode(tools)


def agent(state: State):
    response = llm_with_tools.invoke(state["messages"])

    print("\nAgent response:")
    print(response)

    return {
        "messages": state["messages"] + [response]
    }


def route(state: State):
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return "end"


graph_builder = StateGraph(State)

graph_builder.add_node("agent", agent)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "agent")

graph_builder.add_conditional_edges(
    "agent",
    route,
    {
        "tools": "tools",
        "end": END
    }
)

graph_builder.add_edge("tools", "agent")

graph = graph_builder.compile()


question = input("Question: ")

result = graph.invoke({
    "messages": [
        HumanMessage(content=question)
    ]
})

print("\nAnswer:")

print(result["messages"][-1].content)