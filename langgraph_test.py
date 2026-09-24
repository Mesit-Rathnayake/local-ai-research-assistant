from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    message: str


def greet(state: State):
    return {
        "message": state["message"] + " -> Hello from LangGraph!"
    }


graph_builder = StateGraph(State)

graph_builder.add_node("greet", greet)

graph_builder.add_edge(START, "greet")
graph_builder.add_edge("greet", END)

graph = graph_builder.compile()

result = graph.invoke({
    "message": "User"
})

print(result)
