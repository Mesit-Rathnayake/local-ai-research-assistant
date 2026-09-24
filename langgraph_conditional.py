from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    question: str
    relevant: bool
    result: str


def retrieve(state: State):
    question = state["question"]

    if "python" in question.lower():
        return {
            "relevant": True
        }

    return {
        "relevant": False
    }


def generate_answer(state: State):
    return {
        "result": "I found relevant information."
    }


def retry(state: State):
    return {
        "result": "I could not find relevant information."
    }


def route_after_retrieval(state: State):
    if state["relevant"]:
        return "generate"

    return "retry"


graph_builder = StateGraph(State)

graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate_answer)
graph_builder.add_node("retry", retry)

graph_builder.add_edge(START, "retrieve")

graph_builder.add_conditional_edges(
    "retrieve",
    route_after_retrieval,
    {
        "generate": "generate",
        "retry": "retry"
    }
)
graph_builder.add_edge("generate", END)
graph_builder.add_edge("retry", END)

graph = graph_builder.compile()

question = input("Question: ")

result = graph.invoke({
    "question": question,
    "relevant": False,
    "result": ""
})

print("\nResult:")
print(result["result"])
