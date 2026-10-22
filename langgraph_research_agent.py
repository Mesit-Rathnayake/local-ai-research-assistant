
from typing import TypedDict

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from memory import init_db, load_messages, save_message
from tools import search_documents


class ResearchState(TypedDict):
    question: str
    context: str
    has_evidence: bool
    answer: str
    history: list


llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)


def retrieve_node(state: ResearchState) -> dict:
    context = search_documents.invoke({
        "query": state["question"]
    })

    has_evidence = (
        "No relevant information was found in the documents."
        not in context
    )

    return {
        "context": context,
        "has_evidence": has_evidence
    }


def route_after_retrieval(state: ResearchState):
    if state["has_evidence"]:
        return "generate"

    return "refuse"


def generate_node(state: ResearchState) -> dict:
    messages = [
        SystemMessage(
            content=(
                "You are a research assistant. Answer using only the "
                "provided document context. Do not add unsupported facts."
            )
        ),
        HumanMessage(
            content=(
                f"Document context:\n{state['context']}\n\n"
                f"Question: {state['question']}"
            )
        )
    ]

    response = llm.invoke(messages)

    return {
        "answer": response.content
    }


def refuse_node(state: ResearchState) -> dict:
    return {
        "answer": (
            "I don't have enough information in the provided "
            "documents to answer that question."
        )
    }


def save_memory_node(state: ResearchState) -> dict:
    save_message("user", state["question"])
    save_message("assistant", state["answer"])

    return {}


graph_builder = StateGraph(ResearchState)
graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("generate", generate_node)
graph_builder.add_node("refuse", refuse_node)
graph_builder.add_node("save_memory", save_memory_node)
graph_builder.add_edge(START, "retrieve")
graph_builder.add_conditional_edges(
    "retrieve",
    route_after_retrieval,
    {
        "generate": "generate",
        "refuse": "refuse"
    }
)
graph_builder.add_edge("generate", "save_memory")
graph_builder.add_edge("refuse", "save_memory")
graph_builder.add_edge("save_memory", END)
research_graph = graph_builder.compile()


if __name__ == "__main__":
    init_db()

    print("LangGraph Research Agent ready. Type 'exit' to quit.")

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() in ["exit", "quit"]:
            break

        if not question:
            continue

        result = research_graph.invoke({
            "question": question,
            "context": "",
            "has_evidence": False,
            "answer": "",
            "history": load_messages()
        })

        print("\nAnswer:")
        print(result["answer"])
