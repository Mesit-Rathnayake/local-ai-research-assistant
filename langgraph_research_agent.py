
import re
from typing import TypedDict

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from memory import init_db, load_messages, save_message
from tools import retrieve_documents


class ResearchState(TypedDict):
    question: str
    search_query: str
    context: str
    documents: list[dict]
    has_evidence: bool
    answer: str
    history: list[dict]


llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)


def retrieve_node(state: ResearchState) -> dict:
    documents = retrieve_documents(state["search_query"])
    context = "\n\n".join(
        f'[Source ID: {document["id"]}]\n'
        f'[Source: {document["source"]}'
        f'{f", Page {document["page"]}" if document["page"] is not None else ""}]\n'
        f'{document["content"]}'
        for document in documents
    )

    return {
        "documents": documents,
        "context": context,
        "has_evidence": bool(documents)
    }


def route_after_retrieval(state: ResearchState):
    if state["has_evidence"]:
        return "generate"

    return "refuse"


def format_citations(answer: str, documents: list[dict]) -> str:
    source_map = {
        str(document["id"]): (
            f'{document["source"]}'
            f'{f", Page {document["page"]}" if document["page"] is not None else ""}'
        )
        for document in documents
    }

    def replace_citation(match):
        source_id = match.group(1)
        if source_id not in source_map:
            return ""

        return f'[Source: {source_map[source_id]}]'

    return re.sub(r"\[(\d+)\]", replace_citation, answer)


def generate_node(state: ResearchState) -> dict:
    messages = [
        SystemMessage(
            content="""You are a research assistant.

Answer using only the retrieved document excerpts as factual evidence.

Rules:
- Answer the question clearly and concisely.
- Cite factual claims using the source IDs shown in the excerpts.
- Use citation markers like [1] or [4].
- Only use source IDs that appear in the supplied excerpts.
- Do not invent sources, page numbers, or facts.
- If the evidence is insufficient, say so."""
        ),
        HumanMessage(
            content=(
                f"Document context:\n{state['context']}\n\n"
                f"Question: {state['question']}"
            )
        )
    ]

    response = llm.invoke(messages)
    answer = format_citations(response.content, state["documents"])

    return {
        "answer": answer
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
            "search_query": question,
            "context": "",
            "documents": [],
            "has_evidence": False,
            "answer": "",
            "history": load_messages()
        })

        print("\nAnswer:")
        print(result["answer"])
