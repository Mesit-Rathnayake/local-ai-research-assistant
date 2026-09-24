from typing import TypedDict

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

from langgraph.graph import END, START, StateGraph

from memory import init_db, load_messages, save_message

CHROMA_PATH = "chroma_db"


class RAGState(TypedDict):
    question: str
    rewritten_query: str
    documents: list
    answer: str
    conversation_history: list
    relevant: bool
    retry_count: int


embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vector_store = Chroma(
    persist_directory=CHROMA_PATH,
    collection_name="research_papers",
    embedding_function=embeddings
)

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)

rewrite_prompt = ChatPromptTemplate.from_template("""
Rewrite the user's question into a standalone search query for a document retrieval system.

Use the conversation history to resolve references such as:
- it
- they
- this
- that
- these
- those

If this is a retry, create a different search query using alternative keywords
or terminology that could retrieve relevant information.

Do not answer the question.

Return ONLY the search query.

Conversation history:
{history}

Original question:
{question}

Retry attempt:
{retry_count}
""")

answer_prompt = ChatPromptTemplate.from_template("""
You are a research assistant.

Answer the question using ONLY the provided context.

Each context section contains its source document and page information.

If the answer cannot be found in the context, say:
"I don't have enough information in the provided documents."

Do not use your own knowledge to answer.

Context:
{context}

Question:
{question}

Answer:
""")


def rewrite_query(state: RAGState):
    history = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in state["conversation_history"]
    )

    messages = rewrite_prompt.format_messages(
        history=history,
        question=state["question"],
        retry_count=state["retry_count"]
    )

    rewritten_query = llm.invoke(messages).content.strip()

    print("\nSearch query:")
    print(rewritten_query)

    return {
        "rewritten_query": rewritten_query
    }


def retrieve_documents(state: RAGState):
    results = vector_store.similarity_search_with_relevance_scores(
        state["rewritten_query"],
        k=3
    )

    relevant_results = [
        (document, score)
        for document, score in results
        if score >= 0.5
    ]

    return {
        "documents": relevant_results,
        "relevant": len(relevant_results) > 0
    }


def route_after_retrieval(state: RAGState):
    if state["relevant"]:
        return "generate"

    if state["retry_count"] < 1:
        return "retry"

    return "give_up"


def generate_answer(state: RAGState):
    context_parts = []

    for document, score in state["documents"]:
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page")

        if page is not None:
            source_info = f"{source} | Page {page + 1}"
        else:
            source_info = source

        context_parts.append(
            f"[Source: {source_info}]\n"
            f"{document.page_content}"
        )

    context = "\n\n".join(context_parts)

    messages = answer_prompt.format_messages(
        context=context,
        question=state["question"]
    )

    answer = llm.invoke(messages).content.strip()

    print("\nAnswer:")
    print(answer)

    print("\nSources:")

    for document, score in state["documents"]:
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page")

        if page is not None:
            print(f"- {source}, Page {page + 1}")
        else:
            print(f"- {source}")

    return {
        "answer": answer
    }


def retry(state: RAGState):
    retry_count = state["retry_count"] + 1

    alternative_query = (
        f"{state['question']} "
        f"definition concepts related terminology"
    )

    print(f"\nRetrying retrieval... Attempt {retry_count}")
    print("Alternative search query:")
    print(alternative_query)

    return {
        "retry_count": retry_count,
        "rewritten_query": alternative_query
    }


def give_up(state: RAGState):
    answer = "I don't have enough information in the provided documents."

    print("\nAnswer:")
    print(answer)

    return {
        "answer": answer
    }


def save_memory(state: RAGState):
    save_message("user", state["question"])
    save_message("assistant", state["answer"])

    return {
        "conversation_history": load_messages()
    }


init_db()

graph_builder = StateGraph(RAGState)

graph_builder.add_node("rewrite_query", rewrite_query)
graph_builder.add_node("retrieve_documents", retrieve_documents)
graph_builder.add_node("generate", generate_answer)
graph_builder.add_node("retry", retry)
graph_builder.add_node("give_up", give_up)
graph_builder.add_node("save_memory", save_memory)

graph_builder.add_edge(START, "rewrite_query")
graph_builder.add_edge("rewrite_query", "retrieve_documents")

graph_builder.add_conditional_edges(
    "retrieve_documents",
    route_after_retrieval,
    {
        "generate": "generate",
        "retry": "retry",
        "give_up": "give_up"
    }
)
graph_builder.add_edge("generate", "save_memory")
graph_builder.add_edge("retry", "retrieve_documents")
graph_builder.add_edge("give_up", "save_memory")
graph_builder.add_edge("save_memory", END)

graph = graph_builder.compile()

conversation_history = load_messages()

while True:
    question = input("\nQuestion: ")

    if question.lower() in ["exit", "quit"]:
        break

    result = graph.invoke({
        "question": question,
        "rewritten_query": "",
        "documents": [],
        "answer": "",
        "conversation_history": conversation_history,
        "relevant": False,
        "retry_count": 0
    })

    conversation_history = result["conversation_history"]
