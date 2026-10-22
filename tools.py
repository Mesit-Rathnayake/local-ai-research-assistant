from langchain_core.tools import tool

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


CHROMA_PATH = "chroma_db"


embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vector_store = Chroma(
    persist_directory=CHROMA_PATH,
    collection_name="research_papers",
    embedding_function=embeddings
)


@tool
def search_documents(query: str) -> str:
    """Search the local research documents for information relevant to a query."""

    results = vector_store.similarity_search_with_relevance_scores(
        query,
        k=5
    )

    relevant_results = [
        (document, score)
        for document, score in results
        if score >= 0.4
    ]

    if not relevant_results:
        return "No relevant information was found in the documents."

    output = []

    for document, score in relevant_results:
        source = document.metadata.get(
            "source",
            "Unknown source"
        )

        page = document.metadata.get("page")

        if page is not None:
            source_info = f"{source}, Page {page + 1}"
        else:
            source_info = source

        output.append(
            f"[Source: {source_info}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(output)


def retrieve_documents(query: str) -> list[dict]:
    results = vector_store.similarity_search_with_relevance_scores(
        query,
        k=5
    )

    relevant_results = [
        (document, score)
        for document, score in results
        if score >= 0.4
    ]

    documents = []

    for index, (document, score) in enumerate(relevant_results, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page")

        documents.append({
            "id": index,
            "source": source,
            "page": page + 1 if page is not None else None,
            "content": document.page_content,
            "score": score
        })

    return documents