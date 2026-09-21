from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma_db"

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

prompt = ChatPromptTemplate.from_template("""
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

while True:
    question = input("\nQuestion: ")

    if question.lower() in ["exit", "quit"]:
        break

    results = vector_store.similarity_search_with_relevance_scores(
        question,
        k=3
    )

    relevant_results = [
        (document, score)
        for document, score in results
        if score >= 0.5
    ]

    if not relevant_results:
        print("\nAnswer:")
        print("I don't have enough information in the provided documents.")
        continue

    context_parts = []

    for document, score in relevant_results:
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

    messages = prompt.format_messages(
        context=context,
        question=question
    )

    response = llm.invoke(messages)

    print("\nAnswer:")
    print(response.content)

    print("\nSources:")

    shown_sources = set()

    for document, score in relevant_results:
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page")

        if page is not None:
            source_info = f"{source}, Page {page + 1}"
        else:
            source_info = source

        if source_info not in shown_sources:
            print(f"- {source_info}")
            shown_sources.add(source_info)