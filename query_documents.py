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


while True:
    question = input("\nQuestion: ")

    if question.lower() in ["exit", "quit"]:
        break

    results = vector_store.similarity_search_with_relevance_scores(
        question,
        k=3
    )

    print("\nRetrieved Documents:")

    for i, (document, score) in enumerate(results):
        print(f"\n--- Result {i + 1} | Score: {score:.4f} ---")
        print(document.page_content)

        print("\nMetadata:")
        print(document.metadata)