from tools import search_documents


result = search_documents.invoke(
    {
        "query": "What is RAG?"
    }
)

print(result)