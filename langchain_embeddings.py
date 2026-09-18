from langchain_ollama import OllamaEmbeddings


embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


text = "I enjoy programming in Go."

vector = embeddings.embed_query(text)


print("Vector length:", len(vector))
print("First 10 values:", vector[:10])
