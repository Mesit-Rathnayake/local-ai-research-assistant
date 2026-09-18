import requests
import math

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"


def embed(text):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": text
        }
    )

    response.raise_for_status()

    return response.json()["embedding"]


def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))

    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(x * x for x in b))

    return dot_product / (magnitude_a * magnitude_b)


text1 = "My favorite programming language is Go."
text2 = "I really enjoy programming in Go."
text3 = "I had pizza for dinner yesterday."

embedding1 = embed(text1)
embedding2 = embed(text2)
embedding3 = embed(text3)

print("Go vs Go:")
print(cosine_similarity(embedding1, embedding2))

print("\nGo vs Pizza:")
print(cosine_similarity(embedding1, embedding3))