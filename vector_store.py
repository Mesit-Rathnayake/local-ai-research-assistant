import requests
import math

EMBEDDING_URL = "http://localhost:11434/api/embeddings"
GENERATE_URL = "http://localhost:11434/api/generate"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:3b"


def embed(text):
    response = requests.post(
        EMBEDDING_URL,
        json={
            "model": EMBEDDING_MODEL,
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


def generate_answer(prompt):
    response = requests.post(
        GENERATE_URL,
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    response.raise_for_status()

    return response.json()["response"]


documents = [
    "My favorite programming language is Go.",
    "I am studying Computer Engineering.",
    "I built a project called MailMind.",
    "MailMind is an AI-powered email management application.",
    "I enjoy working with computer vision.",
    "My Academia is a university learning management system.",
    "I use Ollama to run local AI models."
]


document_embeddings = []

for document in documents:
    print(f"Embedding: {document}")
    document_embeddings.append(embed(document))


while True:
    question = input("\nQuestion: ")

    if question.lower() in ["exit", "quit"]:
        break

    question_embedding = embed(question)

    results = []

    for document, document_embedding in zip(
        documents,
        document_embeddings
    ):
        similarity = cosine_similarity(
            question_embedding,
            document_embedding
        )

        results.append((similarity, document))

    results.sort(reverse=True)

    top_results = results[:3]

    context = "\n".join(
        document for similarity, document in top_results
    )

    prompt = f"""
You are a helpful assistant.

Answer the question using only the information provided in the context.

If the answer cannot be found in the context, say:
"I don't have enough information to answer that."

Context:
{context}

Question:
{question}

Answer:
"""

    answer = generate_answer(prompt)

    print("\nRetrieved Context:")
    for similarity, document in top_results:
        print(f"{similarity:.4f} → {document}")

    print("\nAnswer:")
    print(answer)
