import requests
import math


EMBEDDING_URL = "http://localhost:11434/api/embeddings"
GENERATE_URL = "http://localhost:11434/api/generate"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:3b"


documents = [
    """
    MailMind is an AI-powered email management application.
    It automatically analyzes incoming emails and helps identify
    important messages, deadlines, and required actions.
    The application was built using Go and integrates with Gmail.
    """,

    """
    My Academia is a university learning management system.
    It was developed using React, Node.js, and MongoDB.
    The system includes a citation-based RAG chatbot that allows
    students to ask questions about their academic content.
    """,

    """
    I am studying Computer Engineering at the University of Ruhuna.
    I am interested in artificial intelligence, machine learning,
    computer vision, and software engineering.
    """,

    """
    I enjoy working with computer vision.
    I have worked on projects involving image processing,
    facial landmarks, image enhancement, and computer vision models.
    """
]


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


def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))

    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(x * x for x in b))

    return dot_product / (magnitude_a * magnitude_b)


def chunk_text(text, chunk_size=40, overlap=10):
    words = text.split()

    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        if len(chunk.split()) < overlap:
            break

        chunks.append(chunk)

        if end >= len(words):
            break

        start += chunk_size - overlap

    return chunks


chunks = []

for document in documents:
    chunks.extend(chunk_text(document))


print(f"\nCreated {len(chunks)} chunks.")

print("\nGenerating embeddings...")

chunk_embeddings = []

for chunk in chunks:
    chunk_embeddings.append(embed(chunk))

print("Embeddings ready.")


while True:
    question = input("\nQuestion: ")

    if question.lower() in ["exit", "quit"]:
        break

    question_embedding = embed(question)

    results = []

    for chunk, chunk_embedding in zip(chunks, chunk_embeddings):
        similarity = cosine_similarity(
            question_embedding,
            chunk_embedding
        )

        results.append((similarity, chunk))

    results.sort(reverse=True)

    top_results = results[:3]

    context = "\n\n".join(
        chunk for similarity, chunk in top_results
    )

    prompt = f"""
You are a helpful assistant.

Answer the question using only the information provided
in the context.

If the answer cannot be found in the context, say:
"I don't have enough information to answer that."

Context:
{context}

Question:
{question}

Answer:
"""

    answer = generate_answer(prompt)

    print("\nRetrieved Chunks:")

    for similarity, chunk in top_results:
        print(f"\n{similarity:.4f} → {chunk}")

    print("\nAnswer:")
    print(answer)
