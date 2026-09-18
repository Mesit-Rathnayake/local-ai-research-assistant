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


all_chunks = []

for document in documents:
    chunks = chunk_text(document)

    all_chunks.extend(chunks)


for i, chunk in enumerate(all_chunks):
    print(f"\nChunk {i + 1}:")
    print(chunk)
