from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


documents = [
    "MailMind is an AI-powered email management application. It automatically analyzes incoming emails and helps identify important messages, deadlines, and required actions. The application was built using Go and integrates with Gmail.",

    "My Academia is a university learning management system. It was developed using React, Node.js, and MongoDB. The system includes a citation-based RAG chatbot that allows students to ask questions about their academic content.",

    "I am studying Computer Engineering at the University of Ruhuna. I am interested in artificial intelligence, machine learning, computer vision, and software engineering.",

    "I enjoy working with computer vision. I have worked on projects involving image processing, facial landmarks, image enhancement, and computer vision models."
]


embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


vector_store = Chroma.from_texts(
    texts=documents,
    embedding=embeddings,
    collection_name="research_assistant"
)


retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


while True:
    question = input("\nQuestion: ")

    if question.lower() in ["exit", "quit"]:
        break

    results = retriever.invoke(question)

    print("\nRetrieved Documents:")

    for i, document in enumerate(results):
        print(f"\n{i + 1}. {document.page_content}")
