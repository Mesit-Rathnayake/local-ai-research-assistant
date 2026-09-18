from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


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
    collection_name="research_assistant_rag"
)


retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.5,
        "k": 3
    }
)


llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)


prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Answer the question using only the provided context.

If the answer cannot be found in the context, say:
"I don't have enough information to answer that."

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

    retrieved_documents = retriever.invoke(question)

    context = "\n\n".join(
        document.page_content
        for document in retrieved_documents
    )

    formatted_prompt = prompt.invoke({
        "context": context,
        "question": question
    })

    response = llm.invoke(formatted_prompt)

    print("\nAnswer:")
    print(response.content)
