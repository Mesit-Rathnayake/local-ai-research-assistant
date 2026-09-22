from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

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

rewrite_prompt = ChatPromptTemplate.from_template("""
Rewrite the user's current question into a standalone search query.

Use the conversation history to resolve references such as:
- it
- they
- this
- that
- these
- those

Do not answer the question.

Return ONLY the rewritten search query.

Conversation history:
{history}

Current question:
{question}
""")

prompt = ChatPromptTemplate.from_template("""
You are a research assistant.

Use the conversation history to understand the user's current question.

Answer the question using ONLY the provided document context.

If the answer cannot be found in the document context, say:
"I don't have enough information in the provided documents."

Do not use your own knowledge to answer.

Conversation history:
{history}

Document context:
{context}

Current question:
{question}

Answer:
""")

conversation_history = []

while True:
    question = input("\nQuestion: ")

    if question.lower() in ["exit", "quit"]:
        break

    history = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in conversation_history
    )

    rewrite_messages = rewrite_prompt.format_messages(
        history=history,
        question=question
    )

    rewritten_query = llm.invoke(rewrite_messages).content.strip()

    print("\nSearch query:")
    print(rewritten_query)

    results = vector_store.similarity_search_with_relevance_scores(
        rewritten_query,
        k=3
    )

    relevant_results = [
        (document, score)
        for document, score in results
        if score >= 0.5
    ]

    if not relevant_results:
        answer = "I don't have enough information in the provided documents."

        print("\nAnswer:")
        print(answer)

        conversation_history.append({
            "role": "user",
            "content": question
        })

        conversation_history.append({
            "role": "assistant",
            "content": answer
        })

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

    history = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in conversation_history
    )

    messages = prompt.format_messages(
        history=history,
        context=context,
        question=question
    )

    response = llm.invoke(messages)

    answer = response.content

    print("\nAnswer:")
    print(answer)

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

    conversation_history.append({
        "role": "user",
        "content": question
    })

    conversation_history.append({
        "role": "assistant",
        "content": answer
    })
