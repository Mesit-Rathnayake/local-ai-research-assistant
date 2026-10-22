
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from memory import init_db, save_message, load_messages
from tools import search_documents


llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)

init_db()
conversation_history = load_messages()

print("Local Research Agent ready. Type 'exit' to quit.")

while True:
    question = input("\nQuestion: ").strip()

    if question.lower() in ["exit", "quit"]:
        break

    if not question:
        continue

    save_message("user", question)

    search_result = search_documents.invoke({
        "query": question
    })

    if "No relevant information was found" in search_result:
        answer = (
            "I don't have enough information in the provided documents "
            "to answer that question."
        )
    else:
        messages = [
            SystemMessage(
                content="""You are a research assistant answering questions from retrieved documents.

Use the supplied excerpts as your evidence.

Instructions:
- Read all excerpts before answering.
- Combine relevant information from multiple excerpts when needed.
- Answer directly if the excerpts contain evidence that addresses the question.
- Do not require an excerpt to contain the exact wording of the question.
- Do not add factual claims that are unsupported by the excerpts.
- If the excerpts genuinely contain no evidence relevant to the question, say that the documents do not contain enough information.
- Keep the answer clear and concise."""
            )
        ]

        messages.extend(
            HumanMessage(
                content=f"Previous conversation:\n{message['content']}"
            )
            for message in conversation_history[-6:]
        )

        messages.append(
            HumanMessage(
                content=(
                    f"Question: {question}\n\n"
                    f"Retrieved document excerpts:\n{search_result}"
                )
            )
        )

        response = llm.invoke(messages)
        answer = response.content

    print("\nAnswer:")
    print(answer)

    save_message("assistant", answer)

    conversation_history.extend([
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ])
