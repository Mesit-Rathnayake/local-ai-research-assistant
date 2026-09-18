import sqlite3
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"

DB_NAME = "memory.db"


def init_db():
    connection = sqlite3.connect(DB_NAME)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


def save_message(role, content):
    connection = sqlite3.connect(DB_NAME)

    connection.execute(
        "INSERT INTO messages (role, content) VALUES (?, ?)",
        (role, content)
    )

    connection.commit()
    connection.close()


def load_messages():
    connection = sqlite3.connect(DB_NAME)

    messages = connection.execute(
        "SELECT role, content FROM messages ORDER BY id"
    ).fetchall()

    connection.close()

    return messages


def ask_llm(prompt):
    save_message("user", prompt)

    messages = load_messages()

    conversation = []

    for role, content in messages:
        conversation.append(f"{role.capitalize()}: {content}")

    full_prompt = "\n".join(conversation)

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": full_prompt,
            "stream": False
        }
    )

    response.raise_for_status()

    answer = response.json()["response"]

    save_message("assistant", answer)

    return answer


init_db()

while True:
    question = input("\nYou: ")

    if question.lower() in ["exit", "quit"]:
        break

    answer = ask_llm(question)

    print(f"\nAI: {answer}")