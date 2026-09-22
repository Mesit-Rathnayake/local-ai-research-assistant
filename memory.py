import sqlite3

DB_PATH = "conversation.db"


def init_db():
    connection = sqlite3.connect(DB_PATH)

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
    connection = sqlite3.connect(DB_PATH)

    connection.execute(
        "INSERT INTO messages (role, content) VALUES (?, ?)",
        (role, content)
    )

    connection.commit()
    connection.close()


def load_messages():
    connection = sqlite3.connect(DB_PATH)

    cursor = connection.execute(
        "SELECT role, content FROM messages ORDER BY id"
    )

    messages = [
        {
            "role": role,
            "content": content
        }
        for role, content in cursor.fetchall()
    ]

    connection.close()

    return messages
