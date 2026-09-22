from memory import init_db, save_message, load_messages

init_db()

save_message("user", "My name is Mesith.")
save_message("assistant", "Nice to meet you, Mesith.")

messages = load_messages()

for message in messages:
    print(f"{message['role']}: {message['content']}")
