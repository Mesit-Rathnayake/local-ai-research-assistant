from pathlib import Path

from langchain_core.documents import Document

file_path = Path("documents/research_notes.txt")
documents = [
    Document(
        page_content=file_path.read_text(encoding="utf-8"),
        metadata={"source": str(file_path)},
    )
]

print("Number of documents:", len(documents))

for document in documents:
    print("\nContent:")
    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)