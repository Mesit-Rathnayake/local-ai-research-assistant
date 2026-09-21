from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

file_path = Path("documents/research_paper.pdf")
reader = PdfReader(file_path)

documents = [
    Document(
        page_content=page.extract_text() or "",
        metadata={"source": str(file_path), "page": page_number},
    )
    for page_number, page in enumerate(reader.pages)
]

print("Number of pages:", len(documents))

for i, document in enumerate(documents[:3]):
    print(f"\n--- Page {i + 1} ---")
    print(document.page_content[:1000])

    print("\nMetadata:")
    print(document.metadata)