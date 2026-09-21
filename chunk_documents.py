from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from text_cleaner import clean_text

file_path = Path("documents/research_paper.pdf")
reader = PdfReader(file_path)

documents = [
    Document(
        page_content=page.extract_text() or "",
        metadata={"source": str(file_path), "page": page_number},
    )
    for page_number, page in enumerate(reader.pages)
]

for document in documents:
    document.page_content = clean_text(document.page_content)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print("Original pages:", len(documents))
print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk.page_content)

    print("\nMetadata:")
    print(chunk.metadata)