from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from text_cleaner import clean_text

DOCUMENTS_PATH = Path("documents")
CHROMA_PATH = "chroma_db"

documents = []

for file_path in DOCUMENTS_PATH.iterdir():
    if file_path.suffix.lower() == ".pdf":
        reader = PdfReader(file_path)
        loaded_documents = [
            Document(
                page_content=page.extract_text() or "",
                metadata={"source": file_path.name, "page": page_number},
            )
            for page_number, page in enumerate(reader.pages)
        ]
    elif file_path.suffix.lower() in [".txt", ".md"]:
        loaded_documents = [
            Document(
                page_content=file_path.read_text(encoding="utf-8"),
                metadata={"source": file_path.name},
            )
        ]
    else:
        continue

    for document in loaded_documents:
        document.page_content = clean_text(document.page_content)
        document.metadata["source"] = file_path.name

    documents.extend(loaded_documents)

print("Documents loaded:", len(documents))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)
chunks = splitter.split_documents(documents)

print("Total chunks:", len(chunks))

embeddings = OllamaEmbeddings(model="nomic-embed-text")

Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_PATH,
    collection_name="research_papers",
)

print("\nIngestion complete.")
print("Documents:", len(documents))
print("Chunks:", len(chunks))
print("Vector database:", CHROMA_PATH)
