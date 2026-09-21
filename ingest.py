from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from text_cleaner import clean_text

PDF_PATH = "documents/research_paper.pdf"
CHROMA_PATH = "chroma_db"

file_path = Path(PDF_PATH)
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
    chunk_overlap=100,
)
chunks = splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_PATH,
    collection_name="research_papers",
)

print("Ingestion complete.")
print("Pages:", len(documents))
print("Chunks:", len(chunks))
print("Vector database:", CHROMA_PATH)
