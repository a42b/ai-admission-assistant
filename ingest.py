import shutil
from pathlib import Path

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


DOCS_DIR = Path("documents")
DB_DIR = Path("chroma_db")

EMBED_MODEL = "nomic-embed-text"


def load_documents():
    documents = []

    for file_path in DOCS_DIR.iterdir():
        if file_path.name == ".gitkeep":
            continue

        suffix = file_path.suffix.lower()

        if suffix == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents.extend(loader.load())
            print(f"Loaded TXT: {file_path.name}")

        elif suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
            documents.extend(loader.load())
            print(f"Loaded PDF: {file_path.name}")

        else:
            print(f"Skipped unsupported file: {file_path.name}")

    return documents


def main():
    if not DOCS_DIR.exists():
        DOCS_DIR.mkdir(parents=True)

    documents = load_documents()

    if not documents:
        print("No documents found in documents/")
        return

    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)
        print("Old chroma_db removed")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
    )

    print(f"Indexed {len(chunks)} chunks into {DB_DIR}")


if __name__ == "__main__":
    main()