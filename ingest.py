import shutil
from pathlib import Path

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

DOCS_DIR = Path("documents")
DB_DIR = Path("chroma_db")

EMBED_MODEL = "nomic-embed-text"


def load_documents():
    documents = []
    if not DOCS_DIR.exists():
        return documents

    for file_path in DOCS_DIR.iterdir():
        if file_path.name == ".gitkeep":
            continue

        suffix = file_path.suffix.lower()

        # Специальная обработка для табличных файлов (стоимость и проходные баллы)
        if "infopk" in file_path.name.lower() or "вартість" in file_path.name.lower():
            if suffix == ".txt":
                try:
                    text = file_path.read_text(encoding="utf-8")
                    lines = [line.strip() for line in text.split("\n") if line.strip()]
                    for line in lines:
                        documents.append(
                            Document(
                                page_content=line,
                                metadata={"source": file_path.name}
                            )
                        )
                    print(f"Loaded Table TXT (line-by-line): {file_path.name}")
                    continue
                except Exception as e:
                    print(f"Error loading table {file_path.name}: {e}")
                    continue

        # Обработка стандартных текстовых файлов и PDF
        if suffix == ".txt":
            try:
                loader = TextLoader(str(file_path), encoding="utf-8")
                documents.extend(loader.load())
                print(f"Loaded TXT: {file_path.name}")
            except Exception as e:
                print(f"Error loading TXT {file_path.name}: {e}")

        elif suffix == ".pdf":
            try:
                loader = PyPDFLoader(str(file_path))
                documents.extend(loader.load())
                print(f"Loaded PDF: {file_path.name}")
            except Exception as e:
                print(f"Error loading PDF {file_path.name}: {e}")

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

    # Оптимальный размер чанка для академических документов
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
    )

    chunks = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
    )

    print(f"Successfully indexed {len(chunks)} chunks into {DB_DIR}")


if __name__ == "__main__":
    main()