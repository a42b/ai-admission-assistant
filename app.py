from pathlib import Path
import re

import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


DB_DIR = "chroma_db"
DOCS_DIR = Path("documents")

LLM_MODEL = "qwen3:8b"
EMBED_MODEL = "nomic-embed-text"


PROMPT = """
Ти — офіційний штучний інтелект-асистент для абітурієнтів Національного технічного університету України «Київський політехнічний інститут імені Ігоря Сікорського» (КПІ).

Твоє завдання — відповідати на питання користувача виключно на основі наданого контексту.
Якщо в контексті є структурований довідник спеціальностей, використовуй його дані повністю, не пропускаючи коди, назви та цифри.
Відповідай українською мовою, чітко, структуровано (використовуй марковані списки для переліків).

Важливі правила:
1. ФІОТ — це факультет інформатики та обчислювальної техніки. На ньому є спеціальності 121, 123 та 126.
2. Кафедра ОТ — це кафедра обчислювальної техніки всередині ФІОТ. Вона готує ТІЛЬКИ за спеціальностями 121 та 123. 
3. Спеціальність 126 — це загальнофакультетський напрям ФІОТ, до кафедри ОТ вона відношення не має. Не звужуй відповідь про ФІОТ лише до кафедри ОТ.
4. Використовуй тільки факти, дати, ціни та бали, які чітко вказані в контексті. Не вигадуй та не доповнюй інформацію від себе.
5. Якщо інформації за запитом взагалі немає в контексті, ти ПОВИНЕН відповісти строго цією фразою:
"У наданих документах я не знайшов точної відповіді на це питання."

Контекст:
{context}

Питання користувача:
{question}

Відповідь:
"""


def normalize(text: str) -> str:
    return text.lower().replace("’", "'").replace("`", "'")


def question_keywords(question: str):
    q = normalize(question)
    keywords = re.findall(r"[a-zA-Zа-яА-ЯіїєґІЇЄҐ0-9]+", q)
    extra = []

    if "фіот" in q or "факультет" in q:
        extra += ["фіот", "факультет", "f2", "f6", "f7", "121", "126", "123"]
    if "кафедр" in q or "кафедра от" in q:
        extra += ["кафедра", "от", "f2", "f7", "121", "123"]
    if "варт" in q or "кошту" in q or "ціна" in q or "контракт" in q:
        extra += ["вартість", "навчання", "грн", "f2", "f6", "f7", "121", "126", "123"]
    if "прохід" in q or "бал" in q:
        extra += ["прохідний", "бал", "2024", "2023", "фіот", "f2", "f6", "f7"]

    return list(dict.fromkeys(keywords + extra))


def score_text(text: str, keywords: list[str]) -> int:
    t = normalize(text)
    score = 0
    for kw in keywords:
        if kw and kw in t:
            score += 1
    return score


@st.cache_data(show_spinner=False)
def load_full_txt_docs():
    docs = []
    if not DOCS_DIR.exists():
        return docs
    for path in sorted(DOCS_DIR.glob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": str(path), "source_name": path.name},
                    )
                )
        except Exception:
            continue
    return docs


@st.cache_resource(show_spinner=False)
def load_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)


@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatOllama(model=LLM_MODEL, temperature=0.1)


@st.cache_data(show_spinner=False)
def load_chroma_documents_as_docs():
    vectorstore = load_vectorstore()
    data = vectorstore.get()
    docs = []
    for text, metadata in zip(data.get("documents", []), data.get("metadatas", [])):
        docs.append(Document(page_content=text, metadata=metadata or {}))
    return docs


def get_keyword_docs(question: str, limit: int = 4):
    q = normalize(question)
    keywords = question_keywords(question)
    candidates = []

    for doc in load_full_txt_docs():
        score = score_text(doc.page_content, keywords)
        if "fiot_summary" in doc.metadata.get("source", ""):
            continue  # Этот файл мы обрабатываем отдельно через роутинг
        if score > 0:
            candidates.append((score, doc))

    candidates.sort(key=lambda x: x[0], reverse=True)
    result = []
    seen = set()
    for score, doc in candidates:
        key = (doc.metadata.get("source", ""), doc.page_content[:300])
        if key not in seen:
            seen.add(key)
            result.append(doc)
        if len(result) >= limit:
            break
    return result


def get_vector_docs(question: str, limit: int = 3):
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": limit, "fetch_k": 15, "lambda_mult": 0.5},
    )
    return retriever.invoke(question)


def deduplicate_docs(docs):
    result = []
    seen = set()
    for doc in docs:
        key = (doc.metadata.get("source", ""), doc.page_content[:300])
        if key not in seen:
            seen.add(key)
            result.append(doc)
    return result


def format_context(docs):
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_name", doc.metadata.get("source", "Документ"))
        parts.append(f"\n\n--- ДОКУМЕНТ {i} (Джерело: {source}) ---\n{doc.page_content}")
    return "\n".join(parts)


def retrieve_docs(question: str):
    q = normalize(question)
    docs = []

    # АРХІТЕКТУРНИЙ РОУТИНГ: Якщо питання стосується ФІОТ, балів, цін або спеціальностей
    if any(word in q for word in ["фіот", "факультет", "спеціальн", "бал", "варт", "ціна", "контракт", "121", "123", "126"]):
        for doc in load_full_txt_docs():
            if "fiot_summary" in doc.metadata.get("source", ""):
                docs.append(doc)

    # Додаємо трохи загального контексту з інших джерел для гнучкості
    docs.extend(get_keyword_docs(question, limit=3))
    docs.extend(get_vector_docs(question, limit=2))

    docs = deduplicate_docs(docs)
    return docs[:4]  # Возвращаем строго 4 самых качественных чанка


def answer_question(question: str):
    docs = retrieve_docs(question)
    context = format_context(docs)

    llm = load_llm()
    prompt = ChatPromptTemplate.from_template(PROMPT)
    chain = prompt | llm

    response = chain.invoke({"context": context, "question": question})
    return response.content, docs


# Интерфейс Streamlit для проверки
st.set_page_config(page_title="AI Admission Assistant", page_icon="🎓")
st.title("🎓 AI Admission Assistant")
st.caption("Асистент для абітурієнтів КПІ на основі документів")

question = st.text_input("Введи питання:")

if st.button("Запитати") and question.strip():
    with st.spinner("Шукаю відповідь..."):
        answer, docs = answer_question(question.strip())
    st.subheader("Відповідь")
    st.write(answer)