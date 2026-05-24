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
Відповідай українською мовою, чітко, лаконічно та структуровано.

Важливі правила:
1. ФІОТ — це факультет інформатики та обчислювальної техніки.
2. Кафедра ОТ — це кафедра обчислювальної техніки всередині ФІОТ, а не весь факультет.
3. Якщо користувач питає про факультет ФІОТ загалом, відповідай лише про факультет.
4. Якщо користувач питає про кафедру ОТ, відповідай саме про кафедру ОТ.
5. Ніколи не називай ФІОТ "фізико-інформаційним факультетом".
6. Використовуй тільки факти, дати, ціни та бали, які чітко вказані в контексті. Не вигадуй та не доповнюй інформацію від себе.
7. Якщо в одному джерелі є інформація про факультет, а в іншому про кафедру, не змішуй їх в одну сутність.
8. Якщо точної відповіді на питання в контексті немає або інформація відсутня, ти ПОВИНЕН відповісти строго цією фразою:
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
        extra += ["прохідний", "бал", "2024", "2023", "2022", "фіот", "f2", "f6", "f7"]

    if "126" in q or "f6" in q:
        extra += ["126", "f6", "інформаційні", "системи", "технології"]

    if "121" in q or "f2" in q:
        extra += ["121", "f2", "інженерія", "програмного", "забезпечення"]

    if "123" in q or "f7" in q:
        extra += ["123", "f7", "комп'ютерна", "інженерія"]

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
                        metadata={
                            "source": str(path),
                            "source_name": path.name,
                        },
                    )
                )
        except Exception:
            continue

    return docs


@st.cache_resource(show_spinner=False)
def load_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
    )


@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatOllama(
        model=LLM_MODEL,
        temperature=0.1,
    )


@st.cache_data(show_spinner=False)
def load_chroma_documents_as_docs():
    vectorstore = load_vectorstore()
    data = vectorstore.get()

    docs = []

    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])

    for text, metadata in zip(documents, metadatas):
        docs.append(
            Document(
                page_content=text,
                metadata=metadata or {},
            )
        )

    return docs


def get_docs_by_source_contains(source_part: str):
    all_docs = load_chroma_documents_as_docs()
    result = []

    for doc in all_docs:
        source = doc.metadata.get("source", "")
        if source_part.lower() in source.lower():
            result.append(doc)

    return result


def get_keyword_docs(question: str, limit: int = 5):
    q = normalize(question)
    keywords = question_keywords(question)

    candidates = []

    for doc in load_full_txt_docs():
        score = score_text(doc.page_content, keywords)
        source = doc.metadata.get("source", "")

        if "programs_fiot" in source and (
            "фіот" in q or "факультет" in q or "f6" in q or "126" in q
        ):
            score += 20

        if "fiot_ot_bachelor" in source and (
            "кафедр" in q or "кафедра от" in q
        ):
            score += 20

        if score > 0:
            candidates.append((score, doc))

    for doc in load_chroma_documents_as_docs():
        score = score_text(doc.page_content, keywords)
        source = doc.metadata.get("source", "")

        if "Вартість" in source and (
            "варт" in q or "кошту" in q or "ціна" in q or "контракт" in q
        ):
            score += 30

        if "infopk" in source and (
            "бал" in q or "прохід" in q or "126" in q or "f6" in q
        ):
            score += 30

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


def get_vector_docs(question: str, limit: int = 4):
    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": limit,
            "fetch_k": 20,
            "lambda_mult": 0.5,
        },
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
        source = doc.metadata.get("source", "невідомо")
        parts.append(
            f"\n\n--- ДОКУМЕНТ / ФРАГМЕНТ {i} ---\n"
            f"Джерело: {source}\n"
            f"{doc.page_content}"
        )

    return "\n".join(parts)


def retrieve_docs(question: str):
    q = normalize(question)
    docs = []

    if "варт" in q or "кошту" in q or "ціна" in q or "контракт" in q:
        docs.extend(get_docs_by_source_contains("Вартість"))

    if "прохід" in q or "бал" in q:
        docs.extend(get_docs_by_source_contains("infopk"))

    if "фіот" in q or "факультет" in q or "f6" in q or "126" in q:
        for doc in load_full_txt_docs():
            source = doc.metadata.get("source", "")
            if "programs_fiot" in source:
                docs.append(doc)

    if "кафедр" in q or "кафедра от" in q:
        for doc in load_full_txt_docs():
            source = doc.metadata.get("source", "")
            if "fiot_ot_bachelor" in source:
                docs.append(doc)

    keyword_docs = get_keyword_docs(question, limit=5)
    docs.extend(keyword_docs)

    vector_docs = get_vector_docs(question, limit=4)
    docs.extend(vector_docs)

    docs = deduplicate_docs(docs)

    # Максимальный лимит контекста — 5 чанков для стабильности локальной модели
    return docs[:5]


def answer_question(question: str):
    docs = retrieve_docs(question)
    context = format_context(docs)

    llm = load_llm()
    prompt = ChatPromptTemplate.from_template(PROMPT)
    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    return response.content, docs


# Оставляем интерфейс Streamlit для локальной отладки данных
st.set_page_config(
    page_title="AI Admission Assistant",
    page_icon="🎓",
)

st.title("🎓 AI Admission Assistant")
st.caption("Асистент для абітурієнтів КПІ на основі документів")

st.info(f"Модель: {LLM_MODEL}. База знань: {DB_DIR}")

if not DOCS_DIR.exists():
    st.warning("Папка documents не знайдена.")
    st.stop()

if not Path(DB_DIR).exists():
    st.warning("База знань ще не створена. Спочатку запусти: python ingest.py")
    st.stop()

question = st.text_input("Введи питання:")

if st.button("Запитати") and question.strip():
    with st.spinner("Шукаю відповідь у документах..."):
        try:
            answer, docs = answer_question(question.strip())
        except Exception as e:
            st.error("Сталася помилка під час відповіді.")
            st.exception(e)
            st.stop()

    st.subheader("Відповідь")
    st.write(answer)

    with st.expander("Використані фрагменти документів"):
        for i, doc in enumerate(docs, start=1):
            st.markdown(f"**Фрагмент {i}**")
            st.caption(doc.metadata.get("source", "невідоме джерело"))
            st.write(doc.page_content)