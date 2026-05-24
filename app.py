from pathlib import Path

import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate


DB_DIR = "chroma_db"

LLM_MODEL = "qwen3:8b"
EMBED_MODEL = "nomic-embed-text"


PROMPT = """
Ти — асистент для абітурієнтів КПІ ім. Ігоря Сікорського.

Відповідай українською мовою, чітко, структуровано і без зайвих припущень.
Використовуй тільки інформацію з наданого контексту.
Не вигадуй фактів, дат, вартості навчання, назв спеціальностей, освітніх програм або правил вступу.

Важливі правила:
1. ФІОТ — це факультет інформатики та обчислювальної техніки.
2. Кафедра ОТ — це кафедра обчислювальної техніки всередині ФІОТ, а не весь факультет.
3. Якщо користувач питає про факультет ФІОТ загалом, відповідай про факультет загалом.
4. Якщо користувач питає про кафедру ОТ, відповідай саме про кафедру ОТ.
5. Не називай ФІОТ фізико-інформаційним факультетом.
6. Якщо у контексті є кілька джерел, надай відповідь на основі найбільш релевантних.
7. Якщо відповідь у контексті є, не кажи, що її немає.
8. Якщо точної відповіді справді немає в контексті, скажи:
"У наданих документах я не знайшов точної відповіді на це питання."

Контекст:
{context}

Питання користувача:
{question}

Відповідь:
"""


def format_docs(docs):
    parts = []

    for doc in docs:
        source = doc.metadata.get("source", "невідомо")
        text = doc.page_content
        parts.append(f"Джерело: {source}\n{text}")

    return "\n\n".join(parts)


@st.cache_data(show_spinner=False)
def load_full_text_documents():
    """
    Завантажує всі .txt документи повністю.
    Це допомагає моделі бачити головні текстові джерела без ризику,
    що Chroma витягне тільки неповний фрагмент.
    """
    docs_dir = Path("documents")
    texts = []

    if not docs_dir.exists():
        return ""

    for path in sorted(docs_dir.glob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                texts.append(
                    f"\n\n====================\n"
                    f"Файл: {path.name}\n"
                    f"====================\n"
                    f"{text}"
                )
        except Exception as e:
            texts.append(
                f"\n\n====================\n"
                f"Файл: {path.name}\n"
                f"Помилка читання файлу: {e}\n"
            )

    return "\n".join(texts)


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


def retrieve_docs(question: str):
    """
    RAG-пошук по Chroma.
    Використовується для PDF та додаткових фрагментів документів.
    """
    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 10,
            "fetch_k": 40,
            "lambda_mult": 0.35,
        },
    )

    return retriever.invoke(question)


def build_context(question: str):
    """
    Гібридний контекст:
    1. Повні .txt документи.
    2. Релевантні фрагменти з Chroma, зокрема з PDF.
    """
    full_text_context = load_full_text_documents()

    docs = retrieve_docs(question)
    rag_context = format_docs(docs)

    context = (
        "ПОВНІ ТЕКСТОВІ ДОКУМЕНТИ:\n"
        f"{full_text_context}\n\n"
        "РЕЛЕВАНТНІ ФРАГМЕНТИ З ВЕКТОРНОЇ БАЗИ:\n"
        f"{rag_context}"
    )

    return context, docs


def answer_question(question: str):
    context, docs = build_context(question)

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


st.set_page_config(
    page_title="AI Admission Assistant",
    page_icon="🎓",
)

st.title("🎓 AI Admission Assistant")
st.caption("Асистент для абітурієнтів КПІ на основі документів")

st.info(f"Модель: {LLM_MODEL}. База знань: {DB_DIR}")

if not Path("documents").exists():
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