from pathlib import Path

import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate


DB_DIR = "chroma_db"

LLM_MODEL = "qwen3.6:27b"
EMBED_MODEL = "nomic-embed-text"


PROMPT = """
Ти — асистент для абітурієнтів КПІ ім. Ігоря Сікорського.

Відповідай українською мовою, чітко і зрозуміло.
Використовуй тільки інформацію з наданого контексту.
Не вигадуй фактів, дат, вартості навчання або правил вступу.

Якщо в контексті немає відповіді, скажи:
"У наданих документах я не знайшов точної відповіді на це питання."

Контекст:
{context}

Питання користувача:
{question}
"""


def format_docs(docs):
    return "\n\n".join(
        f"Джерело: {doc.metadata.get('source', 'невідомо')}\n{doc.page_content}"
        for doc in docs
    )


@st.cache_resource
def load_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
    )


@st.cache_resource
def load_llm():
    return ChatOllama(
        model=LLM_MODEL,
        temperature=0.2,
    )


def answer_question(question: str):
    vectorstore = load_vectorstore()
    llm = load_llm()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    docs = retriever.invoke(question)
    context = format_docs(docs)

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

if not Path(DB_DIR).exists():
    st.warning("База знань ще не створена. Спочатку запусти: python ingest.py")
    st.stop()

question = st.text_input("Введи питання:")

if st.button("Запитати") and question.strip():
    with st.spinner("Шукаю відповідь у документах..."):
        answer, docs = answer_question(question)

    st.subheader("Відповідь")
    st.write(answer)

    with st.expander("Використані фрагменти документів"):
        for i, doc in enumerate(docs, start=1):
            st.markdown(f"**Фрагмент {i}**")
            st.caption(doc.metadata.get("source", "невідоме джерело"))
            st.write(doc.page_content)