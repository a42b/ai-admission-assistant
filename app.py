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

# ЄДИНИЙ ПРОМПТ, ЯКИЙ КЕРУЄ І ТЕКСТОМ, І ГОЛОСОМ ОДНОЧАСНО
PROMPT = """
Ти — офіційний штучний інтелект-асистент для абітурієнтів Національного технічного університету України «Київський політехнічний інститут імені Ігоря Сікорського» (КПІ).
Твоє завдання — відповідати на питання користувача виключно на основі наданого контексту українською мовою.

Залежно від вказаного РЕЖИМУ ВЗАЄМОДІЇ, формуй відповідь за такими правилами:

СТАНДАРТНИЙ РЕЖИМ (text):
- Відповідай детально, чітко та структуровано.
- Використовуй марковані списки для переліків, абзаци та точні цифри.
- Обов'язково вказуй шифри спеціальностей та офіційні назви програм.

ГОЛОСОВИЙ РЕЖИМ (voice):
- Формуй відповідь як коротку, розмовну репліку для телефону (максимум 2-3 речення).
- Пиши ТІЛЬКИ суцільний текст. Жодних списків (*, -, 1.), дужок, лапок, знаків плюс чи абзаців.
- Замість технічних шифрів пиши слова (наприклад, замість "121" пиши "сто двадцять перша спеціальність").
- Округляй прохідні бали до цілих (наприклад, замість "158.615" кажи "близько ста п'ятдесяти дев'яти балів").

Загальні залізобетонні бізнес-правила:
1. ФІОТ — це факультет інформатики та обчислювальної техніки. На ньому є спеціальності 121, 123 та 126.
2. Кафедра ОТ — це лише одна з кафедр всередині ФІОТ, яка готує ТІЛЬКИ за спеціальностями 121 та 123. 
3. Спеціальність 126 — це загальнофакультетський напрям ФІОТ, до кафедри ОТ вона не належить.
4. Якщо інформації взагалі немає в контексті, в режимі "text" скажи: "У наданих документах я не знайшов точної відповіді на це питання.", а в режимі "voice" скажи коротко: "На жаль, у мене немає цих даних. Бажаєте, я з'єднаю вас з оператором?".

Контекст:
{context}

РЕЖИМ ВЗАЄМОДІЇ: {mode}
Питання користувача: {question}

Відповідь асистента:
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
                docs.append(Document(page_content=text, metadata={"source": str(path), "source_name": path.name}))
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
            continue
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
    if any(word in q for word in ["фіот", "факультет", "спеціальн", "бал", "варт", "ціна", "контракт", "121", "123", "126"]):
        for doc in load_full_txt_docs():
            if "fiot_summary" in doc.metadata.get("source", ""):
                docs.append(doc)
    docs.extend(get_keyword_docs(question, limit=3))
    docs.extend(get_vector_docs(question, limit=2))
    docs = deduplicate_docs(docs)
    return docs[:4]

# Функція генерації тепер просто передає режим (text або voice) всередину єдиного промпту
def answer_question(question: str, mode: str = "text"):
    docs = retrieve_docs(question)
    context = format_context(docs)
    llm = load_llm()
    
    prompt = ChatPromptTemplate.from_template(PROMPT)
    chain = prompt | llm

    response = chain.invoke({
        "context": context, 
        "question": question,
        "mode": mode  # Передаємо режим ('text' або 'voice') прямо в шаблон промпту
    })
    return response.content, docs
def main():
    st.set_page_config(page_title="AI Admission Assistant", page_icon="🎓")
    st.title("🎓 AI Admission Assistant")
    st.caption("Асистент для абітурієнтів КПІ на основі документів (Текстовий режим)")

    question = st.text_input("Введи питання:")

    if st.button("Запитати") and question.strip():
        with st.spinner("Шукаю відповідь у документах..."):
            answer, _ = answer_question(question.strip(), mode="text")

        st.subheader("Відповідь")
        st.write(answer)


if __name__ == "__main__":
    main()