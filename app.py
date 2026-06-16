from pathlib import Path
import re
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

DB_DIR = "chroma_db"
DOCS_DIR = Path("documents")

LLM_MODEL = "qwen3:8b"  # Або "qwen3:1.5b", якщо тимчасово працюєш на CPU
EMBED_MODEL = "nomic-embed-text"

# ЄДИНИЙ СИСТЕМНИЙ ПРОМПТ ДЛЯ ВСІЄЇ СИСТЕМИ
PROMPT = """
Ти — AI-консультант приймальної комісії ФІОТ КПІ.

Твоє завдання — не переказувати документи, а знайти в контексті потрібний факт і відповісти тільки на питання користувача.

Головні правила:
- Відповідай коротко, чітко і тільки по суті питання.
- Не копіюй великі фрагменти контексту.
- Не переказуй усю інформацію з документа.
- Використовуй контекст лише як джерело фактів.
- Якщо користувач питає про перелік, дай тільки перелік без додаткових пояснень.
- Не додавай прохідні бали, вартість навчання, кафедри, форми навчання, освітні програми, міжнародні можливості або інші деталі, якщо користувач прямо про це не запитував.
- Якщо користувач просить коротко — відповідай одним реченням.
- Якщо користувач просить детальніше — тоді можна дати розгорнуту відповідь.
- Максимальна довжина відповіді у звичайному режимі — 3 речення.
- Відповідай українською мовою.

Правила для голосового режиму:
- У режимі voice відповідай ще коротше: 1–2 речення.
- Не використовуй списки, нумерацію, дужки та складні конструкції.
- Формулюй відповідь так, щоб її було зручно слухати.

Приклад правильної відповіді:

Питання: Які спеціальності є на ФІОТ?
Відповідь: На ФІОТ є три спеціальності: 121 «Інженерія програмного забезпечення», 123 «Комп’ютерна інженерія» та 126 «Інформаційні системи та технології».

Приклад неправильної відповіді:
Не треба додавати кафедри, прохідні бали, вартість навчання, освітні програми чи інші деталі, якщо користувач про це не просив.

Якщо інформації немає в контексті, у режимі text скажи:
"У наданих документах я не знайшов точної відповіді на це питання."

Якщо інформації немає в контексті, у режимі voice скажи:
"На жаль, у мене немає цих даних."

Контекст:
{context}

Режим взаємодії: {mode}
Питання користувача: {question}

Коротка відповідь:
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
        extra += ["proхідний", "бал", "2024", "2023", "фіот", "f2", "f6", "f7"]
    return list(dict.fromkeys(keywords + extra))

def score_text(text: str, keywords: list[str]) -> int:
    t = normalize(text)
    score = 0
    for kw in keywords:
        if kw and kw in t:
            score += 1
    return score

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

def load_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

def load_llm():
    return ChatOllama(model=LLM_MODEL, temperature=0.1)

def get_keyword_docs(question: str, limit: int = 4):
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
    for _, doc in candidates:
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

def answer_question(question: str, mode: str = "text"):
    docs = retrieve_docs(question)
    context = format_context(docs)
    llm = load_llm()
    
    prompt = ChatPromptTemplate.from_template(PROMPT)
    chain = prompt | llm

    response = chain.invoke({
        "context": context, 
        "question": question,
        "mode": mode
    })
    return response.content, docs