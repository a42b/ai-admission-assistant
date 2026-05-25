import os
import streamlit as st
import speech_recognition as sr
import requests
from gtts import gTTS

# Наш єдиний і незмінний текстовий ендпоінт
API_URL_TEXT = "http://localhost:8000/api/ask"

AUDIO_DIR = "static_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

st.set_page_config(page_title="AI Admission Assistant", layout="centered")
st.title("AI Admission Assistant ФІОТ КПІ")
st.caption("Уніфікований діалоговий інтерфейс приймальної комісії")

# Ініціалізація історії чату в пам'яті Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Ініціалізація стану голосового режиму
if "voice_active" not in st.session_state:
    st.session_state.voice_active = False

# Функція розпізнавання мови (STT)
def record_and_recognize():
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 1.0
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True

    with sr.Microphone() as source:
        st.info("Голосовий чат активний. Слухаю вас... Говоріть.")
        recognizer.adjust_for_ambient_noise(source, duration=0.8)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
            st.info("Розпізнаю мовлення...")
            return recognizer.recognize_google(audio, language="uk-UA")
        except sr.WaitTimeoutError:
            return None
        except Exception:
            return None

# Єдина функція відправки запиту на бекенд та генерації аудіо
def send_to_backend(question):
    # Додаємо повідомлення користувача в історію чату
    st.session_state.messages.append({"role": "user", "text": question})
    
    try:
        # Робимо ОДИН спільний запит до API бекенду
        # Передаємо режим 'voice', щоб qwen генерував лаконічні репліки без сміття
        res = requests.post(
            API_URL_TEXT,
            json={"question": question, "mode": "voice"},
            timeout=180,
        )
        
        if res.status_code == 200:
            answer = res.json().get("answer", "")
            
            # Додаємо відповідь асистента в історію
            st.session_state.messages.append({"role": "assistant", "text": answer})
            return answer
        else:
            st.error("Помилка сервера бекенду.")
            return None
    except Exception as e:
        st.error(f"Не вдалося з'єднатися з api.py: {e}")
        return None

# --- БЛОК КЕРУВАННЯ РЕЖИМАМИ (Кнопки) ---
col1, col2 = st.columns(2)

with col1:
    if not st.session_state.voice_active:
        if st.button("Розпочати голосовий чат", use_container_width=True):
            st.session_state.voice_active = True
            st.rerun()
    else:
        if st.button("Припинити голосовий чат", use_container_width=True):
            st.session_state.voice_active = False
            st.rerun()

with col2:
    if st.button("Очистити історію чату", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.write("---")

# --- ВІДОБРАЖЕННЯ ІСТОРІЇ ЧАТУ ---
# Показуємо всі попередні повідомлення на екрані у вигляді діалогу
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["text"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["text"])

# --- ЛОГІКА РОБОТИ РЕЖИМІВ ---

# 1. Сценарій: Активовано Голосовий Чат
if st.session_state.voice_active:
    voice_question = record_and_recognize()
    
    if voice_question:
        # Обробляємо запит
        answer_text = send_to_backend(voice_question)
        
        if answer_text:
            # Озвучуємо отриману відповідь на місці за допомогою gTTS
            tts = gTTS(text=answer_text, lang='uk', slow=False)
            audio_file = os.path.join(AUDIO_DIR, "temp_reply.mp3")
            tts.save(audio_file)
            
            # Оновлюємо сторінку, щоб повідомлення з'явилися в чаті, і програємо звук
            st.rerun()
            
    # Якщо звуковий файл існує і останнє повідомлення від асистента — даємо його прослухати
    audio_file = os.path.join(AUDIO_DIR, "temp_reply.mp3")
    if os.path.exists(audio_file) and st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        st.audio(audio_file, format="audio/mp3", autoplay=True)

# 2. Сценарій: Звичайний Текстовий Чат (діє, коли голосовий режим вимкнено)
else:
    # Стандартний рядок введення Streamlit Chat Input як у ChatGPT
    if text_question := st.chat_input("Задайте ваше питання тут..."):
        send_to_backend(text_question)
        st.rerun()