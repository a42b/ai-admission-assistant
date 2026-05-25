import os
import streamlit as st
import speech_recognition as sr
import requests
from gtts import gTTS
import time

# Наш єдиний і незмінний текстовий ендпоінт
API_URL_TEXT = "http://localhost:8000/api/ask"

AUDIO_DIR = "static_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

st.set_page_config(page_title="AI Admission Assistant", layout="centered")
st.title("AI Admission Assistant ФІОТ КПІ")
st.caption("Уніфікований діалоговий інтерфейс приймальної комісії (Текст / Автоматичний Голос)")

# Ініціалізація історії чату в пам'яті Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Ініціалізація стану голосового режиму
if "voice_active" not in st.session_state:
    st.session_state.voice_active = False

# Стан, який контролює, чи потрібно нам зараз слухати мікрофон
if "should_listen" not in st.session_state:
    st.session_state.should_listen = False

# Функція розпізнавання мови (STT)
def record_and_recognize():
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 1.0
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True

    with sr.Microphone() as source:
        st.info("Мікрофон відкрито. Говоріть...")
        recognizer.adjust_for_ambient_noise(source, duration=0.8)
        try:
            # Слухаємо фрази (таймаут 5 секунд на початок розмови, максимум 8 секунд на фразу)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
            st.info("Очікуйте, обробка мовлення...")
            return recognizer.recognize_google(audio, language="uk-UA")
        except sr.WaitTimeoutError:
            return None
        except Exception:
            return None

# Спільна функція відправки запиту на бекенд та генерації аудіо
def send_to_backend(question):
    # Додаємо повідомлення користувача в історію чату
    st.session_state.messages.append({"role": "user", "text": question, "audio_path": None})
    
    try:
        # Робимо запит до єдиного API бекенду в режимі voice
        res = requests.post(
            API_URL_TEXT,
            json={"question": question, "mode": "voice"},
            timeout=180,
        )
        
        if res.status_code == 200:
            answer = res.json().get("answer", "")
            
            # Одразу синтезуємо аудіофайл для цієї конкретної відповіді
            tts = gTTS(text=answer, lang='uk', slow=False)
            
            # Створюємо унікальне ім'я файлу на основі мітки часу, щоб доріжки в чаті не перезаписувалися
            filename = f"reply_{int(time.time())}.mp3"
            audio_path = os.path.join(AUDIO_DIR, filename)
            tts.save(audio_path)
            
            # Додаємо відповідь асистента та шлях до її аудіофайлу в історію
            st.session_state.messages.append({"role": "assistant", "text": answer, "audio_path": audio_path})
            return answer, audio_path
        else:
            st.error("Помилка сервера бекенду.")
            return None, None
    except Exception as e:
        st.error(f"Не вдалося з'єднатися з api.py: {e}")
        return None, None

# --- БЛОК КЕРУВАННЯ РЕЖИМАМИ (Кнопки) ---
col1, col2 = st.columns(2)

with col1:
    if not st.session_state.voice_active:
        if st.button("Розпочати голосовий чат", use_container_width=True):
            st.session_state.voice_active = True
            st.session_state.should_listen = True  # Дозволяємо слухати мікрофон
            st.rerun()
    else:
        if st.button("Припинити голосовий чат", use_container_width=True):
            st.session_state.voice_active = False
            st.session_state.should_listen = False # Повністю закриваємо мікрофон
            st.rerun()

with col2:
    if st.button("Очистити історію чату", use_container_width=True):
        st.session_state.messages = []
        st.session_state.should_listen = st.session_state.voice_active
        st.rerun()

st.write("---")

# --- ВІДОБРАЖЕННЯ ІСТОРІЇ ЧАТУ З ОДНОЧАСНИМ ЗБЕРЕЖЕННЯМ ДОРІЖОК ---
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["text"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["text"])
            # Якщо у відповіді є збережений аудіофайл, відображаємо його доріжку в чаті
            if msg.get("audio_path") and os.path.exists(msg["audio_path"]):
                st.audio(msg["audio_path"], format="audio/mp3")

# --- АВТОМАТИЧНА ЛОГІКА ДІАЛОГУ ---

# Сценарій А: Голосовий чат увімкнено і система готова слухати
if st.session_state.voice_active and st.session_state.should_listen:
    voice_question = record_and_recognize()
    
    if voice_question:
        # 1. СТОП МІКРОФОН: Тимчасово блокуємо прослуховування, поки йде обробка та озвучка
        st.session_state.should_listen = False
        
        # 2. Обробляємо питання на бекенді та зберігаємо унікальний аудіофайл
        answer_text, audio_path = send_to_backend(voice_question)
        
        if answer_text:
            # Оновлюємо сторінку, щоб повідомлення з'явилося на екрані
            st.rerun()
    else:
        # Якщо користувач промовчав, робимо легку паузу і перезапускаємо цикл прослуховування
        time.sleep(1)
        st.rerun()

# Сценарій Б: Щойно згенерувалася нова відповідь бота, і мікрофон заблоковано
elif st.session_state.voice_active and not st.session_state.should_listen:
    # Перевіряємо, чи останнє повідомлення належить роботу
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        last_msg = st.session_state.messages[-1]
        
        if last_msg.get("audio_path") and os.path.exists(last_msg["audio_path"]):
            # Автоматично запускаємо програвання звуку на екрані (без кліків)
            # Параметр autoplay=True змусить браузер прочитати файл самостійно
            st.audio(last_msg["audio_path"], format="audio/mp3", autoplay=True)
            
            # Обчислюємо приблизну тривалість звучання відповіді (середня швидкість мовлення: 13 символів на секунду)
            # Це дозволяє заблокувати мікрофон саме на той час, поки робот говорить
            audio_duration = max(2.0, len(last_msg["text"]) / 13.0)
            time.sleep(audio_duration)
            
            # 3. СТАРТ МІКРОФОН: Робот закінчив говорити, знову відкриваємо мікрофон для користувача
            st.session_state.should_listen = True
            st.rerun()

# Сценарій В: Класичний текстовий чат (коли голосовий чат повністю вимкнено)
elif not st.session_state.voice_active:
    if text_question := st.chat_input("Задайте ваше питання тут..."):
        send_to_backend(text_question)
        st.rerun()