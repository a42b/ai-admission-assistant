import os
import streamlit as st
import speech_recognition as sr
import requests
from gtts import gTTS
import time

API_URL_TEXT = "http://localhost:8000/api/ask"
AUDIO_DIR = "static_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

st.set_page_config(page_title="AI Admission Assistant", layout="centered")
st.title("AI Admission Assistant ФІОТ КПІ")
st.caption("Уніфікований діалоговий інтерфейс приймальної комісії (Текст / Автоматичний Голос)")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "voice_active" not in st.session_state:
    st.session_state.voice_active = False

if "should_listen" not in st.session_state:
    st.session_state.should_listen = False

def record_and_recognize():
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 1.0
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True

    with sr.Microphone() as source:
        st.info("Мікрофон відкрито. Говоріть...")
        recognizer.adjust_for_ambient_noise(source, duration=0.8)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
            st.info("Очікуйте, обробка мовлення...")
            return recognizer.recognize_google(audio, language="uk-UA")
        except sr.WaitTimeoutError:
            return None
        except Exception:
            return None

def send_to_backend(question):
    st.session_state.messages.append({"role": "user", "text": question, "audio_path": None})
    try:
        res = requests.post(
            API_URL_TEXT,
            json={"question": question, "mode": "voice"},
            timeout=180,
        )
        if res.status_code == 200:
            answer = res.json().get("answer", "")
            tts = gTTS(text=answer, lang='uk', slow=False)
            filename = f"reply_{int(time.time())}.mp3"
            audio_path = os.path.join(AUDIO_DIR, filename)
            tts.save(audio_path)
            st.session_state.messages.append({"role": "assistant", "text": answer, "audio_path": audio_path})
            return answer, audio_path
        else:
            st.error("Помилка сервера бекенду.")
            return None, None
    except Exception as e:
        st.error(f"Не вдалося з'єднатися з api.py: {e}")
        return None, None

col1, col2 = st.columns(2)

with col1:
    if not st.session_state.voice_active:
        if st.button("Розпочати голосовий чат", use_container_width=True):
            st.session_state.voice_active = True
            st.session_state.should_listen = True
            st.rerun()
    else:
        if st.button("Припинити голосовий чат", use_container_width=True):
            st.session_state.voice_active = False
            st.session_state.should_listen = False
            st.rerun()

with col2:
    if st.button("Очистити історію чату", use_container_width=True):
        st.session_state.messages = []
        st.session_state.should_listen = st.session_state.voice_active
        st.rerun()

st.write("---")

for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["text"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["text"])
            if msg.get("audio_path") and os.path.exists(msg["audio_path"]):
                st.audio(msg["audio_path"], format="audio/mp3")

if st.session_state.voice_active and st.session_state.should_listen:
    voice_question = record_and_recognize()
    if voice_question:
        st.session_state.should_listen = False
        answer_text, audio_path = send_to_backend(voice_question)
        if answer_text:
            st.rerun()
    else:
        time.sleep(1)
        st.rerun()

elif st.session_state.voice_active and not st.session_state.should_listen:
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        last_msg = st.session_state.messages[-1]
        if last_msg.get("audio_path") and os.path.exists(last_msg["audio_path"]):
            st.audio(last_msg["audio_path"], format="audio/mp3", autoplay=True)
            audio_duration = max(2.0, len(last_msg["text"]) / 13.0)
            time.sleep(audio_duration)
            st.session_state.should_listen = True
            st.rerun()

elif not st.session_state.voice_active:
    if text_question := st.chat_input("Задайте ваше питання тут..."):
        send_to_backend(text_question)
        st.rerun()