import streamlit as st
import speech_recognition as sr
import requests

API_URL_TEXT = "http://localhost:8000/api/ask"
API_URL_VOICE = "http://localhost:8000/api/ask_voice_stream"

st.set_page_config(page_title="Voice Assistant Tester", page_icon="🎙️")
st.title("🎙️ Голосовий асистент ФІОТ КПІ")
st.caption("Демонстраційний інтерфейс для перевірки телефону (STT/TTS)")

def record_and_recognize():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Слухаю вас... Говоріть.")
        recognizer.adjust_for_ambient_noise(source, duration=0.8)
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio, language="uk-UA")
    except Exception:
        return None

if st.button("🎤 Поставити питання голосом"):
    question = record_and_recognize()
    if question:
        st.markdown(f"**Ви запитали:** *«{question}»*")
        
        with st.spinner("Голосовий асистент формує відповідь..."):
            try:
                # Тягнемо аудіо відповіді
                res_voice = requests.post(API_URL_VOICE, json={"question": question, "mode": "voice"})
                # Тягнемо текст відповіді для виведення на екран
                res_text = requests.post(API_URL_TEXT, json={"question": question, "mode": "voice"}).json()
                
                if res_voice.status_code == 200:
                    audio_file = "static_audio/temp_reply.mp3"
                    with open(audio_file, "wb") as f:
                        f.write(res_voice.content)
                    
                    st.subheader("🗣️ Модель відповіла:")
                    st.warning(res_text["answer"])
                    st.audio(audio_file, format="audio/mp3")
            except Exception as e:
                st.error("Перевірте, чи запущено сервер у терміналі за допомогою команди: python api.py")