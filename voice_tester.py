import os
import streamlit as st
import speech_recognition as sr
import requests
from gtts import gTTS

API_URL_TEXT = "http://localhost:8000/api/ask"

AUDIO_DIR = "static_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)


st.set_page_config(page_title="Voice Assistant Tester", page_icon="🎙️")

st.title("🎙️ Голосовий асистент ФІОТ КПІ")
st.caption("Демонстраційний інтерфейс голосового асистента для абітурієнтів")


def record_and_recognize():
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 1.0
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True

    with sr.Microphone() as source:
        st.info("Слухаю вас... Говоріть.")
        recognizer.adjust_for_ambient_noise(source, duration=0.8)

        try:
            audio = recognizer.listen(
                source,
                timeout=5,
                phrase_time_limit=8,
            )
        except sr.WaitTimeoutError:
            st.warning("Я не почув мовлення. Спробуйте ще раз.")
            return None

    st.info("Розпізнаю мовлення...")

    try:
        return recognizer.recognize_google(audio, language="uk-UA")
    except sr.UnknownValueError:
        st.warning("Не вдалося розпізнати мовлення. Спробуйте сказати чіткіше.")
        return None
    except sr.RequestError as e:
        st.error(f"Помилка сервісу розпізнавання мовлення: {e}")
        return None


def make_audio(text: str):
    audio_file = os.path.join(AUDIO_DIR, "voice_answer.mp3")

    tts = gTTS(
        text=text,
        lang="uk",
        slow=False,
    )

    tts.save(audio_file)
    return audio_file


def ask_api(question: str):
    response = requests.post(
        API_URL_TEXT,
        json={
            "question": question,
            "mode": "text",
        },
        timeout=180,
    )

    if response.status_code != 200:
        raise RuntimeError(f"API error {response.status_code}: {response.text}")

    data = response.json()
    return data.get("answer", "")


if st.button("🎤 Поставити питання голосом"):
    question = record_and_recognize()

    if question:
        st.markdown(f"**Ви запитали:** *«{question}»*")

        with st.spinner("Асистент формує відповідь..."):
            try:
                answer = ask_api(question)

                if not answer:
                    st.error("API повернуло порожню відповідь.")
                    st.stop()

                audio_file = make_audio(answer)

                st.subheader("🗣️ Модель відповіла:")
                st.success(answer)
                st.audio(audio_file, format="audio/mp3")

            except Exception as e:
                st.error(f"Помилка у voice_tester.py: {e}")