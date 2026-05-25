import os
import streamlit as st
import speech_recognition as sr
import requests
from gtts import gTTS  # Додали локальний синтез прямо у тестер

# Нам тепер потрібен ЛИШЕ ОДИН ендпоінт — текстовий
API_URL_TEXT = "http://localhost:8000/api/ask"

AUDIO_DIR = "static_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

st.set_page_config(page_title="Voice Assistant Tester", page_icon="🎙️")
st.title("🎙️ Голосовий асистент ФІОТ КПІ")
st.caption("Демонстраційний інтерфейс для перевірки телефонного сценарію (STT/TTS)")


def record_and_recognize():
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 1.0
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True

    with sr.Microphone() as source:
        st.info("Слухаю вас... Говоріть.")
        recognizer.adjust_for_ambient_noise(source, duration=0.8)

        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
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
        st.error(f"Помилка розпізнавання мовлення: {e}")
        return None


if st.button("🎤 Поставити питання голосом"):
    question = record_and_recognize()

    if question:
        st.markdown(f"**Ви запитали:** *«{question}»*")

        with st.spinner("Голосовий асистент формує відповідь..."):
            try:
                # КРОК 1: Робимо ОДИН запит до бекенду, щоб отримати ідеальну коротку відповідь
                res_text = requests.post(
                    API_URL_TEXT,
                    json={"question": question, "mode": "voice"},
                    timeout=180,
                )

                if res_text.status_code != 200:
                    st.error(f"Помилка текстового API: {res_text.status_code} — {res_text.text}")
                    st.stop()

                # Дістаємо текст, який згенерувала модель qwen
                answer = res_text.json().get("answer", "")

                # КРОК 2: Відображаємо цей текст на екрані (він гарантовано єдиний)
                st.subheader("🗣️ Модель відповіла:")
                st.success(answer)

                # КРОК 3: Одразу озвучуємо цей самий текст за допомогою gTTS
                # Перетворюємо текст на аудіопотік українською мовою
                tts = gTTS(text=answer, lang='uk', slow=False)
                
                audio_file = os.path.join(AUDIO_DIR, "temp_reply.mp3")
                tts.save(audio_file)

                # КРОК 4: Виводимо аудіоплеєр із цим файлом
                st.audio(audio_file, format="audio/mp3")

            except Exception as e:
                st.error(f"Помилка у voice_tester.py: {e}")