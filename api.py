import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from gtts import gTTS
from app import answer_question

app = FastAPI(
    title="KPI Admission API",
    description="Єдиний бекенд для обробки текстових та голосових запитів",
    version="2.0.0"
)

AUDIO_DIR = "static_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

class QuestionRequest(BaseModel):
    question: str
    mode: str = "text"

class ApiResponse(BaseModel):
    mode: str
    answer: str
    sources: list[str]

@app.post("/api/ask", response_model=ApiResponse)
async def ask_bot(payload: QuestionRequest):
    """ Повертає текстовий JSON (підтримує як mode='text', так і mode='voice') """
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Питання порожнє")
    
    req_mode = "voice" if payload.mode.lower() == "voice" else "text"
    
    try:
        answer, docs = answer_question(payload.question.strip(), mode=req_mode)
        sources = list(set([doc.metadata.get("source_name", "Документ") for doc in docs]))
        return ApiResponse(mode=req_mode, answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ask_voice_stream")
async def ask_bot_voice_stream(payload: QuestionRequest):
    """ Приймає питання і одразу повертає .mp3 файл з лаконічною аудіо-відповіддю """
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Питання порожнє")
    
    try:
        answer, _ = answer_question(payload.question.strip(), mode="voice")
        
        tts = gTTS(text=answer, lang='uk', slow=False)
        audio_path = os.path.join(AUDIO_DIR, "response.mp3")
        tts.save(audio_path)
        
        return FileResponse(audio_path, media_type="audio/mp3", filename="response.mp3")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)