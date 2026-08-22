import asyncio
import os
import time
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

from .engine import RAGEngine

engine = RAGEngine()

@asynccontextmanager
async def lifespan(_: FastAPI):
    # Serve health/STT immediately while the heavier embedding model warms in RAM.
    load_task = asyncio.create_task(asyncio.to_thread(engine.load))
    yield
    if not load_task.done(): load_task.cancel()

app = FastAPI(title="GoaVaani Voice RAG", version="1.0.0", lifespan=lifespan)
origins = [item.strip() for item in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173").split(",")]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=["GET","POST"], allow_headers=["*"])

class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    language: str = Field(default="en", pattern="^(en|hi|te)$")
    mode: str = Field(default="fast", pattern="^(fast|enhanced)$")

@app.get("/health")
async def health():
    stt_provider = "elevenlabs" if os.getenv("ELEVENLABS_API_KEY") else None
    all_indexes_ready = engine.ready and all(language in engine.metadata for language in ("en", "hi", "te"))
    return {"status":"ok","model_ready":engine.ready,"model_state":"ready" if engine.ready else "warming","index_ready":all_indexes_ready,"retrieval_mode":"lexical" if engine.lightweight else "semantic","stt_ready":bool(stt_provider),"stt_provider":stt_provider,"groq_ready":bool(os.getenv("GROQ_API_KEY")),"languages":["en","hi","te"]}

@app.post("/api/ask")
async def ask(payload: AskRequest):
    return await engine.ask(payload.question.strip(), payload.language, payload.mode)

@app.get("/api/suggestions")
async def suggestions(language: str = Query("en", pattern="^(en|hi|te)$")):
    rows = engine.metadata.get(language, [])
    unique = []
    seen = set()
    if rows:
        stride = max(1, len(rows) // 30)
        for row in rows[::stride]:
            question = " ".join((row.get("source_query") or "").split())
            if 8 <= len(question) <= 180 and question.casefold() not in seen:
                unique.append(question); seen.add(question.casefold())
            if len(unique) == 8: break
    return {"language":language,"index_ready":bool(rows),"questions":unique}

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=.15, min=.15, max=.5), reraise=True)
async def elevenlabs_transcribe(data: bytes, filename: str, content_type: str, language: str):
    start = time.perf_counter()
    provider_language = {"en":"eng", "hi":"hin", "te":"tel"}[language]
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post("https://api.elevenlabs.io/v1/speech-to-text", headers={"xi-api-key":os.environ["ELEVENLABS_API_KEY"]}, files={"file":(filename,data,content_type)}, data={"model_id":"scribe_v2","language_code":provider_language,"tag_audio_events":"false","diarize":"false"})
        response.raise_for_status()
    payload = response.json(); payload["latency_ms"] = round((time.perf_counter()-start)*1000,3)
    return payload

@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = Query("en", pattern="^(en|hi|te)$")):
    provider = "elevenlabs" if os.getenv("ELEVENLABS_API_KEY") else None
    if not provider: raise HTTPException(503,"Configure ELEVENLABS_API_KEY")
    data = await file.read()
    if not data or len(data) > 20_000_000: raise HTTPException(400,"Audio must be between 1 byte and 20 MB")
    try:
        result = await elevenlabs_transcribe(data,file.filename or "question.webm",file.content_type or "audio/webm",language)
        text = result.get("text","")
        return {"text":text.strip(),"language":result.get("language_code",language),"confidence":result.get("language_probability"),"provider":provider,"timings_ms":{"speech_to_text":result["latency_ms"]}}
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", {})
            provider_message = detail.get("message") if isinstance(detail, dict) else str(detail)
        except Exception:
            provider_message = ""
        message = f"Speech provider returned {exc.response.status_code}"
        if provider_message: message += f": {provider_message[:240]}"
        raise HTTPException(502,message) from exc
    except Exception as exc: raise HTTPException(502,"Speech transcription failed after retry") from exc

@app.post("/api/voice-ask")
async def voice_ask(file: UploadFile = File(...), language: str = Query("en", pattern="^(en|hi|te)$"), mode: str = Query("fast", pattern="^(fast|enhanced)$")):
    """One measured audio-to-final-answer request, including STT and optional Groq."""
    wall_start = time.perf_counter()
    transcript = await transcribe(file, language)
    if not transcript["text"]: raise HTTPException(422,"No speech was recognized")
    result = await engine.ask(transcript["text"], language, mode)
    rag_total = result["timings_ms"].pop("total", 0.0)
    result["timings_ms"]["speech_to_text"] = transcript["timings_ms"]["speech_to_text"]
    result["timings_ms"]["rag_total"] = rag_total
    result["timings_ms"]["total"] = round((time.perf_counter()-wall_start)*1000,3)
    result["transcript"] = transcript["text"]
    result["transcript_language"] = transcript["language"]
    result["stt_provider"] = transcript["provider"]
    return result
