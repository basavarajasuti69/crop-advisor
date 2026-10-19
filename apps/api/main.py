from __future__ import annotations

import os
import sys
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents.graph import run_advice
from speech.tts_adapter import tts_adapter
from speech.stt import stt_adapter

APP_NAME = "crop-advisor"
REGION = os.getenv("AWS_REGION", "ap-south-1")

app = FastAPI(title=APP_NAME, version="0.1.0")


class AdviceRequest(BaseModel):
    image_url: str | None = None
    image_bytes: bytes | None = None
    crop: str | None = None
    language: str = "en"
    location: str | None = None


class AdviceResponse(BaseModel):
    case_id: str
    crop: str
    language: str
    diagnosis: dict[str, Any]
    advice: str
    citations: list[str]
    confidence: float
    safe: bool
    escalate: bool


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "region": REGION}


@app.post("/advice", response_model=AdviceResponse)
async def advice(req: AdviceRequest) -> AdviceResponse:
    state = await run_advice(req)
    return AdviceResponse(
        case_id=state.case_id,
        crop=state.crop or "",
        language=state.language,
        diagnosis={
            "predictions": state.predictions,
            "confidence": state.confidence,
            "quality_ok": state.quality_ok,
            "quality_issue": state.quality_issue,
        },
        advice=state.answer or "",
        citations=state.citations,
        confidence=state.confidence,
        safe=state.safe,
        escalate=state.escalate,
    )


class SpeakRequest(BaseModel):
    text: str
    language: str = "en"


@app.post("/speak")
async def speak(req: SpeakRequest):
    """Synthesize text to speech audio (MP3). Falls across Polly -> Sarvam -> Google."""
    audio = tts_adapter.synthesize(req.text, req.language)
    if audio is None:
        raise HTTPException(status_code=503, detail="TTS unavailable")
    return Response(content=audio, media_type="audio/mpeg")


class TranscribeRequest(BaseModel):
    audio_b64: str
    language: str = "en"


@app.post("/transcribe")
async def transcribe(req: TranscribeRequest):
    """Transcribe base64-encoded audio to text."""
    import base64

    audio_bytes = base64.b64decode(req.audio_b64)
    text = stt_adapter.transcribe(audio_bytes, req.language)
    if text is None:
        raise HTTPException(status_code=503, detail="STT unavailable")
    return {"text": text, "language": req.language}
async def advice_image(file: UploadFile = File(...), language: str = "en") -> AdviceResponse:
    contents = await file.read()
    state = await run_advice(
        AdviceRequest(image_url=file.filename, image_bytes=contents, language=language)
    )
    return AdviceResponse(
        case_id=state.case_id,
        crop=state.crop or "",
        language=state.language,
        diagnosis={
            "predictions": state.predictions,
            "confidence": state.confidence,
            "quality_ok": state.quality_ok,
            "quality_issue": state.quality_issue,
        },
        advice=state.answer or "",
        citations=state.citations,
        confidence=state.confidence,
        safe=state.safe,
        escalate=state.escalate,
    )
