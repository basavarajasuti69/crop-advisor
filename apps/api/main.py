from __future__ import annotations

import base64
import os
import sys
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, field_validator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents.classifier import (
    _artifacts_dir,
    crop_from_label,
    load_class_names,
    load_model_metadata,
)
from agents.graph import run_advice
from speech.tts_adapter import tts_adapter
from speech.stt import stt_adapter

APP_NAME = "crop-advisor"
REGION = os.getenv("AWS_REGION", "ap-south-1")


def _cors_origins(raw: str | None = None) -> list[str]:
    """Parse CA_CORS_ORIGINS into a list of allowed origins.

    Whitespace and trailing slashes are removed. A browser sends the
    Origin header without a trailing slash, so a configured
    "https://x.netlify.app/" would never match and CORS would fail.
    """
    value = os.getenv("CA_CORS_ORIGINS", "http://localhost:3000") if raw is None else raw
    origins: list[str] = []
    for entry in value.split(","):
        origin = entry.strip().rstrip("/")
        if origin:
            origins.append(origin)
    return origins


app = FastAPI(title=APP_NAME, version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class AdviceRequest(BaseModel):
    image_url: str | None = None
    image_bytes: bytes | None = None
    crop: str | None = None
    language: str = "en"
    location: str | None = None

    @field_validator("image_bytes", mode="before")
    @classmethod
    def _decode_base64_image(cls, value: Any) -> Any:
        """Pydantic v2 encodes a JSON string into bytes as UTF-8, not base64.

        Clients send the image base64-encoded, so decode it here; otherwise the
        bytes arrive as garbage and the image cannot be opened.
        """
        if isinstance(value, str):
            try:
                return base64.b64decode(value, validate=True)
            except (ValueError, TypeError):
                return value
        return value


DISCLAIMER = (
    "This is an informational estimate, not a guaranteed diagnosis, and it is "
    "not a substitute for an agricultural officer. The score shown is an "
    "uncalibrated model confidence, not the probability of a disease."
)


def _model_disclaimer() -> str:
    """The disclaimer, naming the dataset the deployed model was trained on.

    Read from the export's own manifest, so the text cannot claim a
    dataset the model was not trained on.
    """
    metadata = load_model_metadata()
    dataset = metadata.get("dataset")
    if not dataset:
        return DISCLAIMER
    return (
        f"This is an informational estimate from a model trained on {dataset}. "
        "It is not a guaranteed diagnosis and it is not a substitute for an "
        "agricultural officer. The score shown is an uncalibrated model "
        "confidence, not the probability of a disease."
    )


class AdviceResponse(BaseModel):
    case_id: str
    crop: str
    language: str
    diagnosis: dict[str, Any]
    advice: str
    citations: list[str]
    confidence: float
    confidence_is_calibrated: bool = False
    safe: bool
    escalate: bool
    disclaimer: str = DISCLAIMER


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "region": REGION}


@app.get("/ready")
async def ready() -> dict[str, Any]:
    """Model readiness check.

    Reports whether the classifier artifacts are present and loadable,
    without exposing sensitive information. Used by deployment health
    checks and by the frontend to decide whether to enable analysis.
    """
    metadata = load_model_metadata()
    classes = metadata.get("classes") or []
    d = _artifacts_dir()
    onnx_path = d / "classifier.onnx"
    classes_path = d / "classes.json"
    artifacts_present = onnx_path.exists() and classes_path.exists()
    return {
        "status": "ready" if artifacts_present else "not_ready",
        "model": {
            "num_classes": metadata.get("num_classes") or len(classes),
            "dataset": metadata.get("dataset"),
            "confidence_is_calibrated": metadata.get("confidence_is_calibrated", False),
            "artifacts_present": artifacts_present,
        },
        "region": REGION,
    }


@app.get("/model")
async def model_info() -> dict[str, Any]:
    """What the deployed classifier can actually recognise.

    Everything here is read from the export's own manifest and class
    list, so the description cannot drift away from the artifact that
    is actually being served.
    """
    metadata = load_model_metadata()
    classes = metadata.get("classes") or load_class_names()
    crops = sorted({
        crop for crop in (crop_from_label(name) for name in classes) if crop
    })
    return {
        "classes": classes,
        "crops": crops,
        "crop_specific": bool(crops),
        "num_classes": metadata.get("num_classes") or len(classes),
        "trained_on": metadata.get("dataset"),
        "metrics": metadata.get("metrics"),
        "confidence_is_calibrated": metadata.get(
            "confidence_is_calibrated", False
        ),
        "disclaimer": _model_disclaimer(),
    }


@app.post("/advice", response_model=AdviceResponse)
async def advice(req: AdviceRequest) -> AdviceResponse:
    state = await run_advice(req)
    return AdviceResponse(
        case_id=state.case_id,
        crop=state.crop or "",
        language=state.language,
        diagnosis={
            "predictions": state.predictions,
            "top_predictions": state.predictions[:3],
            "confidence": state.confidence,
            "confidence_is_calibrated": False,
            "guidance_available": state.guidance_available,
            "quality_ok": state.quality_ok,
            "quality_issue": state.quality_issue,
        },
        advice=state.answer or "",
        citations=state.citations,
        confidence=state.confidence,
        confidence_is_calibrated=False,
        safe=state.safe,
        escalate=state.escalate,
        disclaimer=_model_disclaimer(),
    )


class SpeakRequest(BaseModel):
    text: str
    language: str = "en"


@app.post("/speak")
async def speak(req: SpeakRequest):
    """Synthesize text to speech audio (MP3). Falls across Polly -> Sarvam -> Google."""
    audio = tts_adapter.synthesize(req.text, req.language)
    if audio is None:
        providers = ", ".join(tts_adapter.configured_providers(req.language)) or "none configured"
        raise HTTPException(
            status_code=503,
            detail=f"TTS unavailable: no speech provider could serve '{req.language}' ({providers}).",
        )
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
@app.post("/advice/image", response_model=AdviceResponse)
async def advice_image(
    file: UploadFile = File(...),
    language: str = Form("en"),
    crop: str = Form(""),
) -> AdviceResponse:
    contents = await file.read()
    state = await run_advice(
        AdviceRequest(image_url=file.filename, image_bytes=contents, language=language, crop=crop or None)
    )
    return AdviceResponse(
        case_id=state.case_id,
        crop=state.crop or "",
        language=state.language,
        diagnosis={
            "predictions": state.predictions,
            "top_predictions": state.predictions[:3],
            "confidence": state.confidence,
            "confidence_is_calibrated": False,
            "guidance_available": state.guidance_available,
            "quality_ok": state.quality_ok,
            "quality_issue": state.quality_issue,
        },
        advice=state.answer or "",
        citations=state.citations,
        confidence=state.confidence,
        confidence_is_calibrated=False,
        safe=state.safe,
        escalate=state.escalate,
        disclaimer=_model_disclaimer(),
    )
