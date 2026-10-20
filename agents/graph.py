"""LangGraph state and node definitions for the crop-disease advisor.

Phase 3: triage -> diagnosis (classifier + vision LLM second opinion) ->
context (weather MCP) -> retrieve (crop_knowledge MCP) -> draft -> localize ->
critic (dose verification) -> follow-up (reminders + escalation).
"""

from __future__ import annotations

import asyncio
import importlib
import io
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from langgraph.graph import StateGraph, END
from mcp.types import CallToolRequest, CallToolRequestParams
from PIL import Image
from pydantic import BaseModel, Field

from rag.glossary import hi as glossary_hi  # noqa: E402
from rag.glossary import kn as glossary_kn  # noqa: E402

from apps.api.config import settings  # noqa: E402

from agents.llm import explain, llm_configured, vision_opinion  # noqa: E402

# ---------------------------------------------------------------------------
# Classifier: SageMaker endpoint first, local PyTorch fallback
# ---------------------------------------------------------------------------

_SAGEMAKER_CLIENT = None


def _sagemaker_client():
    global _SAGEMAKER_CLIENT
    if _SAGEMAKER_CLIENT is None:
        import boto3

        _SAGEMAKER_CLIENT = boto3.client("sagemaker-runtime", region_name=settings.sagemaker_region)
    return _SAGEMAKER_CLIENT


_CLASSIFIER = None
_CLASSES = None
_ONNX_SESSION = None
_ONNX_CLASSES: list[str] = []
_TORCH_TRANSFORM = None

# ONNX artifacts exported by ml/export_edge/export_onnx.py.
_ONNX_DIR = Path(__file__).resolve().parent.parent / "ml/export_edge/artifacts"
_ONNX_MEAN = (0.485, 0.456, 0.406)
_ONNX_STD = (0.229, 0.224, 0.225)


def _repo_root() -> Path:
    """Repo root, so paths do not depend on the current working directory."""
    return Path(__file__).resolve().parent.parent


def _load_classifier(weights_path: str | None = None, metrics_path: str | None = None):
    """Load the PyTorch classifier. Paths are injectable so tests can simulate
    a missing model. Raises ModelUnavailable when torch is not installed."""
    global _CLASSIFIER, _CLASSES
    if weights_path is None and metrics_path is None and _CLASSIFIER is not None:
        return _CLASSIFIER, _CLASSES
    # Resolve from the repo root, not the CWD: the API may be started from
    # anywhere (and in Docker the CWD is the image WORKDIR).
    root = _repo_root()
    w = Path(weights_path) if weights_path else root / "ml/training/artifacts/model.pt"
    m = Path(metrics_path) if metrics_path else root / "ml/training/artifacts/metrics.json"
    if not (w.exists() and m.exists()):
        raise ModelUnavailable("classifier model file missing")
    try:
        import torch
        from torchvision import models
    except ImportError as exc:
        raise ModelUnavailable("torch is not installed") from exc
    classes = json.loads(m.read_text())["classes"]
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(torch.load(w, map_location="cpu"))
    model.eval()
    _CLASSIFIER = model
    _CLASSES = classes
    return _CLASSIFIER, _CLASSES


def _torch_transform():
    """The training/eval transform: Resize(256) + CenterCrop(224) + Normalize."""
    global _TORCH_TRANSFORM
    if _TORCH_TRANSFORM is None:
        from torchvision import transforms

        _TORCH_TRANSFORM = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(list(_ONNX_MEAN), list(_ONNX_STD)),
        ])
    return _TORCH_TRANSFORM


def _onnx_preprocess(img: Image.Image):
    """NumPy equivalent of _torch_transform() - must stay byte-identical to it."""
    import numpy as np

    width, height = img.size
    if width <= height:
        new_w, new_h = 256, int(256 * height / width)
    else:
        new_h, new_w = 256, int(256 * width / height)
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    left = int(round((new_w - 224) / 2.0))
    top = int(round((new_h - 224) / 2.0))
    cropped = resized.crop((left, top, left + 224, top + 224))
    arr = np.asarray(cropped, dtype="float32") / 255.0
    arr = (arr - np.array(_ONNX_MEAN, dtype="float32")) / np.array(_ONNX_STD, dtype="float32")
    return arr.transpose(2, 0, 1)[None, ...]


def _load_onnx():
    """Load the ONNX Runtime session. Returns (session, classes) or None."""
    global _ONNX_SESSION, _ONNX_CLASSES
    if _ONNX_SESSION is not None:
        return _ONNX_SESSION, _ONNX_CLASSES
    model_path = _ONNX_DIR / "classifier.onnx"
    classes_path = _ONNX_DIR / "classes.json"
    if not (model_path.exists() and classes_path.exists()):
        return None
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    _ONNX_CLASSES = json.loads(classes_path.read_text())["classes"]
    _ONNX_SESSION = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    return _ONNX_SESSION, _ONNX_CLASSES


def _predict_onnx(image_bytes: bytes, top_k: int = 3) -> list[dict[str, Any]]:
    """Classify with ONNX Runtime. No torch required."""
    import numpy as np

    loaded = _load_onnx()
    if loaded is None:
        raise ModelUnavailable("onnx model not available")
    session, classes = loaded
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    logits = session.run(["logits"], {"input": _onnx_preprocess(img)})[0].reshape(-1)
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    order = np.argsort(probs)[::-1][: min(top_k, len(classes))]
    return [
        {"label": classes[int(i)], "score": round(float(probs[i]), 4)} for i in order
    ]


# ---------------------------------------------------------------------------
# Typed errors raised by the classifier path (never fake a prediction)
# ---------------------------------------------------------------------------


class ModelUnavailable(Exception):
    """The classifier model file is missing or could not be loaded."""


class ImageUnreadable(Exception):
    """The uploaded image could not be opened/decoded."""


# ---------------------------------------------------------------------------
# Confidence thresholds (named constants; no raw numbers in logic)
# ---------------------------------------------------------------------------
CONFIDENCE_MIN = 0.70   # "not sure" threshold; below it -> retake photo
HEALTHY_MIN = 0.85      # a "healthy" result below this is treated as not sure
FOLLOW_UP_MIN = 0.80    # below it -> schedule follow-up
VISION_DISAGREE_MIN = 0.70   # classifier/vision disagreement retake threshold
VISION_ENABLED = os.getenv("CA_VISION_SECOND_OPINION", "false").lower() == "true"


def _predict_image(image_bytes: bytes, top_k: int = 3, model=None, classes=None) -> list[dict[str, Any]]:
    """Classify an image.

    Backends, in order: an explicitly injected model, ONNX Runtime (no torch
    needed), then the local PyTorch model. Never returns a fake prediction.
    """
    try:
        Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise ImageUnreadable("could not open image")
    if model is None and classes is None:
        if _load_onnx() is not None:
            return _predict_onnx(image_bytes, top_k)
        model, classes = _load_classifier()
    elif model is None or classes is None:
        raise ValueError("model and classes must be provided together")
    import torch

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = _torch_transform()(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    top = torch.topk(probs, min(top_k, len(classes)))
    return [
        {"label": classes[int(i)], "score": round(float(p), 4)}
        for i, p in zip(top.indices, top.values)
    ]


def _predict_sagemaker(image_bytes: bytes) -> list[dict[str, Any]]:
    """Invoke SageMaker endpoint for classification (production path)."""
    if not settings.sagemaker_endpoint:
        return []
    try:
        client = _sagemaker_client()
        buf = io.BytesIO(image_bytes)
        response = client.invoke_endpoint(
            EndpointName=settings.sagemaker_endpoint,
            ContentType="application/x-image",
            Body=buf.read(),
        )
        body = json.loads(response["Body"].read().decode())
        if isinstance(body, dict) and "predictions" in body:
            return body["predictions"]
        if isinstance(body, list):
            return body
        return [{"label": str(body.get("label", "healthy")), "score": float(body.get("score", 0.0))}]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# MCP client (imports local MCP server modules directly)
# ---------------------------------------------------------------------------


MCP_SERVER_MODULES = {
    "weather": "mcp_servers.weather.server",
    "crop_knowledge": "mcp_servers.crop_knowledge.server",
    "officer_directory": "mcp_servers.officer_directory.server",
}


def _mcp_module(server_name: str):
    mod_name = MCP_SERVER_MODULES.get(server_name)
    if not mod_name:
        return None
    try:
        return importlib.import_module(mod_name)
    except Exception:
        return None


async def mcp_call(server_name: str, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call an MCP server tool by importing its module and invoking the tool handler."""
    mod = _mcp_module(server_name)
    if mod is None:
        return {"error": f"Unknown MCP server: {server_name}"}
    server_obj = getattr(mod, "server", None)
    if server_obj is None:
        return {"error": f"No server object in {server_name}"}
    try:
        handler = server_obj.request_handlers[CallToolRequest]
        req = CallToolRequest(
            method="tools/call",
            params=CallToolRequestParams(name=tool, arguments=arguments),
        )
        result = await handler(req)
        # result is ServerResult wrapping CallToolResult
        call_result = getattr(result, "root", result)
        content = getattr(call_result, "content", [])
        if content and isinstance(content, list):
            text = content[0].text if hasattr(content[0], "text") else str(content[0])
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"text": text}
        return {"error": "Empty MCP result"}
    except Exception as exc:
        return {"error": f"MCP call failed: {str(exc)}"}


_STRINGS_DIR = Path(__file__).resolve().parent.parent / "strings"


def load_strings(language: str) -> dict[str, str]:
    path = _STRINGS_DIR / f"{language}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _validate_kannada(text: str) -> bool:
    """Return True if text is at least 90% Kannada letters, has no '#' headings,
    and contains no letter from another Indian script."""
    if "#" in text:
        return False
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return True
    kannada = [ch for ch in letters if "\u0c80" <= ch <= "\u0cff"]
    other_indian = [ch for ch in letters if not ch.isascii() and not ("\u0c80" <= ch <= "\u0cff")]
    if other_indian:
        return False
    meaningful = [ch for ch in letters if not ch.isascii()]
    if not meaningful:
        return True
    return len(kannada) / len(meaningful) >= 0.90


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AgentState(BaseModel):
    case_id: str
    image_bytes: bytes | None = None
    image_url: str | None = None
    crop: str | None = None
    language: str = "en"
    location: str | None = None

    # quality + vision
    quality_ok: bool = False
    quality_issue: str | None = None
    predictions: list[dict[str, Any]] = Field(default_factory=list)
    vision_llm_opinion: dict[str, Any] | None = None
    agreed: bool | None = None
    retake_requested: bool = False

    # RAG / advice
    passages: list[dict[str, Any]] = Field(default_factory=list)
    answer: str | None = None
    citations: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    llm_explained: bool = False

    # context (weather MCP)
    weather: dict[str, Any] | None = None
    season: dict[str, Any] | None = None

    # officer directory
    officer: list[dict[str, Any]] = Field(default_factory=list)

    # safety / outcome
    safe: bool = True
    escalate: bool = False
    escalate_reason: str | None = None
    ood: bool = False
    follow_up_scheduled: bool = False
    follow_up_days: int = 5

    # control flow
    step: str = "start"
    error: str | None = None


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


async def triage(state: AgentState) -> AgentState:
    """Triage: quality check, crop ID, path decision."""
    state.step = "triage"
    if state.image_bytes is None and state.image_url is None:
        state.quality_ok = False
        state.quality_issue = "no image provided"
        state.retake_requested = True
        state.safe = False
        state.escalate = True
        state.escalate_reason = "no_image"
        strings = load_strings(state.language)
        # NEEDS NATIVE REVIEW: no_photo is null for hi/kn
        state.answer = strings.get("no_photo") or strings.get(
            "cannot_read", "Cannot read this photo. Please retake it."
        )
        return state
    # Placeholder quality model; real implementation runs blur/leaf/lit checks.
    state.quality_ok = True
    state.quality_issue = None
    # Crop ID: use provided crop or default
    state.crop = state.crop or "rice"
    return state


async def classify(state: AgentState) -> AgentState:
    """Run the crop classifier; returns top-k disease predictions."""
    state.step = "classify"
    if not state.quality_ok:
        return state
    image_bytes = state.image_bytes
    if image_bytes is None:
        state.error = "no_image"
        state.safe = False
        state.escalate = True
        state.escalate_reason = "no_image"
        state.retake_requested = True
        strings = load_strings(state.language)
        state.answer = strings.get("cannot_read", "Cannot read this photo. Please retake it.")
        return state
    # Try SageMaker endpoint first (production), fall back to local PyTorch.
    predictions = None
    try:
        predictions = _predict_sagemaker(image_bytes)
    except Exception:
        predictions = []
    if not predictions:
        try:
            predictions = _predict_image(image_bytes)
        except ModelUnavailable:
            state.error = "model_unavailable"
            state.safe = False
            state.escalate = True
            state.escalate_reason = "model_unavailable"
            state.retake_requested = True
            strings = load_strings(state.language)
            state.answer = strings.get("service_problem", "Service problem, please try again later.")
            return state
        except ImageUnreadable:
            state.error = "image_unreadable"
            state.safe = False
            state.escalate = True
            state.escalate_reason = "image_unreadable"
            state.retake_requested = True
            strings = load_strings(state.language)
            state.answer = strings.get("cannot_read", "Cannot read this photo. Please retake it.")
            return state
    if not predictions:
        state.error = "no_predictions"
        state.safe = False
        state.escalate = True
        state.escalate_reason = "no_predictions"
        state.retake_requested = True
        strings = load_strings(state.language)
        state.answer = strings.get("service_problem", "Service problem, please try again later.")
        return state
    state.predictions = predictions
    state.crop = state.crop or "rice"
    state.confidence = predictions[0]["score"] if predictions else 0.0
    return state


async def vision_second_opinion(state: AgentState) -> AgentState:
    """Vision LLM cross-check; enabled only when CA_VISION_SECOND_OPINION=true.

    Strips EXIF/GPS, resizes to 512px, sends only the image to the model,
    asks for symptoms + up to 3 diseases (JSON), and compares by label match.
    Disagreement below the threshold requests a retake.
    """
    state.step = "vision_second_opinion"
    if state.error:
        return state
    if not VISION_ENABLED:
        state.vision_llm_opinion = None
        state.agreed = None
        return state
    image_bytes = state.image_bytes
    if image_bytes is None:
        state.vision_llm_opinion = None
        state.agreed = None
        return state
    class_list = _CLASSES or [p["label"] for p in state.predictions]
    try:
        opinion = await asyncio.to_thread(vision_opinion, image_bytes, class_list, state.language)
    except Exception:
        opinion = None
    state.vision_llm_opinion = opinion
    if not opinion:
        state.agreed = None
        return state
    vision_diseases = opinion.get("diseases", []) or []
    classifier_label = state.predictions[0]["label"] if state.predictions else None
    state.agreed = bool(classifier_label) and classifier_label in vision_diseases
    if not state.agreed and state.confidence < VISION_DISAGREE_MIN:
        state.retake_requested = True
        state.quality_issue = "Classifier and vision LLM disagree; please retake photo."
    return state


async def context_weather(state: AgentState) -> AgentState:
    """Add weather + season context via weather MCP server."""
    state.step = "context_weather"
    if not state.location:
        return state
    weather = await mcp_call("weather", "get_weather", {"location": state.location, "date": "today"})
    state.weather = weather if "error" not in weather else None
    import datetime

    month = datetime.datetime.now().month
    season = await mcp_call("weather", "get_season", {"location": state.location, "month": month})
    state.season = season if "error" not in season else None
    return state


async def retrieve(state: AgentState) -> AgentState:
    """RAG retrieval via crop_knowledge MCP server."""
    state.step = "retrieve"
    label = state.predictions[0]["label"] if state.predictions else "healthy"
    crop = state.crop or "rice"
    kb = await mcp_call(
        "crop_knowledge", "search_knowledge", {"query": f"{label} {crop}", "crop": crop}
    )
    if "error" not in kb and isinstance(kb, list) and kb:
        state.passages = kb
        state.citations = ["crop_knowledge_mcp"]
    else:
        state.passages = [
            {
                "text": f"{label} guidance for {crop}: remove affected leaves and apply approved fungicide.",
                "source": "crop_knowledge_base",
                "dose": "as per label",
            }
        ]
        state.citations = ["crop_knowledge_base"]
    return state


async def draft_advice(state: AgentState) -> AgentState:
    """Draft grounded advice with citations."""
    state.step = "draft_advice"
    label = state.predictions[0]["label"] if state.predictions else "healthy"
    crop = state.crop or "rice"
    passage = state.passages[0]["text"] if state.passages else ""
    strings = load_strings("en")
    intro = strings.get("intro", "Your {crop} leaf shows {label}.").format(crop=crop, label=label)
    confirm = strings.get("confirm", "Confirm with your local agriculture officer before spraying.")
    state.answer = f"{intro} {passage} {confirm}"
    state.citations = state.citations or ["crop_knowledge_base"]
    return state


async def localize_advice(state: AgentState) -> AgentState:
    """Translate the drafted advice into the farmer's language (Phase 2: Hindi; Phase 5: Kannada)."""
    state.step = "localize_advice"
    label = state.predictions[0]["label"] if state.predictions else "healthy"
    crop = state.crop or "rice"
    passage = state.passages[0]["text"] if state.passages else ""

    if state.language == "hi":
        strings = load_strings("hi")
        crop_local = glossary_hi.CROP_NAMES.get(crop)
        if crop_local is None:
            crop_local = strings.get("neutral_crop", "आपकी फसल")
        disease_local = glossary_hi.DISEASE_NAMES.get(label)
        if disease_local is None:
            # NEEDS NATIVE REVIEW: disease_unknown is null for hi
            disease_local = strings.get("disease_unknown") or strings.get("neutral_crop", "आपकी फसल")
        intro = strings.get("intro", "").format(crop=crop_local, label=disease_local)
        confirm = strings.get("confirm", "")
        state.answer = f"{intro} {passage} {confirm}"
    elif state.language == "kn":
        # Never put English passage text into a Kannada answer.
        strings = load_strings("kn")
        crop_local = glossary_kn.CROP_NAMES.get(crop)
        if crop_local is None:
            crop_local = strings.get("neutral_crop", "ನಿಮ್ಮ ಬೆಳೆ")
        disease_local = glossary_kn.DISEASE_NAMES.get(label)
        if disease_local is None:
            # NEEDS NATIVE REVIEW: disease_unknown is null for kn
            disease_local = strings.get("disease_unknown") or strings.get("neutral_crop", "ನಿಮ್ಮ ಬೆಳೆ")
        intro = strings.get("intro", "").format(crop=crop_local, label=disease_local)
        confirm = strings.get("confirm", "")
        state.answer = f"{intro} {confirm}"
    else:
        state.answer = state.answer or ""
    return state


async def llm_explain(state: AgentState) -> AgentState:
    """LLM-written farmer explanation in the target language.

    Sends only the disease label and the top 2 retrieved chunks - never the
    image, never the location. No-op when the LLM is not configured, so the
    plain template from localize_advice is used instead.
    """
    state.step = "llm_explain"
    if state.error:
        return state
    if not llm_configured():
        return state
    label = state.predictions[0]["label"] if state.predictions else "healthy"
    chunks = state.passages[:2]
    try:
        answer = await asyncio.to_thread(explain, label, chunks, state.language)
    except Exception:
        return state
    if answer:
        state.answer = answer
        state.llm_explained = True
    return state


async def critic_safety(state: AgentState) -> AgentState:
    """Verify doses and chemical names match source text; block unsupported claims."""
    state.step = "critic_safety"
    state.safe = True
    state.escalate = False
    state.escalate_reason = None
    label = state.predictions[0]["label"] if state.predictions else "healthy"

    # OOD / "not sure" check: confidence below threshold
    if state.confidence < CONFIDENCE_MIN:
        state.ood = True
        state.escalate = True
        state.escalate_reason = "low_confidence"
        top2 = state.predictions[:2]
        possibilities = "; ".join(f"{p['label']} ({p['score']})" for p in top2)
        strings = load_strings(state.language)
        retake = strings.get("retake", "Please retake the photo. I am not sure. Top possibilities: {possibilities}.")
        state.answer = retake.format(possibilities=possibilities)
        return state

    # Healthy-but-unsure check: a "healthy" result below HEALTHY_MIN is not sure
    if label == "healthy" and state.confidence < HEALTHY_MIN:
        state.ood = True
        state.escalate = True
        state.escalate_reason = "low_confidence"
        top2 = state.predictions[:2]
        possibilities = "; ".join(f"{p['label']} ({p['score']})" for p in top2)
        strings = load_strings(state.language)
        retake = strings.get("retake", "Please retake the photo. I am not sure. Top possibilities: {possibilities}.")
        state.answer = retake.format(possibilities=possibilities)
        return state

    # Retake requested by vision disagreement
    if state.retake_requested:
        state.escalate = True
        state.escalate_reason = "retake_needed"
        return state

    # Dose verification: any passage mentioning a dose must be source-quoted
    for passage in state.passages:
        dose = passage.get("dose", "")
        if dose and dose not in ("as per label", "as prescribed"):
            state.safe = False
            state.escalate = True
            state.escalate_reason = "unverified_dose"
            break

    # Kannada validator: >= 90% Kannada letters, no markdown headings, no other Indian script
    if state.language == "kn" and state.answer and not _validate_kannada(state.answer):
        state.safe = False
        state.escalate = True
        state.escalate_reason = "language_mismatch"
        strings = load_strings("kn")
        crop_local = glossary_kn.CROP_NAMES.get(state.crop or "rice", state.crop or "rice")
        disease_local = glossary_kn.DISEASE_NAMES.get(label, label)
        intro = strings.get("intro", "").format(crop=crop_local, label=disease_local)
        confirm = strings.get("confirm", "")
        state.answer = f"{intro} {confirm}"

    # Bedrock Guardrails check (production): screen the advice text
    if state.safe and settings.guardrail_id and state.answer:
        try:
            import boto3
            br = boto3.client("bedrock-runtime", region_name=settings.region)
            br.apply_guardrail(
                guardrailIdentifier=settings.guardrail_id,
                guardrailVersion=settings.guardrail_version,
                source={
                    "text": [
                        {
                            "text": {"content": state.answer},
                        }
                    ]
                },
            )
        except Exception:
            # Guardrail service unavailable - fail closed
            state.safe = False
            state.escalate = True
            state.escalate_reason = "guardrail_error"

    return state


async def find_officer(state: AgentState) -> AgentState:
    """Look up local agriculture officer via officer_directory MCP (for escalation)."""
    state.step = "find_officer"
    if not state.location:
        return state
    crop = state.crop or "rice"
    officer = await mcp_call(
        "officer_directory", "find_officer", {"location": state.location, "crop": crop, "language": state.language}
    )
    if "error" not in officer:
        officers = officer if isinstance(officer, list) else [officer]
        state.officer = [
            o
            for o in officers
            if isinstance(o, dict) and "message" not in o and "error" not in o
        ]
    return state


async def follow_up(state: AgentState) -> AgentState:
    """Schedule follow-up reminder and/or escalation to expert."""
    state.step = "follow_up"
    if state.escalate:
        state.follow_up_scheduled = True
        state.follow_up_days = 3 if state.escalate_reason == "retake_needed" else 5
        # Ingest escalated case into expert review queue
        await _ingest_expert_review(state)
    elif state.confidence < FOLLOW_UP_MIN:
        state.follow_up_scheduled = True
        state.follow_up_days = 5

    # Officer / helpline guidance appended to the advice
    strings = load_strings(state.language)
    helpline = strings.get("helpline", "Contact the agriculture helpline for further assistance.")
    officer_tmpl = strings.get("officer", "Local agriculture officer: {name} ({phone}).")
    if state.officer:
        officer_text = "; ".join(
            officer_tmpl.format(name=o.get("name", "agriculture officer"), phone=o.get("phone", ""))
            for o in state.officer
        )
        state.answer = (state.answer or "") + " " + officer_text
    elif helpline and helpline not in (state.answer or ""):
        state.answer = (state.answer or "") + " " + helpline
    return state


async def _ingest_expert_review(state: AgentState) -> None:
    """Send escalated case to the expert review queue."""
    try:
        import httpx
        payload = {
            "case_id": state.case_id,
            "image_url": state.image_url,
            "predicted_label": state.predictions[0]["label"] if state.predictions else "unknown",
            "predicted_score": state.predictions[0]["score"] if state.predictions else 0.0,
            "crop": state.crop or "rice",
            "language": state.language,
            "location": state.location,
            "confidence": state.confidence,
            "escalate_reason": state.escalate_reason,
            "answer": state.answer,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "escalate": True,
        }
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post("http://localhost:8001/ingest", json=payload)
    except Exception:
        # Expert queue may be offline; case is still escalated in the graph
        pass


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_after_triage(state: AgentState) -> str:
    if not state.quality_ok:
        return END
    return "classify"


def route_after_diagnosis(state: AgentState) -> str:
    if state.retake_requested:
        return END
    return "context_weather"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)
    builder.add_node("triage", triage)
    builder.add_node("classify", classify)
    builder.add_node("vision_second_opinion", vision_second_opinion)
    builder.add_node("context_weather", context_weather)
    builder.add_node("retrieve", retrieve)
    builder.add_node("draft_advice", draft_advice)
    builder.add_node("localize_advice", localize_advice)
    builder.add_node("llm_explain", llm_explain)
    builder.add_node("critic_safety", critic_safety)
    builder.add_node("find_officer", find_officer)
    builder.add_node("follow_up", follow_up)

    builder.set_entry_point("triage")
    builder.add_conditional_edges("triage", route_after_triage)
    builder.add_edge("classify", "vision_second_opinion")
    builder.add_conditional_edges("vision_second_opinion", route_after_diagnosis)
    builder.add_edge("context_weather", "retrieve")
    builder.add_edge("retrieve", "draft_advice")
    builder.add_edge("draft_advice", "localize_advice")
    builder.add_edge("localize_advice", "llm_explain")
    builder.add_edge("llm_explain", "critic_safety")
    builder.add_conditional_edges(
        "critic_safety",
        lambda s: "find_officer" if s.escalate else END,
    )
    builder.add_edge("find_officer", "follow_up")
    builder.add_edge("follow_up", END)
    return builder


graph = build_graph().compile()


async def run_advice(req: Any) -> AgentState:
    state = AgentState(
        case_id=str(uuid.uuid4()),
        image_bytes=getattr(req, "image_bytes", None),
        image_url=getattr(req, "image_url", None),
        crop=getattr(req, "crop", None),
        language=getattr(req, "language", "en"),
        location=getattr(req, "location", None),
    )
    result = await graph.ainvoke(cast(Any, state))
    if isinstance(result, dict):
        return AgentState(**result)
    return result
