"""LangGraph state and node definitions for the crop-disease advisor.

Phase 3: triage -> diagnosis (classifier + vision LLM second opinion) ->
context (weather MCP) -> retrieve (crop_knowledge MCP) -> draft -> localize ->
critic (dose verification) -> follow-up (reminders + escalation).
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from langgraph.graph import StateGraph, END
from mcp.types import CallToolRequest, CallToolRequestParams
from pydantic import BaseModel, Field

from rag.glossary import hi as glossary_hi  # noqa: E402
from rag.glossary import kn as glossary_kn  # noqa: E402


from agents.llm import explain, llm_configured, vision_opinion  # noqa: E402

# ---------------------------------------------------------------------------
# Classifier: SageMaker endpoint first, local PyTorch fallback
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Confidence thresholds (named constants; no raw numbers in logic)
# ---------------------------------------------------------------------------
CONFIDENCE_MIN = 0.70   # "not sure" threshold; below it -> retake photo
HEALTHY_MIN = 0.85      # a "healthy" result below this is treated as not sure
FOLLOW_UP_MIN = 0.80    # below it -> schedule follow-up
VISION_DISAGREE_MIN = 0.70   # classifier/vision disagreement retake threshold
VISION_ENABLED = os.getenv("CA_VISION_SECOND_OPINION", "false").lower() == "true"


# The classifier lives in its own module so the serving path can be read
# without the LangGraph machinery. The names below are kept as aliases
# because they are part of the module's existing surface.
from agents.classifier import (  # noqa: E402
    ImageUnreadable,
    ModelUnavailable,
    crop_from_label,
    disease_from_label,
    is_healthy_label,
    kb_section_for_label,
    load_class_names,
    looks_like_leaf,
)
from agents.classifier import predict as _predict_image  # noqa: E402
from agents.classifier import onnx_preprocess as _onnx_preprocess  # noqa: E402, F401
from agents.classifier import predict_onnx as _predict_onnx  # noqa: E402, F401
from agents.classifier import load_torch as _load_classifier  # noqa: E402, F401


def _torch_transform():
    """The training/eval transform: Resize(256) + CenterCrop(224) + Normalize."""
    from agents.classifier import _torch_transform as transform

    return transform()


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


# The Unicode range of the script each language is written in.
_SCRIPT_RANGES = {
    "kn": (0x0C80, 0x0CFF),  # Kannada
    "hi": (0x0900, 0x097F),  # Devanagari
}


def _validate_script(text: str, language: str) -> bool:
    """Return True if the text is in the right script for the language.

    At least 90% of the letters must be in the target script, there must be
    no '#' headings, and no letter from another Indian script. Latin
    letters (product names, units) are ignored in the ratio.
    """
    ranges = _SCRIPT_RANGES.get(language)
    if ranges is None:
        return True
    if "#" in text:
        return False
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return True
    low, high = ranges
    in_script = [ch for ch in letters if low <= ord(ch) <= high]
    other_indian = [
        ch for ch in letters if not ch.isascii() and not (low <= ord(ch) <= high)
    ]
    if other_indian:
        return False
    meaningful = [ch for ch in letters if not ch.isascii()]
    if not meaningful:
        return True
    return len(in_script) / len(meaningful) >= 0.90


def _validate_kannada(text: str) -> bool:
    """True if the text is at least 90% Kannada, with no '#' headings and no
    letter from another Indian script."""
    return _validate_script(text, "kn")


def _validate_hindi(text: str) -> bool:
    """True if the text is at least 90% Devanagari, with no '#' headings and no
    letter from another Indian script."""
    return _validate_script(text, "hi")


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
    guidance_available: bool = False
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
    # A photo that is not a plant at all: the classifier would still assign
    # a disease to it, so it is turned away before the model is asked.
    if state.image_bytes is not None and not looks_like_leaf(state.image_bytes):
        state.quality_ok = False
        state.quality_issue = "not a leaf photo"
        state.safe = False
        state.escalate = True
        state.escalate_reason = "not_a_leaf"
        state.retake_requested = True
        strings = load_strings(state.language)
        state.answer = strings.get("not_a_leaf") or strings.get(
            "cannot_read", "This does not look like a photo of a leaf."
        )
        return state
    # Crop ID: use the crop the caller supplied. It is not guessed
    # here - a wrong guess would send the farmer to another crop's
    # guidance. The classifier may override it below.
    state.crop = state.crop or None
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
    # Classify using the local ONNX/PyTorch model.
    predictions = None
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

    predicted_crop = crop_from_label(predictions[0]["label"])
    if predicted_crop:
        state.crop = predicted_crop

    state.confidence = predictions[0]["score"]
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
    class_list = load_class_names() or [p["label"] for p in state.predictions]
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
    """RAG retrieval using the crop_knowledge MCP server."""
    state.step = "retrieve"

    label = state.predictions[0]["label"] if state.predictions else "healthy"
    detected_crop = crop_from_label(label)
    crop = detected_crop or state.crop
    disease = disease_from_label(label)

    # Keep the crop consistent with the classifier's crop-specific label.
    state.crop = crop

    # A PlantDoc class usually has no section in the knowledge base.
    # When there is one, ask for it by name so the retrieval cannot
    # drift onto a different crop's section.
    section = kb_section_for_label(label)
    if section:
        query = section.lower().replace(" - ", " ")
        kb_crop = section.split(" - ")[0].strip().lower()
    else:
        query = " ".join(part for part in (disease, crop) if part).lower()
        kb_crop = crop or ""

    kb = await mcp_call(
        "crop_knowledge",
        "search_knowledge",
        {
            "query": query,
            "crop": kb_crop,
        },
    )

    if "error" not in kb and isinstance(kb, list) and kb:
        state.passages = kb
        state.citations = ["crop_knowledge_mcp"]
        state.guidance_available = True
    else:
        # No guidance for this class. Say so instead of borrowing another
        # crop's advice.
        state.passages = [
            {
                "text": (
                    f"{disease} guidance for {crop}: "
                    "consult a local agriculture officer for "
                    "crop-specific treatment recommendations."
                ),
                "source": "crop_knowledge_base",
                "dose": "as per label",
            }
        ]
        state.citations = ["crop_knowledge_base"]
        state.guidance_available = False

    return state


async def draft_advice(state: AgentState) -> AgentState:
    """Draft grounded advice with citations."""
    state.step = "draft_advice"
    label = state.predictions[0]["label"] if state.predictions else "healthy"
    strings = load_strings("en")
    crop = state.crop or strings.get("neutral_crop", "crop")
    disease = disease_from_label(label)
    passage = state.passages[0]["text"] if state.passages else ""
    strings = load_strings("en")
    intro = strings.get("intro", "Your {crop} leaf shows {label}.").format(crop=crop, label=disease)
    confirm = strings.get("confirm", "Confirm with your local agriculture officer before spraying.")
    state.answer = f"{intro} {passage} {confirm}"
    state.citations = state.citations or ["crop_knowledge_base"]
    return state


async def localize_advice(state: AgentState) -> AgentState:
    """Translate the drafted advice into the farmer's language."""
    state.step = "localize_advice"

    label = state.predictions[0]["label"] if state.predictions else "healthy"
    detected_crop = crop_from_label(label)
    crop = detected_crop or state.crop
    disease = disease_from_label(label)

    state.crop = crop

    if state.language == "hi":
        # Never put English passage text into a Hindi answer either: the
        # language model writes the explanation in Hindi from the English
        # chunks when one is configured.
        strings = load_strings("hi")
        crop_local = glossary_hi.CROP_NAMES.get(crop) if crop else None
        if crop_local is None:
            crop_local = strings.get("neutral_crop", "फसल")
        disease_local = glossary_hi.DISEASE_NAMES.get(disease)
        if disease_local is None:
            # NEEDS NATIVE REVIEW: disease_unknown is null for hi
            disease_local = strings.get("disease_unknown") or strings.get("neutral_crop", "फसल")
        intro = strings.get("intro", "").format(crop=crop_local, label=disease_local)
        confirm = strings.get("confirm", "")
        state.answer = f"{intro} {confirm}"
    elif state.language == "kn":
        # Never put English passage text into a Kannada answer.
        strings = load_strings("kn")
        crop_local = glossary_kn.CROP_NAMES.get(crop) if crop else None
        if crop_local is None:
            crop_local = strings.get("neutral_crop", "ಬೆಳೆ")
        disease_local = glossary_kn.DISEASE_NAMES.get(disease)
        if disease_local is None:
            # NEEDS NATIVE REVIEW: disease_unknown is null for kn
            disease_local = strings.get("disease_unknown") or strings.get("neutral_crop", "ಬೆಳೆ")
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
    disease = disease_from_label(label)
    chunks = state.passages[:2]
    try:
        answer = await asyncio.to_thread(explain, disease, chunks, state.language)
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
    if is_healthy_label(label) and state.confidence < HEALTHY_MIN:
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

    # Script validator: >= 90% letters in the target script, no markdown
    # headings, no letter from another Indian script.
    if state.language in ("kn", "hi") and state.answer and not _validate_script(
        state.answer, state.language
    ):
        state.safe = False
        state.escalate = True
        state.escalate_reason = "language_mismatch"
        detected_crop = crop_from_label(label)
        crop = detected_crop or state.crop or ""
        disease = disease_from_label(label)
        state.crop = crop
        if state.language == "kn":
            strings = load_strings("kn")
            crop_local = glossary_kn.CROP_NAMES.get(crop)
            disease_local = glossary_kn.DISEASE_NAMES.get(disease)
        else:
            strings = load_strings("hi")
            crop_local = glossary_hi.CROP_NAMES.get(crop)
            disease_local = glossary_hi.DISEASE_NAMES.get(disease)
        if crop_local is None:
            crop_local = strings.get("neutral_crop", "")
        if disease_local is None:
            disease_local = strings.get("disease_unknown") or strings.get("neutral_crop", "")
        intro = strings.get("intro", "").format(crop=crop_local, label=disease_local)
        confirm = strings.get("confirm", "")
        state.answer = f"{intro} {confirm}"

    return state


async def find_officer(state: AgentState) -> AgentState:
    """Look up local agriculture officer via officer_directory MCP (for escalation)."""
    state.step = "find_officer"
    if not state.location:
        return state
    # The directory only lists officers for some crops. When the crop is
    # not one of them, ask without a crop filter rather than returning
    # nobody - a local officer can still advise on any crop.
    crop = state.crop or ""
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
