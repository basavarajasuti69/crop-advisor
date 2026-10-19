"""OpenAI-compatible LLM client for farmer-facing explanations (Nemotron 3.5 Super VL).

Reads CA_LLM_BASE_URL, CA_LLM_API_KEY and CA_LLM_MODEL_NAME from environment.
Never sends the leaf image or the farmer's location to the model.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import httpx

STRINGS_DIR = Path(__file__).resolve().parent.parent / "strings"

BASE_URL = os.getenv("CA_LLM_BASE_URL", "").rstrip("/")
API_KEY = os.getenv("CA_LLM_API_KEY", "")
MODEL_NAME = os.getenv("CA_LLM_MODEL_NAME", "")
REQUEST_TIMEOUT = 10.0

DOSE_UNIT_RE = re.compile(r"\d+\s*(ml|g|kg|l|per\s+acre|per\s+litre)\b", re.IGNORECASE)
HEADING_RE = re.compile(r"^\s*#+\s", re.MULTILINE)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")

# Common English words that are never treated as invented chemicals.
COMMON_ENGLISH = {
    "The","This","That","These","Those","Remove","Apply","Avoid","Continue","Confirm",
    "Wear","Do","Not","Spray","Before","After","And","Or","But","If","Then","Your",
    "Crop","Leaf","Shows","Is","Are","Was","Were","Be","Been","Being","Have","Has","Had",
    "Does","Did","Will","Would","Could","Should","May","Might","Can","Cannot",
    "No","Yes","Please","For","To","In","On","At","By","With","From","Of","As","An","A",
    "It","Its","We","You","Your","He","She","They","Them","Their","There","Here",
    "When","Where","Why","How","What","Which","Who","Whom","Whose","Into","During",
    "Until","While","About","Against","Between","Under","Over","Again","Further","Then",
    "Once","All","Any","Both","Each","Few","More","Most","Other","Some","Such","Nor",
    "Only","Own","Same","So","Than","Too","Very","Just","Now","Healthy","Leaf","Blight",
    "Rust","Curl","Virus","Disease","Rice","Wheat","Cotton","Maize","Tomato","Plants",
    "Leaves","Stems","Fruit","Flowers","Irrigation","Fertilization","Standard","Nitrogen",
    "Fungicide","Label","Directions","Officer","Spraying","Treatment","Lesions","Spots",
    "Yellow","Orange","Brown","Tan","White","Green","Curling","Thickening","Vein",
    "Swelling","Infected","Plants","Cure","Control","Whiteflies","Rotate","Crops",
    "Margins","Approved","Needed","Symptoms","Visible","Possible","Steps","Numbered",
}

def load_strings(language: str) -> dict[str, str]:
    path = STRINGS_DIR / f"{language}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _plain_template(disease_label: str, chunks: list[Any], language: str) -> str:
    strings = load_strings(language)
    crop = strings.get("neutral_crop", "your crop")
    intro = strings.get("intro", "Your {crop} leaf shows {label}.").format(crop=crop, label=disease_label)
    confirm = strings.get("confirm", "")
    guidance = ""
    if language == "en" and chunks:
        first = chunks[0]
        first_text = first.get("text", str(first)) if isinstance(first, dict) else str(first)
        guidance = first_text[:200]
    elif language == "en":
        guidance = strings.get("no_guidance", "")
    return " ".join(p for p in (intro, guidance, confirm) if p).strip()


def _script_fraction(text: str, ranges: list[tuple[int, int]]) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return True
    in_script = [ch for ch in letters if any(lo <= ord(ch) <= hi for lo, hi in ranges)]
    return len(in_script) / len(letters) >= 0.90


def _validate_text(text: str, language: str) -> bool:
    """Existing validators: no markdown headings, script check."""
    if HEADING_RE.search(text):
        return False
    if language == "kn":
        from agents.graph import _validate_kannada  # noqa: PLC0415
        return _validate_kannada(text)
    if language == "hi":
        return _script_fraction(text, [(0x0900, 0x097F)])
    if language == "en":
        return _script_fraction(text, [(0x0041, 0x005A), (0x0061, 0x007A)])
    return True


def _extract_known_tokens(chunks: list[Any]) -> set[str]:
    tokens: set[str] = set()
    for c in chunks:
        s = c.get("text", str(c)) if isinstance(c, dict) else str(c)
        for w in WORD_RE.findall(s):
            tokens.add(w.lower())
    return tokens


def _reject_dose_chemical(text: str, chunks: list[Any], language: str = "en") -> tuple[bool, str]:
    """Reject a number followed by a dose unit not in chunks, or a chemical name
    not in chunks. Also reject Latin letters in kn/hi output (English leak)."""
    chunk_text = " ".join(c.get("text", str(c)) if isinstance(c, dict) else str(c) for c in chunks)
    known = _extract_known_tokens(chunks)
    # Dose units: only reject if not grounded in a retrieved chunk.
    for m in DOSE_UNIT_RE.finditer(text):
        if m.group(0).lower() not in chunk_text.lower():
            return True, "invented dose unit"
    # Chemical names: uppercase word not in chunks.
    for w in WORD_RE.findall(text):
        if len(w) > 2 and w[0].isupper() and w.lower() not in known and w not in COMMON_ENGLISH:
            return True, f"invented chemical {w}"
    # English leak: Latin letters in a kn/hi answer (excluding dose units already checked).
    if language in ("kn", "hi"):
        cleaned = DOSE_UNIT_RE.sub("", text)
        for ch in cleaned:
            if ch.isascii() and ch.isalpha():
                return True, "english leak"
    return False, ""


def _build_prompt(disease_label: str, chunks: list[Any], language: str) -> list[dict[str, str]]:
    chunk_text = "\n".join(
        c.get("text", str(c)) if isinstance(c, dict) else str(c) for c in chunks
    )
    system = (
        "You are a careful agricultural advisor. Write a short farmer-friendly explanation "
        "in the target language. Use ONLY the provided chunks. Do not invent doses, chemical "
        "names, or treatments unless they appear in the chunks. Short sentences, at most 5 "
        "numbered steps. Do not add a safety line or helpline; those are added separately. "
        "Target language: {language}.\n\nCHUNKS:\n{chunks}\n\nDISEASE: {label}"
    ).format(language=language, chunks=chunk_text, label=disease_label)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "Write the explanation now."},
    ]


def _llm_chat(messages: list[dict[str, str]]) -> str:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 400,
    }
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        resp = client.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
    return data["choices"][0]["message"]["content"]


def explain(disease_label: str, chunks: list[Any], language: str) -> str:
    """Return a farmer-facing explanation in the target language.

    Never sends the image or the farmer's location. Falls back to a plain
    template if the LLM cannot be called, times out, or fails validation.
    """
    chunks = chunks or []
    strings = load_strings(language)
    safety = strings.get("confirm", "")
    helpline = strings.get("helpline", "")

    def finalize(core: str) -> str:
        parts = [p for p in (core, safety, helpline) if p]
        return " ".join(parts).strip()

    # No chunks to ground the explanation → plain template, no model call.
    if not chunks:
        return finalize(_plain_template(disease_label, chunks, language))

    # No LLM configured → plain template immediately (no timeout).
    if not BASE_URL or not API_KEY or not MODEL_NAME:
        return finalize(_plain_template(disease_label, chunks, language))

    try:
        messages = _build_prompt(disease_label, chunks, language)
        content = _llm_chat(messages)
    except Exception:
        return finalize(_plain_template(disease_label, chunks, language))

    if not content or not content.strip():
        return finalize(_plain_template(disease_label, chunks, language))

    content = content.strip()

    # Reject invented doses and chemicals not in the retrieved chunks
    # (checked on the model output, not on the trusted safety line).
    rejected, _reason = _reject_dose_chemical(content, chunks, language)
    if rejected:
        return finalize(_plain_template(disease_label, chunks, language))

    # Append the fixed safety line + helpline (the model never writes these).
    final = finalize(content)

    # Run existing validators on the final text.
    if not _validate_text(final, language):
        return finalize(_plain_template(disease_label, chunks, language))

    return final
