from __future__ import annotations

import contextlib
import io
import json
from unittest.mock import patch

import httpx
import pytest
from PIL import Image

from agents import llm as llm_mod


@contextlib.contextmanager
def _llm_env(base="http://llm.example", key="test-key", model="nemotron-3.5-super-vl"):
    """Set the LLM env config for the duration of the block, then restore it."""
    saved = (llm_mod.BASE_URL, llm_mod.API_KEY, llm_mod.MODEL_NAME)
    llm_mod.BASE_URL = base
    llm_mod.API_KEY = key
    llm_mod.MODEL_NAME = model
    try:
        yield
    finally:
        llm_mod.BASE_URL, llm_mod.API_KEY, llm_mod.MODEL_NAME = saved


@contextlib.contextmanager
def _no_llm_env():
    """No LLM configured -> explain() must use the plain template."""
    saved = (llm_mod.BASE_URL, llm_mod.API_KEY, llm_mod.MODEL_NAME)
    llm_mod.BASE_URL = ""
    llm_mod.API_KEY = ""
    llm_mod.MODEL_NAME = ""
    try:
        yield
    finally:
        llm_mod.BASE_URL, llm_mod.API_KEY, llm_mod.MODEL_NAME = saved


def _chunks():
    return [
        {
            "text": "remove affected leaves and apply approved fungicide.",
            "source": "crop_knowledge_base",
            "dose": "as per label",
        },
        {"text": "avoid excess nitrogen. rotate crops.", "source": "crop_knowledge_base"},
    ]


def _plain_template(language="kn"):
    with _no_llm_env():
        return llm_mod.explain("leaf_blight", _chunks(), language)


# ---------- 1. empty chunks: plain template, model never called ----------

def test_explain_empty_chunks_returns_plain_template_without_model_call():
    with patch.object(llm_mod, "_llm_chat") as mock_chat:
        with _llm_env():
            out = llm_mod.explain("leaf_blight", [], "kn")
    mock_chat.assert_not_called()
    assert out is not None
    assert "ಕೃಷಿ ಅಧಿಕಾರಿ" in out


# ---------- 2. success: model text + appended safety line and helpline ----------

@patch.object(llm_mod, "_llm_chat")
def test_explain_success_appends_safety_and_helpline(mock_chat):
    model_text = "ನಿಮ್ಮ ಎಲೆಯಲ್ಲಿ ಎಲೆ ಕಲುಷ ಕಾಣಿಸುತ್ತಿದೆ. ಬಾಧಿತ ಎಲೆಗಳನ್ನು ತೆಗೆಯಿರಿ."
    mock_chat.return_value = model_text
    with _llm_env():
        out = llm_mod.explain("leaf_blight", _chunks(), "kn")
    assert model_text in out
    assert "ಕೃಷಿ ಅಧಿಕಾರಿ" in out
    assert "ಹೆಲ್ಪ್‌ಲೈನ್" in out
    mock_chat.assert_called_once()


# ---------- 3. API error: falls back to the plain template ----------

@patch.object(llm_mod, "_llm_chat")
def test_explain_api_error_falls_back(mock_chat):
    mock_chat.side_effect = httpx.HTTPStatusError(
        "boom",
        request=httpx.Request("POST", "http://llm.example"),
        response=httpx.Response(502),
    )
    with _llm_env():
        out = llm_mod.explain("leaf_blight", _chunks(), "kn")
    assert out == _plain_template()
    assert "ಕೃಷಿ ಅಧಿಕಾರಿ" in out


# ---------- 4. timeout: falls back to the plain template ----------

@patch.object(llm_mod, "_llm_chat")
def test_explain_timeout_falls_back(mock_chat):
    mock_chat.side_effect = httpx.ConnectTimeout(
        "timed out", request=httpx.Request("POST", "http://llm.example")
    )
    with _llm_env():
        out = llm_mod.explain("leaf_blight", _chunks(), "kn")
    assert out == _plain_template()
    assert "ಕೃಷಿ ಅಧಿಕಾರಿ" in out


# ---------- 5. English leaking into kn: rejected ----------

@patch.object(llm_mod, "_llm_chat")
def test_explain_english_leaking_into_kn_falls_back(mock_chat):
    mock_chat.return_value = "ನಿಮ್ಮ ಎಲೆಯಲ್ಲಿ leaf blight ಕಾಣಿಸುತ್ತಿದೆ."
    with _llm_env():
        out = llm_mod.explain("leaf_blight", _chunks(), "kn")
    assert "leaf blight" not in out.lower()
    assert not any(ch.isascii() and ch.isalpha() for ch in out)
    assert out == _plain_template()


# ---------- 6. invented dose: rejected ----------

@patch.object(llm_mod, "_llm_chat")
def test_explain_invented_dose_rejected(mock_chat):
    mock_chat.return_value = "ನಿಮ್ಮ ಎಲೆಯಲ್ಲಿ ಎಲೆ ಕಲುಷ ಕಾಣಿಸುತ್ತಿದೆ. 5 ml ರೇಸ್ಸು ಹಾಕಿ."
    with _llm_env():
        out = llm_mod.explain("leaf_blight", _chunks(), "kn")
    assert "5 ml" not in out
    assert out == _plain_template()


# ---------- 7. chunk-grounded dose: allowed ----------

@patch.object(llm_mod, "_llm_chat")
def test_explain_chunk_grounded_dose_allowed(mock_chat):
    chunks = [
        {
            "text": "Apply 10 ml fungicide per litre.",
            "source": "crop_knowledge_base",
            "dose": "as per label",
        }
    ]
    model_text = "ನಿಮ್ಮ ಎಲೆಯಲ್ಲಿ ರೋಗ ಕಾಣಿಸುತ್ತಿದೆ. 10 ml ಹಾಕಿ."
    mock_chat.return_value = model_text
    with _llm_env():
        out = llm_mod.explain("leaf_blight", chunks, "kn")
    assert "10 ml" in out
    assert model_text in out
    assert "ಕೃಷಿ ಅಧಿಕಾರಿ" in out


# ---------- no LLM configured ----------

def test_no_llm_configured_plain_template():
    with _no_llm_env():
        out = llm_mod.explain("leaf_blight", _chunks(), "en")
    assert out is not None
    assert "Confirm with your local agriculture officer" in out


# ---------- Vision second opinion ----------


def _jpeg_with_exif(size=(1200, 900)):
    img = Image.new("RGB", size, color=(10, 200, 30))
    exif = img.getexif()
    exif[0x010F] = "Canon"  # Make
    exif[0x0110] = "ModelX"  # Model
    exif[0x8825] = {1: "N", 2: (12.34, 56.78)}  # GPSInfo
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


def test_strip_exif_and_resize_removes_metadata_and_caps_512px():
    raw = _jpeg_with_exif()
    assert dict(Image.open(io.BytesIO(raw)).getexif())  # original has EXIF
    out = llm_mod._strip_exif_and_resize(raw)
    img = Image.open(io.BytesIO(out))
    assert max(img.size) <= 512
    assert dict(img.getexif()) == {}  # EXIF/GPS stripped


@patch.object(llm_mod, "_llm_chat")
def test_vision_opinion_sends_only_image_and_returns_json(mock_chat):
    mock_chat.return_value = json.dumps(
        {"symptoms": "tan lesions", "diseases": ["leaf_blight", "healthy"]}
    )
    with _llm_env():
        opinion = llm_mod.vision_opinion(_jpeg_with_exif(), ["healthy", "leaf_blight"], "en")
    assert opinion == {"symptoms": "tan lesions", "diseases": ["leaf_blight", "healthy"]}
    messages = mock_chat.call_args[0][0]
    system = messages[0]["content"]
    assert "healthy, leaf_blight" in system
    assert "JSON" in system
    parts = messages[1]["content"]
    image_part = [p for p in parts if p.get("type") == "image_url"][0]
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")


@patch.object(llm_mod, "_llm_chat")
def test_vision_opinion_returns_none_on_api_error(mock_chat):
    mock_chat.side_effect = httpx.ConnectTimeout(
        "timed out", request=httpx.Request("POST", "http://llm.example")
    )
    with _llm_env():
        assert llm_mod.vision_opinion(_jpeg_with_exif(), ["healthy"], "en") is None


def test_vision_opinion_returns_none_when_not_configured():
    with _no_llm_env():
        assert llm_mod.vision_opinion(_jpeg_with_exif(), ["healthy"], "en") is None
