from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

from agents import llm as llm_mod


def _chunks():
    return [
        {"text": "remove affected leaves and apply approved fungicide.", "source": "crop_knowledge_base", "dose": "as per label"},
        {"text": "avoid excess nitrogen. rotate crops.", "source": "crop_knowledge_base"},
    ]


def test_explain_empty_chunks_returns_plain_template_without_model_call():
    called = []
    with patch.object(llm_mod, "_llm_chat", side_effect=lambda *a, **k: called.append(True) or "") as mock_chat:
        with patch.dict(llm_mod.os.environ, {"CA_LLM_BASE_URL": "http://x", "CA_LLM_API_KEY": "k", "CA_LLM_MODEL_NAME": "m"}):
            llm_mod.BASE_URL = llm_mod.os.getenv("CA_LLM_BASE_URL")
            llm_mod.API_KEY = llm_mod.os.getenv("CA_LLM_API_KEY")
            llm_mod.MODEL_NAME = llm_mod.os.getenv("CA_LLM_MODEL_NAME")
            out = llm_mod.explain("leaf_blight", [], "kn")
    assert not called
    assert out is not None
    assert "ಕೃಷಿ ಅಧಿಕಾರಿ" in out or "ಹೆಲ್ಪ್‌ಲೈನ್" in out


@patch.object(llm_mod, "_llm_chat")
def test_explain_success_appends_safety_and_helpline(mock_chat):
    mock_chat.return_value = "ನಿಮ್ಮ ಎಲೆಯಲ್ಲಿ ಎಲೆ ಕಲುಷ ಕಾಣಿಸುತ್ತಿದೆ. ಬಾಧಿತ ಎಲೆಗಳನ್ನು ತೆಗೆಯಿರಿ."
    with patch.dict(llm_mod.os.environ, {"CA_LLM_BASE_URL": "http://x", "CA_LLM_API_KEY": "k", "CA_LLM_MODEL_NAME": "m"}):
        llm_mod.BASE_URL = llm_mod.os.getenv("CA_LLM_BASE_URL")
        llm_mod.API_KEY = llm_mod.os.getenv("CA_LLM_API_KEY")
        llm_mod.MODEL_NAME = llm_mod.os.getenv("CA_LLM_MODEL_NAME")
        out = llm_mod.explain("leaf_blight", _chunks(), "kn")
    assert "ಕೃಷಿ ಅಧಿಕಾರಿ" in out
    assert "ಹೆಲ್ಪ್‌ಲೈನ್" in out
    assert mock_chat.called


@patch.object(llm_mod, "_llm_chat")
def test_explain_api_error_falls_back(mock_chat):
    mock_chat.side_effect = httpx.HTTPStatusError("boom", request=httpx.Request("POST", "http://x"), response=httpx.Response(502))
    with patch.dict(llm_mod.os.environ, {"CA_LLM_BASE_URL": "http://x", "CA_LLM_API_KEY": "k", "CA_LLM_MODEL_NAME": "m"}):
        llm_mod.BASE_URL = llm_mod.os.getenv("CA_LLM_BASE_URL")
        llm_mod.API_KEY = llm_mod.os.getenv("CA_LLM_API_KEY")
        llm_mod.MODEL_NAME = llm_mod.os.getenv("CA_LLM_MODEL_NAME")
        out = llm_mod.explain("leaf_blight", _chunks(), "kn")
    assert out is not None
    assert "ಕೃಷಿ ಅಧಿಕಾರಿ" in out


@patch.object(llm_mod, "_llm_chat")
def test_explain_timeout_falls_back(mock_chat):
    mock_chat.side_effect = httpx.ConnectTimeout("timed out", request=httpx.Request("POST", "http://x"))
    with patch.dict(llm_mod.os.environ, {"CA_LLM_BASE_URL": "http://x", "CA_LLM_API_KEY": "k", "CA_LLM_MODEL_NAME": "m"}):
        llm_mod.BASE_URL = llm_mod.os.getenv("CA_LLM_BASE_URL")
        llm_mod.API_KEY = llm_mod.os.getenv("CA_LLM_API_KEY")
        llm_mod.MODEL_NAME = llm_mod.os.getenv("CA_LLM_MODEL_NAME")
        out = llm_mod.explain("leaf_blight", _chunks(), "kn")
    assert out is not None
    assert "ಕೃಷಿ ಅಧಿಕಾರಿ" in out


@patch.object(llm_mod, "_llm_chat")
def test_explain_english_leaking_into_kn_falls_back(mock_chat):
    mock_chat.return_value = "ನಿಮ್ಮ ಎಲೆಯಲ್ಲಿ leaf blight ಕಾಣಿಸುತ್ತಿದೆ."
    with patch.dict(llm_mod.os.environ, {"CA_LLM_BASE_URL": "http://x", "CA_LLM_API_KEY": "k", "CA_LLM_MODEL_NAME": "m"}):
        llm_mod.BASE_URL = llm_mod.os.getenv("CA_LLM_BASE_URL")
        llm_mod.API_KEY = llm_mod.os.getenv("CA_LLM_API_KEY")
        llm_mod.MODEL_NAME = llm_mod.os.getenv("CA_LLM_MODEL_NAME")
        out = llm_mod.explain("leaf_blight", _chunks(), "kn")
    assert "leaf blight" not in out.lower()
    assert "ಕೃಷಿ ಅಧಿಕಾರಿ" in out


@patch.object(llm_mod, "_llm_chat")
def test_explain_invented_dose_rejected(mock_chat):
    mock_chat.return_value = "ನಿಮ್ಮ ಎಲೆಯಲ್ಲಿ ಎಲೆ ಕಲುಷ ಕಾಣಿಸುತ್ತಿದೆ. 5 ml ರೇಸ್ಸು ಹಾಕಿ."
    with patch.dict(llm_mod.os.environ, {"CA_LLM_BASE_URL": "http://x", "CA_LLM_API_KEY": "k", "CA_LLM_MODEL_NAME": "m"}):
        llm_mod.BASE_URL = llm_mod.os.getenv("CA_LLM_BASE_URL")
        llm_mod.API_KEY = llm_mod.os.getenv("CA_LLM_API_KEY")
        llm_mod.MODEL_NAME = llm_mod.os.getenv("CA_LLM_MODEL_NAME")
        out = llm_mod.explain("leaf_blight", _chunks(), "kn")
    assert "5 ml" not in out
    assert "ಕೃಷಿ ಅಧಿಕಾರಿ" in out


@patch.object(llm_mod, "_llm_chat")
def test_explain_chunk_grounded_dose_allowed(mock_chat):
    chunks = [{"text": "Apply 10 ml fungicide per litre.", "source": "crop_knowledge_base", "dose": "as per label"}]
    mock_chat.return_value = "ನಿಮ್ಮ ಎಲೆಯಲ್ಲಿ ರೋಗ ಕಾಣಿಸುತ್ತಿದೆ. 10 ml ಹಾಕಿ."
    with patch.dict(llm_mod.os.environ, {"CA_LLM_BASE_URL": "http://x", "CA_LLM_API_KEY": "k", "CA_LLM_MODEL_NAME": "m"}):
        llm_mod.BASE_URL = llm_mod.os.getenv("CA_LLM_BASE_URL")
        llm_mod.API_KEY = llm_mod.os.getenv("CA_LLM_API_KEY")
        llm_mod.MODEL_NAME = llm_mod.os.getenv("CA_LLM_MODEL_NAME")
        out = llm_mod.explain("leaf_blight", chunks, "kn")
    assert "10 ml" in out
    assert "ಕೃಷಿ ಅಧಿಕಾರಿ" in out


def test_no_llm_configured_plain_template():
    with patch.dict(llm_mod.os.environ, {}, clear=True):
        llm_mod.BASE_URL = ""
        llm_mod.API_KEY = ""
        llm_mod.MODEL_NAME = ""
        out = llm_mod.explain("leaf_blight", _chunks(), "en")
    assert out is not None
    assert "Confirm with your local agriculture officer" in out
