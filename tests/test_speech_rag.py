from __future__ import annotations

import asyncio
import base64
import json
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, ".")

from mcp_servers.crop_knowledge import server as ck_server  # noqa: E402
from speech.stt import STTAdapter  # noqa: E402
from speech.tts_adapter import TTSAdapter  # noqa: E402


def _search(query: str, crop: str) -> list[dict]:
    result = asyncio.run(ck_server._search_knowledge({"query": query, "crop": crop}))
    return json.loads(result[0].text)


# ---------- RAG retrieval ----------

@pytest.mark.parametrize(
    "query,crop,expected",
    [
        ("healthy rice", "rice", "Rice - Healthy"),
        ("leaf_blight rice", "rice", "Rice - Leaf Blight"),
        ("healthy wheat", "wheat", "Wheat - Healthy"),
        ("leaf_rust wheat", "wheat", "Wheat - Rust"),
        ("healthy cotton", "cotton", "Cotton - Healthy"),
        ("leaf_curl_virus cotton", "cotton", "Cotton - Leaf Curl Virus"),
        ("healthy maize", "maize", "Maize - Healthy"),
        ("leaf_blight maize", "maize", "Maize - Leaf Blight"),
        ("healthy tomato", "tomato", "Tomato - Healthy"),
        ("blight tomato", "tomato", "Tomato - Blight"),
    ],
)
def test_search_finds_the_chunk_for_every_disease(query, crop, expected):
    """The graph asks for "<disease> <crop>" but the KB is headed
    "<Crop> - <Disease>", so the query has to match token by token."""
    chunks = _search(query, crop)
    assert chunks, f"no chunk for {query!r}"
    assert chunks[0]["text"].startswith(expected)


def test_search_returns_nothing_for_an_unknown_query():
    assert _search("dragon fruit fungus", "rice") == []


def test_search_does_not_match_a_different_crop():
    """A rice query must not return the maize chunk."""
    chunks = _search("leaf_blight rice", "rice")
    assert all("maize" not in chunk["text"].lower() for chunk in chunks)


# ---------- Sarvam text to speech ----------

def _sarvam_adapter() -> TTSAdapter:
    adapter = TTSAdapter()
    adapter._sarvam_key = "test-key"
    adapter._google_creds = None
    return adapter


def test_sarvam_tts_uses_a_valid_model_and_speaker():
    adapter = _sarvam_adapter()
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"audios": [base64.b64encode(b"mp3").decode()]}
        return response

    with patch("httpx.post", side_effect=fake_post):
        out = adapter.synthesize("namaste", "kn")

    assert out == b"mp3"
    body = captured["json"]
    # The API rejects anything outside bulbul:v2 / v3 / v4-flash.
    assert body["model"] in ("bulbul:v2", "bulbul:v3", "bulbul:v4-flash")
    # "meera" is not a valid speaker.
    assert body["speaker"] in TTSAdapter.SARVAM_SPEAKERS.values()
    assert body["target_language_code"] in ("en-IN", "hi-IN", "kn-IN")
    # pitch and loudness are only supported by bulbul:v2.
    if body["model"] != "bulbul:v2":
        assert "pitch" not in body
        assert "loudness" not in body


@pytest.mark.parametrize("language", ["en", "hi", "kn"])
def test_sarvam_tts_covers_every_supported_language(language):
    adapter = _sarvam_adapter()
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["json"] = json
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"audios": [base64.b64encode(b"mp3").decode()]}
        return response

    with patch("httpx.post", side_effect=fake_post):
        assert adapter.synthesize("hello", language) == b"mp3"
    assert captured["json"]["target_language_code"].endswith("-IN")


# ---------- Sarvam speech to text ----------

def test_sarvam_stt_posts_multipart_with_a_bcp47_language_code():
    adapter = STTAdapter()
    adapter._sarvam_key = "test-key"
    captured = {}

    def fake_post(url, headers=None, data=None, files=None, timeout=None):
        captured["url"] = url
        captured["data"] = data
        captured["files"] = files
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"transcript": "namaste"}
        return response

    with patch("httpx.post", side_effect=fake_post):
        out = adapter.transcribe(b"audio-bytes", "kn")

    assert out == "namaste"
    # The endpoint takes multipart/form-data with the raw audio.
    assert "file" in captured["files"]
    assert captured["files"]["file"][1] == b"audio-bytes"
    # Saaras, not Bulbul.
    assert captured["data"]["model"].startswith("saaras:")
    # BCP-47, not the bare language code.
    assert captured["data"]["language_code"] == "kn-IN"
