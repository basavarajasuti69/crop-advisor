from __future__ import annotations

import base64
import io
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from apps.api.main import AdviceRequest, app

client = TestClient(app)


def _png_bytes() -> bytes:
    img = Image.new("RGB", (300, 300), color=(34, 139, 34))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _fake_predictor(calls: list[bytes]):
    """Stand-in classifier. The real weights are a build artifact (gitignored),
    so tests must not depend on a model being present."""

    def _predict(image_bytes, top_k=3, model=None, classes=None):
        calls.append(image_bytes)
        # Raises if the bytes are not a decodable image.
        Image.open(io.BytesIO(image_bytes)).verify()
        return [{"label": "healthy", "score": 0.9}]

    return _predict


def test_advice_request_decodes_base64_image_bytes():
    raw = _png_bytes()
    req = AdviceRequest(image_bytes=base64.b64encode(raw).decode())
    assert req.image_bytes == raw


def test_advice_request_accepts_raw_bytes():
    raw = _png_bytes()
    req = AdviceRequest(image_bytes=raw)
    assert req.image_bytes == raw


def test_advice_endpoint_reads_base64_image():
    """A base64 image must reach the classifier intact, not as UTF-8 garbage."""
    calls: list[bytes] = []
    with patch("agents.graph._predict_sagemaker", return_value=[]), patch(
        "agents.graph._predict_image", side_effect=_fake_predictor(calls)
    ):
        response = client.post(
            "/advice",
            json={
                "image_bytes": base64.b64encode(_png_bytes()).decode(),
                "crop": "rice",
                "language": "en",
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert len(calls) == 1
    assert body["diagnosis"]["predictions"][0]["label"] == "healthy"
    assert body["confidence"] == 0.9
    assert body["diagnosis"]["quality_ok"] is True
    assert "Cannot read this photo" not in body["advice"]


def test_advice_image_endpoint_accepts_multipart_upload():
    calls: list[bytes] = []
    with patch("agents.graph._predict_sagemaker", return_value=[]), patch(
        "agents.graph._predict_image", side_effect=_fake_predictor(calls)
    ):
        response = client.post(
            "/advice/image",
            files={"file": ("leaf.png", _png_bytes(), "image/png")},
            data={"crop": "rice", "language": "en"},
        )
    assert response.status_code == 200
    assert len(calls) == 1
    assert response.json()["confidence"] == 0.9
    assert "Cannot read this photo" not in response.json()["advice"]


def test_advice_endpoint_without_image_returns_readable_message():
    """No photo at all must not produce an empty advice string."""
    response = client.post("/advice", json={"crop": "rice", "language": "en"})
    assert response.status_code == 200
    body = response.json()
    assert body["advice"], "advice must not be empty"
    assert "upload" in body["advice"].lower()
    assert body["safe"] is False
    assert body["escalate"] is True
    assert body["diagnosis"]["quality_issue"] == "no image provided"


def test_advice_endpoint_without_image_is_localized():
    for language, marker in (("hi", "फोटो"), ("kn", "ಫೋಟೊ")):
        response = client.post("/advice", json={"crop": "rice", "language": language})
        assert response.status_code == 200
        advice = response.json()["advice"]
        assert advice, f"{language} advice must not be empty"
        assert marker in advice
        assert not any(c.isascii() and c.isalpha() for c in advice)


def test_speak_reports_configured_providers_when_unavailable():
    response = client.post("/speak", json={"text": "hello", "language": "kn"})
    assert response.status_code == 503
    assert "TTS unavailable" in response.json()["detail"]
