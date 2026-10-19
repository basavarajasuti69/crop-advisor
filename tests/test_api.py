from __future__ import annotations

import base64
import io

from fastapi.testclient import TestClient
from PIL import Image

from apps.api.main import AdviceRequest, app

client = TestClient(app)


def _png_bytes() -> bytes:
    img = Image.new("RGB", (300, 300), color=(34, 139, 34))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


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
    assert body["diagnosis"]["quality_ok"] is True
    assert body["diagnosis"]["confidence"] > 0
    assert "Cannot read this photo" not in body["advice"]


def test_advice_image_endpoint_accepts_multipart_upload():
    response = client.post(
        "/advice/image",
        files={"file": ("leaf.png", _png_bytes(), "image/png")},
        data={"crop": "rice", "language": "en"},
    )
    assert response.status_code == 200
    assert "Cannot read this photo" not in response.json()["advice"]


def test_speak_reports_configured_providers_when_unavailable():
    response = client.post("/speak", json={"text": "hello", "language": "kn"})
    assert response.status_code == 503
    assert "TTS unavailable" in response.json()["detail"]
