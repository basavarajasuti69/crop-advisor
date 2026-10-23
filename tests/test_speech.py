from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

from speech.tts_adapter import TTSAdapter


def _adapter(sarvam_key="sarvam-key", google_creds=None) -> TTSAdapter:
    adapter = TTSAdapter()
    adapter._sarvam_key = sarvam_key
    adapter._google_creds = google_creds
    return adapter


def test_sarvam_receives_kn_in_language_code():
    adapter = _adapter()
    with patch("httpx.post") as post:
        post.return_value = MagicMock(status_code=200)
        post.return_value.json.return_value = {"audios": [base64.b64encode(b"kn-mp3").decode()]}
        out = adapter.synthesize("text", "kn")
    assert out == b"kn-mp3"
    assert post.call_args.kwargs["json"]["target_language_code"] == "kn-IN"


def test_sarvam_receives_hi_in_language_code():
    adapter = _adapter()
    with patch("httpx.post") as post:
        post.return_value = MagicMock(status_code=200)
        post.return_value.json.return_value = {"audios": [base64.b64encode(b"hi-mp3").decode()]}
        out = adapter.synthesize("text", "hi")
    assert out == b"hi-mp3"
    assert post.call_args.kwargs["json"]["target_language_code"] == "hi-IN"


def test_google_receives_kn_in_language_code():
    adapter = _adapter(sarvam_key=None, google_creds="creds.json")
    fake_tts = MagicMock()
    fake_tts.TextToSpeechClient.return_value.synthesize_speech.return_value.audio_content = b"kn-mp3"
    with patch.dict("sys.modules", {"google": MagicMock(), "google.cloud": MagicMock()}):
        with patch("google.cloud.texttospeech", fake_tts, create=True):
            out = adapter.synthesize("text", "kn")
    assert out == b"kn-mp3"
    voice_kwargs = fake_tts.VoiceSelectionParams.call_args.kwargs
    assert voice_kwargs["language_code"] == "kn-IN"


def test_configured_providers_lists_sarvam_and_google():
    adapter = _adapter(sarvam_key="k", google_creds="g")
    assert adapter.configured_providers("en") == ["sarvam", "google"]
    assert adapter.configured_providers("kn") == ["sarvam", "google"]


def test_configured_providers_empty_without_credentials():
    adapter = _adapter(sarvam_key=None, google_creds=None)
    assert adapter.configured_providers("en") == []
    assert adapter.configured_providers("kn") == []
