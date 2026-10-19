from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

from speech.tts_adapter import TTSAdapter


def _adapter(sarvam_key="sarvam-key", google_creds=None) -> TTSAdapter:
    adapter = TTSAdapter()
    adapter._sarvam_key = sarvam_key
    adapter._google_creds = google_creds
    return adapter


def _polly_client(audio: bytes = b"mp3") -> MagicMock:
    client = MagicMock()
    resp = MagicMock()
    resp.__getitem__.return_value.read.return_value = audio
    client.synthesize_speech.return_value = resp
    return client


def test_polly_uses_voice_for_language():
    adapter = _adapter(sarvam_key=None)
    client = _polly_client()
    adapter._polly = client
    assert adapter.synthesize("namaste", "hi") == b"mp3"
    assert client.synthesize_speech.call_args.kwargs["VoiceId"] == "Aditi"


def test_kannada_is_never_sent_to_polly():
    """Polly has no Kannada voice - it would read the text with an English voice."""
    adapter = _adapter(sarvam_key=None)
    client = _polly_client()
    adapter._polly = client
    assert adapter.synthesize("text", "kn") is None
    client.synthesize_speech.assert_not_called()


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
    adapter._polly = MagicMock()  # Polly fails -> falls through to Sarvam
    adapter._polly.synthesize_speech.side_effect = RuntimeError("polly down")
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
    assert adapter.configured_providers("en") == ["aws_polly", "sarvam", "google"]
    # Polly has no Kannada voice, so it is not offered for Kannada.
    assert adapter.configured_providers("kn") == ["sarvam", "google"]


def test_configured_providers_empty_without_credentials():
    adapter = _adapter(sarvam_key=None, google_creds=None)
    assert adapter.configured_providers("en") == ["aws_polly"]
    assert adapter.configured_providers("kn") == []
